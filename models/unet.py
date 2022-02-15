import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import BackboneResnet18VGG
from .block import UpBlock, VGG16Block, Out
import sys
from .layers import PixelShuffle_ICNR, PooledSelfAttention2d

class Unet(nn.Module):
    def __init__(self, 
                backbone_class = BackboneResnet18VGG,
                encoder_args = {},
                decoder_args = {},
                pretrained = False, **kwargs):
        super(Unet, self).__init__()

        self.backbone = backbone_class(encoder_args, decoder_args)
        self.base_model = self.backbone.base_model
        self.features_name = self.backbone.features_name
        self.blocks = self.backbone.blocks
        self.out_conv = self.backbone.out_conv

    def forward(self, x):
        x, features_value = self.forward_backbone(x)
        # print(self.features_name)
        for i, block in enumerate(self.blocks):
            name = self.features_name[i]
            x = block(x, features_value[name])
            # print(name, x.shape)
        x = self.out_conv(x)
        return x

    def forward_backbone(self, x):
        features_value = {}
        features_value["x"] = x
        if x.shape[1] != self.backbone.input_channel:
            if self.backbone.input_channel == 3:
                x = torch.cat([x, x, x], 1)
        for name, child in self.base_model.named_children():
            x = child(x)
            # print(name, x.shape)
            if name in self.features_name:
                features_value[name] = x
            if name == self.backbone.last_layer:
                break
        return x, features_value

class UnetAdaptation(Unet):
    def __init__(self, 
            backbone_class = BackboneResnet18VGG,
            encoder_args = {},
            decoder_args = {},
            pretrained = False,
            ds = False):
        super(UnetAdaptation, self).__init__(backbone_class, encoder_args, decoder_args, pretrained)
        self.ds = ds
        if pretrained is not None:
            state_dict = torch.load(pretrained)["net"]
            self.load_state_dict(state_dict)
    def forward(self, x):
        x, features_value = self.forward_backbone(x)
        if self.ds == False:
            return x
        else:
            for i, block in enumerate(self.blocks):
                name = self.features_name[i]
                x = block(x, features_value[name])
                # print(name, x.shape)
            y = self.out_conv(x)
            
            return x, y


from .ocr import SpatialGather_Module, SpatialOCR_Module
from torch.nn import BatchNorm2d
BN_MOMENTUM = 0.01

class UnetOCR(nn.Module):
    def __init__(self, 
                classes,
                ocr_mid_channels,
                ocr_key_channels,

                backbone_class = BackboneResnet18VGG,
                encoder_args = {},
                decoder_args = {},
                pretrained = False):
        super(UnetOCR, self).__init__()

        self.backbone = backbone_class(encoder_args, decoder_args)
        self.base_model = self.backbone.base_model
        self.features_name = self.backbone.features_name
        self.blocks = self.backbone.blocks
        self.out_conv = self.backbone.out_conv[:-1]

        self.classes = classes
        last_inp_channels = self.backbone.list_decoder_channel[-1]
        # Define the auxiliary conv layer to provide the soft object regions
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels,
                      out_channels=last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels,
                      out_channels=self.classes,
                      kernel_size=1, stride=1, padding=0)
        )

        # Define the conv layers for computing the pixel representations
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(last_inp_channels, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
        )

        self.ocr_gather_head = SpatialGather_Module(cls_num=classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1, dropout=0.05)
        # Define the conv layers for computing the final segmentation map
        self.cls_head = nn.Conv2d(ocr_mid_channels, classes, kernel_size=1,
                                  stride=1, padding=0, bias=True)

    def forward(self, x):
        x, features_value = self.forward_backbone(x)
        # print(self.features_name)
        for i, block in enumerate(self.blocks):
            name = self.features_name[i]
            x = block(x, features_value[name])
            # print(name, x.shape)
        x = self.out_conv(x)
#         import pdb; pdb.set_trace()

        out_aux = self.aux_head(x)

        feats = self.conv3x3_ocr(x)
        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)
        out = self.cls_head(feats)
        if self.training:
            return out_aux, out
        else:
            return out

    def forward_backbone(self, x):
        features_value = {}
        features_value["x"] = x
        if x.shape[1] != self.backbone.input_channel:
            x = torch.cat([x, x, x], 1)
        for name, child in self.base_model.named_children():
            x = child(x)
            # print(name, x.shape)
            if name in self.features_name:
                features_value[name] = x
            if name == self.backbone.last_layer:
                break
        return x, features_value
