import torch.nn as nn
from torchvision.models import resnet18, resnet101, densenet161, densenet121, resnet34, vgg16

from .block import VGG16Block, UpBlock, Out
from .block import Resnet18BlocksUp, UpLayer
from .block import Resnet101BlockUp

import sys
from .layers import PixelShuffle_ICNR 

import timm
class Backbone:
    def __init__(self, encoder_args, decoder_args):
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args

        self.base_model = None
        self.features_name = None
        self.last_layer = None
        self.input_channle = None
        self.up_class = None
        self.out_conv_class = None
        self.list_encoder_channel = None
        self.list_decoder_channel = [512, 256, 128, 64]

    def initial_decoder(self):
        self.list_decoder_channel = [self.encoder_output] + self.list_decoder_channel
        self.blocks = nn.ModuleList()
        for i in range(len(self.list_encoder_channel)):
            input_channel_encoder = self.list_encoder_channel[i]
            input_channel_decoder = self.list_decoder_channel[i]
            output_channel = self.list_decoder_channel[i+1]
            up_block = self.up_class(input_channel_encoder,
                                     input_channel_decoder,
                                     output_channel,
                                     **self.decoder_args)
            self.blocks.append(up_block)
        self.out_conv = self.out_conv_class(self.list_decoder_channel[-1], 1)

class BackboneOriginal(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneOriginal, self).__init__(encoder_args, decoder_args)        
        self.base_model = vgg16(**encoder_args).features  
        self.features_name = ["22", "15", "8", "3"]
        self.last_layer = "29"
        self.input_channel = 3
        self.encoder_output = 512
        self.list_encoder_channel = [512, 256, 128, 64]
        self.up_class = UpBlock
        self.out_conv_class = Out
        self.initial_decoder()

class BackboneResnet18VGG(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneResnet18VGG, self).__init__(encoder_args, decoder_args)        
        self.base_model = resnet18(**encoder_args)
        self.features_name = ["layer3", "layer2", "layer1","relu"]
        self.last_layer = "layer4"
        self.input_channel = 3
        self.encoder_output = 512
        self.list_encoder_channel = [256, 128, 64, 64]
        self.up_class = UpBlock
        self.out_conv_class = Out
        self.initial_decoder()
        self.out_conv = nn.Sequential(
            PixelShuffle_ICNR(64),
            nn.Conv2d(64, 1, kernel_size = 1, stride = 1)
        )
class BackboneResnet34VGG(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneResnet34VGG, self).__init__(encoder_args, decoder_args)
        self.base_model = resnet34(**encoder_args)
        self.features_name = ["layer3", "layer2", "layer1","relu"]
        self.last_layer = "layer4"
        self.input_channel = 3
        self.encoder_output = 512
        self.list_encoder_channel = [256, 128, 64, 64]
        self.up_class = UpBlock
        self.out_conv_class = Out
        self.initial_decoder()
        self.out_conv = nn.Sequential(
            PixelShuffle_ICNR(64),
            nn.Conv2d(64, 1, kernel_size = 1, stride = 1)
        )

class BackboneResnet101VGG(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneResnet101VGG, self).__init__(encoder_args, decoder_args)
        self.base_model = resnet101(**encoder_args)
        self.features_name = ["layer3", "layer2", "layer1","relu"]
        self.last_layer = "layer4"
        self.input_channel = 3
        self.encoder_output = 2048
        self.list_encoder_channel = [1024, 512, 256, 64]
        self.up_class = UpBlock
        self.out_conv_class = Out
        self.initial_decoder()
        self.out_conv = nn.Sequential(
            PixelShuffle_ICNR(64),
            nn.Conv2d(64, 1, kernel_size = 1, stride = 1)
        )

class BackboneDensenet161VGG(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneDensenet161VGG, self).__init__(encoder_args, decoder_args)        
        self.base_model = densenet161(**encoder_args).features
        self.features_name = ["denseblock3","denseblock2","denseblock1","relu0"]
        self.last_layer = "denseblock4"
        self.input_channel = 3
        self.encoder_output = 2208
        self.list_encoder_channel = [2112, 768, 384, 96]
        self.up_class = UpBlock
        self.out_conv_class = Out
        self.initial_decoder()
        self.out_conv = nn.Sequential(
            PixelShuffle_ICNR(64),
            nn.Conv2d(64, 1, kernel_size = 1, stride = 1)
        )
class BackboneDensenet121VGG(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneDensenet121VGG, self).__init__(encoder_args, decoder_args)        
        self.base_model = densenet121(**encoder_args).features
        self.features_name = ["denseblock3","denseblock2","denseblock1","relu0"]
        self.last_layer = "denseblock4"
        self.input_channel = 3
        self.encoder_output = 1024
        self.list_encoder_channel = [1024, 512, 256, 64]
        self.up_class = UpBlock
        self.out_conv_class = Out
        self.initial_decoder()
        try:
            num_classes = decoder_args["num_classes"]
        except:
            num_classes = 1
        self.out_conv = nn.Sequential(
            PixelShuffle_ICNR(64),
            nn.Conv2d(64, num_classes, kernel_size = 1, stride = 1)
        )
from torchvision.models import resnext101_32x8d
import torch
class BackboneResnext101VGG(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneResnext101VGG, self).__init__(encoder_args, decoder_args)
        if encoder_args["pretrained"] == "Instagram":   
            self.base_model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        else:
            self.base_model = resnext101_32x8d(pretrained=True)
        self.features_name = ["layer3","layer2","layer1","relu"]
        self.last_layer = "layer4"
        self.input_channel = 3
        self.encoder_output = 2048
        self.list_encoder_channel = [1024, 512, 256, 64]
        self.up_class = UpBlock
        self.out_conv_class = Out
        self.initial_decoder()
        self.out_conv = nn.Sequential(
            PixelShuffle_ICNR(64),
            nn.Conv2d(64, 1, kernel_size = 1, stride = 1)
        )
class BackboneEfficientB0VGG(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneEfficientB0VGG, self).__init__(encoder_args, decoder_args)        
        self.create_model(**encoder_args)
        self.features_name = ["blocks_4","blocks_2","blocks_1","act1"]
        self.last_layer = "act2"
        self.input_channel = 3
        self.encoder_output = 1280
        self.list_encoder_channel = [112, 40, 24, 32]
        self.up_class = UpBlock
        self.out_conv_class = Out
        self.initial_decoder()
        self.out_conv = nn.Sequential(
            PixelShuffle_ICNR(64),
            nn.Conv2d(64, 1, kernel_size = 1, stride = 1)
        )
    def create_model(self, **encoder_args):
        m = timm.create_model("efficientnet_b0", **encoder_args)
        list_layers = {}
        for name, child in m.named_children():
            i = 0
            for name_, child_ in child.named_children():
                list_layers[name+"_"+name_]=child_
                i+=1
            if i==0:
                list_layers[name]=child
        self.base_model = nn.ModuleDict(list_layers)

class BackboneEfficientB7VGG(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackboneEfficientB7VGG, self).__init__(encoder_args, decoder_args)        
        self.create_model(**encoder_args)
        self.features_name = ["blocks_4","blocks_2","blocks_1","act1"]
        self.last_layer = "act2"
        self.input_channel = 3
        self.encoder_output = 2560
        self.list_encoder_channel = [224, 80, 48, 64]
        self.up_class = UpBlock
        self.out_conv_class = Out
        self.initial_decoder()
        self.out_conv = nn.Sequential(
            PixelShuffle_ICNR(64),
            nn.Conv2d(64, 1, kernel_size = 1, stride = 1)
        )
    def create_model(self, **encoder_args):
        type_ = encoder_args["type"]
        encoder_args.pop("type")
        if type_ == "ns":
            m = timm.create_model("tf_efficientnet_b7_ns", **encoder_args)
        else:
            m = timm.create_model("tf_efficientnet_b7", **encoder_args)
        list_layers = {}
        for name, child in m.named_children():
            i = 0
            for name_, child_ in child.named_children():
                list_layers[name+"_"+name_]=child_
                i+=1
            if i==0:
                list_layers[name]=child
        self.base_model = nn.ModuleDict(list_layers)

class BackBoneResnet18(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackBoneResnet18, self).__init__(encoder_args, decoder_args)
        self.base_model = resnet18(**encoder_args)
        self.features_name = ["layer3", "layer2", "layer1", "relu"]
        self.last_layer = "layer4"
        self.input_channel = 3
        self.list_channels = [512, 256, 128, 64, 64, 1]
        self.up_class = Resnet18BlocksUp
        self.out_conv_class = UpLayer
        self.initial_decoder()


class BackBoneResnet101(Backbone):
    def __init__(self, encoder_args, decoder_args):
        super(BackBoneResnet101, self).__init__(encoder_args, decoder_args)
        self.encoder_args = encoder_args
        self.base_model = resnet101(**encoder_args)
        self.features_name = ["layer3", "layer2", "layer1", "relu"]
        self.last_layer = "layer4"
        self.input_channel = 3
        self.list_channels = [2048, 1024, 512, 256, 64, 1]
        self.num_bottleneck = [23, 4, 3]
        self.up_class = Resnet101BlockUp
        self.out_conv_class = UpLayer
        self.initial_decoder()

    def initial_decoder(self):
        list_channels = self.list_channels
        assert len(list_channels) > 3
        self.blocks = nn.ModuleList()
        for i in range(len(list_channels)-3):
            input_channel = list_channels[i]
            output_channel = list_channels[i+1]
            up_block = self.up_class(input_channel,
                                     output_channel,
                                     num_bottleneck = self.num_bottleneck[i],
                                     **self.decoder_args)
            self.blocks.append(up_block)
        self.blocks.append(
            self.up_class(
                list_channels[-3], list_channels[-2],
                last_block=True
            )
        )
        self.out_conv = self.out_conv_class(list_channels[-2], list_channels[-1])