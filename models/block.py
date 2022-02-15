
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from .layers import PixelShuffle_ICNR 

def crop_combine(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x2 = F.pad(x2, [
        -diffX//2, -diffX//2,
        -diffY//2, -diffY//2
    ])
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x2 = F.pad(x2, [
        -diffX, 0,
        -diffY, 0
    ])
    return x2
class VGG16Block(nn.Module):
    def __init__(self, list_channels, batch_norm = True, padding = 1):
        super().__init__()
        assert len(list_channels) > 1

        list_layers = []
        for i in range(len(list_channels) - 1):
            input_channel = list_channels[i]
            output_channel = list_channels[i+1]
            list_layers.append(nn.Conv2d(input_channel, output_channel, 3, padding = padding))
            if batch_norm:
                list_layers.append(nn.BatchNorm2d(output_channel))
            list_layers.append(nn.ReLU(inplace=True))
        self.multi_conv = nn.Sequential(*list_layers)
        
    def forward(self, x):
        return self.multi_conv(x)
class Out(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.out_conv = nn.Conv2d(input_channel, output_channel, kernel_size = 1)

    def forward(self, x):
        return self.out_conv(x)

class UpBlock(nn.Module):
    def __init__(self, input_channel_encoder,
                       input_channel_decoder,
                       output_channel,
                       batch_norm = True, 
                       padding = 1,
                       bilinear = True,
                       pixel_shuffle = False,
                       middle_channel = None,
                       num_classes = 0):
        super(UpBlock, self).__init__()
        self.input_channel_encoder = input_channel_encoder
        self.input_channel_decoder = input_channel_decoder
        self.output_channel = output_channel

        self.conv_encoder = nn.Conv2d(input_channel_encoder, output_channel, kernel_size=1, stride=1)
        self.conv_decoder = nn.Conv2d(input_channel_decoder, output_channel, kernel_size=1, stride=1)
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2,
                            mode = "bilinear",
                            align_corners=True),
            )
        else:
            if pixel_shuffle:
                self.up = nn.Sequential(
                    PixelShuffle_ICNR(output_channel)
                )
            else:
                self.up = nn.ConvTranspose2d(output_channel, 
                                        output_channel,
                                        kernel_size = 2,
                                        stride = 2)
        self.conv_block = VGG16Block([output_channel*2, output_channel, output_channel], batch_norm, padding)
    def forward(self, x1, x2):
        if self.input_channel_decoder != self.output_channel:
            x1 = self.conv_decoder(x1)
        if self.input_channel_encoder != self.output_channel:
            x2 = self.conv_encoder(x2)
        x1 = self.up(x1)
        x =  torch.cat([x2, x1], dim = 1)
        return self.conv_block(x)

class UpLayer(nn.Module):
    def __init__(self, input_channel, output_channel, bilinear = True,  kernel_size =3, stride = 2, padding = 1, pixel_shuffle = False):
        super(UpLayer, self).__init__()
        if bilinear:
            self.up_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride,
                            mode = "bilinear",
                            align_corners=True),
                nn.Conv2d(input_channel, output_channel, kernel_size = kernel_size, padding = padding)
            )
        else:
            if pixel_shuffle:
                self.up_layer = nn.Sequential(
                    nn.Conv2d(input_channel, output_channel, kernel_size = 1),
                    PixelShuffle_ICNR(output_channel)
                )
            else:
                self.up_layer = nn.ConvTranspose2d(input_channel, 
                                                input_channel,
                                                kernel_size = 2,
                                                stride = stride)
    def forward(self, x):
        return self.up_layer(x)

class Resnet18Block(nn.Module):
    def __init__(self, input_channel, output_channel, up_sample = False, padding = 1, bilinear = True, pixel_shuffle = False):
        super(Resnet18Block, self).__init__()
        self.bilinear = bilinear
        self.up_sample = up_sample
        if up_sample:
            self.conv1 = UpLayer(input_channel, output_channel, kernel_size=3, padding= padding, bilinear = bilinear, pixel_shuffle = pixel_shuffle)
            self.up = nn.Sequential(
                UpLayer(input_channel, output_channel, kernel_size=1, padding= 0, bilinear = bilinear, pixel_shuffle = pixel_shuffle),
                nn.BatchNorm2d(output_channel)
            )
        else:
            self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding = padding)
            self.up  = nn.Conv2d(input_channel, output_channel, 3, padding = padding)           
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(output_channel, output_channel, 3, padding= padding)
        self.bn2 = nn.BatchNorm2d(output_channel)
    def forward(self, x):
        residual = self.up(x)
        # print(residual.shape,x.shape)
        x = self.relu(self.bn1(self.conv1(x)))
        # print(residual.shape,x.shape)
        x = self.bn2(self.conv2(x))
        x = torch.add(residual, x)
        # print(residual.shape,x.shape)
        return self.relu(x)

class Resnet18BlocksUp(nn.Module):
    def __init__(self, input_channel, output_channel, padding = 1, bilinear = True, pixel_shuffle = False):
        super(Resnet18BlocksUp, self).__init__()
        self.block1 = Resnet18Block(input_channel, output_channel, up_sample = True, padding = 1, bilinear = bilinear, pixel_shuffle = pixel_shuffle)
        self.block2 = Resnet18Block(output_channel*2, output_channel, up_sample = False, padding = 1, bilinear = bilinear, pixel_shuffle = pixel_shuffle)
    def forward(self, x1, x2):
        x1 = self.block1(x1)
        x2 = crop_combine(x1, x2)
        x = torch.cat([x1, x2], 1)
        return self.block2(x)


class Resnet101Bottleneck(nn.Module):
    def __init__(self, input_channel, middle_channel, output_channel, 
                       up_sample = False, 
                       padding = 1, 
                       bilinear = True):
        super(Resnet101Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channel, middle_channel, kernel_size = 1, stride= 1)
        if up_sample:
            self.up = nn.Sequential(
                UpLayer(input_channel, output_channel, kernel_size =3, stride = 2, padding = padding),
                nn.BatchNorm2d(output_channel))
        else:
            self.up = nn.Conv2d(input_channel, output_channel, kernel_size = 1, stride = 1)
        self.bn1 = nn.BatchNorm2d(middle_channel)

        if up_sample:
            self.conv2 = UpLayer(middle_channel, middle_channel, kernel_size =3, stride = 2, padding = padding)
        else:
            self.conv2 = nn.Conv2d(middle_channel, middle_channel, kernel_size = 3, stride= 1, padding = padding)
        
        self.bn2 = nn.BatchNorm2d(middle_channel)
        self.conv3 = nn.Conv2d(middle_channel, output_channel, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.up(x)
        # print(residual.shape)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x+residual
        return self.relu(x)

class Resnet101BlockUp(nn.Module):
    def __init__(self, input_channel, output_channel, 
                       last_block = False,
                       num_bottleneck = 2, 
                       padding = 1, 
                       bilinear = True):
        super(Resnet101BlockUp, self).__init__()
        list_bottlenecks = []
        if last_block == True:
            self.up_bottleneck = UpLayer(input_channel, output_channel)
            list_bottlenecks.append(nn.Conv2d(output_channel*2, output_channel, kernel_size=1, stride=1))
        else:
            self.up_bottleneck = Resnet101Bottleneck(input_channel, input_channel//4, output_channel, up_sample=True)
            bottleneck = Resnet101Bottleneck(input_channel, output_channel//4, output_channel, padding = padding)
            list_bottlenecks.append(bottleneck)
            for i in range(num_bottleneck-1):
                bottleneck = Resnet101Bottleneck(output_channel, output_channel//4, output_channel, padding = padding)
                list_bottlenecks.append(bottleneck)
        self.blocks = nn.Sequential(*list_bottlenecks)
    def forward(self, x1, x2):
        x1 = self.up_bottleneck(x1)
        x2 = crop_combine(x1, x2)
        x = torch.cat([x1, x2], 1)
        return self.blocks(x)
            