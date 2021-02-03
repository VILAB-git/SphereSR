import torch
import torch.nn as nn
from model_utils.padding import *
from model_utils.pooling import pooling, pooling_old
from model_utils.unpooling import unpooling
from model_utils.convolution import PHDConv2d
import torch.nn.functional as F



class PHDnet_resNet(nn.Module):
    def __init__(self):
        super(PHDnet_resNet, self).__init__()
        self.pooling = pooling()
        self.unpooling = unpooling()

        self.block_enc1 = ResBlock_enc(3, 6, 30)
        self.block_enc2 = ResBlock_enc(30, 32, 64)
        self.block_enc3 = ResBlock_enc(64, 128, 256)
        self.block_enc4 = ResBlock_enc(256, 512, 512)
        self.block_enc5 = ResBlock_enc(512, 512, 1024)

        self.block_dec1 = ResBlock_dec(1024,512,512)
        self.block_dec2 = ResBlock_dec(512, 512, 256)
        self.block_dec3 = ResBlock_dec(256,128,64)
        self.block_dec4 = ResBlock_dec(64,32,30)
        self.block_dec5 = ResBlock_dec(30,6, 1, Final=True)


    def forward(self, input, level):       #input.size = [batch* 5, c, h ,w]  level=7

        x0 = input
        x0 = self.block_enc1(x0, level)
        identity1 = x0
        x0 = self.pooling(x0, level)
        x0 = self.block_enc2(x0, level-1)
        identity2 = x0
        x0 = self.pooling(x0, level-1)
        x0 = self.block_enc3(x0, level-2)
        identity3 = x0
        x0 = self.pooling(x0, level-2)
        x0 = self.block_enc4(x0, level-3)
        identity4 = x0
        x0 = self.pooling(x0, level-3)
        x0 = self.block_enc5(x0, level-4)
        identity5 = x0
        x0 = self.pooling(x0, level-4)

        x0 = self.unpooling(x0, level-5)
        x0 = x0 + identity5
        x0 = self.block_dec1(x0, level-4)
        x0 = self.unpooling(x0, level-4)
        x0 = x0 + identity4
        x0 = self.block_dec2(x0, level-3)
        x0 = self.unpooling(x0, level-3)
        x0 = x0 + identity3
        x0 = self.block_dec3(x0, level-2)
        x0 = self.unpooling(x0, level-2)
        x0 = x0 + identity2
        x0 = self.block_dec4(x0, level-1)
        x0 = self.unpooling(x0, level-1)
        x0 = x0 + identity1
        x0 = self.block_dec5(x0, level)

        

        return x0


class ResBlock_enc(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ResBlock_enc, self).__init__()

        self.conv1 = PHDConv2d(in_channels=in_channels, out_channels=hidden_channels)
        self.conv2 = PHDConv2d(in_channels=hidden_channels, out_channels=out_channels)

        self.batchnorm1 = nn.BatchNorm2d(hidden_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.padding = padding()

    def forward(self, x, level):


        x0 = x
        x1 = self.padding.get_padding(x0, level)
        x2 = self.conv1(x1)
        x3 = self.batchnorm1(x2)
        x4 = self.relu(x3)

        x0 = x4
        x1 = self.padding.get_padding(x0, level)
        x2 = self.conv2(x1)
        x3 = self.batchnorm2(x2)
        x4 = self.relu(x3)


        return x4


class ResBlock_dec(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, Final=False):
        super(ResBlock_dec, self).__init__()

        self.conv1 = PHDConv2d(in_channels=in_channels, out_channels=hidden_channels)

        if Final==False :
            self.conv2 =  PHDConv2d(in_channels=hidden_channels, out_channels=out_channels)
        else:
            self.conv2 = PHDConv2d(in_channels=hidden_channels, out_channels=out_channels, bias=True)
        
        self.batchnorm1 = nn.BatchNorm2d(hidden_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        self.padding = padding()
        self.Final = Final

    def forward(self, x, level):


        x0 = x
        x1 = self.padding.get_padding(x0, level)
        x2 = self.conv1(x1)
        x3 = self.batchnorm1(x2)
        x4 = self.relu(x3)

        x0 = x4
        x1 = self.padding.get_padding(x0, level)
        x2 = self.conv2(x1)

        if self.Final == False :
            x3 = self.batchnorm2(x2)
            x4 = self.relu(x3)
        else :
            x4 = x2


        return x4
