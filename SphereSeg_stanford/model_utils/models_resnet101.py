import torch
import torch.nn as nn
from model_utils.padding import *
from model_utils.pooling import pooling, pooling_old
from model_utils.unpooling import unpooling
from model_utils.convolution import PHDConv2d
import torch.nn.functional as F
import matplotlib.pyplot as plt


def feature_plot(x0, cnt_img=40):
    feature_list = torch.Tensor([])
    for k in range(x0.shape[1]//cnt_img):
        for i in range(k*cnt_img,k*cnt_img+cnt_img):
            if i==k*cnt_img:
                feature = torch.cat((x0[0,i].cpu(),x0[1,i].cpu(),x0[2,i].cpu(),x0[3,i].cpu(),x0[4,i].cpu()),dim=1)
            else :
                tmp = torch.cat((x0[0,i].cpu(),x0[1,i].cpu(),x0[2,i].cpu(),x0[3,i].cpu(),x0[4,i].cpu()),dim=1)
                feature = torch.cat((feature,tmp),dim=1)
        if k==0:
            feature_list = feature
        else:
            feature_list = torch.cat((feature_list,feature),dim=0)
    return feature_list



class PHDnet_resNet(nn.Module):
    def __init__(self):
        super(PHDnet_resNet, self).__init__()

        self.pooling = pooling()
        self.unpooling = unpooling()

        self.initblock = InitialBlock(3, 64)

        self.resblock_enc1_1 = ResBlock_enc(64, 64, 256, initial = True)
        self.resblock_enc1_2 = ResBlock_enc(256, 64, 256)
        self.resblock_enc1_3 = ResBlock_enc(256, 64, 256)


        self.resblock_enc2_1 = ResBlock_enc_pooling(256, 128, 512)
        self.resblock_enc2_2 = ResBlock_enc(512, 128, 512)
        self.resblock_enc2_3 = ResBlock_enc(512, 128, 512)
        self.resblock_enc2_4 = ResBlock_enc(512, 128, 512)

        self.resblock_enc3_1 = ResBlock_enc_pooling(512, 256, 1024)
        self.resblock_enc3_2 = ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_3 = ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_4 = ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_5 = ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_6 = ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_7 = ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_8 = ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_9 = ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_10= ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_11= ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_12= ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_13= ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_14= ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_15= ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_16= ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_17= ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_18= ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_19= ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_20= ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_21= ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_22= ResBlock_enc(1024, 256, 1024)
        self.resblock_enc3_23= ResBlock_enc(1024, 256, 1024)


        self.resblock_enc4_1 = ResBlock_enc_pooling(1024, 512, 2048)
        self.resblock_enc4_2 = ResBlock_enc(2048, 512, 2048)
        self.resblock_enc4_3 = ResBlock_enc(2048, 512, 2048)

        self.resblock_dec1 = ResBlock_dec(2048, 256)
        self.resblock_dec2 = ResBlock_dec(256,128)
        self.resblock_dec3 = ResBlock_dec(128,64)
        self.resblock_dec4 = ResBlock_dec(64,32)
        self.resblock_dec5 = ResBlock_dec(32, 13, Final = True)
        






    def forward(self, input, level):       #input.size = [batch* 5, c, h ,w]  level=7


        x0 = input
        x0 = self.initblock(x0, level)
        x0 = self.pooling(x0, level)
        x0 = self.pooling(x0, level-1)
        

        x0 = self.resblock_enc1_1(x0, level-2)
        x0 = self.resblock_enc1_2(x0, level-2)
        x0 = self.resblock_enc1_3(x0, level-2)

        x0 = self.resblock_enc2_1(x0, level-2)
        x0 = self.resblock_enc2_2(x0, level-3)
        x0 = self.resblock_enc2_3(x0, level-3)
        x0 = self.resblock_enc2_4(x0, level-3)

        x0 = self.resblock_enc3_1(x0, level-3)
        x0 = self.resblock_enc3_2(x0, level-4)
        x0 = self.resblock_enc3_3(x0, level-4)
        x0 = self.resblock_enc3_4(x0, level-4)
        x0 = self.resblock_enc3_5(x0, level-4)
        x0 = self.resblock_enc3_6(x0, level-4)
        x0 = self.resblock_enc3_7(x0, level-4)
        x0 = self.resblock_enc3_8(x0, level-4)
        x0 = self.resblock_enc3_9(x0, level-4)
        x0 = self.resblock_enc3_10(x0, level-4)
        x0 = self.resblock_enc3_11(x0, level-4)
        x0 = self.resblock_enc3_12(x0, level-4)
        x0 = self.resblock_enc3_13(x0, level-4)
        x0 = self.resblock_enc3_14(x0, level-4)
        x0 = self.resblock_enc3_15(x0, level-4)
        x0 = self.resblock_enc3_16(x0, level-4)
        x0 = self.resblock_enc3_17(x0, level-4)
        x0 = self.resblock_enc3_18(x0, level-4)
        x0 = self.resblock_enc3_19(x0, level-4)
        x0 = self.resblock_enc3_20(x0, level-4)
        x0 = self.resblock_enc3_21(x0, level-4)
        x0 = self.resblock_enc3_22(x0, level-4)
        x0 = self.resblock_enc3_23(x0, level-4)


        x0 = self.resblock_enc4_1(x0, level-4)
        x0 = self.resblock_enc4_2(x0, level-5)
        x0 = self.resblock_enc4_3(x0, level-5)

        x0 = self.resblock_dec1(x0, level-5)
        x0 = self.resblock_dec2(x0, level-4)
        x0 = self.resblock_dec3(x0, level-3)
        x0 = self.resblock_dec4(x0, level-2)
        x0 = self.resblock_dec5(x0, level-1)

        # import pdb;pdb.set_trace()


        return x0





class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()


        self.conv1 = PHDConv2d(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = PHDConv2d(in_channels=out_channels, out_channels=out_channels)

        self.batchnorm1 = nn.BatchNorm2d(out_channels)
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



class ResBlock_enc(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, initial = False):
        super(ResBlock_enc, self).__init__()

        self.initial = initial

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels,kernel_size = 1)
        self.conv2 = PHDConv2d(in_channels=hidden_channels, out_channels=hidden_channels)
        self.conv3 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels,kernel_size = 1)
        
        if self.initial == True:
            self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size = 1)
            self.batchnorm4 = nn.BatchNorm2d(out_channels)
        
        self.batchnorm1 = nn.BatchNorm2d(hidden_channels)
        self.batchnorm2 = nn.BatchNorm2d(hidden_channels)
        self.batchnorm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.padding = padding()

    def forward(self, x, level):

        identity = x.clone()
        x1 = x
        x2 = self.conv1(x1)
        x3 = self.batchnorm1(x2)
        x4 = self.relu(x3)

        x0 = x4
        x1 = self.padding.get_padding(x0, level)
        x2 = self.conv2(x1)
        x3 = self.batchnorm2(x2)
        x4= self.relu(x3)

        x1 = x4
        x2 = self.conv3(x1)
        x3 = self.batchnorm3(x2)

        if self.initial == True:
            identity = self.conv4(identity)
            identity = self.batchnorm4(identity)

        x3 = x3 + identity
        x4 = self.relu(x3)

        # plt.imsave('./feature_img/identity_level'+str(level)+'3output_.png',feature_plot(x4,cnt_img=60))

        return x4




class ResBlock_enc_pooling(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ResBlock_enc_pooling, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels,kernel_size = 1)
        self.conv2 = PHDConv2d(in_channels=hidden_channels, out_channels=hidden_channels)
        self.conv3 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels,kernel_size = 1)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size = 1)
        
        
        self.batchnorm1 = nn.BatchNorm2d(hidden_channels)
        self.batchnorm2 = nn.BatchNorm2d(hidden_channels)
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        self.batchnorm4 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.padding = padding()
        self.pooling = pooling()

    def forward(self, x, level):


        identity = x
        identity = self.conv4(identity)
        identity = self.pooling(identity, level)
        identity = self.batchnorm4(identity)



        x1 = x
        x2 = self.conv1(x1)
        x2 = self.pooling(x2, level)
        x3 = self.batchnorm1(x2)
        x4 = self.relu(x3)

        x0 = x4
        x1 = self.padding.get_padding(x0, level-1)
        x2 = self.conv2(x1)
        x3 = self.batchnorm2(x2)
        x4= self.relu(x3)

        x1 = x4
        x2 = self.conv3(x1)
        x3 = self.batchnorm3(x2)


        x3 = x3 + identity
        x4 = self.relu(x3)

        return x4




class ResBlock_dec(nn.Module):
    def __init__(self, in_channels, out_channels, Final=False):
        super(ResBlock_dec, self).__init__()

        self.conv1 = PHDConv2d(in_channels=in_channels, out_channels=out_channels)

        if Final==False :
            self.conv2 =  PHDConv2d(in_channels=out_channels, out_channels=out_channels)
        else:
            self.conv2 = PHDConv2d(in_channels=out_channels, out_channels=out_channels, bias=True)


        self.conv3 = PHDConv2d(in_channels=in_channels, out_channels=out_channels)
        
        
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        self.padding = padding()
        self.unpooling = unpooling()
        self.Final = Final

    def forward(self, x, level):

        x = self.unpooling(x, level)
        identity = x.clone()
        x0 = x
        x1 = self.padding.get_padding(x0, level+1)
        x2 = self.conv1(x1)
        x3 = self.batchnorm1(x2)
        x4 = self.relu(x3)

        x0 = x4
        x1 = self.padding.get_padding(x0, level+1)
        x2 = self.conv2(x1)

        if self.Final == False :
            pass
        else :
            return x2

        x3 = self.batchnorm2(x2)


        identity = self.padding.get_padding(identity, level+1)
        identity = self.conv3(identity)
        identity = self.batchnorm3(identity)


        x3 = x3 + identity
        x4 = self.relu(x3)        


        return x4
