import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
from torch.nn.modules.utils import _pair



class PHDConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):

        super(PHDConv2d, self).__init__()

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = _pair(stride)

        # weight
        weight_trainable = torch.Tensor(out_channels, in_channels, 10) #.to(self.device)
        nn.init.kaiming_uniform_(weight_trainable, mode='fan_in', nonlinearity='relu')
        self.weight = Parameter(weight_trainable)  # adding zero

        #         self._w1_index = [
        #             [0, 1, -],
        #             [-, 2, -],
        #             [3, 4, 5],
        #             [-, 6, 7],
        #             [-, 8, 9]
        #         ]
        #         self._w2_index = [
        #             [ 9, 8, -],
        #             [ 7, 6, -],
        #             [ 5, 4, 3],
        #             [-,  2, -],
        #             [-,  1, 0]
        #         ]

        _w1_index = [
            0,1,4,6,7,8,10,11,13,14
        ]
        _w2_index = [
             14,13,10,8,7,6,4,3,1,0
        ]
        self.register_buffer('w1_index',torch.tensor(_w1_index))  #.to(self.device)
        self.register_buffer('w2_index', torch.tensor(_w2_index))  #.to(self.device)
 

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in = in_channels * 9
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def get_phd_weight(self):
        out_ch, in_ch = self.weight.shape[:2]

        weight1 = torch.zeros(out_ch, in_ch, 15).cuda()
        weight1[:,:,self.w1_index] = self.weight

        weight2 = torch.zeros(out_ch, in_ch, 15).cuda()
        weight2[:,:,self.w2_index] = self.weight

        # import pdb;pdb.set_trace()
        return weight1.view(out_ch, in_ch, 5,3), weight2.view(out_ch, in_ch, 5,3)

    def forward(self, input):
        # x = unfold_padding(input)
        weight1, weight2 = self.get_phd_weight()

        # outputs = [None for _ in range(5)]
        # for i in range(5):
        #     feat1 = F.conv2d(x[i], weight1, self.bias, stride = [2,1])
        #     feat2 = F.conv2d(x[i][:,:,0:], weight2, self.bias, stride = [2,1])
        #     outputs[i] = torch.stack((feat1,feat2),dim=2).view(feat1.size(0),-1,3)
        # outputs = torch.Tensor(outputs)

        feat1 = F.conv2d(input, weight1, self.bias, stride = [2,1])
        feat2 = F.conv2d(input[:,:,1:,:], weight2, self.bias, stride = [2,1])
        output = torch.stack((feat1,feat2),dim=3).view(feat1.size(0),feat1.size(1),-1,feat1.size(-1))
            
        return output


if __name__ == "__main__":
    conv1 = PHDConv2d(in_channels = 1, out_channels = 1, bias=False)

    input = torch.ones(5*1*8*3,dtype = torch.float).view(5,1,8,3).to('cuda')  #sub_img_num, feature_num, height, width
    
    output = conv1(input)

    


    import pdb; pdb.set_trace()

