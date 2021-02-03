import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import itertools



class unpooling(nn.Module):
    def __init__(self):
        super(unpooling, self).__init__()
        self.height_dict = [2**i for i in range(8)]
        self.width_dict = [4*2**i for i in range(8)]
        self.unpooling_look_up = {}
        self.unpool = nn.Upsample(scale_factor=2,mode = 'nearest')
        for level in range(1, 8):
            self.unpooling_look_up[level] = {}
            self.get_level_unpooling_idx(level)

    def get_level_unpooling_idx(self, level):
        dict1, dict2, dict3, dict4 = [], [], [], []
        for i in range(2**level):
            for j in range(2**(level-1)):
                dict1.append([4*i+1,4*i+2])
                dict2.append([2*j,2*j+1]) 
                dict3.append([4*i+2,4*i+1])
                dict4.append([2*j+1,2*j])
        x1_list = list(itertools.chain(*dict1))
        y1_list = list(itertools.chain(*dict2))
        x2_list = list(itertools.chain(*dict3))
        y2_list = list(itertools.chain(*dict4))
        self.unpooling_look_up[level][0] = x1_list
        self.unpooling_look_up[level][1] = y1_list
        self.unpooling_look_up[level][2] = x2_list
        self.unpooling_look_up[level][3] = y2_list

    def forward(self, input, level):
        _input = input.clone()
        _input = self.unpool(_input)
        _input[..., self.unpooling_look_up[level+1][0], self.unpooling_look_up[level+1][1]] = \
        _input[..., self.unpooling_look_up[level+1][2], self.unpooling_look_up[level+1][3]]
        return _input

	

if __name__=='__main__':
    import time
    torch.manual_seed(0)
    level = 1
    h = 4*2**level
    w = 2**level
    # maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    input_gen = torch.randn(5*1*h*w,dtype = torch.float).view(5,1,h,w)
    

    unpool_class = unpooling()
    output = unpool_class(input_gen, level)

    import pdb;pdb.set_trace()
