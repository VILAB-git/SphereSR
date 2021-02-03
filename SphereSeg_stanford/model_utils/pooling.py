import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import itertools



class pooling(nn.Module):
    def __init__(self):
        super(pooling, self).__init__()
        self.height_dict = [2**i for i in range(8)]
        self.width_dict = [4*2**i for i in range(8)]
        self.pooling_look_up = {}
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        for level in range(1, 8):
            self.pooling_look_up[level] = {}
            self.get_level_pooling_idx(level)

    def get_level_pooling_idx(self, level):
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
        self.pooling_look_up[level][0] = x1_list
        self.pooling_look_up[level][1] = y1_list
        self.pooling_look_up[level][2] = x2_list
        self.pooling_look_up[level][3] = y2_list

    def forward(self, input, level):
        _input = input.clone()
        _input[..., self.pooling_look_up[level][0], self.pooling_look_up[level][1]] = \
        _input[..., self.pooling_look_up[level][2], self.pooling_look_up[level][3]]
        # input[..., self.pooling_idx] = 
        # outputs[i] = maxpool(input[i])
        return self.maxpool(_input)

def pooling_old(_input):
    maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    n, c, h, w = _input.size()
    output = _input
    for i in range(h//4):
        for j in range(w//2):
            output[:,:,[4*i+1,4*i+2],[2*j,2*j+1]] = _input[:,:,[4*i+2,4*i+1],[2*j+1,2*j]]
    # print(input.shape)
    return maxpool(output)
	

if __name__=='__main__':
    import time
    torch.manual_seed(0)
    level = 3
    h = 4*2**level
    w = 2**level
    maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=(2,1), padding=0)
    # maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    input_gen = torch.randn(5*1*h*w,dtype = torch.float).view(5,1,h,w)
    pool_class = pooling()
    start = time.time()
    maxpool2 = pooling_old(input_gen)
    resume_time_old = time.time() - start
    print('time(old): ', resume_time_old)
    start = time.time()
    maxpool1 = pool_class(input_gen, level)
    resume_time_new = time.time() - start
    print('time(new): ', resume_time_new)
    print("speed ratio: ", resume_time_old/resume_time_new)
    if (~(maxpool1==maxpool2)).sum()==0:
        print('same result')
    else:
        print('different result')
