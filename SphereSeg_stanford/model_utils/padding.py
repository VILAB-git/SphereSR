
import torch
import numpy as np
import torch.nn.functional as F

class padding(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.index_lookup = {}
        for level in range(1, 8):
            self.index_lookup[level] = {}
            self.get_level_padding_idx(level)

    def get_level_padding_idx(self, level):
        h, w = 4*(2**level), 2**level
        index1_in, index2_in, index3_in, index4_in, index5_in, index6_in = \
            [], [], [], [], [], [] #pink, brown, purple, green, yellow, blue
        index1_out, index2_out, index3_out, index4_out, index5_out, index6_out = \
            [], [], [], [], [], []

        for i in range(h//2):
            index1_in.append([4,h//2-1-i +2,w-1 + 1])
        index1_in.append([3,0+2,w-1+1])

        for i in range(h//2+2):
            if i == 0:
                continue
            else :
                index1_out.append([0, (i+1)%2, i//2 ])

        for i in range(h//2 + 1):
            index2_in.append([4,h -1 - i+2,w-1+1])

        for i in range(h//2 + 1):
            index2_out.append([0, (h//2 + 2) - 1 - i, 0])

        index3_in.append([3,h-1 +2,0+1])
        for i in range(w-1):
            index3_in.append([4, h-1+2, i+1 ])
            index3_in.append([4, h-2+2, i+1 ])
        index3_in.append([4, h-1+2, w-1+1])

        for i in range(h//2):
            index3_out.append([0, (h + 2) -1 - i, 0])

        for i in range(h//2):
            index4_in.append([1,h+3 - (h//2-1-i +2), w+ 1 -(w-1 + 1)])
        index4_in.append([2,h+3 -(0+2),w+1 - (w-1+1)])

        for i in range(h//2+2):
            if i == 0:
                continue
            else :
                index4_out.append([0, h+3 - ((i+1)%2), w+1 - (i//2) ])


        for i in range(h//2 + 1):
            index5_in.append([1,h+3 - (h -1 - i+2),w+1 - (w-1+1)])

        for i in range(h//2 + 1):
            index5_out.append([0,h+3 - ((h//2 + 2) - 1 - i), w+1 - 0])


        index6_in.append([2,h+3 - (h-1+2),w+1 - (0+1)])
        for i in range(w-1):
            index6_in.append([1, h+3 - (h-1+2), w+1 - (i+1) ])
            index6_in.append([1, h+3 - (h-2+2), w+1 - (i+1) ])
        index6_in.append([1, h+3 - (h-1+2), w+1 - (w-1+1)])

        for i in range(h//2):
            index6_out.append([0, h+3 - ((h + 2) -1 - i), w+1 - 0])

        index1_in = np.array(index1_in)
        index2_in = np.array(index2_in)
        index3_in = np.array(index3_in)
        index4_in = np.array(index4_in)
        index5_in = np.array(index5_in)
        index6_in = np.array(index6_in)

        index1_out = np.array(index1_out)
        index2_out = np.array(index2_out)
        index3_out = np.array(index3_out)
        index4_out = np.array(index4_out)
        index5_out = np.array(index5_out)
        index6_out = np.array(index6_out)

        index_in = np.concatenate((index1_in, index2_in, index3_in, index4_in, index5_in, index6_in),axis=0)
        index_out = np.concatenate((index1_out, index2_out, index3_out, index4_out, index5_out, index6_out),axis=0)

        self.index_lookup[level][0] = torch.Tensor(index_out).type(torch.long).to(self.device)
        self.index_lookup[level][1] =  torch.Tensor(index_in).type(torch.long).to(self.device)


    def get_padding(self, input, level):
        input = input.reshape(input.size(0)//5, 5, input.size(1), input.size(2), input.size(3))
        input_pad = F.pad(input, [1,1,2,2])	#padding / [L,R,U,D]
        input_pad_tr = input_pad.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(5):
            input_pad_tr[:, (self.index_lookup[level][0][:,0]+i)%5, \
                             self.index_lookup[level][0][:,1], \
                             self.index_lookup[level][0][:,2]] = \
            input_pad_tr[:, (self.index_lookup[level][1][:,0]+i)%5, \
                             self.index_lookup[level][1][:,1], \
                             self.index_lookup[level][1][:,2]]
        result = input_pad_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*5,result.size(2),result.size(3),result.size(4))
        return result

class padding_old(object):
    def __init__(self, level):
        self.level = level
        h, w = 4*(2**self.level), 2**self.level

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        index1_in = [] #pink
        index2_in = [] #brown
        index3_in = [] #purple
        index4_in = [] #green
        index5_in = [] #yellow
        index6_in = [] #blue

        index1_out = []
        index2_out = []
        index3_out = []
        index4_out = []
        index5_out = []
        index6_out = []

        for i in range(h//2):
            index1_in.append([4,h//2-1-i +2,w-1 + 1])
        index1_in.append([3,0+2,w-1+1])

        for i in range(h//2+2):
            if i == 0:
                    continue
            else :
                    index1_out.append([0, (i+1)%2, i//2 ])


        for i in range(h//2 + 1):
            index2_in.append([4,h -1 - i+2,w-1+1])

        for i in range(h//2 + 1):
            index2_out.append([0, (h//2 + 2) - 1 - i, 0])


        index3_in.append([3,h-1 +2,0+1])
        for i in range(w-1):
            index3_in.append([4, h-1+2, i+1 ])
            index3_in.append([4, h-2+2, i+1 ])
        index3_in.append([4, h-1+2, w-1+1])

        for i in range(h//2):
            index3_out.append([0, (h + 2) -1 - i, 0])
        


        for i in range(h//2):
            index4_in.append([1,h+3 - (h//2-1-i +2), w+ 1 -(w-1 + 1)])
        index4_in.append([2,h+3 -(0+2),w+1 - (w-1+1)])

        for i in range(h//2+2):
            if i == 0:
                continue
            else :
                index4_out.append([0, h+3 - ((i+1)%2), w+1 - (i//2) ])


        for i in range(h//2 + 1):
            index5_in.append([1,h+3 - (h -1 - i+2),w+1 - (w-1+1)])

        for i in range(h//2 + 1):
            index5_out.append([0,h+3 - ((h//2 + 2) - 1 - i), w+1 - 0])


        index6_in.append([2,h+3 - (h-1+2),w+1 - (0+1)])
        for i in range(w-1):
            index6_in.append([1, h+3 - (h-1+2), w+1 - (i+1) ])
            index6_in.append([1, h+3 - (h-2+2), w+1 - (i+1) ])
        index6_in.append([1, h+3 - (h-1+2), w+1 - (w-1+1)])

        for i in range(h//2):
            index6_out.append([0, h+3 - ((h + 2) -1 - i), w+1 - 0])

        index1_in = np.array(index1_in)
        index2_in = np.array(index2_in)
        index3_in = np.array(index3_in)
        index4_in = np.array(index4_in)
        index5_in = np.array(index5_in)
        index6_in = np.array(index6_in)

        index1_out = np.array(index1_out)
        index2_out = np.array(index2_out)
        index3_out = np.array(index3_out)
        index4_out = np.array(index4_out)
        index5_out = np.array(index5_out)
        index6_out = np.array(index6_out)

        self.index_in = np.concatenate((index1_in,index2_in,index3_in,index4_in,index5_in,index6_in),axis=0)
        self.index_out = np.concatenate((index1_out,index2_out,index3_out,index4_out,index5_out,index6_out),axis=0)

        self.index_in = torch.Tensor(self.index_in).type(torch.long).to(self.device)
        self.index_out = torch.Tensor(self.index_out).type(torch.long).to(self.device)


    def get_padding(self, input):
        input_pad = F.pad(input, [1,1,2,2])	#padding / [L,R,U,D]
        input_pad_tr = input_pad.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(5):
            input_pad_tr[:, (self.index_out[:,0]+i)%5, self.index_out[:,1], self.index_out[:,2]] = \
            input_pad_tr[:, (self.index_in[:,0]+i)%5, self.index_in[:,1], self.index_in[:,2]]
        input_pad_result = input_pad_tr.permute(0,1,4,2,3)
        return input_pad_result


def check_time():
    import time
    level = 6
    h, w = 4*(2**level), 2**level
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pad_level = padding(level)
    input = torch.randn(100*5*h*w).view(100,5,1,h,w).to(device)  #sub_img_num, feature_num, height, width
    start = time.time() 
    result = pad_level.get_padding(input)
    # result = pad_level.get_padding2(input)
    resume_time_new = time.time() - start
    print("time(new) :", resume_time_new)
    # print(pad_level2.get_padding(input))
    x0 = torch.zeros(input.size(0),input.size(1),input.size(2),input.size(3)+4,input.size(4)+2).cuda()
    start = time.time() 
    for idx, x in enumerate(input):
        x0[idx] = pad_level.get_padding_old(x)
    resume_time_old = time.time() - start
    print("time(old) :", resume_time_old)
    print("speed ratio: ", resume_time_old/resume_time_new)
    if ((~(x0==result)).sum())==0:
        print('same result')
    else:
        print('different result')

if __name__ == "__main__":
    level = 3 
    h, w = 4*(2**level), 2**level
    pad_old = padding_old(level)
    pad_new = padding()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.randn(100*5*h*w).view(100,5,1,h,w).to(device)  #sub_img_num, feature_num, height, width
    result = pad_old.get_padding(input)
    result2 = pad_new.get_padding(input, level)
    if ((~(result==result2)).sum())==0:
        print('same result')
    else:
        print('different result')
    import pdb; pdb.set_trace()

