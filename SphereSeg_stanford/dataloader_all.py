import numpy as np
import torch.utils.data as data
from torchvision import transforms
from utils.projection_helper import img2ERP, erp2sphere
from scipy.io import savemat, loadmat
from torchvision import datasets
import matplotlib.pyplot as plt
import torch 
import os
from PIL import Image

class Reshape(object):
    def __init__(self):
        pass
        self.Toten = transforms.ToTensor()

    def __call__(self, sample):
        for key in sample.keys():
            if key!='erp_img' and key!='erp_dist' and key!='ico2erp_dist' and key!='erp_mask':
                out_dict = []
                for i in range(5):
                    out_dict.append(self.Toten(sample[key][i]))

                sample[key] = torch.stack(out_dict, dim=0)
        
        return sample

class Normalize(object):
    def __init__(self, mean, std):
        self.norm = transforms.Normalize(mean, std)

    def __call__(self, sample):
        for key in sample.keys():
            if key=='ico_img':
                out_dict = []
                for i in range(5):
                    out_dict.append(self.norm(sample[key][i]))
                sample[key] = torch.stack(out_dict, dim=0)
        return sample

class SphereMono(data.Dataset):
    def __init__(self, img_level, dist_level, data_dir, transform=None, load_erp=False, h_rot = False, mode = 'train', split_num = 1):
        self.face_vector_img = loadmat("./face_vector/level" + str(img_level) + "_vector.mat")["vector"]
        self.face_vector_img_tr = self.face_vector_img.reshape(-1, 3)
        self.face_vector_dist = loadmat("./face_vector/level" + str(dist_level) + "_vector.mat")["vector"]
        self.face_vector_dist_tr = self.face_vector_dist.reshape(-1, 3)
        self.transform = transform
        self.load_erp = load_erp
        self.h_rot = h_rot
        self.data_dir = data_dir
        self.mode = mode
        self.n_class = 13
        self.split_num = split_num

        assert self.split_num in [1,2,3]

        if self.split_num == 1:
            if self.mode == 'train':
                sets = ['area_1','area_2','area_3','area_4','area_6']
            elif self.mode == 'val':
                sets = ['area_5a','area_5b']
            elif self.mode == 'test':
                sets = ['area_5a','area_5b']
                self.arg_map = np.fliplr(np.rot90(np.array(loadmat("./arg_map/level"+str(img_level)+"_arg.mat")["arg_map"],dtype=np.uint64),2))
        elif self.split_num == 2:
            if self.mode == 'train':
                sets = ['area_1','area_3','area_5a','area_5b','area_6']
            elif self.mode == 'val':
                sets = ['area_2','area_4']
            elif self.mode == 'test':
                sets = ['area_2','area_4']
                self.arg_map = np.fliplr(np.rot90(np.array(loadmat("./arg_map/level"+str(img_level)+"_arg.mat")["arg_map"],dtype=np.uint64),2))
        elif self.split_num == 3:
            if self.mode == 'train':
                sets = ['area_2','area_4','area_5a','area_5b']
            elif self.mode == 'val':
                sets = ['area_1','area_3','area_6']
            elif self.mode == 'test':
                sets = ['area_1','area_3','area_6']
                self.arg_map = np.fliplr(np.rot90(np.array(loadmat("./arg_map/level"+str(img_level)+"_arg.mat")["arg_map"],dtype=np.uint64),2))

        
        self.up_img_list = []
        for part in sets:
            up_img_list = os.listdir(os.path.join(data_dir, part, 'rgb'))
            up_img_list.sort()
            up_img_list_temp = [os.path.join(os.path.join(data_dir, part, 'rgb'), up_img_list[i]) for i in range(len(up_img_list))]
            self.up_img_list += up_img_list_temp
        
        self.up_depth_list = []
        for part in sets:
            up_depth_list = os.listdir(os.path.join(data_dir, part, 'semantic_index'))
            up_depth_list.sort()
            up_depth_list_temp = [os.path.join(os.path.join(data_dir, part, 'semantic_index'), up_depth_list[i]) for i in range(len(up_depth_list))]
            self.up_depth_list += up_depth_list_temp
        
        

    def __getitem__(self, idx):
        erp_img = np.array(Image.open(self.up_img_list[idx]), np.uint8)
        erp_dist = np.array(Image.open(self.up_depth_list[idx]))
        
        if self.h_rot == True:
            cut = np.random.randint(0,np.shape(erp_img)[1])
            erp_img = np.concatenate((erp_img[:,cut:,:],erp_img[:,:cut,:]),axis=1)
            erp_dist = np.concatenate((erp_dist[:,cut:],erp_dist[:,:cut]),axis=1) 



        sample = {}

        ico_img = erp2sphere(erp_img, self.face_vector_img_tr)
        ico_img = ico_img.reshape(self.face_vector_img.shape)

        ico_dist = erp2sphere(erp_dist, self.face_vector_dist_tr, method="nearest")
        ico_dist = ico_dist.reshape(self.face_vector_dist.shape[:-1])[..., None]


        ico_dist_mask = (ico_dist == 0)
        
        ico_img_mask0 = (ico_img[:,:,:,0]==0)
        ico_img_mask1 = (ico_img[:,:,:,1]==0)
        ico_img_mask2 = (ico_img[:,:,:,2]==0)
        ico_img_mask = ico_img_mask0 & ico_img_mask1 & ico_img_mask2
        ico_img_mask = np.expand_dims(ico_img_mask,axis=3)

        ico_mask = ico_dist_mask | ico_img_mask



        erp_dist_mask = (erp_dist == 0)

        erp_img_mask0 = (erp_img[:,:,0]==0)
        erp_img_mask1 = (erp_img[:,:,1]==0)
        erp_img_mask2 = (erp_img[:,:,2]==0)
        erp_img_mask = erp_img_mask0 & erp_img_mask1 & erp_img_mask2
        # erp_img_mask = np.expand_dims(erp_img_mask,axis=2)

        erp_mask = erp_dist_mask | erp_img_mask
        # erp_mask = np.expand_dims(erp_mask,axis=2)

        
        # erp_mask = np.concatenate((erp_mask,erp_mask,erp_mask),axis=2)



        # ico_mask = np.expand_dims(ico_mask,axis=0)
        ico_mask = np.concatenate((ico_mask,ico_mask,ico_mask,ico_mask,ico_mask,ico_mask,ico_mask,ico_mask,ico_mask,ico_mask,ico_mask,ico_mask,ico_mask),axis=3)

        # create one-hot encoding
        p, q, r, s = ico_dist.shape
        target = np.zeros((p,q,r,13))
        for c in range(13):

            idx = ico_dist==c+1
            idx = np.squeeze(idx,axis=-1)
            target[...,c][idx] = 1
        ico_dist = target

        erp_dist[erp_dist == 0] = 14
        erp_dist -= 1



        
        
        sample['ico_img'] = ico_img
        sample['ico_dist'] = ico_dist
        sample['ico_mask'] = ico_mask
        sample['erp_img'] = erp_img
        sample['erp_dist'] = erp_dist
        sample['erp_mask'] = erp_mask



        if self.transform:
            sample = self.transform(sample)

            # import pdb;pdb.set_trace()
                

        return sample

    def __len__(self):
        return len(self.up_img_list)

