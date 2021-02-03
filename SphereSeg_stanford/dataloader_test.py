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
    def __init__(self, img_level, dist_level, data_dir, transform=None, load_erp=False, h_rot = False):
        self.face_vector_img = loadmat("./face_vector/level" + str(img_level) + "_vector.mat")["vector"]
        self.face_vector_img_tr = self.face_vector_img.reshape(-1, 3)
        self.face_vector_dist = loadmat("./face_vector/level" + str(dist_level) + "_vector.mat")["vector"]
        self.face_vector_dist_tr = self.face_vector_dist.reshape(-1, 3)


        self.data_dir = '/data/matterport_official_b_0.2/val'
        self.up_img_list = os.listdir(os.path.join(data_dir, 'img', 'cam2'))
        self.up_img_list.sort()
        self.up_depth_list = os.listdir(os.path.join(data_dir, 'depth', 'cam2'))
        self.up_depth_list.sort()
        self.up_img_list_val = [os.path.join(os.path.join(data_dir, 'img', 'cam2'), self.up_img_list[i]) for i in range(len(self.up_img_list))]
        self.up_depth_list_val = [os.path.join(os.path.join(data_dir, 'depth', 'cam2'), self.up_depth_list[i]) for i in range(len(self.up_depth_list))]

        self.data_dir = '/data/matterport_official_b_0.2/test'
        self.up_img_list = os.listdir(os.path.join(data_dir, 'img', 'cam2'))
        self.up_img_list.sort()
        self.up_depth_list = os.listdir(os.path.join(data_dir, 'depth', 'cam2'))
        self.up_depth_list.sort()
        self.up_img_list_test = [os.path.join(os.path.join(data_dir, 'img', 'cam2'), self.up_img_list[i]) for i in range(len(self.up_img_list))]
        self.up_depth_list_test = [os.path.join(os.path.join(data_dir, 'depth', 'cam2'), self.up_depth_list[i]) for i in range(len(self.up_depth_list))]

        self.up_img_list = self.up_img_list_val + self.up_img_list_test
        self.up_depth_list = self.up_depth_list_val + self.up_depth_list_test


        self.transform = transform
        self.load_erp = load_erp
        self.h_rot = h_rot
        self.arg_map = np.fliplr(np.rot90(np.array(loadmat("./arg_map/level"+str(img_level)+"_arg.mat")["arg_map"],dtype=np.uint64),2))
        

    def __getitem__(self, idx):
        erp_img = np.array(Image.open(self.up_img_list[idx]), np.uint8)
        erp_dist = np.array(Image.open(self.up_depth_list[idx]), np.float32)/655.35

        if self.h_rot == True:
            cut = np.random.randint(0,np.shape(erp_img)[1])
            erp_img = np.concatenate((erp_img[:,cut:,:],erp_img[:,:cut,:]),axis=1)
            erp_dist = np.concatenate((erp_dist[:,cut:],erp_dist[:,:cut]),axis=1)



        sample = {}

        ico_img = erp2sphere(erp_img, self.face_vector_img_tr)
        ico_img = ico_img.reshape(self.face_vector_img.shape)

        ico_dist = erp2sphere(erp_dist, self.face_vector_dist_tr, method="nearest")
        ico_dist = ico_dist.reshape(self.face_vector_dist.shape[:-1])[..., None]



        # use arg_map
        ico2erp_dist=ico_dist.reshape(-1)[self.arg_map]
        sample['ico2erp_dist'] = ico2erp_dist


        
        ico_mask = (ico_dist == 100)
        erp_mask = (erp_dist == 100)
        erp_dist[erp_dist==100] = 0
        sample['ico_img'] = ico_img
        sample['ico_dist'] = ico_dist
        sample['ico_mask'] = ico_mask
        sample['erp_img'] = erp_img
        sample['erp_dist'] = erp_dist
        sample['erp_mask'] = erp_mask


        





        if self.transform:
            sample = self.transform(sample)


        return sample

    def __len__(self):
        return len(self.up_img_list)

def plot_check(img_level, dist_level, data_dir):
    trans = transforms.Compose([Reshape(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    train_dataset = SphereMono(img_level, dist_level, data_dir, trans, load_erp=True)
    batch_size = 2
    trn_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    toten = transforms.ToTensor()
    for i, sample in enumerate(trn_loader):
        # sample = train_dataset.__getitem__(15)
        # sample = toTensor(sample)
        ico_img = sample['ico_img']
        ico_mask = sample['ico_mask']
        ico_dist = sample['ico_dist']
        ico_dist[ico_mask] = 0
        ico_img = ico_img.reshape(ico_img.shape[0]*ico_img.shape[1], ico_img.shape[2], ico_img.shape[3], ico_img.shape[4])
        ico_dist = ico_dist.reshape(ico_dist.shape[0]*ico_dist.shape[1], ico_dist.shape[2], ico_dist.shape[3], ico_dist.shape[4])
        # for visualization
        fig, ax = plt.subplots(1, 5)
        disp_img = {}
        for cnt in range(5):
            disp_img[cnt] = np.array(np.transpose(255*ico_img[cnt].cpu().numpy(), (1,2,0)), np.uint8)
            ax[cnt].set_title(f'{cnt}')
            ax[cnt].imshow(disp_img[cnt])
            ax[cnt].set_yticks([], [])
        disp_dist = {}
        fig, ax = plt.subplots(1, 5)
        for cnt in range(5):
            disp_dist[cnt] = np.transpose(ico_dist[cnt].cpu().numpy(), (1,2,0))
            ax[cnt].set_title(f'{cnt}')
            ax[cnt].imshow(disp_dist[cnt].squeeze())
            ax[cnt].set_yticks([], [])
        erp_img = np.array(sample['erp_img'][0].reshape(512, 1024, 3), np.uint8)
        plt.figure()
        plt.imshow(erp_img)
        plt.show()

if __name__=='__main__':
    img_level = 7
    dist_level = 7
    data_dir = '/data/matterport_official_b_0.2/train'
    # plot_check(img_level, dist_level, data_dir)

    trans = transforms.Compose([Reshape(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    train_dataset = SphereMono(img_level, dist_level, data_dir, trans, load_erp=True)
    batch_size = 1
    trn_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    toten = transforms.ToTensor()
    for i, sample in enumerate(trn_loader):
        erp_dist = sample['erp_dist']
        ico2erp_dist = sample['ico2erp_dist']
        err = torch.abs(erp_dist-ico2erp_dist)
        if sample['erp_mask'].sum() == 0:
            break

    plt.imsave('./erp_dist.png',erp_dist.squeeze())
    plt.imsave('./ico2erp_dist.png',ico2erp_dist.squeeze())
    plt.imsave('error_dist.png',err.squeeze())
        