import numpy as np
import torch.utils.data as data
from torchvision import transforms
from utils.projection_helper import *
from scipy.io import savemat, loadmat
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch 
import os
from PIL import Image
from dataloader_all import *   #change
from model_utils.models_resnet101v3 import *
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import time
import argparse



parser = argparse.ArgumentParser(description='spherePHD!',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', default=None, type=str, help='Path of model')
args = parser.parse_args()

if args.path == None:
    sys.exit()

load_path = './save_model/'+args.path+'.pth'
print('load_path : ',load_path)
n_class = 13

def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

def accs(pred, target):
    accs = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        targets = target_inds.sum()
        if targets == 0:
            accs.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            accs.append(float(intersection) / max(targets, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return accs



def index2color(input, mask=None):
    
    rgb_value = np.array([[241, 255, 82],[102, 168, 226],[190, 123, 75],[89, 173, 163],[254, 158, 137],[0, 224, 1],[113, 143, 65],[84, 84, 84],[0, 18, 141 ],[255, 31, 33],[100, 22, 116],[85, 116, 127],[234, 234, 234],[0,0,0]],np.uint8 )
    # ceiling, floor, wall, colum, beam, window, door, table, chair, bookcase, sofa, board, clutter, <UNK>
    output = rgb_value[input[0]]
    if mask != None:
        output[mask] = rgb_value[-1]

    return output




# load_path = './save_model/01-11-14:48:25.pth'
# print('load_path : ',load_path)


if __name__=='__main__':
    batch_size = 1
    img_level = 7
    dist_level = 7
    split_num = 1
    use_cuda = torch.cuda.is_available()
    net = PHDnet_resNet()
    net = torch.nn.DataParallel(net).cuda()
    val_data_dir = '/data/stanford2d3d/data_512'
    trans = transforms.Compose([Reshape(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])  # change normalize value
    val_dataset = SphereMono(img_level, dist_level, val_data_dir, trans, load_erp=False, mode = 'test', split_num=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    num_samples = len(val_loader)
    min_depth = 1e-3


    arg_map = np.fliplr(np.rot90(np.array(loadmat("./arg_map/level"+str(img_level)+"_arg.mat")["arg_map"],dtype=np.uint64),2))

    net.load_state_dict(torch.load(load_path))
    
    # face vector
    face_vector = loadmat("./face_vector/level" + str(dist_level) + "_vector.mat")["vector"]

    
    total_ious = []
    pixel_accs = []
    total_accs = []

    with torch.no_grad():

        net.eval()
        for i, sample in enumerate(val_loader):


            ico_img = sample['ico_img']
            ico_mask = sample['ico_mask']
            erp_dist = sample['erp_dist']
            erp_mask = sample['erp_mask'][0]
            if use_cuda:
                ico_img = ico_img.cuda()
                ico_mask = ico_mask.cuda()
                # ico_dist = ico_dist.cuda()
            ico_img = ico_img.reshape(ico_img.shape[0]*ico_img.shape[1], ico_img.shape[2], ico_img.shape[3], ico_img.shape[4])
            # ico_dist = ico_dist.reshape(ico_dist.shape[0]*ico_dist.shape[1], ico_dist.shape[2], ico_dist.shape[3], ico_dist.shape[4])
            ico_mask = ico_mask.reshape(ico_mask.shape[0]*ico_mask.shape[1], ico_mask.shape[2], ico_mask.shape[3], ico_mask.shape[4])
            
            dist_out = net(ico_img, img_level)


            dist_out_dec = torch.argmax(dist_out,dim=1).cpu().numpy()
            ico2erp_dist_out=np.expand_dims(dist_out_dec.reshape(-1)[arg_map],axis=0)
            pred = ico2erp_dist_out

            # dist_gt_dec = torch.argmax(ico_dist,dim=1).cpu().numpy()
            # ico2erp_dist_gt=np.expand_dims(dist_gt_dec.reshape(-1)[arg_map],axis=0)
            # target = ico2erp_dist_gt
            target = erp_dist.cpu().numpy()


            for k, t in zip(pred, target):
                k = k[~erp_mask]
                t = t[~erp_mask]

                total_ious.append(iou(k, t))
                pixel_accs.append(pixel_acc(k, t))
                total_accs.append(accs(k,t))
                # print(total_ious, pixel_accs)



 #plot result#################################################################################
            # ico2erp_dist_out = index2color(ico2erp_dist_out,mask = erp_mask)
            # ico2erp_dist_gt = index2color(erp_dist,mask = erp_mask)
            # erp_img = sample['erp_img'].squeeze().cpu().numpy()

            # plt.imsave('./test_result/erp'+str(i)+'_img.png',erp_img)
            # plt.imsave('./test_result/erp'+str(i)+'_est.png',ico2erp_dist_out)
            # plt.imsave('./test_result/erp'+str(i)+'_gt.png',ico2erp_dist_gt)

            # import pdb;pdb.set_trace()
##############################################################################################

total_accs = np.array(total_accs).T
accs = np.nanmean(total_accs, axis=1)

total_ious = np.array(total_ious).T  # n_class * val_len
ious = np.nanmean(total_ious, axis=1)

pixel_accs = np.array(pixel_accs).mean()

print("pix_acc: {}, meanIoU: {}, meanAcc: {}, IoUs: {}, Accs: {}".format(pixel_accs, np.nanmean(ious), np.nanmean(accs), ious, accs))

    
