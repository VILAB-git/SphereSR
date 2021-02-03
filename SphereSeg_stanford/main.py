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
parser.add_argument('--pretrain_path', default=None, type=str, help='Path of pretrained model')
args = parser.parse_args()


time_now = time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time()))
writer = SummaryWriter('runs0/'+time_now)   #command  == tensorboard --logdir=runs
print('runs/'+time_now)


def index2color(input, mask ):
    
    rgb_value = np.array([[241, 255, 82],[102, 168, 226],[190, 123, 75],[89, 173, 163],[254, 158, 137],[0, 224, 1],[113, 143, 65],[84, 84, 84],[0, 18, 141 ],[255, 31, 33],[100, 22, 116],[85, 116, 127],[234, 234, 234],[0,0,0]],np.uint8 )
    #<UNK>, ceiling, floor, wall, colum, beam, window, door, table, chair, bookcase, sofa, board, clutter
    output = rgb_value[input[0]]
    output[mask[0]] = rgb_value[-1]

    return output




if __name__=='__main__':
    batch_size = 4
    img_level = 7
    dist_level = 7
    num_epochs = 70
    use_cuda = torch.cuda.is_available()
    net = PHDnet_resNet()
    net = torch.nn.DataParallel(net).cuda()
    learning_rate = 1e-3
    weight_decay = 0
    split_num = 1
    optimizer = optim.Adam(net.parameters(), lr=learning_rate,weight_decay=weight_decay) #add weight decay
    data_dir = '/data/stanford2d3d/data_512'
    trans = transforms.Compose([Reshape(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])  # change normalize value
    train_dataset = SphereMono(img_level, dist_level, data_dir, trans, load_erp=False, h_rot = True, mode = 'train', split_num=1)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last = True)
    val_dataset = SphereMono(img_level, dist_level, data_dir, trans, load_erp=False, mode = 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last = True)
    loss_criterion = torch.nn.BCEWithLogitsLoss() 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)   #add scheduler

    arg_map = np.fliplr(np.rot90(np.array(loadmat("./arg_map/level"+str(img_level)+"_arg.mat")["arg_map"],dtype=np.uint64),2))


    if args.pretrain_path != None:
        print('load pretrained model..')
        net.load_state_dict(torch.load(args.pretrain_path))  #load pretrained model


    
    # face vector
    face_vector = loadmat("./face_vector/level" + str(dist_level) + "_vector.mat")["vector"]
    erp_h = 512
    erp_w = 1024
    uv = xyz2uv(face_vector)
    img_idx = uv2img_idx_v2(uv, erp_h, erp_w)
    # for plotting
    plot_train_rate = 10 
    plot_val_rate = 10 
    
    for epoch in range(num_epochs):
        trn_loss = 0.0 
        val_loss = 0.0
        net.train()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()
            ico_img = sample['ico_img']
            ico_mask = sample['ico_mask']
            ico_dist = sample['ico_dist']
            if use_cuda:
                ico_img = ico_img.cuda()
                ico_mask = ico_mask.cuda()
                ico_dist = ico_dist.cuda()
            ico_img = ico_img.reshape(ico_img.shape[0]*ico_img.shape[1], ico_img.shape[2], ico_img.shape[3], ico_img.shape[4])
            ico_dist = ico_dist.reshape(ico_dist.shape[0]*ico_dist.shape[1], ico_dist.shape[2], ico_dist.shape[3], ico_dist.shape[4])
            ico_mask = ico_mask.reshape(ico_mask.shape[0]*ico_mask.shape[1], ico_mask.shape[2], ico_mask.shape[3], ico_mask.shape[4])



            
            ico_img.requires_grad = False
            ico_mask.requires_grad = False
            ico_dist.requires_grad = False
            dist_out = net(ico_img, img_level)
            
            loss = loss_criterion(dist_out[~ico_mask], ico_dist[~ico_mask])


            optimizer.zero_grad()  
            loss.backward()

            optimizer.step()
            trn_loss += loss.item()



            if (i+1) % plot_train_rate == 0:

                erp_mask = sample['erp_mask']

                print('epoch = ',epoch, ',i = ',i, 'train_loss = ', trn_loss/ plot_train_rate)

                # ico2erp_dist_mask=np.expand_dims(ico_mask[:,:3,:,:].cpu().numpy().reshape(-1)[arg_map],axis=0)

                dist_out_dec = torch.argmax(dist_out,dim=1).cpu().numpy()
                ico2erp_dist_out=np.expand_dims(dist_out_dec.reshape(-1)[arg_map],axis=0)
                ico2erp_dist_out = index2color(ico2erp_dist_out,mask = erp_mask)

                dist_gt_dec = torch.argmax(ico_dist,dim=1).cpu().numpy()
                ico2erp_dist_gt=np.expand_dims(dist_gt_dec.reshape(-1)[arg_map],axis=0)
                ico2erp_dist_gt = index2color(ico2erp_dist_gt, mask = erp_mask)



                # plt.imsave('./test_result/erp_est.png',ico2erp_dist_out)
                # plt.imsave('./test_result/erp_gt.png',ico2erp_dist_gt)


                writer.add_scalar('training loss', trn_loss / plot_train_rate,epoch * len(train_loader) + i)
                writer.add_image(time_now+'/train/img', sample['erp_img'][0].permute(2,0,1), global_step=epoch * len(train_loader) + i)

                # import pdb;pdb.set_trace()
                writer.add_image(time_now+'/train/est', np.transpose(ico2erp_dist_out,(2,0,1)), global_step=epoch * len(train_loader) + i)
                writer.add_image(time_now+'/train/gt', np.transpose(ico2erp_dist_gt,(2,0,1)), global_step=epoch * len(train_loader) + i)
                trn_loss = 0


        with torch.no_grad():

            net.eval()
            for i, sample in enumerate(val_loader):
                ico_img = sample['ico_img']
                ico_mask = sample['ico_mask']
                ico_dist = sample['ico_dist']
                if use_cuda:
                    ico_img = ico_img.cuda()
                    ico_mask = ico_mask.cuda()
                    ico_dist = ico_dist.cuda()
                ico_img = ico_img.reshape(ico_img.shape[0]*ico_img.shape[1], ico_img.shape[2], ico_img.shape[3], ico_img.shape[4])
                ico_dist = ico_dist.reshape(ico_dist.shape[0]*ico_dist.shape[1], ico_dist.shape[2], ico_dist.shape[3], ico_dist.shape[4])
                ico_mask = ico_mask.reshape(ico_mask.shape[0]*ico_mask.shape[1], ico_mask.shape[2], ico_mask.shape[3], ico_mask.shape[4])
                
                dist_out = net(ico_img, img_level)
                loss = loss_criterion(dist_out[~ico_mask], ico_dist[~ico_mask])
                val_loss += loss.item()
                if (i+1) % plot_val_rate == 0:

                    erp_mask = sample['erp_mask']

                    # ico2erp_dist_mask=np.expand_dims(ico_mask[:,:3,:,:].cpu().numpy().reshape(-1)[arg_map],axis=0)

                    dist_out_dec = torch.argmax(dist_out,dim=1).cpu().numpy()
                    ico2erp_dist_out=np.expand_dims(dist_out_dec.reshape(-1)[arg_map],axis=0)
                    ico2erp_dist_out = index2color(ico2erp_dist_out,mask = erp_mask)

                    dist_gt_dec = torch.argmax(ico_dist,dim=1).cpu().numpy()
                    ico2erp_dist_gt=np.expand_dims(dist_gt_dec.reshape(-1)[arg_map],axis=0)
                    ico2erp_dist_gt = index2color(ico2erp_dist_gt, mask = erp_mask)

                    writer.add_image(time_now+'/val/img', sample['erp_img'][0].permute(2,0,1), global_step=epoch * len(train_loader) + i)
                    writer.add_image(time_now+'/val/est', np.transpose(ico2erp_dist_out,(2,0,1)), global_step=epoch * len(train_loader) + i)
                    writer.add_image(time_now+'/val/gt', np.transpose(ico2erp_dist_gt,(2,0,1)), global_step=epoch * len(train_loader) + i)  
        
        
        
        writer.add_scalar('validation loss',val_loss / (i+1), epoch)

        print('epoch = ',epoch, 'val loss= ', val_loss / (i+1))
        save_path = './save_model/'+time_now+'.pth'
        torch.save(net.state_dict(), save_path)

        scheduler.step()
