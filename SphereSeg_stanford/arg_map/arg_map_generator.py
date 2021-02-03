#!!! # make arg_map level (n) on gpu



from scipy.io import savemat, loadmat
import numpy as np
import torch

def distance(a1, a2):
    theta1 = pi/2 - a1[:,1]
    theta2 = pi/2 - a2[:,1]
    phi1 = a1[:,0]
    phi2 = a2[:,0]
    dist = torch.sqrt(torch.Tensor([2]).cuda()) * torch.sqrt(1-(torch.cos(theta1-theta2)+torch.sin(theta1)*torch.sin(theta2)
                                  *(torch.cos(phi1-phi2)-1)))
    return dist

def xyz2uv(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    u = np.arctan2(x, z)
    c = np.sqrt(x * x + z * z)
    v = np.arctan2(y, c)
    return np.stack([u, v], axis=-1)


img_level = 7
face_vector = loadmat("./../face_vector/level" 
                      + str(img_level) + "_vector.mat")["vector"]

uv = xyz2uv(face_vector).reshape(-1,2)

p, q = uv.shape

xy = np.zeros([512,1024,2])
pi = np.pi
for i in range(512):
    for j in range(1024):
#         xy[i,j] = [ -pi + 2*pi/1024*j, pi/2 - pi/512*i]
        xy[i,j] = [-pi+pi*(2*j+1)/(1024+1),pi/2-pi/2*(2*i+1)/(512+1)]


arg_map = torch.zeros([512,1024]).cuda()
xy = torch.Tensor(xy).cuda()
uv = torch.Tensor(uv).cuda()
uv[:,0] = (uv[:,0] -0.0031)*0.9994
uv[:,1] = (uv[:,1] + 0.0025)*0.9994

for i in range(512):
    for j in range(1024):
        dist_list = distance(xy[i,j].unsqueeze(0), uv)
        arg_map[i,j] = dist_list.argmin()
    if i%100==0:
        print(i)
        

tmp = {}
tmp['arg_map'] = arg_map.cpu().numpy()
savemat('./level'+str(img_level)+'_arg.mat', tmp)
