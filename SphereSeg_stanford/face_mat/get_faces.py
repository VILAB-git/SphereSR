import igl
import numpy as np
import math
from numpy.linalg import norm
import matplotlib.pyplot as plt
from functools import partial

from functools import lru_cache
import torch
from copy import copy


class UnfoldVertex(object):
    def __init__(self, unfold_v):
        self.unfold_v = unfold_v
        self.reset()
        
    def __getitem__(self, item):
        pos = self.unfold_v[item][self.cnt[item]]
        self.cnt[item] += 1
        return pos

    def reset(self):
        self.cnt = {key:0 for key in self.unfold_v.keys()}

        
class VertexIdxManager(object):
    def __init__(self, unfold_v):
        self.reg_v = {}
        self.next_v_index = len(unfold_v)
        
    def get_next(self, a, b):
        if a>b:
            a,b = b,a
        key = f'{a},{b}'
        if key not in self.reg_v:
            self.reg_v[key] = self.next_v_index
            self.next_v_index += 1
        return self.reg_v[key]



def get_base_icosahedron():
    t = (1.0 + 5.0 ** .5) / 2.0
    vertices =[-1, t, 0, 1, t, 0, 0, 1, t, -t, 0, 1, -t, 0, -1, 0, 1, -t, t, 0, -1, t, 0,
                1, 0, -1, t, -1, -t, 0, 0, -1, -t, 1, -t, 0]
    faces = [0,2,1, 0,3,2, 0,4,3, 0,5,4, 0,1,5, 
             1,7,6, 1,2,7, 2,8,7, 2,3,8, 3,9,8, 3,4,9, 4,10,9, 4,5,10, 5,6,10, 5,1,6,
             6,7,11, 7,8,11, 8,9,11, 9,10,11, 10,6,11]

    # make every vertex have radius 1.0
    vertices = np.reshape(vertices, (-1, 3)) / (np.sin(2*np.pi/5)*2)
    faces = np.reshape(faces, (-1, 3))
    
    # Rotate vertices so that v[0] = (0, -1, 0), v[1] is on yz-plane
    ry = -vertices[0]
    rx = np.cross(ry, vertices[1])
    rx /= np.linalg.norm(rx)
    rz = np.cross(rx, ry)
    R = np.stack([rx,ry,rz])
    vertices = vertices.dot(R.T)
    return vertices, faces

def get_base_unfold():
    v, f = get_base_icosahedron()
    unfold_v = {i:[] for i in range(12)}

    # edge length
    l = 1/np.sin(2*np.pi/5)
    # height
    h = 3**0.5*l/2

    # v0
    for i in range(5):
        unfold_v[0].append([i*l, 0])

    # v1
    for _ in range(5):
        unfold_v[1].append([-0.5*l, h])
    unfold_v[1][1] = [-0.5*l + 5*l, h]
    unfold_v[1][4] = [-0.5*l + 5*l, h]

    # v2-v5
    for i in range(2, 6):
        for _ in range(5):
            unfold_v[i].append([(0.5 + i - 2)*l, h])

    # v6
    for _ in range(5):
        unfold_v[6].append([-l, 2*h])
    unfold_v[6][1] = [-l + 5*l, 2*h]
    unfold_v[6][2] = [-l + 5*l, 2*h]
    unfold_v[6][4] = [-l + 5*l, 2*h]

    # v7-v10
    for i in range(7, 11):
        for _ in range(5):
            unfold_v[i].append([(i - 7)*l, 2*h])

    # v11
    for i in range(5):
        unfold_v[11].append([(-0.5 + i)*l, 3*h])

    # to numpy
    for i in range(len(unfold_v)):
        unfold_v[i] = np.array(unfold_v[i])
    return unfold_v, f

def unfold_subdivision(unfold_v, faces):
    v_idx_manager = VertexIdxManager(unfold_v)

    new_faces = []
    new_unfold = copy(unfold_v)
    v_obj = UnfoldVertex(unfold_v)
    for (a, b, c) in faces:
        a_pos = v_obj[a]
        b_pos = v_obj[b]
        c_pos = v_obj[c]

        new_a= v_idx_manager.get_next(a, b)
        new_b= v_idx_manager.get_next(b, c)
        new_c= v_idx_manager.get_next(c, a)

        new_a_pos = (a_pos+b_pos)/2
        new_b_pos = (b_pos+c_pos)/2
        new_c_pos = (c_pos+a_pos)/2

        # new faces
        new_faces.append([a, new_a, new_c])
        new_faces.append([b, new_b, new_a])
        new_faces.append([new_a, new_b, new_c])
        new_faces.append([new_b, c, new_c])

        # new vertex
        indices = [new_a, new_b, new_c]
        poses = [new_a_pos, new_b_pos, new_c_pos]
        for (idx, pos) in zip(indices, poses):
            if idx not in new_unfold:
                new_unfold[idx] = []
            for _ in range(3):
                new_unfold[idx].append(pos)
    return new_unfold, new_faces

def get_unfold_icosahedron(level=0):
    if level == 0:
        unfold_v, f = get_base_unfold()
        return unfold_v, f
    # require subdivision
    unfold_v, f = get_unfold_icosahedron(level-1)
    unfold_v, f = unfold_subdivision(unfold_v, f)
    return unfold_v, f

def distort_list(unfold_v):
    np_round = partial(np.round, decimals=9)

    # calculate transform matrix
    new_x = unfold_v[2][0]-unfold_v[0][0]
    edge_len = np.linalg.norm(new_x)
    new_x /= edge_len
    new_y = np.cross([0,0,1], np.append(new_x, 0))[:2]
    R = np.stack([new_x, new_y])

    a = unfold_v[2][0]-unfold_v[0][0]
    b = unfold_v[1][0]-unfold_v[0][0]
    skew = np.eye(2)
    skew[0, 1] = -1/np.tan(np.arccos(a.dot(b)/norm(a)/norm(b)))
    skew[0]/=norm(skew[0])

    T = skew.dot(R)
    # scale adjust
    scale = np.linalg.det(skew)*edge_len
    T /=scale

    # to numpy array for efficient computation
    # np_round to alleviate numerical error when sorting
    distort_unfold = copy(unfold_v)
    five_neighbor = [distort_unfold[i] for i in range(12)]
    five_neighbor = np.array(five_neighbor)
    # Transform
    five_neighbor = np_round(five_neighbor.dot(T.T))
    
    # the same procedure for six_neighbor if len(unfold_v) > 12
    if len(unfold_v)>12:
        six_neighbor = [distort_unfold[i] for i in range(12, len(unfold_v))]
        six_neighbor = np.array(six_neighbor)
        six_neighbor = np_round(six_neighbor.dot(T.T))
    




    distort_list = []
    for it in five_neighbor:
        cnt = 0
        for i in it:
            if i in it[:cnt]:
                distort_list.append(np.array([-1,-1]))
            else:
                distort_list.append(i)
            cnt += 1
            
        distort_list.append(np.array([-1,-1]))
        
    if len(unfold_v)>12:
        for it in six_neighbor:
            cnt = 0
            for i in it:
                if i in it[:cnt]:
                    distort_list.append(np.array([-1,-1]))
                else:
                    distort_list.append(i)
                cnt += 1
    
    return distort_list

if __name__ == "__main__":
    level = 2
    new_unfold, new_faces = get_unfold_icosahedron(level)


    distort_list = distort_list(new_unfold)

    distort_new = np.array(distort_list).reshape(-1,6,2)

    distort_add = np.ones((distort_new.shape[0],distort_new.shape[1],3))

    for i in range(distort_new.shape[0]):
            for j in range(distort_new.shape[1]):
                distort_add[i][j] = np.append(distort_new[i][j],i)


    distort_add = distort_add.reshape(-1,3)
    idxs = []

    for idx, i in enumerate(distort_add):
            if (i[0] == -1 and i[1] == -1):
                idxs.append(False)
            else:
                idxs.append(True)
                    
    distort_add  = distort_add[idxs]



    box1 = []
    box2 = []
    box3 = []
    box4 = []
    box5 = []

    for i in distort_add:
            if (i[0] >=0 and i[0] <= 1) and (i[1] >=0 and i[1] <= 2):
                box1.append(i)
            if (i[0] >=1 and i[0] <= 2) and (i[1] >=-1 and i[1] <= 1):
                box2.append(i)
            if (i[0] >=2 and i[0] <= 3) and (i[1] >=-2 and i[1] <= 0):
                box3.append(i)
            if (i[0] >=3 and i[0] <= 4) and (i[1] >=-3 and i[1] <= -1):
                box4.append(i)
            if (i[0] >=4 and i[0] <= 5) and (i[1] >=-4 and i[1] <= -2):
                box5.append(i)

    box1 = np.array(box1)
    box2 = np.array(box2)
    box3 = np.array(box3)
    box4 = np.array(box4)
    box5 = np.array(box5)

    boxes = []

    for box in [box1,box2,box3,box4,box5]:

            idx = np.argsort(box[:,0])
            new_box = box[idx, :].reshape(2**level +1, -1 ,3)

            output = np.zeros(new_box.shape)
            for idx, i in enumerate(new_box):
                output[idx] = i[np.argsort(i[:,1])]
            boxes.append(output)
            
    boxes = np.array(boxes)

    boxes = boxes[:,:,:,2]


    n, x, y = boxes.shape
    faces = np.zeros((n,2*(y-1),x-1,3))
    for l in range(faces.shape[0]):
            for i in range(faces.shape[1]):
                for j in range(faces.shape[2]):
                    if i%2 == 0:
                        faces[l,i,j] = np.array([boxes[l,j,y-1-i//2],boxes[l,j+1,y-1-i//2],boxes[l,j+1,y-1-i//2-1]])
                    else :
                        faces[l,i,j] = np.array([boxes[l,j,y-1-i//2],boxes[l,j,y-1-i//2-1],boxes[l,j+1,y-1-i//2-1]])
    import scipy.io as sio
    sio.savemat("./level" + str(level) + "_face.mat", {"faces": faces})
    #import pdb; pdb.set_trace()
