import sys
sys.path.append("..")
import numpy as np
import math
import cv2
from numpy.linalg import norm
import matplotlib.pyplot as plt
from functools import partial
from functools import lru_cache
import torch
from utils.icosa_helper import *
from utils.projection_helper import *
from utils.geometry_helper import *
from collections import OrderedDict 
import open3d as o3d


def iterk(k, k1):
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            if i%2 == 0:
                k1[2*i, 2*j, :] = np.append(k[i,j], 0)
                k1[2*i+2, 2*j+1, :] = np.append(k[i,j], 1)
                k1[2*i+1, 2*j+1, :] = np.append(k[i,j], 2)
                k1[2*i, 2*j+1, :] = np.append(k[i,j], 3)
            else:
                k1[2*i+1, 2*j+0, :] = np.append(k[i,j], 0)
                k1[2*i+1, 2*j+1, :] = np.append(k[i,j], 1)
                k1[2*i, 2*j, :] = np.append(k[i,j], 2)
                k1[2*i-1, 2*j, :] = np.append(k[i,j], 3)
    return k1

def main(level):
    k = np.zeros([5, 4])
    k1 = np.zeros([8, 2, 2])
    k2 = np.zeros([16, 4, 3])
    for i in range(5):
        k[i, :] = [15+i, 5+i, 6+i, 0+i]
    k = np.expand_dims(k[0], axis=1)
    if level==0:
        return k
    else:
        for i in range(level):
            print(i)
            if i==0:
                k1 = np.zeros([8, 2, 2])
                k2 = iterk(k, k1)
            else:
                k = k1
                k1 = np.zeros([2**(i+3), 2**(i+1), 2+i])
                print(k1.shape)
                k2 = iterk(k, k1)
        return k2

def get_face_vector(v, f):
    face_vec = np.zeros(f.shape)
    for i in range(len(f)):
        face = f[i]
        v_0 = v[face[0]]
        v_1 = v[face[1]]
        v_2 = v[face[2]]
        face_tmp = 1/3*(v_0 + v_1 + v_2)
        face_vec[i] = face_tmp
    erp_img = cv2.imread('/media/mnt3/matterport_b_0.2/matterport_b_0.2_re/train/img/cam1/0000.jpg')
    level = 8
    v, f = get_icosahedron(level)
    face_vec = get_face_vector(v, f)
    out = erp2sphere(erp_img, face_vec)/255
    if out.ndim == 1:
        color = out[:, np.newaxis].repeat(3, axis=1)
    else:
        color = out
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(face_vec)
    pcd.colors = o3d.utility.Vector3dVector(color)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
    o3d.visualization.draw_geometries([pcd, origin])
    return face_vec

def get_face_index(face_re, k2):
    output = np.zeros((k2.shape[0], k2.shape[1], 3))
    for i in range(k2.shape[0]):
        for j in range(k2.shape[1]):
            for k in range(k2.shape[2]):
                if k==0:
                    temp = face_re[k2[i,j,0]]
                else:
                    temp = temp[k2[i,j,k]]
            output[i, j, :] = temp 
    return output

def xyz2uv(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    u = np.arctan2(x, z)
    c = np.sqrt(x * x + z * z)
    v = np.arctan2(y, c)
    return np.stack([u, v], axis=-1)

def uv2img_idx(uv, erp_img):
    h, w = erp_img.shape[:2]
    delta_w = 2 * np.pi / w
    delta_h = np.pi / h
    x = uv[..., 0] / delta_w + w / 2 - 0.5
    y = uv[..., 1] / delta_h + h / 2 - 0.5
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    return np.stack([y, x], axis=0)

def remap(img, img_idx, cval=[0, 0, 0], method="linear"):
    # interpolation method
    if method == "linear":
        order = 1
    else:
        # nearest
        order = 0

    # remap image
    if img.ndim == 2:
        # grayscale
        x = map_coordinates(img, img_idx, order=order, cval=cval[0])
    elif img.ndim == 3:
        # color
        x = np.zeros([*img_idx.shape[1:], img.shape[2]], dtype=img.dtype)
        for i in range(img.shape[2]):
            x[..., i] = map_coordinates(img[..., i], img_idx, order=order, cval=cval[i])
    else:
        assert False, 'img.ndim should be 2 (grayscale) or 3 (color)'

    return x

def visualize_face_vec(face_vec):
    erp_img = cv2.imread('/media/mnt3/matterport_b_0.2/matterport_b_0.2_re/train/img/cam1/0000.jpg')
    out = erp2sphere(erp_img, face_vec)/255
    if out.ndim == 1:
        color = out[:, np.newaxis].repeat(3, axis=1)
    else:
        color = out
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(face_vec)
    pcd.colors = o3d.utility.Vector3dVector(color)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
    o3d.visualization.draw_geometries([pcd, origin])


def get_face_vector_2(v, f, visualization=True):
    face_vec = np.zeros(f.shape, np.float32)
    f = f.astype(np.uint32)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            face = f[i, j]
            v_0 = v[face[0]]
            v_1 = v[face[1]]
            v_2 = v[face[2]]
            face_tmp = 1/3*(v_0 + v_1 + v_2)
            face_vec[i, j] = face_tmp
    face_vec_tr = np.reshape(face_vec, (-1, 3))
    if visualization:
        face_vec_tr = face_vec_tr[:f.shape[0]*f.shape[1], :]
        visualize_face_vec(face_vec_tr)
    face_vec_tr = face_vec_tr[:f.shape[0]*f.shape[1], :]

    erp_img = cv2.imread('/media/mnt3/matterport_b_0.2/matterport_b_0.2_re/train/img/cam1/0000.jpg')
    out = erp2sphere(erp_img, face_vec_tr)/255
    if out.ndim == 1:
        color = out[:, np.newaxis].repeat(3, axis=1)
    else:
        color = out
    # erp_img = cv2.imread('/media/mnt3/matterport_b_0.2/matterport_b_0.2_re/train/img/cam1/0000.jpg')
    # out = erp2sphere(erp_img, face_vec)/255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(face_vec)
    pcd.colors = o3d.utility.Vector3dVector(color)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
    o3d.visualization.draw_geometries([pcd, origin])
    return face_vec

def shift_face(face_re, pn_array):
    output = np.zeros(face_re.shape)
    for i in range(20):
        for j in range(4):
            if j==0 or ((pn_array[i] == -1) and j==2):
                output[i,j,...] = face_re[i, j, ...]
            else:
                a,b,c,d = face_re[i, j, ...]
                output[i, j, ...] = [d,a,c,b]
    return output

def get_pn_array(pn_array, level):
    base_pn = np.expand_dims(np.array([1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1]), axis=-1)
    idx_array = np.array([20])
    output = np.zeros(idx_array)
    unit_array = np.expand_dims(np.array([1,1,-1,1]), axis=0)
    if level > 0:
        for i in range(level):
            idx_array = np.append(idx_array, 4)
            output = np.expand_dims(np.zeros(idx_array), axis=-1)
            if i==0:
                output = base_pn * unit_array
            else:
                output = np.dot(tmp, unit_array)
            tmp = np.expand_dims(output, axis=-1)
            # print(tmp[j])
        return output
    else:
        return base_pn
    

if __name__=='__main__':
    level = 2
    from scipy.io import savemat, loadmat
    faces = loadmat("../face_mat/level" + str(level) + "_face.mat")
    f = faces["faces"]
    v, _ = get_icosahedron(level)
    face_vec = np.zeros(f.shape)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            for k in range(f.shape[2]):
                face = f[i, j, k]
                v_0 = v[int(face[0])]
                v_1 = v[int(face[1])]
                v_2 = v[int(face[2])]
                face_tmp = 1/3*(v_0 + v_1 + v_2)
                face_vec[i, j, k] = face_tmp
    face_vec_tr = np.reshape(face_vec, (-1, 3))
    savemat("../face_vector/level" + str(level) + "_vector.mat", {"vector":face_vec})
    import pdb; pdb.set_trace()
    # face_vec_tr = face_vec_tr[:f.shape[0]*f.shape[1], :]
    visualize_face_vec(face_vec_tr[:face_vec_tr.shape[0]//40, :])
    # face_vec_tr = face_vec_tr[:f.shape[0]*f.shape[1], :]

    erp_img = cv2.imread('/media/mnt3/matterport_b_0.2/matterport_b_0.2_re/train/img/cam1/0002.jpg')
    out = erp2sphere(erp_img, face_vec_tr)/255
    if out.ndim == 1:
        color = out[:, np.newaxis].repeat(3, axis=1)
    else:
        color = out
    new_color = color.reshape(f.shape)
    fig, ax = plt.subplots(1, 5)
    for cnt in range(5):
        ax[cnt].set_title(f'{cnt}')
        ax[cnt].imshow(new_color[cnt, ...])
        ax[cnt].set_yticks([], [])
    plt.figure()
    plt.title('original image')
    plt.imshow(erp_img)
    plt.show()
    import pdb; pdb.set_trace()
    # erp_img = cv2.imread('/media/mnt3/matterport_b_0.2/matterport_b_0.2_re/train/img/cam1/0000.jpg')
    # out = erp2sphere(erp_img, face_vec)/255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(face_vec)
    pcd.colors = o3d.utility.Vector3dVector(color)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
    o3d.visualization.draw_geometries([pcd, origin])

    import pdb; pdb.set_trace()
    """
    k2 = main(level)
    k2 = k2.astype(np.uint32)
    v, f = get_icosahedron(level)
    # t = xyz2uv(v)
    # import pdb; pdb.set_trace()
    # new_unfold, new_faces = get_unfold_icosahedron(level)
    # face_re = np.reshape(f, (20, 4, 3))
    idx_array = np.array([20])
    for i in range(level):
        idx_array = np.append(idx_array, 4)
    idx_array = np.append(idx_array, 3)
    face_re = np.reshape(f, idx_array)
    pn_array = np.zeros(face_re.shape[:-1])
    pn_array = get_pn_array(pn_array, level-2)
    # output = shift_face(face_re, pn_array)
    vertex_index = get_face_index(face_re, k2)
    get_face_vector_2(v, vertex_index)
    import pdb; pdb.set_trace()
    """
