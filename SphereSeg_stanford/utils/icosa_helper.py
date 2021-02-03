from math import pi
import numpy as np
import cv2

#LOG_FILE = open('angle_info.log', 'w')
LOG_FILE = None
TOP_VERTEX = [pi*1/5, pi*3/5, pi, pi*7/5, pi*9/5]
BOT_VERTEX = [0, pi*2/5, pi*4/5, pi*6/5, pi*8/5]
THETA_RANGE = [[pi/2, pi/2-1.1071], [pi/2-1.1071, -pi/2+1.1071], [-pi/2+1.1071, -pi/2]]
CATEGORY = np.array([[0, 0, 0], [255, 0 , 0], [255, 215, 0], [0, 255, 0], [0, 0 ,255], \
                     [0, 51, 153], [139, 0 , 255], [128, 128, 0], [0, 0 ,128], [255, 229, 180],\
                     [255, 192, 203], [255, 127, 80], [204, 85, 0], [80, 200, 120]])

def get_pixel(img, theta, phi):
    '''
    find the corresponding pixel from panorama picture by angles
    Args:
        img(np.array)   : the numpy object of panoramic image
        theta(float)    : the angle about altitude
        phi(float)      : the angle about longitude
    Returns:
        pixel(np.array) : a numpy array with three channel value(R, G, B)
    '''
    radius = img.shape[1]/(2*pi)
    if phi > pi:
        phi = phi - 2*pi #constrain phi in [-pi, pi]

    # make sure the coordinate is within 0~hight and 0~width
    if abs(radius*theta) >= img.shape[0]/2:
        pixel_y = img.shape[0]-1 if theta > 0 else 0
    else:
        pixel_y = radius*theta + img.shape[0]/2

    if abs(radius*phi) >= img.shape[1]/2:
        pixel_x = img.shape[1]-1 if phi > 0 else 0
    else:
        pixel_x = radius*phi + img.shape[1]/2

    if LOG_FILE == None:
        return img[int(512-pixel_y)][int(pixel_x)]
    else:
        print('%f,%f' %(theta, phi), file=LOG_FILE)
        return img[int(512-pixel_y)][int(pixel_x)]

def construct_triangle(img, vertex, division, layer, upwards=True):
    '''
    Construct the triangular faces of the icosahedron
    Args:
        img (np.array)  : original panoramic pic
        vertex(float)   : the angle of the only vertex without any edge paralleled to equatorial
        division(int)   : how many subdivision on the face
        layer(int)      : the icosahedron can be divided into 3 layers:
                          0 means top, 1 means middle, 2 means bottom
        upwards(bool)   : if the face is point-up, then input true, otherwise input false
    Returns:
        ret(np.array)   : the corresponding array with size (4**division, 3)
    '''
    # vertex is the only point without any edge parallel to equatorial
    theta_max = THETA_RANGE[layer][0]
    theta_min = THETA_RANGE[layer][1]
    step = (theta_max-theta_min) / (2**division)
    # theta_interval split the theta range in evenly-angled
    theta_interval = np.arange(2**division)*step + theta_min + step/3
    theta_interval = theta_interval[::-1]
    ret2 = np.zeros((2**(division), 2*2**division, 3), np.uint8)
    if upwards:
        for i, theta in enumerate(theta_interval):
            phi_max = i / (2**division) * (pi * 1/5)
            # phi_range ~ [vertex-phi_max, vertex+phi_max]
            phi_range = np.arange(2*i + 1) - i
            if np.max(phi_range) != 0:
                phi_range = phi_range / np.max(phi_range) * phi_max + vertex
            else:
                phi_range = phi_range + vertex
            temp = np.zeros((len(phi_range), 3))
            j = 0
            for phi in phi_range:
                ret2[i,j] = get_pixel(img, theta, phi)
                j = j + 1
            j = 0
        return ret2

    else:
        for i, theta in enumerate(theta_interval):
            phi_max = (2**division-i-1) / (2**division) * (pi * 1/5)
            # phi_range ~ [vertex-phi_max, vertex+phi_max]
            n_interval = 2**(division+1) - 1 - 2*i
            phi_range = np.arange(n_interval) - n_interval // 2
            if np.max(phi_range) != 0:
                phi_range = phi_range / np.max(phi_range) * phi_max + vertex
            else:
                phi_range = phi_range + vertex

            temp = np.zeros((len(phi_range), 3))
            j = 0
            for phi in phi_range:
                ret2[i,j] = get_pixel(img, theta, phi)
                j = j + 1
            j = 0
        return ret2

def pano2icosa(img, division=3, output_mat=''):
    '''
    main function of transormation, and save the matrix as output_mat
    Args:
        img(np.array)   : the file path for load-in the original picture
        output_mat(str) : the file path for saving the transformed matrix in format of '.npy'
        division(int)   : how much subdivision should be applied
    Returns:
        None
    '''
    icosahedron = {}

    for face in range(20):
        # top layer of the isocahedron
        if face < 5:
            icosahedron[face] = construct_triangle(img, BOT_VERTEX[face], division, 0, True)
        # middle layer of the isocahedron, and need to haddle the staggered triangles
        elif face < 15:
            if face % 2 == 1:
                icosahedron[face] = construct_triangle(img, BOT_VERTEX[int((face-5)/2)], division, 1, False)
            else:
                icosahedron[face] = construct_triangle(img, TOP_VERTEX[int((face-6)/2)], division, 1, True)
        # bottom layer of the isocahedron
        else:
            icosahedron[face] = construct_triangle(img, TOP_VERTEX[int(face-15)], division, 2, upwards=False)

    if output_mat != '':
        np.save(output_mat, icosahedron)
    if LOG_FILE:
        LOG_FILE.close()
    return icosahedron
