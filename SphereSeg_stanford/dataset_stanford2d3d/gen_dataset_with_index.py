import cv2

import os 
from PIL import Image
import numpy as np

import array
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_index_img( img ):
    ''' Parse a color as a base-256 number and returns the index
    Args:
        color: A 3-tuple in RGB-order where each element \in [0, 255]
    Returns:
        index: an int containing the indec specified in 'color'
    '''
    return img[:,:,0] * 256 * 256 + img[:,:,1] * 256 + img[:,:,2]

""" Label functions """
def load_labels( label_file ):
    """ Convenience function for loading JSON labels """
    with open( label_file ) as f:
        return json.load( f )

def parse_label( label ):
    """ Parses a label into a dict """
    res = {}
    clazz, instance_num, room_type, room_num, area_num = label.split( "_" )
    res[ 'instance_class' ] = clazz
    res[ 'instance_num' ] = int( instance_num )
    res[ 'room_type' ] = room_type
    res[ 'room_num' ] = int( room_num )
    res[ 'area_num' ] = int( area_num )
    return res


areas = ['area_1','area_2','area_3','area_4','area_5a','area_5b','area_6']
parts = ['semantic','rgb','depth','semantic_pretty']


       
classes = ['<UNK>','ceiling','floor','wall','column','beam','window','door','table','chair','bookcase','sofa','board','clutter']
labels = load_labels('/data/stanford2d3d/data_original/assets/semantic_labels.json')

for area in areas:
    for part in parts:
        i = 0

        read_path = './data_original/'+area+'/'+part+'/'
        write_path= './data_512/'+area+'/'+part+'/'
        
        list_dir = os.listdir(read_path)
        list_dir.sort()

        for filename in list_dir:

            print(filename , i)
            if not 'camera' in filename:
                continue

            
            num = '%04d' %i
            i+= 1

            
            if part == 'rgb':
                img = cv2.imread(read_path+filename)
                resize_img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_AREA)
                cv2.imwrite(write_path+num+'.png', resize_img)
            else:
                resize_img = Image.open(read_path+filename).resize((1024,512),Image.NEAREST)
                resize_img.save(write_path+num+'.png')


            #make semantic_index folder
            if part == 'semantic':
                resize_img = np.array(Image.open(read_path+filename).resize((1024,512),Image.NEAREST),np.int64)
                img = resize_img
                img_index = get_index_img(img)
                img_class = np.zeros(img_index.shape)
                max_labels = len(labels)
                        
                for p in range(img_index.shape[0]):
                    for j in range(img_index.shape[1]):
                        if img_index[p,j] >= max_labels:
                            class_num = 0
                        else:
                            class_name = parse_label(labels[img_index[p,j]])['instance_class']
                            class_num = -1
                            for k in range(14):
                                if class_name == classes[k]:
                                    class_num = k
                                    break
                            if class_num == -1:
                                print("error!")

                        img_class[p,j] = class_num
                cv2.imwrite('./data_512/'+area+'/'+part+'_index'+'/'+num+'.png', img_class)
             
            # import pdb;pdb.set_trace()

