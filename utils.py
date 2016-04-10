import numpy as np
import scipy.misc
import os
from datetime import datetime as dt
import argparse
from models import VGG16

mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
def add_mean(img):
    for i in range(3):
        img[0,:,:,i] += mean[i]
    return img

def sub_mean(img):
    for i in range(3):
        img[0,:,:,i] -= mean[i]
    return img

def read_image(path):
    img = scipy.misc.imread(path,  mode='RGB')
    # Resize if ratio is specified
    img = scipy.misc.imresize(img, (224, 224))
    img = img.astype(np.float32)
    img = img[None, ...]
    # Subtract the image mean
    img = sub_mean(img)
    return img

def save_image(im, iteration, out_dir):
    img = im.copy()
    # Add the image mean
    img = add_mean(img)
    img = np.clip(img[0, ...],0,255).astype(np.uint8)
    nowtime = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)    
    scipy.misc.imsave("{}/neural_art_{}_iteration{}.png".format(out_dir, nowtime, iteration), img)
    
def reverseFolder(root_dir, extention):
    filePathes = []
    for dr in os.listdir(root_dir):
        next_object = root_dir + "/" + dr
        
        if os.path.isdir(next_object):
            #print next_object
            ps = reverseFolder(next_object, extention)
            filePathes.extend(ps)
        elif os.path.isfile(next_object):
            if next_object.endswith(extention):
                #print next_object
                #print dr
                filePathes.append(next_object)
                #print getPath(next_object, dr)
                #spectrumToArrays(next_object, dr)
    return filePathes
                
def cleanFolder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
            