from __future__ import division
import math
import json
import random
import pprint
import imageio
import numpy as np

import tensorflow as tf
import os

import scipy.io as sio
from time import gmtime, strftime
from PIL import Image


pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def load_data(image_path, flip=False, is_test=False):
    img_A, img_B = load_image(image_path)
    img_AB = np.concatenate((img_A, img_B), axis=2)
    return img_AB

def load_image(image_path):
    input_data = sio.loadmat(image_path)
    input_img = input_data['mat_6']
    # img_A = input_img[:, :,0:9]
    # img_B = input_img[:, :,9:10]
    # img_A = input_img[:, :,0:8]
    # img_B = input_img[:, :,8:9]
    img_A = input_img[:, :,0:5]
    img_B = input_img[:, :,5:6]

    return img_A, img_B

# def load_image(image_path):
#     input_data = sio.loadmat(image_path)
#     input_img = input_data['mat_9']
#     # img_A = input_img[:, :,0:9]
#     # img_B = input_img[:, :,9:10]
#     img_A = input_img[:, :,0:8]
#     img_B = input_img[:, :,8:9]
 

    # return img_A, img_B


def load_data_2(image_path, flip=False, is_test=False):
    img_AB=sio.loadmat(image_path)['mat_9']
    return img_AB 



def save_mat(path,image_mat):
#    print(image_mat.shape)
    image_mat = softmax2(image_mat)
#    print(image_mat.shape)
    sio.savemat(path,{"mat":image_mat})


def preprocess_A_and_B(img_A, img_B, load_size=404, fine_size=384, flip=True, is_test=False):
    return img_A, img_B

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, images_,size, image_path):
    return imsave(images,images_, size, image_path) 

def imsave(images,images_, size, path):
    return imageio.imsave(path, merge(images, images_,size))


def imread(path, is_grayscale = True):
    if (is_grayscale):
        return imageio.imread(path, flatten = True).astype(np.float)
    else:
        return imageio.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def softmax2(pre):
    arg = np.argmax(pre, axis=-1)
    p = np.zeros(pre.shape)
    for n in range(pre.shape[0]):
        for i in range(pre.shape[1]):
            for j in range(pre.shape[2]):
                p[n, i, j, arg[n, i, j]] = 1
            
    return p

def softmax(pre):
    arg = np.argmax(pre, axis=2)
    p = np.zeros(pre.shape)
    for i in range(pre.shape[0]):
        for j in range(pre.shape[1]):
            p[i, j, arg[i, j]] = 1
    return p

def merge(images,images_,size):
    im_shape = images.shape
    im_shape_=images_.shape
    picb = None
    for p in range(im_shape[0]):
        im = images[p]
        im = softmax(im)
        im_ = images_[p]
#        print(im.shape)
#        print(im_.shape)
        im = im.reshape((im_shape[1], im_shape[2],im_shape[3]))
        im_ = im_.reshape((im_shape_[1], im_shape_[2],im_shape_[3]))
        pic =  im[:, :, 0]
        pic_ =  im_[:, :, 0]
        
        for c in range(1,im_shape[-1]):
            pic = np.concatenate((pic, im[:, :, c]), axis=0)
            pic_ =  np.concatenate((pic_, im_[:, :, c]), axis=0)
        picb = np.concatenate((pic_, pic), axis=1)
        if p == 0:
            pic_all = picb
        else:
            pic_all = np.concatenate((pic_all, picb), axis=1)
            
    return  pic_all





def inverse_transform(images):
    return (images+1.)/2.

def flip(im):
    return np.fliplr(im)

def data_augment(im):
    return rand_rotate(im),flip(im),tissue_augment(im,is_mask)#new_im