#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:20:52 2024
Objective:
    - Resize the datasize from 512x512 to 256x256
    - Store the images in UNet_256 directory
@author: dagi
"""
import os 
import cv2

# In[] Soource Directories
src_train_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet/Train/images/"
src_val_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet/Val/images/"
src_test_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet/Test/images/"

# In[] Dest Directories
dest_train_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Train/Images/"
dest_val_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Val/Images/"
dest_test_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Test/Images/"

# In[] Create list from source directory
train_list = os.listdir(src_train_path)
val_list = os.listdir(src_val_path)
test_list = os.listdir(src_test_path)

# In[9] Iterate over the training list, read image, resize image, save image
new_size = (256, 256)
for img_files in train_list:
    # set the full path
    full_path = src_train_path + img_files
    # load teh image
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    # resize teh image
    img_resize = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
    # save the image
    dest_path = dest_train_path + img_files
    cv2.imwrite(dest_path, img_resize)
    

# In[9] Iterate over the val list, read image, resize image, save image
for img_files in val_list:
    # set the full path
    full_path = src_val_path + img_files
    # load teh image
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    # resize teh image
    img_resize = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
    # save the image
    dest_path = dest_val_path + img_files
    cv2.imwrite(dest_path, img_resize)
     
    
# In[9] Iterate over the test list, read image, resize image, save image
for img_files in test_list:
    # set the full path
    full_path = src_test_path + img_files
    # load teh image
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    # resize teh image
    img_resize = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
    # save the image
    dest_path = dest_test_path + img_files
    cv2.imwrite(dest_path, img_resize)
         
###############################################################################
# In[] Soource Directories
src_train_path_mask = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet/Train/masks/"
src_val_path_mask = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet/Val/masks/"
src_test_path_mask = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet/Test/masks/"

# In[] Dest Directories
dest_train_path_mask = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Train/Masks/"
dest_val_path_mask = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Val/Masks/"
dest_test_path_mask = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Test/Masks/"

# In[] Create list from source directory
train_list_mask = os.listdir(src_train_path_mask)
val_list_mask = os.listdir(src_val_path_mask)
test_list_mask = os.listdir(src_test_path_mask)

# In[9] Iterate over the training list, read image, resize image, save image
for img_files in train_list_mask:
    # set the full path
    full_path = src_train_path_mask + img_files
    # load teh image
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    # resize teh image
    img_resize = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
    # save the image
    dest_path = dest_train_path_mask + img_files
    cv2.imwrite(dest_path, img_resize)
    

# In[9] Iterate over the val list, read image, resize image, save image
for img_files in val_list_mask:
    # set the full path
    full_path = src_val_path_mask + img_files
    # load teh image
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    # resize teh image
    img_resize = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
    # save the image
    dest_path = dest_val_path_mask + img_files
    cv2.imwrite(dest_path, img_resize)
     

# In[9] Iterate over the test list, read image, resize image, save image
for img_files in test_list_mask:
    # set the full path
    full_path = src_test_path_mask + img_files
    # load teh image
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    # resize teh image
    img_resize = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
    # save the image
    dest_path = dest_test_path_mask + img_files
    cv2.imwrite(dest_path, img_resize)
    