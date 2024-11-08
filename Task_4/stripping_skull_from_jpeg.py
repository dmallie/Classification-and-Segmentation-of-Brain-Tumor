#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:10:25 2024
objective:
    - convert .tif file in the test_imgs to .nii
    - apply skullstripping on the .nii files end up creating .nii.gz
    - extract skull removed files from .nii.gz and save as .jpg
@author: dagi
"""
import nibabel as nib 
import numpy as np 
import os 
from skull_strip import skull_strip 
from PIL import Image

########### PHASE I FROM .TIF to .NII ######################################
# In[] Set routing path to source directory which are populated with files .tif
test_jpeg_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Test/Images/"
train_jpeg_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Train/Images/"
val_jpeg_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Val/Images/"

src_path = [test_jpeg_path, train_jpeg_path, val_jpeg_path]
# In[] Create list of .tif files from src_tif_path 
test_jpeg_list = os.listdir(test_jpeg_path)
train_jpeg_list = os.listdir(train_jpeg_path)
val_jpeg_list = os.listdir(val_jpeg_path) 

src_list = [test_jpeg_list, train_jpeg_list, val_jpeg_list]
# In[] Set Destination path for the NII files
test_nii_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Skull_Stripped/NII/Test/Images/"
train_nii_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Skull_Stripped/NII/Train/Images/"
val_nii_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Skull_Stripped/NII/Val/Images/"

dest_path = [test_nii_path, train_nii_path, val_nii_path]
# In[] iterate over src_tif_list, convert each file to nifti and save them in dest_nii_path
for index, list_dir in enumerate (src_list):
    for jpeg in list_dir:
        # get the full path 
        full_path = src_path[index] + jpeg
        # read the image
        img = Image.open(full_path)
        # Convert it to numpy array
        img_np = np.array(img)
        # Assuming `mri_data` is a numpy array and affine is the transformation matrix
        affine = np.eye(4)  # Example affine matrix; 
        # convert it to NIFTI file
        img_nifti = nib.Nifti1Image(np.array(img_np), affine)
        # set the destination file
        destFile = dest_path[index] + jpeg.replace(".jpg", ".nii")
        # Save the NIFTI file
        nib.save(img_nifti, destFile)

########### PHASE II APPLY SKULL STRIPPING ON .NII FILES ######################################
# In[]Set routing path to skull stripped .nii files
test_skullStripped_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Skull_Stripped/NII/Test/Skull_Stripped/"
train_skullStripped_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Skull_Stripped/NII/Train/Skull_Stripped/"
val_skullStripped_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Skull_Stripped/NII/Val/Skull_Stripped/"

dest_skullStripped_path = [test_skullStripped_path, train_skullStripped_path, val_skullStripped_path]

# In[] Create list of .nii file from skullstripped paths
test_nii_list = os.listdir(test_nii_path)
train_nii_list = os.listdir(train_nii_path)
val_nii_list = os.listdir(val_nii_path)

src_skull_stripped_list = [test_nii_list, train_nii_list, val_nii_list]
# In[] Iterate over src_nii_list and apply skull stripping on each .nii files then them save
for index, nii_list in enumerate(src_skull_stripped_list):
    for nii in nii_list:
        # get the full path
        full_path = dest_path[index] + nii
        # set the destination path
        dest_full_path = dest_skullStripped_path[index] + nii.replace(".nii", ".jpg")    
        # apply skull stripping and then save output in the directory
        skull_strip(full_path, dest_full_path)
        
########### PHASE III Convert .nii.gz FILES in nii_skull_stripped BACK TO .JPG ######################################
# In[] Set routing path to destination directory
dest_test_jpg = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Skull_Stripped/JPEG/Test/Images/"
dest_train_jpg = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Skull_Stripped/JPEG/Train/Images/"
dest_val_jpg = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Skull_Stripped/JPEG/Val/Images/"

dest_jpg_path = [dest_test_jpg, dest_train_jpg, dest_val_jpg]

# In[] create list outof the dest_skullStripped_path
test_skullStripped_list = os.listdir(test_skullStripped_path)
test_skullStripped_list = os.listdir(train_skullStripped_path)
test_skullStripped_list = os.listdir(val_skullStripped_path)

# In[] iterate over dest_skullstripped_list which are .nii files & r skullstripped
for nii_stripped in dest_skullstripped_list:
    # get the full path to the file
    full_path = dest_skullstripped_path_nii + nii_stripped
    # read the nifit file
    nifti_data = nib.load(full_path)
    img_nifti = nifti_data.get_fdata()
    # image is along the second axis
    img_np = img_nifti[:,:,0]
    # Normalize the slie to be in range[0, 255]
    img_normalized = 255 * ((img_np - np.min(img))/(np.max(img) - np.min(img)))
    # convert to uint8 
    img_normalized = img_normalized.astype(np.uint8)
    # Convert to an image using PIL and save
    img = Image.fromarray(img_normalized)
    # Settin the destination path
    savePath = dest_jpg_path+ nii_stripped.split('.')[0] + ".jpg"
    img.save(savePath)


