#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:26:23 2024
Objective:
    - Prepare data with MRI scans to test the accuracy of the models as a whole
    - The dataset is called Test and has two sub directories: Classification & Segmentation
    - Classification is composed of four classes: No Tumor, Pituitary, Glioma, Meningioma 
    - Segmentation will hold the generated masks 
    - Data is downloaded from kaggle. by RORO YASEEN: https://www.kaggle.com/datasets/roroyaseen/brain-tumor-data-mri
    - The Data has not been used for training, validatio as well as evaluation purposes
    - The data for no tumor scan is brought from another dataset in kaggle which is prepared for segmentation task
         but only slices where tumors not detected in their respective mask files 
    - Source for No Tumor dataset:  https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
@author: dagi
"""
import os 
import shutil
import cv2 
from tqdm import tqdm 

# In[] Set source path
glioma = "/media/Linux/Downloads/Brain Tumor/Brain tumor data MRI RoRO YASEEN/New folder (6)/val/Glioma/"
pituitary = "/media/Linux/Downloads/Brain Tumor/Brain tumor data MRI RoRO YASEEN/New folder (6)/val/Meningioma/"
meningioma = "/media/Linux/Downloads/Brain Tumor/Brain tumor data MRI RoRO YASEEN/New folder (6)/val/Pituitary tumor/"
no_tumor = "/media/Linux/Downloads/Brain Tumor/Brain_Tumor kaggle/kaggle_3m/"

src_path = [glioma, pituitary, meningioma]
# In[] Set Destination path
dest_glioma = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Test/Classification/Glioma/"
dest_pituitary = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Test/Classification/Pituitary/"
dest_meningioma = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Test/Classification/Meningioma/"
dest_no_tumor = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Test/Classification/No Tumor/"

dest_path = [dest_glioma, dest_pituitary, dest_meningioma]
# In[] Populate the No Tumor directory
no_tumor_path = []
for root_dir, dirs, files in os.walk(no_tumor):
    for each_dir in dirs:
        dir_path = os.path.join(root_dir, each_dir)
        dir_files = os.listdir(dir_path)
        for each_files in dir_files:
            if "_mask.tif" in each_files:
                continue
            else:
                full_path = dir_path + "/" + each_files
                full_path_mask = dir_path + "/" + each_files.replace(".tif", "_mask.tif")
                # load the image
                img = cv2.imread(full_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # load the mask image
                img_mask = cv2.imread(full_path_mask, cv2.IMREAD_GRAYSCALE)
                # Check if there is a tumor in the mask file
                if (img_mask == 255).any():
                    # tumor is detected so we we don't need this file
                    continue 
                else:
                    # Tumor is not detected as such we can copy and paste teh file to no_tumor directory
                    dest_path = dest_no_tumor + each_files.replace(".tif", ".jpg")
                    cv2.imwrite(dest_path, img) 
# In[] Populate the other directories with scans from the validation sets
for index, path in enumerate(tqdm(src_path, desc="Copy and Paste fles...", leave=False)):
    # create list from the directory path
    path_list = os.listdir(path)
    # iterate over the list
    for each_files in path_list:
        # get the full path
        full_path = path + each_files 
        # copy and paste the file
        shutil.copy(full_path, dest_path[index] )
        