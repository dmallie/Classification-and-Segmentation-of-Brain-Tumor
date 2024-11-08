#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:06:28 2024
Objective:
    - Splits the Malignant tumor dataset into Galioma and Meningioma
    - Splits the respective dataset into Train 70%, Val 20% and Test 10%
    - Augment the Train dataset using the different augmentation techinique
@author: dagi
"""
import os
import random
import shutil

# In[] Route path to source 1
src_path_1_galioma = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Brain_tumor_dataset_by_cheng/glioma/mri/"
src_path_1_meningioma = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Brain_tumor_dataset_by_cheng/meningioma/mri/"

# In[] Route path to source 2
src_path_2_galioma_train = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/Training/glioma_tumor/"
src_path_2_galioma_test = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/Testing/glioma_tumor/"

src_path_2_meningioma_train = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/Training/meningioma_tumor/"
src_path_2_meningioma_test = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/Testing/meningioma_tumor/"

src_path_galioma = [src_path_1_galioma, src_path_2_galioma_test, src_path_2_galioma_train]
src_path_meningioma = [src_path_1_meningioma, src_path_2_meningioma_test, src_path_2_meningioma_train]

# In[] Route path to the destination directory
model_3_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Model_3/"
dest_train_galioma = model_3_path + "Train/Galioma/"
dest_train_meningioma = model_3_path + "Train/Meningioma/"

dest_val_galioma = model_3_path + "Val/Galioma/"
dest_val_meningioma = model_3_path + "Val/Meningioma/"

dest_test_Galioma = model_3_path + "Test/Galioma/"
dest_test_meningioma = model_3_path + "Test/Meningioma/"

# In[] Creating lists from directory
galioma_list = []
meningioma_list = []

for galioma_path in src_path_galioma:
    path_list = []
    # create a list from the path
    dir_list = os.listdir(galioma_path)
    # set the full path for each mri scans
    for elements in dir_list:
        #set full path
        full_path = galioma_path + elements
        path_list.append(full_path)
        
     # concatenate the list with galioma_list
    galioma_list = galioma_list + path_list
    
for meningioma_path in src_path_meningioma:
    # set path_list to empty
    path_list = []
    # create list from the path
    dir_list = os.listdir(meningioma_path)
    # set the full_path for each mri scans
    for elements in dir_list:
        # set full path
        full_path = meningioma_path + elements
        path_list.append(full_path)
    # concatenate list with meningioma_list
    meningioma_list = meningioma_list + path_list
    
# In[] Populate the destination of galioma directories with mri scans
training_border_index = int(0.7*len(galioma_list))
val_border_index = training_border_index + int(0.2*len(galioma_list))

# shuffle the galioma_list to mix the two sources of the data
random.shuffle(galioma_list)    
for index, src_path in enumerate(galioma_list):
    # check the index and decide to which category the scan should go to
    if index < training_border_index:
        dest_path = dest_train_galioma
        shutil.copy(src_path, dest_path)
    elif index <  val_border_index:
        dest_path = dest_val_galioma
        shutil.copy(src_path, dest_path)
    else:
        dest_path = dest_test_Galioma
        shutil.copy(src_path, dest_path)

# In[] Populate the destination of Meningioma directories with mri scans
training_border_index = int(0.7*len(meningioma_list))
val_border_index = training_border_index + int(0.2*len(meningioma_list))

# shuffle the meningioma_list to mix the two sources of the data
random.shuffle(meningioma_list)    
for index, src_path in enumerate(meningioma_list):
    # check the index and decide to which category the scan should go to
    if index < training_border_index:
        dest_path = dest_train_meningioma
        shutil.copy(src_path, dest_path)
    elif index <  val_border_index:
        dest_path = dest_val_meningioma
        shutil.copy(src_path, dest_path)
    else:
        dest_path = dest_test_meningioma
        shutil.copy(src_path, dest_path)
    
    
    
    
    
    
    
    
    
    
    
    
    