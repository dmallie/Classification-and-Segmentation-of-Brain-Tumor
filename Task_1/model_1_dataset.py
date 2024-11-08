#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:45:01 2024
Objective:
    - Creates Train, val and test dataset from no_tumor direcrtory
    - Size of noTumor dataset is 5398: 70% train, 20% val, 10% test
    - Tumor is composed of all three types of tumors
@author: dagi
"""
import os 
import shutil 
import random

# In[] Set the routing path
src_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/No Tumor/"
src_list = os.listdir(src_path)

noTumor_dest_train = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Model_1/Train/healthy/"
noTumor_dest_val = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Model_1/Val/healthy/"
noTumor_dest_test = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Model_1/Test/healthy/"

src_path_glioma = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/Training/glioma_tumor/" 
src_path_melingioma = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/Training/meningioma_tumor/"
src_path_pituitary = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/Training/pituitary_tumor/"

glioma_list = os.listdir(src_path_glioma) # 6613
melingioma_list = os.listdir(src_path_melingioma) # 6708
pituitary_list = os.listdir(src_path_pituitary) # 6189

tumor_dest_train = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Model_1/Train/tumor/"
tumor_dest_val = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Model_1/Val/tumor/"
tumor_dest_test = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Model_1/Test/tumor/"

# In[]  Populate the healthy or notumor training, validation and test directories
train_size = int(0.7 * len(src_list))
val_size = int(0.2 * len(src_list))
src_list_index = list(range(len(src_list)))

train_healthy = random.sample(src_list, train_size)
for elem in train_healthy:
    src_list.remove(elem)

val_healthy = random.sample(src_list, val_size)
for elem in val_healthy:
    src_list.remove(elem)

test_healthy = src_list 
# In[] copy and paste the selected elemnts to their destination
for mri in train_healthy:
    # set full path
    full_path = src_path + mri
    # copy and paste the file
    shutil.copy(full_path, noTumor_dest_train)
    
for mri in val_healthy:
    # set full path
    full_path = src_path + mri
    # copy and paste the file
    shutil.copy(full_path, noTumor_dest_val)

for mri in test_healthy:
    # set full path
    full_path = src_path + mri
    # copy and paste the file
    shutil.copy(full_path, noTumor_dest_test)
    
# In[] To populate the tumor directory
"""
- Merge the three lists of tumors in one list during which update the list to full path
- Randomly select trin, val and test mri's from list
"""
tumor = []
for glioma in glioma_list:
    # set full path
    full_path = src_path_glioma + glioma 
    # append full path to tumor list
    tumor.append(full_path)

for melingioma in melingioma_list:
    # set full path
    full_path = src_path_melingioma + melingioma 
    # append full path to tumor list
    tumor.append(full_path)

for pituitary in pituitary_list:
    # set full path
    full_path = src_path_pituitary + pituitary 
    # append full path to tumor list
    tumor.append(full_path)

# In[] Total size of MRI with tumor is larger than 19K but the final dataset we need is 6.5 K
tumor_dataset_size = 6500 # since it's very large we select only fraction of it

tumor_train_size = int(tumor_dataset_size *  0.7)
tumor_val_size = int(tumor_dataset_size * 0.2)
tumor_test_size = tumor_dataset_size - (tumor_train_size + tumor_val_size)

tumor_train_list = []
tumor_val_list = []
tumor_test_list = []

tumor_train_list = random.sample(tumor, tumor_train_size)
for elements in tumor_train_list:
    tumor.remove(elements)
    
tumor_val_list = random.sample(tumor, tumor_val_size)
for elements in tumor_val_list:
    tumor.remove(elements)
    
tumor_test_list = random.sample(tumor, tumor_test_size)

# In[] Move these slected files to their respective directory
for elements in tumor_train_list:
    # copy and paste the file to the respective directory
    shutil.copy(elements, tumor_dest_train)


for elements in tumor_val_list:
    # copy and paste the file to the respective directory
    shutil.copy(elements, tumor_dest_val)
       
for elements in tumor_test_list:
    # copy and paste the file to the respective directory
    shutil.copy(elements, tumor_dest_test)
 


































