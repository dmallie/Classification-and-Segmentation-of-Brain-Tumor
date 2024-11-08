#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:22:18 2024
Objective:
    - Creating dataset which helps to classify brain tumors as benign and malignant
    - Dataset is composed from two datasets which are downloaded from Kaggle
    - Benign types of tumors are also called Pituitary
    - malignant types are glioma and meningioma
@author: dagi
"""
import os
import shutil
import random 
from tqdm import tqdm 

# In[] Route path to source 1: Dataset prepared by Cheng and uploaded in a matlab format
src_path_1_glioma = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Brain_tumor_dataset_by_cheng/glioma/mri/"
src_path_1_pituitary = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Brain_tumor_dataset_by_cheng/pituitary/mri/"
src_path_1_meningioma = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Brain_tumor_dataset_by_cheng/meningioma/mri/"

# In[] Route path to Source 2: Dataset is prepared by MALICK
src_path_2_glioma_train = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/Training/glioma_tumor/"
src_path_2_glioma_test = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/Testing/glioma_tumor/"

src_path_2_pituitary_train = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/Training/pituitary_tumor/"
src_path_2_pituitary_test  ="/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/Testing/pituitary_tumor/"

src_path_2_meningioma_train = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/Training/meningioma_tumor/"
src_path_2_meningioma_test = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/Testing/meningioma_tumor/"

src_path_pituitary = [src_path_1_pituitary, src_path_2_pituitary_test, src_path_2_pituitary_train]
src_path_glioma = [src_path_1_glioma, src_path_2_glioma_test, src_path_2_glioma_train]
src_path_meningioma = [src_path_1_meningioma, src_path_2_meningioma_test, src_path_2_meningioma_train]

# In[] Route path to destination directory
model_2_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Model_2/"

dest_train_benign = model_2_path + "Train/benign/"
dest_train_malignant = model_2_path + "Train/malignant/"

dest_val_benign = model_2_path + "Val/benign/"
dest_val_malignant = model_2_path + "Val/malignant/"

dest_test_benign = model_2_path + "Test/benign/"
dest_test_malignant = model_2_path + "Test/malignant/"

# In[] Create lists from the directories
pituitary_list_1 = os.listdir(src_path_1_pituitary)
pituitary_list_2 = os.listdir(src_path_2_pituitary_test)
pituitary_list_3 = os.listdir(src_path_2_pituitary_train)

pituitary_list = [pituitary_list_1, pituitary_list_2, pituitary_list_3]

meningioma_list_1 = os.listdir(src_path_1_meningioma)
meningioma_list_2 = os.listdir(src_path_2_meningioma_test)
meningioma_list_3 = os.listdir(src_path_2_meningioma_train)

meningioma_list = [meningioma_list_1, meningioma_list_2, meningioma_list_3]

glioma_list_1 = os.listdir(src_path_1_glioma)
glioma_list_2 = os.listdir(src_path_2_glioma_test)
glioma_list_3 = os.listdir(src_path_2_glioma_train)

glioma_list = [glioma_list_1, glioma_list_2, glioma_list_3]

# In[] By mergining the lists together to create benign and malignant lists
benign = []
malignant = []

for index, source in enumerate (pituitary_list):
    # source defines the source path of the file
    src_path = src_path_pituitary[index]
    # Iterate over pituitary list
    for elements in source:
        # get the fullpath of the file
        full_path = src_path + elements
        # append the path to benign list
        benign.append(full_path)

# In[] populate malignant list by full paths from glioma tumor paths
for index, source in enumerate(glioma_list):
    # source defines the source path of the file
    src_path = src_path_glioma[index]
    # Iterate over pituitary list
    for elements in source:
        # get the fullpath of the file
        full_path = src_path + elements
        # append the path to benign list
        malignant.append(full_path)
        
# In[] populate malignant list by full paths from meningioma tumor paths
for index, source in enumerate(meningioma_list):
    # source defines the source path of the file
    src_path = src_path_meningioma[index]
    # Iterate over pituitary list
    for elements in source:
        # get the fullpath of the file
        full_path = src_path + elements
        # append the path to benign list
        malignant.append(full_path)
        
# In[] Shake and shuffle the benign and malignant lists to increase the randomness
random.shuffle(benign)
random.shuffle(malignant)

# In[] train is 70%, val is 20% and test is 10 of the total size
# Now we populate the training, val and test directories of benign tumors
training_limit = int(0.7 * len(benign))
val_limit = training_limit + int(0.2 *  len(benign))

for index, elements in enumerate(tqdm(benign, desc="Benign", leave=False)):
    if index <= training_limit:
        shutil.copy(elements, dest_train_benign)
    elif(index <= val_limit):
        shutil.copy(elements, dest_val_benign)
    else:
        shutil.copy(elements, dest_test_benign)
    
# In[] We Populate the malignant tumor to their respective directories
# malignant_size = 9000

training_limit = int(0.7 * len(malignant))
val_limit = training_limit + int(0.2 * len(malignant))

for index, elements in enumerate(tqdm(malignant, desc="Malignant", leave=False)):
    # if index is < training_size then it falls in to training category
    if index <= training_limit:
        shutil.copy(elements, dest_train_malignant)
    elif index <= val_limit:
        # if index value falls between training_size and val_size paste to validation category
        shutil.copy(elements, dest_val_malignant)
    else:
        # Otherwise paste it to test directory
        shutil.copy(elements, dest_test_malignant)



















