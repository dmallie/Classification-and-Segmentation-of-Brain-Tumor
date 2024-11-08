#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:34:42 2024
Objective:
    - This script calculates the mean and std of each datasets: Train, Val & Test
    - Read images from each directory & calculate its mean and std values
@author: dagi
"""
import cv2
import os 
import numpy as np
from tqdm import tqdm

# In[] Calculate average mean values for a given dataset
def calculate_mean(dataset):
    # Initialise the parameters 
    # mean_sum = 0
    sum_img = None # accummulate the sum of pixel values of the entire dataset
    # std_sum = 0 
    # std_value = 0

    # Iterate over the dataset, read each image and collect the sum value of each pixels
    for img_name in dataset:
        # Read the image 
        img_ = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        if len(img_.shape) == 3:
            img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        else:
            img = img_
        # Resize the image to 256x256 for consistency
        img = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
        # accumulate teh sum of pixel values of each individual pixels
        if sum_img is None:
            sum_img = img / 255
        else:
            sum_img += img/255
        
    # calculating the mean
    mean_img = sum_img / len(dataset)

    # Calculate  the mean value of pixels for each channel
    mean_pixel_value = np.mean(mean_img, axis=(0, 1))

    # mean_pixel_value = [0.171481, 0.171481, 0.171481]
    return mean_pixel_value, mean_img

# In[] Function which calculate the standard deviation of pixel values in images of given dataset
def calculate_std(dataset, mean_img):
    # For standard deviation
    sum_squared_img = None
    squared_diff = 0

    # Calculating STD
    # Go through the same dataset of training mri images
    for img_path in dataset:
        # Read the image
        img_ = cv2.imread(img_path)
        # Convert BGR image to RGB and Grayscale to RGB
        if len(img_.shape) == 3:
            img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        else:
            img = img_
        # Resize the image to 256x256
        img = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
    
        # Accumulate the squared differences from the mean image
        squared_diff = (img/255 - mean_img) ** 2
        
        if sum_squared_img is None:
            sum_squared_img = squared_diff
        else:
            sum_squared_img += squared_diff
    
    # Calculating the variance
    variance = sum_squared_img / len(dataset)

    # Standard Deviation
    std = np.sqrt(np.mean(variance, axis = (0, 1)))

    # std = [0.151678, 0.151678, 0.151678]
    return std 


# In[] Set Route path
root_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Model_3/"
training_path = root_path + "Train/"
val_path = root_path + "Val/"
test_path = root_path + "Test/"

# In[] Create list from root_path
def create_list(path):
    dataset_path = []
    for dirPath, dirNames, fileNames in os.walk(path):
        for file in fileNames:
            if file.endswith('.jpg' ) or file.endswith('.tif'):
                full_path = dirPath + "/" + file
                dataset_path.append(full_path)
    return dataset_path

# In[] Create list for training, val and test datasets
training_list = create_list(training_path)
val_list = create_list(val_path)
test_list = create_list(test_path)

datasets = [training_list, val_list, test_list]

# In[] Initialise the parameters 
mean_img = []
mean_list = []
std_list = []
height = 256 
width = 256

# In[] Calculate the mean and standard deviation of each datasets
for data in tqdm(datasets, desc="Calculating mean", leave=False):
    mean, img_mean = calculate_mean(data)
    mean_list.append(mean)
    mean_img.append(img_mean)

for index, data in enumerate(tqdm(datasets, desc="Calculating std", leave=False)):
    std = calculate_std(data, mean_img[index])
    std_list.append(std)
    
# training dataset: mean values: 0.17617, std values: 0.15991
# validation dataset: mean values: 0.17753, std values : 0.16255
# test dataset: mean values: 0.17626, std values: 0.0.16147
