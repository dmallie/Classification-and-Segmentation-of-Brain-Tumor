#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 04:51:40 2024
Objective:
    - Using the test set the model segement out the tumor and save the mask 
    file in the Segmented directory
@author: dagi
"""
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from torch.utils.data import DataLoader
from UNET_architecture import UNet
import cv2
from custom_dataset import UNetDataset
from tqdm import tqdm
import numpy as np 

# In[] Routing path to the Source directory
src_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Test/Images/"
mask_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Test/Masks/"
src_list = os.listdir(src_path)

# In[] Destination folder
dest_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Test/Segmented/"

# In[] Setting Hyperparameters
WIDTH = 256 
HEIGHT = 256 
OUTPUT_SHAPE = 1
BATCH  = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# In[] Set Transform Functions
transform_fn = A.Compose([
                        A.Resize(height=HEIGHT, width=WIDTH),
                        # A.ToGray(p=1.0), # p=1.0 ensures that the grayscale transform is always applied
                        A.Normalize(
                            mean = [0.15602],
                            std = [0.131627],
                            max_pixel_value = 1.0),
                        ToTensorV2(),
                        ])
                        
# In[] Setting the dataset and dataloader
dataset = UNetDataset(src_path, mask_path, transform_fn)
data_loader = DataLoader(dataset = dataset,
                              batch_size = BATCH,
                              shuffle = True,
                              num_workers = 4,
                              pin_memory = True)

# In[] Load the model 
model_path = "best_model.pth"
model = UNet(in_channels= 1, out_channels=OUTPUT_SHAPE)
# load the saved dict
saved_state_dict = torch.load(model_path, weights_only=True)
# load teh state_dict into the model
model.load_state_dict(saved_state_dict)

model = model.to(DEVICE)
# In[] Evaluation Loop
model.eval() # set the model to evaluation mode

for img_name in tqdm(src_list, desc="Testing", leave=False):
    # set the full path of the image
    full_path = src_path + img_name
    # set the destination path
    full_dest_path = dest_path + img_name.replace(".jpg", ".png")
    ##########################################################
    # load the image
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    # convet img to numpy array and standardize the values
    img_standardized = np.array(img) / 255.0
    # Transform the image
    img_transformed = transform_fn(image = img_standardized)["image"].unsqueeze(0).to(DEVICE) # add batch dimension
    
    ##############################################################
    # Perform the evaluation task
    with torch.no_grad():
        # Forward pass
        predictions = torch.sigmoid(model(img_transformed))
        # convert probabilites to 0 or 1
        binary_predictions = (predictions > 0.5).float()
    ##############################################################
    # move the binary predictions to cpu and numpy
    mask = binary_predictions.squeeze(0).squeeze(0).cpu().detach().numpy()
    # convert mask to uint8 and values between 0 and 255
    mask = (mask * 255).astype(np.uint8)
    # save the mask file
    cv2.imwrite(full_dest_path, mask)
    