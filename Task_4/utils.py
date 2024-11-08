#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:35:04 2024
Objective:
    - Compute the accuracy of the segmentation model:
        - by calculating the dice_loss
        - by calculating IntersectionOverunion 
        - by caculaatign the average difference between the centroids interms of pixels
        - by calculating the average difference in the area of the bounding rectangles
@author: dagi
"""
import torch
import torchvision
import torch.nn as nn
# from dataset import BrainTumorDataset
from torch.utils.data import DataLoader 
import numpy as np 

# In[2] Set augmenation pipeline
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# In[3] 
"""
Objectives
- Creates training_dataset
- Creates training_dataloader
- Creates validation_dataset
- Creates validation_dataloader
"""
def get_loaders(train_ds, 
                val_ds,
                batch_size
                ):
    # Create Train & validation dataset
    # train_ds = CarvanaDataset(train_dir, train_maskdir, train_transform )
    # val_ds = CarvanaDataset(val_dir, val_maskdir, val_transform)
    
    # Create Training and validation DataLoader
    train_dataloader = DataLoader(dataset = train_ds,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 4,
                                  pin_memory = True)
    val_dataloader = DataLoader(dataset = val_ds,
                                batch_size = batch_size,
                                shuffle = False,
                                num_workers = 4,
                                pin_memory = True
                                )
    return train_dataloader, val_dataloader 

# In[5]
def save_predictions_as_imgs(loader, model, folder ="saved_images/", device="cuda"):
    model.eval()
    save_path = "/media/dagi/Linux/Mallie_Dagmawi/PyTorch/data/Brain MRI Segmentation/" + folder
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
        torchvision.utils.save_image(preds, f"{save_path}/pred_{idx}.png")
        torchvision.utils.save_image(y, f"{save_path}{idx}.png")
    model.train()

# In[] Get the coordinates & centroid of the bounding rectangle
def bounding_rectangle(orig_mask, seg_mask):
    # Convert to torch tensor datatype
    orig_mask = torch.tensor(orig_mask)
    seg_mask = torch.tensor(seg_mask)
    # extract only the nonzero values 
    orig_nonzero = torch.nonzero(orig_mask)
    seg_nonzero = torch.nonzero(seg_mask)
    # if the nonzero is empty then we assign (0,0) (0, 0) to the rectangle
    if len(orig_nonzero) == 0:
        orig_top_left = (0, 0)
        orig_bottom_right = (0, 0)
    # Else get the coordinates for the top_left and bottom_right corners
    else:
        orig_top_left  = torch.min(orig_nonzero, dim=0)[0]
        orig_bottom_right = torch.max(orig_nonzero, dim=0)[0]
    
    if len(seg_nonzero) == 0:
        seg_top_left = (0, 0)
        seg_bottom_right = (0, 0)
    else:
        seg_top_left = torch.min(seg_nonzero, dim=0)[0]
        seg_bottom_right = torch.max(seg_nonzero, dim=0)[0]
    # Calculate the center points of the two rectangles
    # top_left = (top_most, left_most)
    # bottom_right = (bottom_most, right_most)
    # center_x = left_most + (right_most - left_most)/2
    # center_y = top_most + (bottom_most - top_most)/2
    # center = (center_y, center_x)
    # center_orig = (y1, x1), center_seg = (y2, x2)
    center_x1 = int(orig_top_left[1] + (orig_bottom_right[1] - orig_top_left[1])/2)
    center_y1 = int(orig_top_left[0] + (orig_bottom_right[0] - orig_top_left[0])/2)
    # center for the segmented box
    center_x2 = int(seg_top_left[1] + (seg_bottom_right[1] - seg_top_left[1])/2)
    center_y2 = int(seg_top_left[0] + (seg_bottom_right[0] - seg_top_left[0])/2)
 #   print(f"orig_top_left: {orig_top_left}")
    orig_top_left = orig_top_left
    orig_bottom_right = orig_bottom_right
    # collect the bounding corder of the rectangle
    orig_coord = (orig_top_left, orig_bottom_right)
    seg_coord = (seg_top_left, seg_bottom_right)
    # calculate the centroid of the rectangle
    center_x_orig = (orig_bottom_right[1] - orig_top_left[1])/2
    center_y_orig = (orig_bottom_right[0] - orig_top_left[0])/2
    orig_center = (center_y1, center_x1)
    
    seg_center = (center_y2, center_x2)
    #print(f"type(orig_coord): {type(orig_coord)}")
    # return the values
    return orig_coord, orig_center, seg_coord, seg_center

# In[] Calculates the area of teh rectangle
def get_area(orig_coord, seg_coord):
    # unpack the orig_coord & seg_coord
    orig_top_left = orig_coord[0]
    orig_bottom_right = orig_coord[1]
    
    seg_top_left = seg_coord[0]
    seg_bottom_right = seg_coord[1]
    # calculate the width of both rectangles
    orig_width = orig_bottom_right[1] - orig_top_left[1]
    seg_width = seg_bottom_right[1] - seg_top_left[1]
    # claculate the height of both rectangles
    orig_height = orig_bottom_right[0] - orig_top_left[0]
    seg_height = seg_bottom_right[0] - seg_top_left[0]

    # calculate the area
    orig_area = orig_height * orig_width
    seg_area = seg_height * seg_width
    
    # calculate the ratio of the two areas
    if  seg_area == 0:
        return 0
    if orig_area > seg_area:
        return np.round((seg_area/orig_area)*100, 3)
    else:
        return np.round((orig_area/seg_area)*100, 3)


# In[] Calculates distance between the centroids
def get_dist_centroid(orig_center, seg_center):
    # calcualte the a in the pythagores theorm a² + b² = c²
    a = abs(orig_center[0] - seg_center[0])
    # calcualte the b in the pythagores theorm a² + b² = c²
    b = abs(orig_center[1] - seg_center[1])
    # calculate the c which is the distance b/n the centroids
    c = np.sqrt(a**2 + b**2)
    # return the distance
    return np.round(c, 3)

# In[] Pixel wise comparison 
def pixelwise_comparison(orig_mask, seg_mask):
    # Convert to torch tensor
    orig_mask = torch.tensor(orig_mask)
    seg_mask = torch.tensor(seg_mask)
    # select only nonzero values
    orig_nonzero = torch.nonzero(orig_mask)
    seg_nonzero = torch.nonzero(seg_mask)
    # calculate the percentile difference between the two binary file
    if len(seg_nonzero) == 0:
        return 0
    if len(seg_nonzero) > len(orig_nonzero):
        return np.round((len(orig_nonzero)/len(seg_nonzero))*100, 3)
    else:
        return np.round((len(seg_nonzero)/len(orig_nonzero))*100, 3)

# In[] Dice Loss: calculates the dice loss between two binary masks
def dice_loss(orig_mask, seg_mask, epsilon = 1e-6):
    # convert to float32 numpy array
    orig_mask = orig_mask.astype(np.float32)/255.
    seg_mask = seg_mask.astype(np.float32)/255.
    # Compute the intersection that is the sum of element wise multiplication
    intersection = np.sum(orig_mask *  seg_mask)
    # Compute the union between the two masks that is sum of element wise addition
    union = np.sum(orig_mask) + np.sum(seg_mask) + epsilon
    # Compute the dice coefficient
    dice_coef = (2 * intersection)/union
    # Calculate the dice loss
    dice_loss = 1 - dice_coef
    # return the dice loss
    return np.round(dice_loss, 3)

# In[] IntersectionOverUnion IoU
def intersection_over_union(orig_mask, seg_mask, eps = 1e-6):
    # convert to float32 numpy array
    orig_mask = orig_mask.astype(np.uint8)/255.
    seg_mask = seg_mask.astype(np.uint8)/255.
        
    # Calculate the intersection (logical AND)
    intersection = np.logical_and(orig_mask, seg_mask)
    
    # Calculate the union (logical OR)
    union = np.logical_or(orig_mask, seg_mask)
    
    # Sum the intersection and union
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)
    
    # Compute the IoU and subtract from 1 to make it a loss
    iou = intersection_sum /(union_sum + eps)
    # return the loss
    return np.round(1 - iou, 3)    

# In[] BCE  adn Dice Loss
def bce_dice_loss(pred, target, bce_weight=0.5, smooth=1e-6):
    # Ensure predictions are in the range [0, 1]
    pred = np.clip(pred, smooth, 1 - smooth)

    # 1. Binary Cross-Entropy (BCE) Loss
    bce_loss = -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))

    # 2. Dice Loss
    intersection = np.sum(pred * target)
    dice_loss = 1 - (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

    # 3. Combined BCE + Dice Loss
    bce_dice_loss = bce_weight * bce_loss + (1 - bce_weight) * dice_loss

    return bce_dice_loss
