#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:06:53 2024
Objective:
    - This script evaluates the performance of the model on test dataset
@author: dagi
"""
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image 
import torch.nn as nn 
from utils import accuracy_calculator
from tqdm import tqdm 

# In[] functin that counts the total number of mri scans
def count_scans(directory):
    counted_file = 0
    testFiles = []
    for root, dirs, files in os.walk(directory):
        # set full file path
        print(f"root: {root}")
        for scan in files:
            full_path = root + "/" + scan
            # print(f"full_path: {full_path}")
            testFiles.append(full_path)
        counted_file += len(files)
    return counted_file, testFiles

# In[] Set route path
testPath = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Model_2/Test/"
no_test, testFiles = count_scans(testPath)

# In[] Setting Hyperparameter
WIDTH = 256
HEIGHT = 256
OUTPUT_SHAPE = 1
BATCH = 1

# In[] Set transform function
transform_fn = transforms.Compose([
                        # Resize the image to fit the model
                        transforms.Resize(size=(HEIGHT, WIDTH)),
                        # Make sure each image is in grayscale
                        transforms.Grayscale(num_output_channels=1),
                        # Convert image to tensor object
                        transforms.ToTensor(),
                        # Normalize the 3-channeled tensor object
                        transforms.Normalize(mean=[0.18442], std = [0.16092])
                ])

# In[] Setting up the dataset and dataloader
test_dataset = datasets.ImageFolder(
                    root = testPath,
                    transform = transform_fn,
                    target_transform=None)


test_dataloader = DataLoader(
                        dataset = test_dataset,
                        batch_size = BATCH,
                        num_workers = 4,
                        shuffle = False,
                        pin_memory = True)

# In[] Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "best_model_2.pth"
model = torch.load(model_path, weights_only=False)

model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
# In[] Evaluation loop
actual_label = []
pred_label = []
crr_preds = 0
running_loss = 0.0

correct_predictions = 0
incorrect_predictions = 0
# Set the model to evaluation mode
model.eval()
with torch.no_grad(): # disable gradient calculations
    for images, labels in tqdm(test_dataloader, desc="Testing", leave=False): # iterate over the dataloader
        # Move data to cuda device
        images, labels = images.to(device), labels.to(device)
        # 2. Forward Propagation
        y_pred = model(images).squeeze(dim=1) # add batch size after squeezing
        # Calculate the loss
        loss = criterion(y_pred, labels.float())
        # Convert the logits to binary predictions
        pred_labels = (torch.sigmoid(y_pred) >= 0.5).long() 
        # Accumulate the values in the actual_label and pred_label lists
        actual_label.append(labels.cpu())
        pred_label.append(pred_labels.cpu())
    # Collect the output of an epoch
    actual_label = torch.cat(actual_label)
    pred_label = torch.cat(pred_label)
    running_loss = loss/len(test_dataloader)   
    # Calculate the average loss and accuracy
    crr_preds = accuracy_calculator(pred_label, actual_label)

# In[]   
print(f"Correct_predictions: {crr_preds}/{len(test_dataloader)} ie an accuracy of {crr_preds/len(test_dataloader):.3f}%")
accuracy = crr_preds/len(test_dataloader)
txt_lines = ["Model_2's Performance\n", f"\t- Model loss: {running_loss:.3f}\n", 
             f"\t- Model's classification accuracy: {accuracy:.3f}\n", f"\t- Model's prediction: {crr_preds}/{len(test_dataloader)}"]
txt_path = "model_2_test_summary.txt"
with open(txt_path, 'w') as f:
    for line in txt_lines:
        f.write(line)
        
f.close()




