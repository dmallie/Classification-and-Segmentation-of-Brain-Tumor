#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:45:24 2024

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

# In[] Set route path
testPath = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Task_1_classification/"
# no_test, testFiles = count_scans(testPath)

# In[] Setting Hyperparameter
WIDTH = 256
HEIGHT = 256
OUTPUT_SHAPE = 1
BATCH = 1

# In[] Set transform function
transform_fn = transforms.Compose([
                        # Resize the image to fit the model
                        transforms.Resize(size=(HEIGHT, WIDTH)),
                        # Convert image to grayscale 
                        transforms.Grayscale(num_output_channels=1),
                        # Convert image to tensor object
                        transforms.ToTensor(),
                        # Normalize the 3-channeled tensor object
                        transforms.Normalize(mean=[0.17812], std = [0.17263])
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
model_path = "/home/dagi/Documents/PyTorch/MIP/final_project_3/Task_1/best_model.pth"
model = torch.load(model_path, weights_only=False)

model = model.to(device)

loss_fn = nn.BCEWithLogitsLoss()

# In[] Evaluation loop
actual_label = []
pred_label = []
crr_predictions = 0
running_loss = 0.0
loss = 0.0

correct_predictions = 0
incorrect_predictions = 0

with torch.no_grad():
    for image, label in tqdm(test_dataloader, desc="Evaluating the model", leave=False):
        # Move data to cuda device
        image, label = image.to(device), label.to(device)
# 2. Forward Propagation
        y_pred = model(image).squeeze(dim=1) # add batch size after squeezing
        
        loss += loss_fn(y_pred, label.float())
        # Convert the logits to binary predictions
        pred_labels = (torch.sigmoid(y_pred) >= 0.5).long() 
        # Accumulate the values in the actual_label and pred_label lists
        actual_label.append(label.cpu())
        pred_label.append(pred_labels.cpu())
    # Collect the output of an epoch
    actual_label = torch.cat(actual_label)
    pred_label = torch.cat(pred_label)
    running_loss = loss/len(test_dataloader) 
    # Calculate the average loss and accuracy
    correct_preds = accuracy_calculator(pred_label, actual_label)
    
# In[]
print(f"Correct_predictions: {correct_preds}/{len(test_dataloader)} ie an accuracy of {correct_preds/len(test_dataloader):.3f}%")
accuracy = correct_preds/len(test_dataloader)
txt_lines = ["Model_1 detects tumor from a brain MRI scan.\n","Model_1's Performance\n", f"\t- Model loss: {running_loss:.3f}\n", 
             f"\t- Model's classification accuracy: {accuracy:.3f}\n", f"\t- Model's prediction: {correct_preds}/{len(test_dataloader)}"]
txt_path = "model_1_test_summary.txt"
with open(txt_path, 'w') as f:
    for line in txt_lines:
        f.write(line)
        
f.close()
print(f"Correct_predictions: {correct_preds}/{len(pred_label)} ie an accuracy of {correct_preds/len(pred_label):.3f}")
