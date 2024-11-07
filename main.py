#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:28:42 2024
Objective:
    - Combine the different tasks and models to evaluate the Test dataset with one shot
    
@author: dagi
"""
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import torch.nn as nn 
from Task_3.utils import accuracy_calculator
from Task_4.UNET_architecture import UNet
from ModelWithoutClassifierLayer import RemoveClassifierLayer
from torchinfo import summary
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
testPath = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Test/Classification/"
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

# In[] Set paths for the different models we have
model_1_path = "/home/dagi/Documents/PyTorch/MIP/final_project_3/Task_1/best_model.pth"
model_2_path = "/home/dagi/Documents/PyTorch/MIP/final_project_3/Task_2/best_model_2.pth"
model_3_path = "/home/dagi/Documents/PyTorch/MIP/final_project_3/Task_3/best_model_3.pth"
model_4_path = "/home/dagi/Documents/PyTorch/MIP/final_project_3/Task_4/best_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.BCEWithLogitsLoss() 

# In[] Load the model
model_1 = torch.load(model_1_path, weights_only=False)
model_2 = torch.load(model_2_path, weights_only=False)
model_3 = torch.load(model_3_path, weights_only=False)

model_1 = model_1.to(device)
model_2 = model_2.to(device)
model_3 = model_3.to(device)

# In[] iterate over teh dataloader
true_labels = []
predicted_labels = []
running_loss = 0.0
model_1.eval()
model_2.eval()
model_3.eval()

with torch.no_grad():
    for image, label in tqdm(test_dataloader, desc="Evaluating the model", leave=False):
        # Move data to cuda device
        image = image.to(device)
        # Configure the label value to meet the model criteria 
        if label != 1:
            # the mri scan has tumor
            model_label = 1 # model_1 thinks 1 is tumor in the scan
        else:
            # free from tumor
            model_label = 0
            
        model_1_label = torch.tensor(model_label, dtype=torch.int32) # convert int to tensor
        model_1_label = model_1_label.unsqueeze(0) # convert to 1-dimensional tensor
        model_1_label = model_1_label.to(device) # move the label to cuda device
        # Task 1: Classify the image as Healthy or have tumor
        model_1_pred = model_1(image).squeeze(dim=1) # forward propagation & discard batch size
        # convert the logits to binary predictions
        pred_label = (torch.sigmoid(model_1_label) >= 0.5).long()
        print(f"pred_label: {pred_label}")
        break 
        # If the model couldn't find Tumor in the scan operation ends here 
        if pred_label == 0: # model decided that no tumor on the mri scan
            # Accumulate the label value in the true_labels list
            true_labels.append(label.item())
            predicted_labels.append(1) # dataloader assigned 1 for No Tumor
            loss = loss_fn(model_1_pred, model_1_label.float()) # Calculate the loss value       
            running_loss += loss 
            continue
        else: # time to determine whether the tumor is pituitary (benign) or malignant
            # Configure the label value 
            if label != 3: # label is malignant
                model_label = 1
            else: # true lable is pituitary
                model_label = 0
            # Assign the right value to the label
            model_2_label = torch.tensor(model_label, dtype=torch.int32)
            model_2_label = model_2_label.unsqueeze(0) # convert to a 1-dimensional tensor
            model_2_label = model_2_label.to(device) # move the label to cuda device
            # Task 2: Classify the image as Benign or Malignant
            model_2_pred = model_2(image).squeeze(dim=1) # forward propagation then discard batch size
            # convert the logits to binary predictions
            pred_label = (torch.sigmoid(model_2_label) >= 0.5).long()
            # if the tumor is happened to be pituitary
            if pred_label == 0:
                true_labels.append(label.item())
                predicted_labels.append(3)
                loss = loss_fn(model_2_pred, model_2_label.float()) # Calculate the loss value       
                running_loss += loss 
                continue
            else: # Tumor is malignant
                # Configure the label value
                if label == 0: # Tumor is Glioma
                    model_label = 0
                else: # Tumor is meningioma
                    model_label = 2
                # Assign the right value to the label
                model_3_label = torch.tensor(model_label, dtype=torch.int32)
                model_3_label = model_3_label.unsqueeze(0) # convert to a 1-dimensional tensor
                model_3_label = model_3_label.to(device)
                # Task 3: Categorize the tumor as Glioma or Meningioma
                model_3_pred = model_3(image).squeeze(dim=1) 
                # Convert the logits to binary predictions
                pred_label = (torch.sigmoid(model_3_pred) >= 0.5).long()
                
                true_labels.append(label.item())     
                loss = loss_fn(model_3_pred, model_3_label.float())
                running_loss += loss
                if pred_label == 0: # Tumor is Glioma
                    predicted_labels.append(0)
                else: # Tumor is Meningioma
                    predicted_labels.append(2)
                    
# In[]
correct_predictions = accuracy_calculator(predicted_labels, true_labels)

print(f"Correct_predictions: {correct_predictions}/{len(test_dataloader)} ie an accuracy of {correct_predictions/len(test_dataloader):.3f}%")
accuracy = correct_predictions/len(test_dataloader)
txt_lines = ["System's Performance\n", f"\t- Model loss: {running_loss:.3f}\n", 
             f"\t- Model's classification accuracy: {accuracy:.3f}\n", f"\t- Model's prediction: {correct_predictions}/{len(test_dataloader)}"]
txt_path = "summary.txt"
with open(txt_path, 'w') as f:
    for line in txt_lines:
        f.write(line)
        
f.close()
# In[]
summary(model = model_1_,
        input_size = (BATCH, 1, HEIGHT, WIDTH),
        col_names = ["input_size", "output_size", "trainable"],
        col_width = 20,
        row_settings = ["var_names"])

# In[]


model_4 = UNet(in_channels=1, out_channels=1)
model_4_saved_dict = torch.load(model_4_path, weights_only=True)
# load the state dict onto the model
model_4.load_state_dict(model_4_saved_dict)

model_1 = model_1.to(device)
model_2 = model_2.to(device)
model_3 = model_3.to(device)
model_4 = model_4.to(device)

# In[] Set loss function

loss_fn = nn.BCEWithLogitsLoss()
# In[] Evaluation loop
actual_label = []
pred_label = []
crr_predictions = 0
running_loss = 0.0
loss = 0.0

correct_predictions = 0
incorrect_predictions = 0
# Set the model to evaluation mode
model.eval()
with torch.no_grad(): # disable gradient calculations
    for images, labels in test_dataloader: # iterate over the dataloader
        # Move data to cuda device
        images, labels = images.to(device), labels.to(device)
        # 2. Forward Propagation
        y_pred = model(images).squeeze(dim=1) # add batch size after squeezing
        
        loss += loss_fn(y_pred, labels.float())
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


