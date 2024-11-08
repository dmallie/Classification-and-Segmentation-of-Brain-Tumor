#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:06:07 2024

@author: dagi
"""
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import torch.nn as nn 
from utils import accuracy_calculator
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
                        # Normalize the tensor object
                        transforms.Normalize(mean=[0.1805], std = [0.1627])
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

# In[] Helper function to set labels for each model
def get_model_label(task, label):
    """Function to determine model-specific labels."""
    if task == "model_1":
        return 1 if label != 1 else 0  # Adjust based on your label needs
    elif task == "model_2":
        return 1 if label != 3 else 0
    elif task == "model_3":
        return 0 if label == 0 else 1
    else:
        raise ValueError("Unknown task.")

# In[] Helper function for model prediction and thresholding
def predict_with_threshold(model, image, threshold=0.5):
    logits = model(image).squeeze(dim=1)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).long()
    return predictions, logits

# In[] Prepare models and evaluation mode
model_1.eval()
model_2.eval()
model_3.eval()

true_labels = []
predicted_labels = []
running_loss = 0.0

count_pituitary = 0
labels = []
with torch.no_grad():
    for index, (image, label) in enumerate(tqdm(test_dataloader, desc="Evaluating the model", leave=False)):
        image, label = image.to(device), label.to(device)

        # Model 1 prediction
        model_1_label = torch.tensor(get_model_label("model_1", label.item())).unsqueeze(0).to(device)
        model_1_pred, model_1_logits = predict_with_threshold(model_1, image)
        if model_1_pred == 0:  # No tumor found
            true_labels.append(label.item())
            predicted_labels.append(1)
            running_loss += loss_fn(model_1_logits, model_1_label.float())
            continue  # Skip to the next image if no tumor is found

        # Model 2 prediction
        model_2_label = torch.tensor(get_model_label("model_2", label.item())).unsqueeze(0).to(device)
        if model_2_label == 0:
            count_pituitary += 1
            labels.append(label.item())
        model_2_pred, model_2_logits = predict_with_threshold(model_2, image)
        if model_2_pred == 0:  # Tumor is pituitary
            true_labels.append(label.item())
            predicted_labels.append(3)
            running_loss += loss_fn(model_2_logits, model_2_label.float())
            continue

        # Model 3 prediction
        model_3_label = torch.tensor(get_model_label("model_3", label.item())).unsqueeze(0).to(device)
        model_3_pred, model_3_logits = predict_with_threshold(model_3, image)
        true_labels.append(label.item())
        running_loss += loss_fn(model_3_logits, model_3_label.float())
        predicted_labels.append(0 if model_3_pred == 0 else 2)  # Append prediction for Glioma or Meningioma


# In[] 
glioma, healthy, meningioma, pituitary = 0, 0, 0, 0
for index in range(len(true_labels)):
    if true_labels[index] == 0: # glioma
        if predicted_labels[index] == 0:
            glioma += 1 
    elif true_labels[index] == 1: # healthy
        if predicted_labels[index] == 1:
            healthy += 1 
    elif true_labels[index] == 2:# meningioma
        if predicted_labels[index] == 2:
            meningioma += 1 
    elif true_labels[index] == 3: # Pituitary
        if predicted_labels[index] == 3:
            pituitary += 1 

# In[]
healthy_size = len(os.listdir(testPath+"Healthy/"))
pituitary_size = len(os.listdir(testPath+"Pituitary/"))
glioma_size = len(os.listdir(testPath+"Glioma/"))
meningioma_size = len(os.listdir(testPath + "Meningioma/"))

print(f"Healthy: {healthy} ie {healthy/healthy_size:.3f}\t Pituitary: {pituitary} ie {pituitary/pituitary_size:.3f}\nGlioma: {glioma} ie {glioma/glioma_size:.3f}\t Meningioma: {meningioma} ie {meningioma/meningioma_size:.3f}")

# In[] Calculate metrics after loop
correct_predictions = accuracy_calculator(predicted_labels, true_labels)
accuracy = correct_predictions / len(test_dataloader)
average_loss = running_loss / len(test_dataloader)

# Log performance
txt_lines = [
    "System's Performance\n",
    f"\t- Model loss: {average_loss:.3f}\n",
    f"\t- Model's classification accuracy: {accuracy:.3f}\n",
    f"\t- Model's prediction: {correct_predictions}/{len(test_dataloader)}\n",
    f"\t- Accuracy of Healthy Dataset: {healthy}/{healthy_size} ie {healthy/healthy_size*100:.3f}%\n",
    f"\t- Accuracy of Pituitary Dataset: {pituitary}/{pituitary_size} ie {pituitary/pituitary_size*100:.3f}%\n",
    f"\t- Accuracy of Glioma Dataset: {glioma}/{glioma_size} ie {glioma/glioma_size*100:.3f}%\n",
    f"\t- Accuracy of Meningioma Dataset: {meningioma}/{meningioma_size} ie {meningioma/meningioma_size*100:.3f}%\n",
]
with open("summary.txt", 'w') as f:
    f.writelines(txt_lines)

print(f"Correct predictions: {correct_predictions}/{len(test_dataloader)}")
print(f"Accuracy: {accuracy:.3f}, Loss: {average_loss:.3f}")