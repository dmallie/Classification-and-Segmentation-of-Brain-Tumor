#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:09:43 2024
Objective:
    - Setup Training Loop : training_loop
    - Setup Validation Loop : val_loop
    - Setup the Loop which combines them : combo_loop
@author: dagi
"""
import torch
import torch.optim as optim
from tqdm import tqdm
from early_stoppage import EarlyStopping
from utils import precision, accuracy_calculator

# In[] Set Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
earlyStopping = EarlyStopping(patience=15, min_delta=0.01)

# In[] Training loop
def training_loop(model, dataloader, optimizer, criterion):
    # To accumulate the model's output
    actual_labels = []
    predicted_labels = []
    # Reset parameters to calculate the loss & accuracy
    running_loss = 0
    # 1. Activate the training mode
    model.train()
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # 2. Forward pass
        predictions = model(images)
        loss = criterion(predictions.squeeze(dim=1), labels.float())
        
        # 3. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        # 4. Update the weights
        optimizer.step()
        
        # update the loss
        running_loss += loss.item()
        
        # Convert the output to prediction probabilities 
        output = (torch.sigmoid(predictions) >= 0.5).long() 
        
        predicted_labels.append(output.cpu())
        actual_labels.append(labels.cpu())
        
    # Collect the output of an epoch
    actual_labels = torch.cat(actual_labels)
    predicted_labels = torch.cat(predicted_labels)
    
    # Calculate the average loss and accuracy
    epoch_loss = running_loss / len(dataloader)
    epoch_precision = precision(predicted_labels, actual_labels)
    epoch_accuracy =  accuracy_calculator(predicted_labels, actual_labels)
    return epoch_loss, epoch_precision, epoch_accuracy/len(actual_labels)

# In[] Validation Loop
def validation_loop(model, dataloader, criterion):
    running_loss = 0.0
    actual_labels = []
    predicted_labels = []
    # Set model's mode to evaluation
    model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            # Move the data to the cuda device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            predictions = model(images)
            # Calculate the loss
            loss = criterion(predictions.squeeze(dim=1), labels.float())
            
            # convert predictions to probabilities
            output = (torch.sigmoid(predictions) >= 0.5).long()
            running_loss += loss.item()
            
            actual_labels.append(labels.cpu())
            predicted_labels.append(output.cpu())
            
    # Collect the output of an epoch
    actual_labels = torch.cat(actual_labels)
    predicted_labels = torch.cat(predicted_labels)
    
    # Calculate the average loss and accuracy
    epoch_loss = running_loss / len(dataloader)
    epoch_precision = precision(predicted_labels, actual_labels)
    epoch_accuracy = accuracy_calculator(predicted_labels, actual_labels)
    return epoch_loss, epoch_precision, epoch_accuracy/len(actual_labels)
    
# In[] Main loop
def main_loop(model, train_dataloader, val_dataloader,
              optimizer, criterion, epochs, scheduler,save_path="best_model.pth"):
    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        
        # Training
        train_loss, train_precision, train_accuracy = training_loop(model, train_dataloader, optimizer, criterion)
        print(f"Training Loss: {train_loss:.4f}\t Training Accuracy: {train_accuracy:.2f}\t Training Precision: {train_precision:.2f}")
        
        # Validation
        val_loss, val_precision, val_accuracy = validation_loop(model, val_dataloader, criterion)
        print(f"Validation Loss: {val_loss:.4f}\t Validation Accuracy: {val_accuracy:.2f}\t Val Precision: {val_precision:.2f}")
        
            
        # Step the scheduler if validation loss improves
        scheduler.step(val_loss)

        # Early stopping
        earlyStopping(val_loss)
        if earlyStopping.early_stop:
            print("Early stopping")
            break

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, save_path)
            print("Model saved!")
        
        print("-" * 30)
    return train_accuracy, val_accuracy