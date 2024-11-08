#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 22:49:20 2024

@author: dagi
"""

# In[] Setup the accuracy calculator
def accuracy_calculator(y_pred = None, y_true = None):
    # find the labels of predicted values from y_pred
    # pred_labels = round(y_pred)
    # print(f"len(pred_labels): {len(pred_labels)}")
    # compare the predicted labels with true labels & count correctly predicted labels
    # correct_predictions = torch.eq(y_true, pred_labels).sum().item()
    correct_predictions = 0
    wrong_predictions = 0
    for index in range(len(y_pred)):
        if round(round(y_pred[index])) == y_true[index]:
            correct_predictions += 1
        else:
            wrong_predictions += 1
    # calculate the incorrect ones
    return correct_predictions
