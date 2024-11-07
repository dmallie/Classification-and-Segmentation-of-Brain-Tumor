#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:54:58 2024
Objective:
    - Remove the classifier layer while retaining the trained base layer
@author: dagi
"""
import torch.nn as nn 
import torch 

# In[]
class RemoveClassifierLayer(nn.Module):
    def __init__(self, model):
        super(RemoveClassifierLayer, self).__init__()
        # Load model up to (but not including) the FC layer
        self.model_base = nn.Sequential(*list(model.children())[:-1])
        
    def forward(self, x):
        x = self.model_base(x)
        return x.squeeze()  # Remove extra dimensions if needed

# In[]
class IntegratedModel(nn.Module):
    def __init__(self, model1, model2, model3, num_classes):
        super(IntegratedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.num_classes = num_classes
        
        self.height = 256 
        self.width = 256
        # Assume each model output is a vector of the same size
        # Adjust input size if the concatenated output size differs
        output_size = self.model1(torch.randn(1, 1, self.height, self.width)).view(-1).size(0) \
                      + self.model2(torch.randn(1, 1, self.height, self.width)).view(-1).size(0) \
                      + self.model3(torch.randn(1, 1, self.height, self.width)).view(-1).size(0)

        # Final classification layer
        self.classifier = nn.Linear(output_size, self.num_classes)

    def forward(self, x):
        # Forward pass through each model
        x1 = self.model1(x).view(x.size(0), -1)  # Flatten outputs
        x2 = self.model2(x).view(x.size(0), -1)
        x3 = self.model3(x).view(x.size(0), -1)

        # Concatenate outputs
        x_concat = torch.cat((x1, x2, x3), dim=1)

        # Final classifier layer
        out = self.classifier(x_concat)
        return out