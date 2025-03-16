import argparse
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
# from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split,TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from collections import defaultdict
import json 
from time import time
from tqdm import tqdm
from datetime import datetime
import cv2
import random
from PIL import Image
import torch.nn.functional as F
from matplotlib.cm import inferno

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.model.eval()

    def save_gradients(self, grad):
        self.gradients = grad

    def save_activations(self, act):
        self.activations = act

    def forward_backward_pass(self, x, y):
        x = x.unsqueeze(0)  # Add batch dimension
        y = torch.tensor([y], dtype=torch.long).to(x.device)

        # Forward pass
        self.activations = None
        self.gradients = None
        handle_forward = self.target_layer.register_forward_hook(lambda m, i, o: self.save_activations(o))
        handle_backward = self.target_layer.register_full_backward_hook(lambda m, i, o: self.save_gradients(o[0]))

        output = self.model(x)
        loss = F.cross_entropy(output, y)
        self.model.zero_grad()
        loss.backward()

        handle_forward.remove()
        handle_backward.remove()

        return output

    def generate_heatmap(self, x):
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3], keepdim=True)
        self.activations *= pooled_gradients  # Weight the channels by the gradients
        heatmap = torch.mean(self.activations, dim=1).squeeze().cpu()
        heatmap = F.relu(heatmap)  # Apply ReLU to zero-out negatives
        heatmap /= torch.max(heatmap)  # Normalize
        return heatmap.detach().numpy()

    def overlay_heatmap(self, heatmap, image, alpha=0.5):
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)  # Convert to 8-bit
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * alpha + image * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        return superimposed_img

    def process_image(self, img_tensor, class_idx, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
        inv_normalize = transforms.Normalize(mean=[-m/s for m, s in zip(norm_mean, norm_std)], std=[1/s for s in norm_std])
        img_normalized = inv_normalize(img_tensor)
        return transforms.ToPILImage()(img_normalized)

def grad_cam_analysis(model, data_loader, target_layer, correct=True, mapping=None, num_images=5):
    cam = GradCAM(model, target_layer)
    results = [[] for _ in range(len(mapping))]
    device = next(model.parameters()).device

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        for img, label in zip(x, y):
            label_idx = label.item()
            if label_idx in mapping:
                output = cam.forward_backward_pass(img, label)
                pred = output.argmax(dim=1).item()
                if (correct and label_idx == pred) or (not correct and label_idx != pred):
                    if len(results[mapping[label_idx]]) < num_images:
                        heatmap = cam.generate_heatmap(img)
                        processed_img = cam.process_image(img, label_idx)
                        heatmap_img = cam.overlay_heatmap(heatmap, np.array(processed_img))
                        results[mapping[label_idx]].append(heatmap_img)
    return results

# Example usage:
# model = YourModel()
target_layer = model.module.layer3  # or whichever layer you choose
# val_loader = YourValidationDataLoader()
# test_loader = YourTestDataLoader()
mapping = {2:0, 7:1, 13:2, 18:3, 19:4, 22:5, 23:6}

correctly_classified = grad_cam_analysis(model, val_loader, target_layer, correct=True, mapping=mapping)
incorrectly_classified = grad_cam_analysis(model, test_loader, target_layer, correct=False, mapping=mapping)
print('complete')
