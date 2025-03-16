import argparse
import os
import pandas as pd
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
from DL_A1_param import args
from scratch_norm import NoNorm, BatchNorm, InstanceNorm, LayerNorm, GroupNorm, BatchInstanceNorm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
test = args.test_data_file
test_file_paths = [os.path.join(test, file) for file in os.listdir(test)]

transform = transforms.Compose([
    transforms.Resize((256,256)),  # Resize the image to a common size
    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5), (0.5,0.5,0.5))  # Normalize pixel values
])

# Load and preprocess each image
test_tensors = []
for c,file_path in enumerate(test_file_paths):
    try:
    # Open the image using PIL
        img = Image.open(file_path)
    
    # Apply the defined transformation
        img_tensor = transform(img)
    
        if c==0:
            test_tensors = img_tensor.unsqueeze(0)
            # print(resume_tensors)
        else:
    # Add the preprocessed image tensor to the list
            # print(resume_tensors.shape,img_tensor.unsqueeze(0).shape)
            test_tensors = torch.cat((test_tensors,img_tensor.unsqueeze(0)))
    except:
        pass
        
test_dataset = TensorDataset(test_tensors, torch.ones(int(len(test_tensors))))
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
n = args.n

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = layer_normalization(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = layer_normalization(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                layer_normalization(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.layers = nn.ModuleList([
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
         layer_normalization(16),
         nn.ReLU(inplace=True),
        self.make_layer(block, 16, layers[0], stride=1),
        self.make_layer(block, 32, layers[1], stride=2),
         self.make_layer(block, 64, layers[2], stride=2),
         nn.AvgPool2d(kernel_size = 8),
        nn.Flatten(),
        nn.Linear(4096, num_classes)])

    def make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
        return x

def ResNet18(num_classes):
    return ResNet(ResidualBlock, [n, n, n],num_classes)
    
    
def eval(test_loader):

    net = ResNet18(num_classes=25)
#     print(net)
    checkpoint = torch.load(args.model_file,map_location=device)
    
    net = net.to(device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:]
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3,weight_decay=1e-4, momentum=0.9)
    schedulers = [
        optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=- 1, verbose=False),
        optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, verbose=False)
        ]
    scheduler =  schedulers[1] #Check for self.epochs param
    
#     val_accuracy_tracker = defaultdict(list)
    test_accuracy_tracker = defaultdict(list)
    micro_f1  = defaultdict(list)
    macro_f1 = defaultdict(list)


    
    net.eval()

    val_loss = []
    predictions = []
    for idx, batch in enumerate(test_loader):
            val_pred = []
            val_label = []
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)

                
           

            _, pred = outputs.max(1)
            val_pred.extend(pred.detach().cpu())
            val_label.extend(labels.detach().cpu())
            predictions.append(pred.item())
        
#     loss_tracker["val"].append(np.mean(val_loss))
    val_accuracy = accuracy_score(val_label,val_pred)
    print(val_accuracy)
    test_accuracy_tracker["val"].append(val_accuracy)
    micro_f1["val"].append(f1_score(val_label,val_pred,average='micro'))
    macro_f1["val"].append(f1_score(val_label,val_pred,average='macro'))

    item = {"image_class" : predictions}
    df = pd.DataFrame(item)
    print('here')
    df.to_csv(args.output_file,index=False)
    t1 = time()

        
# for i in ['in','ln','gn','nn']:
if args.normalization == 'inbuilt':
        def layer_normalization(dim):
            return nn.BatchNorm2d(dim)
        
        eval(test_loader)
elif args.normalization == 'bn':
        def layer_normalization(dim):
            return nn.BatchNorm2d(dim) 
        eval(test_loader) 
elif args.normalization == 'nn':
        def layer_normalization(dim):
            return NoNorm()
        eval(test_loader)
elif args.normalization == 'in':
        def layer_normalization(dim):
            return InstanceNorm(dim)
        eval(test_loader)  
elif args.normalization == 'ln':
        def layer_normalization(dim):
            return LayerNorm(dim)  
        eval(test_loader)  
elif args.normalization == 'gn':
        def layer_normalization(dim):
            return GroupNorm(dim)
        eval(test_loader)
elif args.normalization == 'bin':
        def layer_normalization(dim):
            return BatchInstanceNorm(dim)
        eval(test_loader)

print(f'testing for {args.normalization}')


