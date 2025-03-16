import os
import numpy as np
import torch
print(torch.__version__)
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
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

n = args.n
print(n)
def data_loader(train_path,val_path):
    
    transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),          
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
    ])
  
    
    train_dataset = ImageFolder(root=train_path, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset =  ImageFolder(root=val_path, transform=transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader,val_loader

tp = os.path.join(args.test_data_file,'train')
vp = os.path.join(args.test_data_file,'val')
train_loader, val_loader= data_loader(tp,vp)
print(1)

def layer_normalization(dim):
    if args.normalization == "torch_bn":
        return nn.BatchNorm2d(dim)

    elif args.normalization == "BN":
        return BatchNorm(num_features=dim)

    elif args.normalization == "NN":
        return NoNorm()

    elif args.normalization == "IN":
        return InstanceNorm(num_features=dim)

    elif args.normalization == "LN":
        return LayerNorm(num_features=dim)
    
    elif args.normalization == "GN":
        return GroupNorm(num_features=dim)

    elif args.normalization == "BIN":
        return BatchInstanceNorm(num_features=dim)



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

def train(train_loader, val_loader):

    net = ResNet(n_classes = args.r)
#     print(net)
    net = net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3,weight_decay=1e-4, momentum=0.9)
    schedulers = [
        optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=- 1),
        optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
        ]
    scheduler =  schedulers[1] #Check for self.epochs param
    
    loss_tracker = defaultdict(list)
    accuracy_tracker = defaultdict(list)
    micro_f1  = defaultdict(list)
    macro_f1 = defaultdict(list)
    time_tracker = defaultdict(list)
    ft_quantile_tracker = defaultdict(list)

    best_accuracy = -1
    best_accu_epoch = -1

    print("\n\n---------------------------- MODEL TRAINING BEGINS ----------------------------")
        
    t0 = time()
    
    for epoch in range(50):
        print("\n#------------------ Epoch: %d ------------------#" % epoch)

        train_loss = []
        correct_pred = 0
        total_samples = 0
        train_pred = []
        train_label = []
        val_pred = []
        val_label = []
        net.train()
        for idx, batch in enumerate(train_loader):
#             print(idx)

            optimizer.zero_grad()
            
            images, labels = batch
            # print(images)
            images = images.to(device)
            labels = labels.to(device)
            # print(images)
            outputs = net(images)
            # print(len(outputs),len(labels))
            loss = criterion(outputs, labels)
            
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            _, pred = outputs.max(1)
#             print(pred.detach().cpu(),labels.detach().cpu())
            train_pred.extend(pred.detach().cpu())
            train_label.extend(labels.detach().cpu())


        loss_tracker["train"].append(np.mean(train_loss))
        accuracy_tracker["train"].append(accuracy_score(train_label,train_pred))
        micro_f1["train"].append(f1_score(train_label,train_pred,average='micro'))
        macro_f1["train"].append(f1_score(train_label,train_pred,average='macro'))
#         print(time()-t0)
        scheduler.step()
        print()
        print("validating...")
        net.eval()
        correct_pred = 0
        total_samples = 0
        val_loss = []
        feature_list = []
        for idx, batch in enumerate(val_loader):
            val_pred = []
            val_label = []
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)

            loss = criterion(outputs, labels)        
            val_loss.append(loss.item())

            _, pred = outputs.max(1)
            val_pred.extend(pred.detach().cpu())
            val_label.extend(labels.detach().cpu())

        
        loss_tracker["val"].append(np.mean(val_loss))
        val_accuracy = accuracy_score(val_label,val_pred)
        accuracy_tracker["val"].append(val_accuracy)
        micro_f1["val"].append(f1_score(val_label,val_pred,average='micro'))
        macro_f1["val"].append(f1_score(val_label,val_pred,average='macro'))


        t1 = time()

        print("Epoch: {}, Total Time Elapsed: {}Mins, Train Loss: {}, Train Accuracy: {},microf1_train :{}, macrof1_train: {},  Validation Loss: {}, Validation Accuracy: {},microf1_val: {}, macrof1_val: {}".format(epoch, round((t1-t0)/60,2), round(loss_tracker["train"][-1],3), round(accuracy_tracker["train"][-1],3), round(micro_f1['train'][-1],3), round(macro_f1['train'][-1],3), round(loss_tracker["val"][-1],3), round(accuracy_tracker["val"][-1],3), round(micro_f1['val'][-1],3), round(macro_f1['val'][-1],3)))
       
    item = {'train_loss' : loss_tracker['train'],
           'train_accuracy' : accuracy_tracker['train'],
           'microf1_train' : micro_f1['train'],
           'macrof1_train' : macro_f1['train'],
            'val_loss' : loss_tracker['val'],
           'val_accuracy' : accuracy_tracker['val'],
           'microf1_val' : micro_f1['val'],
           'macrof1_val' : macro_f1['val']}
    df = pd.DataFrame(item)
    df.to_csv(os.path.join(args.output_file,f'resnet_{args.normalization}_stats.csv'),index=False)
    torch.save(net.state_dict(), os.path.join(args.output_file,f"resnet_{args.normalization}.pth"))
    return

print(2)
train(train_loader, val_loader)

