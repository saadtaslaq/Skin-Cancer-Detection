import numpy as np 
from random import shuffle
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import time
import torchvision.models as models


#Set device to GPU_indx if GPU is avaliable
GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')
print(device)

data_path ="data"
batch_size = 32


class SkinDataset(Dataset):

    def __init__(self, root_data_dir, training = True):

        self.to_tensor = torchvision.transforms.ToTensor()
        self.dir = root_data_dir
        self.is_train = training

        self.data = []
        if self.is_train:
            n_benign = 1440
            n_malignant = 1197
            train_type = '/train/'
        else:
            n_benign = 360
            n_malignant = 300
            train_type = '/test/'

        for i in range(n_benign):
            image = Image.open(self.dir + train_type + 'benign/' + str(i) + '.jpg')
            image = self.to_tensor(image)
            self.data.append((image, 0))

        for i in range(n_benign):
            image = Image.open(self.dir + train_type + 'malignant/' + str(i) + '.jpg')
            image = self.to_tensor(image)
            self.data.append((image, 1))


    # returns a signle image and label pair            
    def __getitem__(self, index):
        # # load from training
        # if self.is_train:
        #     # load from benign (1440 samples)
        #     if index < 1440:
        #         image = Image.open(self.dir + '/train/benign/'+ str(index) + '.jpg')
        #         image = self.to_tensor(image)
        #         return image, self.to_tensor(0).type(torch.LongTensor)
        #     # load from malignant
        #     else:
        #         image = Image.open(self.dir + '/train/malignant/'+ str(index-1440) + '.jpg')
        #         image = self.to_tensor(image)
        #         return image, self.to_tensor(1).type(torch.LongTensor)
        # # load from testing
        # else:
        #     if index < 360:
        #         image = Image.open(self.dir + '/test/benign/'+ str(index) + '.jpg')
        #         image = self.to_tensor(image)
        #         return image, self.to_tensor(0).type(torch.LongTensor)
        #     else:
        #         image = Image.open(self.dir + '/test/malignant/'+ str(index-360) + '.jpg')
        #         image = self.to_tensor(image)
        #         return image, self.to_tensor(1).type(torch.LongTensor)
        return self.data[index]

    def __len__(self):
        if self.is_train:
            return 2637  # 1440+1197
        else:
            return 660  # 360+300


## Create a dataset and dataloader

dataset_train = SkinDataset(data_path)
dataset_test = SkinDataset(data_path, False)
data_loader_train = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle = True)
data_loader_test = DataLoader(dataset = dataset_test, batch_size=batch_size, shuffle = False)

## Sanity Check

dataloader_iterable = iter(data_loader_train)
image, label = next(dataloader_iterable)

plt.figure(figsizse = (20,10))
img_out = torchvision.utils.make_grid(((image+1)/2)[0:4], 4)
lbl_out = torchvision.utils.make_grid(label[0:4].unsqueeze(1), 4).float()

out = torch.cat((img_out, lbl_out), 1)

plt.imshow(out.numpy().transpose((1, 2, 0)))

## Create model and Optimizer

model = Alex_net_alike().to(device)
learning_rate = 1e-4
n_epochs = 40
optimiser = optim.Adam(model.parameters(), lr = learning_rate)

loss_fun = nn.CrossEntropyLoss()


## Model just a resnet right now
res_net = models.resnet18(pretrained=False, num_classes=2).to(device)
Alex_net = models.alexnet(pretrained=False, num_classes=2).to(device)

# Train




# Evaluate





