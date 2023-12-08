import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F
import os
from PIL import Image
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataframe = pd.read_csv('./data.csv')
test_df = pd.read_csv('./testdata.csv')

from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(train_dataframe, shuffle=True, test_size=0.01, stratify=train_dataframe['Label'])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, train='train', transform=None):
        if train == 'train':
            self.image_list = []
            self.label_list = []
            self.other_list = []
            path = './dataset/{}/{}'
            for index, row in dataframe.iterrows():
                image_path = row['Image']
                image_label = row['Label']
                image_age = row['Age']
                image_gender = row['Gender']
                image_race = row['Race']
                image = Image.open(path.format(image_label, image_path)).convert('RGB')
                # if there is transform, apply transform
                if transform != None:
                    image = transform(image)
                self.image_list.append(image)
                self.label_list.append(image_label)
                self.other_list.append((image_age, image_gender, image_race))

        elif train == 'test':
            self.image_list = []
            self.label_list = []  # 이미지의 경로
            self.other_list = []
            path = './testset/{}'
            for index, row in dataframe.iterrows():
                image_path = row['Image']
                image_gender = row['Gender']
                image_race = row['Race']
                image = Image.open(path.format(image_path)).convert('RGB')
                if transform != None:
                    image = transform(image)
                self.image_list.append(image)
                self.label_list.append(image_path)
                self.other_list.append((image_gender, image_race))
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return self.image_list[idx], self.label_list[idx], self.other_list[idx]

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.6029, 0.4615, 0.3949),(0.2195, 0.1960, 0.1866))
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.6029, 0.4615, 0.3949),(0.2195, 0.1960, 0.1866))
])

train_dataset = CustomDataset(train_df, train='train', transform=train_transform) # 5138
valid_dataset = CustomDataset(valid_df, train='train', transform=test_transform) # 2202
test_dataset = CustomDataset(test_df, train='test', transform=test_transform) # 822

print(len(train_dataset))
print(len(valid_dataset))
print(len(test_dataset))

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

classes = ['1 ~ 10', '11 ~ 20', '21 ~ 30', '31 ~ 40', '41 ~ 50']

data, label, other =  next(iter(train_loader))
print('train loader')
print(data.shape, label.shape)

image, file_name, other = next(iter(test_loader))
print('test loader')
print(image.shape, len(file_name))

import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

#=================================do1==========================================================================================================

class ConvNet_do1(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(True))
        self.cn2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn7 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.5))
        self.cn8 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.fc1 = nn.Sequential(nn.Linear(266 , 32),
                                 nn.BatchNorm1d(32),
                                 nn.ReLU(True),
                                 nn.Dropout(0.5),

                                 nn.Linear(32, 5))

    def forward(self, x, other_gender, other_race):
        x = self.cn1(x)
        x = self.cn2(x)
        x = self.cn3(x)
        x = self.cn4(x)
        x = self.cn5(x)
        x = self.cn6(x)
        x = self.cn7(x)
        x = self.cn8(x)

        x = x.view(x.size(0), -1)

        other_gender = torch.nn.functional.one_hot(other_gender, 5)
        other_race = torch.nn.functional.one_hot(other_race, 5)

        other_gender = other_gender.view(-1, 5)
        other_race = other_race.view(-1, 5)

        x = torch.cat((x, other_gender), dim=1)
        x = torch.cat((x, other_race), dim=1)

        x = self.fc1(x)

        return x

#=================================do2==========================================================================================================

class ConvNet_do2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(True))
        self.cn2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.5))
        self.cn7 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.5))
        self.cn8 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.fc1 = nn.Sequential(nn.Linear(266 , 32),
                                 nn.BatchNorm1d(32),
                                 nn.ReLU(True),
                                 nn.Dropout(0.5),

                                 nn.Linear(32, 5))

    def forward(self, x, other_gender, other_race):
        x = self.cn1(x)
        x = self.cn2(x)
        x = self.cn3(x)
        x = self.cn4(x)
        x = self.cn5(x)
        x = self.cn6(x)
        x = self.cn7(x)
        x = self.cn8(x)

        x = x.view(x.size(0), -1)

        other_gender = torch.nn.functional.one_hot(other_gender, 5)
        other_race = torch.nn.functional.one_hot(other_race, 5)

        other_gender = other_gender.view(-1, 5)
        other_race = other_race.view(-1, 5)

        x = torch.cat((x, other_gender), dim=1)
        x = torch.cat((x, other_race), dim=1)

        x = self.fc1(x)

        return x

class ConvNet_do2_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(True))
        self.cn2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(3, 3), nn.ReLU(True), nn.Dropout(0.55))
        self.cn7 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.55))
        self.cn8 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.fc1 = nn.Sequential(nn.Linear(266 , 32),
                                 nn.BatchNorm1d(32),
                                 nn.ReLU(True),
                                 nn.Dropout(0.5),

                                 nn.Linear(32, 5))

    def forward(self, x, other_gender, other_race):
        x = self.cn1(x)
        x = self.cn2(x)
        x = self.cn3(x)
        x = self.cn4(x)
        x = self.cn5(x)
        x = self.cn6(x)
        x = self.cn7(x)
        x = self.cn8(x)

        x = x.view(x.size(0), -1)

        other_gender = torch.nn.functional.one_hot(other_gender, 5)
        other_race = torch.nn.functional.one_hot(other_race, 5)

        other_gender = other_gender.view(-1, 5)
        other_race = other_race.view(-1, 5)

        x = torch.cat((x, other_gender), dim=1)
        x = torch.cat((x, other_race), dim=1)

        x = self.fc1(x)

        return x


class ConvNet_do2_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(True))
        self.cn2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(3, 3), nn.ReLU(True), nn.Dropout(0.5))
        self.cn7 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.6))
        self.cn8 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.fc1 = nn.Sequential(nn.Linear(266 , 32),
                                 nn.BatchNorm1d(32),
                                 nn.ReLU(True),
                                 nn.Dropout(0.5),

                                 nn.Linear(32, 5))

    def forward(self, x, other_gender, other_race):
        x = self.cn1(x)
        x = self.cn2(x)
        x = self.cn3(x)
        x = self.cn4(x)
        x = self.cn5(x)
        x = self.cn6(x)
        x = self.cn7(x)
        x = self.cn8(x)

        x = x.view(x.size(0), -1)

        other_gender = torch.nn.functional.one_hot(other_gender, 5)
        other_race = torch.nn.functional.one_hot(other_race, 5)

        other_gender = other_gender.view(-1, 5)
        other_race = other_race.view(-1, 5)

        x = torch.cat((x, other_gender), dim=1)
        x = torch.cat((x, other_race), dim=1)

        x = self.fc1(x)

        return x

#=================================do3==========================================================================================================

class ConvNet_do3(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(True))
        self.cn2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2),
                                 nn.ReLU(True))
        self.cn6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2),
                                 nn.ReLU(True), nn.Dropout(0.5))
        self.cn7 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2),
                                 nn.ReLU(True), nn.Dropout(0.5))
        self.cn8 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2),
                                 nn.ReLU(True), nn.Dropout(0.5))
        self.fc1 = nn.Sequential(nn.Linear(266, 32),
                                 nn.BatchNorm1d(32),
                                 nn.ReLU(True),
                                 nn.Dropout(0.5),

                                 nn.Linear(32, 5))

    def forward(self, x, other_gender, other_race):
        x = self.cn1(x)
        x = self.cn2(x)
        x = self.cn3(x)
        x = self.cn4(x)
        x = self.cn5(x)
        x = self.cn6(x)
        x = self.cn7(x)
        x = self.cn8(x)

        x = x.view(x.size(0), -1)

        other_gender = torch.nn.functional.one_hot(other_gender, 5)
        other_race = torch.nn.functional.one_hot(other_race, 5)

        other_gender = other_gender.view(-1, 5)
        other_race = other_race.view(-1, 5)

        x = torch.cat((x, other_gender), dim=1)
        x = torch.cat((x, other_race), dim=1)

        x = self.fc1(x)

        return x

#=================================do4==========================================================================================================

class ConvNet_do4(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(True))
        self.cn2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn7 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.5))
        self.cn8 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.fc1 = nn.Sequential(nn.Linear(266 , 32),
                                 nn.BatchNorm1d(32),
                                 nn.ReLU(True),
                                 nn.Dropout(0.5),

                                 nn.Linear(32, 5))

    def forward(self, x, other_gender, other_race):
        x = self.cn1(x)
        x = self.cn2(x)
        x = self.cn3(x)
        x = self.cn4(x)
        x = self.cn5(x)
        x = self.cn6(x)
        x = self.cn7(x)
        x = self.cn8(x)

        x = x.view(x.size(0), -1)

        other_gender = torch.nn.functional.one_hot(other_gender, 5)
        other_race = torch.nn.functional.one_hot(other_race, 5)

        other_gender = other_gender.view(-1, 5)
        other_race = other_race.view(-1, 5)

        x = torch.cat((x, other_gender), dim=1)
        x = torch.cat((x, other_race), dim=1)

        x = self.fc1(x)

        return x
#=================================do5==========================================================================================================

class ConvNet_do5(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(True))
        self.cn2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.25))
        self.cn6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn7 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.5))
        self.cn8 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.fc1 = nn.Sequential(nn.Linear(266 , 32),
                                 nn.BatchNorm1d(32),
                                 nn.ReLU(True),
                                 nn.Dropout(0.5),

                                 nn.Linear(32, 5))

    def forward(self, x, other_gender, other_race):
        x = self.cn1(x)
        x = self.cn2(x)
        x = self.cn3(x)
        x = self.cn4(x)
        x = self.cn5(x)
        x = self.cn6(x)
        x = self.cn7(x)
        x = self.cn8(x)

        x = x.view(x.size(0), -1)

        other_gender = torch.nn.functional.one_hot(other_gender, 5)
        other_race = torch.nn.functional.one_hot(other_race, 5)

        other_gender = other_gender.view(-1, 5)
        other_race = other_race.view(-1, 5)

        x = torch.cat((x, other_gender), dim=1)
        x = torch.cat((x, other_race), dim=1)

        x = self.fc1(x)

        return x

#=================================do6==========================================================================================================

class ConvNet_do6(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(True))
        self.cn2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.25))
        self.cn3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.25))
        self.cn4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.25))
        self.cn5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.25))
        self.cn6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.25))
        self.cn7 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.25))
        self.cn8 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.25))
        self.fc1 = nn.Sequential(nn.Linear(266 , 32),
                                 nn.BatchNorm1d(32),
                                 nn.ReLU(True),
                                 nn.Dropout(0.5),

                                 nn.Linear(32, 5))

    def forward(self, x, other_gender, other_race):
        x = self.cn1(x)
        x = self.cn2(x)
        x = self.cn3(x)
        x = self.cn4(x)
        x = self.cn5(x)
        x = self.cn6(x)
        x = self.cn7(x)
        x = self.cn8(x)

        x = x.view(x.size(0), -1)

        other_gender = torch.nn.functional.one_hot(other_gender, 5)
        other_race = torch.nn.functional.one_hot(other_race, 5)

        other_gender = other_gender.view(-1, 5)
        other_race = other_race.view(-1, 5)

        x = torch.cat((x, other_gender), dim=1)
        x = torch.cat((x, other_race), dim=1)

        x = self.fc1(x)

        return x



#=================================do7==========================================================================================================

class ConvNet_do7(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(True))
        self.cn2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.1))
        self.cn3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.1))
        self.cn4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.1))
        self.cn5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.1))
        self.cn6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.1))
        self.cn7 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.1))
        self.cn8 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.1))
        self.fc1 = nn.Sequential(nn.Linear(266 , 32),
                                 nn.BatchNorm1d(32),
                                 nn.ReLU(True),
                                 nn.Dropout(0.5),

                                 nn.Linear(32, 5))

    def forward(self, x, other_gender, other_race):
        x = self.cn1(x)
        x = self.cn2(x)
        x = self.cn3(x)
        x = self.cn4(x)
        x = self.cn5(x)
        x = self.cn6(x)
        x = self.cn7(x)
        x = self.cn8(x)

        x = x.view(x.size(0), -1)

        other_gender = torch.nn.functional.one_hot(other_gender, 5)
        other_race = torch.nn.functional.one_hot(other_race, 5)

        other_gender = other_gender.view(-1, 5)
        other_race = other_race.view(-1, 5)

        x = torch.cat((x, other_gender), dim=1)
        x = torch.cat((x, other_race), dim=1)

        x = self.fc1(x)

        return x

#=================================do8==========================================================================================================

class ConvNet_do8(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(True))
        self.cn2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn7 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn8 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.fc1 = nn.Sequential(nn.Linear(266 , 32),
                                 nn.BatchNorm1d(32),
                                 nn.ReLU(True),
                                 nn.Dropout(0.5),

                                 nn.Linear(32, 5))

    def forward(self, x, other_gender, other_race):
        x = self.cn1(x)
        x = self.cn2(x)
        x = self.cn3(x)
        x = self.cn4(x)
        x = self.cn5(x)
        x = self.cn6(x)
        x = self.cn7(x)
        x = self.cn8(x)

        x = x.view(x.size(0), -1)

        other_gender = torch.nn.functional.one_hot(other_gender, 5)
        other_race = torch.nn.functional.one_hot(other_race, 5)

        other_gender = other_gender.view(-1, 5)
        other_race = other_race.view(-1, 5)

        x = torch.cat((x, other_gender), dim=1)
        x = torch.cat((x, other_race), dim=1)

        x = self.fc1(x)

        return x

# ======================================do9=================================================

class ConvNet_do9(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Sequential(nn.Conv2d(3, 64, 3), nn.BatchNorm2d(64), nn.ReLU(True))
        self.cn2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn4 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.ReLU(True))
        self.cn5 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.Dropout(0.5))
        self.cn6 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2), nn.ReLU(True))

        self.fc1 = nn.Sequential(nn.Linear(9226 , 512),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(True),
                                 nn.Dropout(0.5),

                                 nn.Linear(512, 32),
                                 nn.BatchNorm1d(32),
                                 nn.ReLU(True),
                                 nn.Dropout(0.5),

                                 nn.Linear(32, 5))

    def forward(self, x, other_gender, other_race):
        x = self.cn1(x)
        x = self.cn2(x)
        x = self.cn3(x)
        x = self.cn4(x)
        x = self.cn5(x)
        x = self.cn6(x)

        x = x.view(x.size(0), -1)

        other_gender = torch.nn.functional.one_hot(other_gender, 5)
        other_race = torch.nn.functional.one_hot(other_race, 5)

        other_gender = other_gender.view(-1, 5)
        other_race = other_race.view(-1, 5)

        x = torch.cat((x, other_gender), dim=1)
        x = torch.cat((x, other_race), dim=1)

        x = self.fc1(x)

        return x

# 앙상블 실험에 사용할 모든 모델 구조 불러오기
model_1 = ConvNet_do1()
model_2 = ConvNet_do2()
model_3 = ConvNet_do3()
model_4 = ConvNet_do4()
model_5 = ConvNet_do5()
model_6 = ConvNet_do6()
model_7 = ConvNet_do7()
model_8 = ConvNet_do8()
model_9 = ConvNet_do9()
model_2_1 = ConvNet_do2_1()
model_2_2 = ConvNet_do2_2()
# 모델을 GPU에 올리기
model_1.to(device)
model_2.to(device)
model_3.to(device)
model_4.to(device)
model_5.to(device)
model_6.to(device)
model_7.to(device)
model_8.to(device)
model_9.to(device)
model_2_1.to(device)
model_2_2.to(device)
# 저장되어 있는 모델(.pt) load
model_1.load_state_dict(torch.load('C:/Users/user/PycharmProjects/dl/model/do1.pt'))
model_2.load_state_dict(torch.load('C:/Users/user/PycharmProjects/dl/69best_model_0609/do_gh/do2.pt'))
model_3.load_state_dict(torch.load('C:/Users/user/PycharmProjects/dl/model/do3.pt'))
model_4.load_state_dict(torch.load('C:/Users/user/PycharmProjects/dl/model/do4.pt'))
model_5.load_state_dict(torch.load('C:/Users/user/PycharmProjects/dl/model/do5.pt'))
model_6.load_state_dict(torch.load('C:/Users/user/PycharmProjects/dl/model/do6.pt'))
model_7.load_state_dict(torch.load('C:/Users/user/PycharmProjects/dl/69best_model_0609/do_gh/do7.pt'))
model_8.load_state_dict(torch.load('C:/Users/user/PycharmProjects/dl/69best_model_0609/do_gh/do8.pt'))
model_9.load_state_dict(torch.load('C:/Users/user/PycharmProjects/dl/69best_model_0609/do_gh/do9.pt'))
model_2_1.load_state_dict(torch.load('C:/Users/user/PycharmProjects/dl/69best_model_0609/do_gh/do2-1.pt'))
model_2_1.load_state_dict(torch.load('C:/Users/user/PycharmProjects/dl/model/do2-2.pt'))

id_list = []
pred_list = []
with torch.no_grad():
    # 모델을 평가모드로 설정
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    model_5.eval()
    model_6.eval()
    model_7.eval()
    model_8.eval()
    model_9.eval()
    model_2_1.eval()
    model_2_2.eval()

    for data, file_name, other in test_loader:
        data = data.to(device)

        other_gender = other[0].to(device)
        other_race = other[1].to(device)

        # 각 모델마다 logits 값 구하기
        logits_model1 = model_1(data, other_gender, other_race)
        logits_model2 = model_2(data, other_gender, other_race)
        logits_model3 = model_3(data, other_gender, other_race)
        logits_model4 = model_4(data, other_gender, other_race)
        logits_model5 = model_5(data, other_gender, other_race)
        logits_model6 = model_6(data, other_gender, other_race)
        logits_model7 = model_7(data, other_gender, other_race)
        logits_model8 = model_8(data, other_gender, other_race)
        logits_model9 = model_9(data, other_gender, other_race)
        logits_model2_1 = model_2_1(data, other_gender, other_race)
        logits_model2_2 = model_2_2(data, other_gender, other_race)

        # 다양한 모델 조합과 가중치를 주어 실험하기
        logits = logits_model2_1 * (0.16) + logits_model2 * (0.23) + logits_model7 * (0.25) + logits_model8 * (0.14) + logits_model9 * (0.20)

        ps = F.softmax(logits, dim=1)
        top_p, top_class = ps.topk(1, dim=1)

        id_list += list(file_name)
        pred_list += top_class.T.tolist()[0]

handout_result = pd.DataFrame({'Id': id_list, 'Category': pred_list})
handout_result.to_csv('C:/Users/user/PycharmProjects/dl/csv/last_try.csv', index=False)