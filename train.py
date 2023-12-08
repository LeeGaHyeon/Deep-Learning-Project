import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split, KFold
import torch.optim as optim
import torch.nn.functional as F
import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# 현재 사용 가능한 cuda 장치가 있는 경우 'cuda'를, 그렇지 않은 경우 'cpu'를 할당
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# CUDA를 활용할 수 있는 가상환경을 셋팅하였으므로 cuda가 출력
print(device)

# './data.csv' 파일에서 훈련 데이터를 읽어와 train_dataframe 변수에 저장
train_dataframe = pd.read_csv('./data.csv')
# './testdata.csv' 파일에서 테스트 데이터를 읽어와 test_df 변수에 저장
test_df = pd.read_csv('./testdata.csv')
# train_test_split 함수를 사용하여 훈련 데이터를 훈련 세트와 검증 세트로 분리
train_df, valid_df = train_test_split(train_dataframe, shuffle=True, test_size=0.01, stratify=train_dataframe['Label'])

# template-code.jpynb
class CustomDataset(torch.utils.data.Dataset):
    # 초기화 __init__ 함수
    def __init__(self, dataframe, train='train', transform=None):
        # 훈련 데이터셋인 경우
        if train == 'train':
            self.image_list = [] # 이미지 경로
            self.label_list = [] # 레이블
            self.other_list = [] # 나이, 성별, 인종
            path = './dataset/{}/{}'
            for index, row in dataframe.iterrows():
                image_path = row['Image']
                image_label = row['Label']
                image_age = row['Age']
                image_gender = row['Gender']
                image_race = row['Race']
                image = Image.open(path.format(image_label, image_path)).convert('RGB')
                if transform != None:
                    image = transform(image)
                self.image_list.append(image)
                self.label_list.append(image_label)
                self.other_list.append((image_age, image_gender, image_race))
        # 테스트 데이터셋인 경우
        elif train == 'test':
            self.image_list = [] # 이미지 경로
            self.label_list = [] # 레이블
            self.other_list = [] # 성별, 인종
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
    # 데이터셋의 총 샘플 수 반환
    def __len__(self):
        return len(self.image_list)
    # 주어진 idx에 해당하는 샘플(이미지 리스트, 레이블 리스트, 다른 정보 리스트) 추출하여 반환
    def __getitem__(self, idx):
        return self.image_list[idx], self.label_list[idx], self.other_list[idx]

# 참고자료 : https://deep-learning-study.tistory.com/475
# 훈련 데이터셋의 경로
train_path = "C:/Users/user/PycharmProjects/dl/dataset"
# 이미지를 텐서 형태로 변환하고, datasets.ImageFolder를 사용하여 이미지 폴더에 있는 데이터셋을 변수에 저장
train_ds = datasets.ImageFolder(root = train_path, transform=transforms.ToTensor())
# train_ds에서 각 이미지의 RGB 평균 값을 계산하여 meanRGB에 할당
# (x.numpy()는 이미지를 NumPy 배열로 변환, np.mean 함수를 사용하여 픽셀 값을 축에 따라 평균 계산)
meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in train_ds]
# 표준편차 계산
stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in train_ds]

meanR = np.mean([m[0] for m in meanRGB]) # R채널의 평균 값
meanG = np.mean([m[1] for m in meanRGB]) # G채널의 평균 값
meanB = np.mean([m[2] for m in meanRGB]) # B채널의 평균 값

stdR = np.mean([s[0] for s in stdRGB]) # R채널의 표준편차 값
stdG = np.mean([s[1] for s in stdRGB]) # G채널의 표준편차 값
stdB = np.mean([s[2] for s in stdRGB]) # B채널의 표준편차 값

print(meanR, meanG, meanB)
print(stdR, stdG, stdB)

train_transform = transforms.Compose([
    transforms.Resize(200), # 224*224의 크기로 image Resize
    transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 랜덤 좌우 반전
    transforms.ToTensor(), # tensor로 변환
    transforms.Normalize((0.6029, 0.4615, 0.3949),(0.2195, 0.1960, 0.1866)) # Normalization
])

test_transform = transforms.Compose([
    transforms.Resize(200), # 224*224의 크기로 image Resize
    transforms.ToTensor(), # tensor로 변환
    transforms.Normalize((0.6029, 0.4615, 0.3949),(0.2195, 0.1960, 0.1866)) # Normalization
])

train_dataset = CustomDataset(train_df, train='train', transform=train_transform) # 5138
valid_dataset = CustomDataset(valid_df, train='train', transform=test_transform) # 2202
test_dataset = CustomDataset(test_df, train='test', transform=test_transform) # 822

batch_size = 64
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

'''
다양한 버전의 모델을 설계하였으나, 코드 분량상의 이유로 ensemble.py에 모든 모델의 구조에 대한 코드를 담았습니다.
'''
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

model = ConvNet_do2()
model.to(device)

# # 최종 제출에는 사용하지 않은 FocalLoss
# class FocalLoss(nn.Module): # https://dacon.io/competitions/official/235585/codeshare/1796
#
#     def __init__(self, gamma=2.0, eps=1e-7):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         #print(self.gamma)
#         self.eps = eps
#         self.ce = torch.nn.CrossEntropyLoss(reduction="none")
#
#     def forward(self, input, target):
#         logp = self.ce(input, target)
#         p = torch.exp(-logp)
#         loss = (1 - p) ** self.gamma * logp
#         return loss.mean()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
n_epochs = 30

valid_loss_min = np.Inf # 최소 검증 손실을 무한대로 초기화
train_loss = torch.zeros(n_epochs) # 각 epoch에 대한 훈련 손실 값 초기화
valid_loss = torch.zeros(n_epochs) # 각 epoch에 대한 검증 손실 값 초기화

train_acc = torch.zeros(n_epochs) # 각 epoch에 대한 훈련 정확도 값 초기화
valid_acc = torch.zeros(n_epochs) # 각 epoch에 대한 검증 정확도 값 초기화

for e in range(0, n_epochs): # n_epoch만큼 반복문 실행
    model.train() # 모델을 훈련모드로 설정
    for image, label, other in train_loader: # 훈련 데이터셋에서 이미지, 레이블, 그 외의 정보를 가져온다.
        data = image.to(device) # 이미지를 GPU에 올리는 작업
        label = label.to(device) # 레이블을 GPU에 올리는 작업
        other_gender = other[1].to(device) # 그 외의 정보(성별)를 GPU에 올리는 작업
        other_race = other[2].to(device) # 그 외의 정보(인종)를 GPU에 올리는 작업
        optimizer.zero_grad() # Optimizer 초기화
        logits = model(data, other_gender, other_race) # 모델에 데이터와 그 외의 데이터(성별, 인종)를 logits를 얻는다.
        loss = criterion(logits, label) # logits과 레이블을 사용하여 loss를 계산한다.
        loss.backward() # 역전파를 통해 gradient를 계산한다.
        optimizer.step() # optimizer를 사용하여 gradient update
        train_loss[e] += loss.item() # 훈련 손실 값에 현재 epoch에서의 손실 값을 더한다.

        ps = F.softmax(logits, dim=1) # softmax를 통해 logits를 확률값으로 변환
        top_p, top_class = ps.topk(1, dim=1) # 가장 높은 확률 값을 가진 클래스를 선택
        equals = top_class == label.reshape(top_class.shape) # 선택한 클래스와 레이블을 비교하여 정확하게 예측한 경우 equals변수에 True를 할당
        # torch.mean 함수로 정확도의 평균을 계산하고 detach() 함수를 통해 gradient 연산을 분리하고 CPU로 이동
        train_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()

    train_loss[e] /= len(train_loader) # 훈련 데이터셋의 배치 사이즈로 나누어 훈련 데이터셋에 대한 평균 손실 계산
    train_acc[e] /= len(train_loader) # 훈련 데이터셋의 배치 사이즈로 나누어 훈련 데이터셋에 대한 정확도 계산
    with torch.no_grad(): # gradient 연산을 비활성화 하고 메모리 사용량을 줄이는 역할
        model.eval() # 모델을 평가 모드로 설정
        for data, label, other in valid_loader: # 검증 데이터셋에서 이미지와 레이블, 그 외의 데이터(성별, 인종)을 가져온다.
            # 위와 겹치는 주석은 생략하였습니다.
            data = data.to(device)
            label = label.to(device)
            other_gender = other[1].to(device)
            other_race = other[2].to(device)

            logits = model(data, other_gender, other_race) # 모델에 데이터를 넣음으로써 logits 계산
            loss = criterion(logits, label)
            valid_loss[e] += loss.item()

            ps = F.softmax(logits, dim=1)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == label.reshape(top_class.shape)
            valid_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()

    valid_loss[e] /= len(valid_loader)
    valid_acc[e] /= len(valid_loader)
    scheduler.step() # 한 epoch 당 스케쥴러 업데이트

    # epoch마다 훈련 및 검증 손실, 훈련 및 검증 정확도를 출력
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        e, train_loss[e], valid_loss[e]))
    print('Epoch: {} \tTraining accuracy: {:.6f} \tValidation accuracy: {:.6f}'.format(
        e, train_acc[e], valid_acc[e]))
    # 검증 손실이 감소할 때마다 모델을 저장
    if valid_loss[e] <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss[e]))
        torch.save(model.state_dict(), 'C:/Users/user/PycharmProjects/dl/model/testing.pt') # 모델 저장
        valid_loss_min = valid_loss[e] # 최소 검증 손실(valid_loss_min)을 현재 검증 손실로 업데이트

model.load_state_dict(torch.load('C:/Users/user/PycharmProjects/dl/model/testing.pt')) # 저장된 모델 로드

import matplotlib.pyplot as plt

plt.plot(train_loss, label='training loss')
plt.plot(valid_loss, label='validation loss')
plt.legend()

id_list = []
pred_list = []
with torch.no_grad():
    model.eval()
    for data, file_name, other in test_loader:
        data = data.to(device)

        other_gender = other[0].to(device)
        other_race = other[1].to(device)

        logits = model(data, other_gender, other_race)

        ps = F.softmax(logits, dim=1)
        top_p, top_class = ps.topk(1, dim=1)

        id_list += list(file_name)
        pred_list += top_class.T.tolist()[0]

handout_result = pd.DataFrame({'Id': id_list, 'Category': pred_list})
handout_result.to_csv('C:/Users/user/PycharmProjects/dl/csv/testing.csv', index=False)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

classes = ['1~10', '11~20', '21~30', '31~40', '41~']
classes_cm = [0, 1, 2, 3, 4]
test_loss = 0
y_pred = []
y_true = []
test_acc = 0
with torch.no_grad():
    model.eval()
    for data, labels, other in valid_loader:
        data, labels = data.to(device), labels.to(device)

        other_gender = other[1].to(device)
        other_race = other[2].to(device)
        logits = model(data, other_gender, other_race)
        loss = criterion(logits, labels)
        test_loss += loss.item()

        top_p, top_class = logits.topk(1, dim=1)
        y_pred.extend(top_class.data.cpu().numpy())
        y_true.extend(labels.data.cpu().numpy())
        equals = top_class == labels.reshape(top_class.shape)
        test_acc += torch.sum(equals.type(torch.float)).detach().cpu()

    test_acc /= len(valid_loader.dataset)
    test_acc *= 100

cm = confusion_matrix(y_true, y_pred, labels=classes_cm, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot()
plt.show()
print('Test accuracy : {}'.format(test_acc))

'''
grid search 구현 
'''
#
# import itertools
#
# # 하이퍼파라미터 조합 정의
# learning_rates = [0.001, 0.01, 0.1]
# weight_decays = [0.01, 0.001, 0.0001]
# gammas = [0.95, 0.9, 0.85]
#
# # 최적의 하이퍼파라미터와 결과 초기화
# best_params = None
# best_loss = np.Inf
#
# valid_loss_min = np.Inf # 최소 검증 손실을 무한대로 초기화
# train_loss = torch.zeros(n_epochs) # 각 epoch에 대한 훈련 손실 값 초기화
# valid_loss = torch.zeros(n_epochs) # 각 epoch에 대한 검증 손실 값 초기화
#
# train_acc = torch.zeros(n_epochs) # 각 epoch에 대한 훈련 정확도 값 초기화
# valid_acc = torch.zeros(n_epochs) # 각 epoch에 대한 검증 정확도 값 초기화
# # 모든 조합에 대해 반복
# for lr, wd, gamma in itertools.product(learning_rates, weight_decays, gammas):
#     criterion = FocalLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
#
#     n_epochs = 10
#
#     valid_loss_min = np.Inf
#     train_loss = torch.zeros(n_epochs)
#     valid_loss = torch.zeros(n_epochs)
#     train_acc = torch.zeros(n_epochs)
#     valid_acc = torch.zeros(n_epochs)
#
#     for e in range(0, n_epochs):
#         model.train()
#         for image, label, other in train_loader:
#             data = image.to(device)
#             label = label.to(device)
#             other_gender = other[1].to(device)
#             other_race = other[2].to(device)
#             optimizer.zero_grad()
#             logits = model(data, other_gender, other_race)
#             loss = criterion(logits, label)
#             loss.backward()
#             optimizer.step()
#             train_loss[e] += loss.item()
#
#             ps = F.softmax(logits, dim=1)
#             top_p, top_class = ps.topk(1, dim=1)
#             equals = top_class == label.reshape(top_class.shape)
#             train_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()
#
#         train_loss[e] /= len(train_loader)
#         train_acc[e] /= len(train_loader)
#         with torch.no_grad():
#             model.eval()
#             for data, label, other in valid_loader:
#                 data = data.to(device)
#                 label = label.to(device)
#                 other_gender = other[1].to(device)
#                 other_race = other[2].to(device)
#                 logits = model(data, other_gender, other_race)
#                 loss = criterion(logits, label)
#                 valid_loss[e] += loss.item()
#
#                 ps = F.softmax(logits, dim=1)
#                 top_p, top_class = ps.topk(1, dim=1)
#                 equals = top_class == label.reshape(top_class.shape)
#                 valid_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()
#
#         valid_loss[e] /= len(valid_loader)
#         valid_acc[e] /= len(valid_loader)
#         scheduler.step()
#
#         # 검증 손실이 최소값인 경우 최적의 하이퍼파라미터로 업데이트
#         if valid_loss[e] < best_loss:
#             best_loss = valid_loss[e]
#             best_params = (lr, wd, gamma)
#
#         # epoch마다 훈련 및 검증 손실, 훈련 및 검증 정확도를 출력
#         print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
#             e, train_loss[e], valid_loss[e]))
#         print('Epoch: {} \tTraining accuracy: {:.6f} \tValidation accuracy: {:.6f}'.format(
#             e, train_acc[e], valid_acc[e]))
#         # 검증 손실이 감소할 때마다 모델을 저장
#         if valid_loss[e] <= valid_loss_min:
#             print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
#                 valid_loss_min,
#                 valid_loss[e]))
#             torch.save(model.state_dict(), 'C:/Users/user/PycharmProjects/dl/model/o8-gridsearch.pt') # 모델 저장
#             valid_loss_min = valid_loss[e] # 최소 검증 손실(valid_loss_min)을 현재 검증 손실로 업데이트
#
# model.load_state_dict(torch.load('C:/Users/user/PycharmProjects/dl/model/o8-gridsearch.pt')) # 저장된 모델 로드
# print("Best Hyperparameters:", best_params)
# import matplotlib.pyplot as plt
#
# plt.plot(train_loss, label='training loss')
# plt.plot(valid_loss, label='validation loss')
# plt.legend()
#
# id_list = []
# pred_list = []
# with torch.no_grad():
#     model.eval()
#     for data, file_name, other in test_loader:
#         data = data.to(device)
#
#         other_gender = other[0].to(device)
#         other_race = other[1].to(device)
#
#         logits = model(data, other_gender, other_race)
#
#         ps = F.softmax(logits, dim=1)
#         top_p, top_class = ps.topk(1, dim=1)
#
#         id_list += list(file_name)
#         pred_list += top_class.T.tolist()[0]
#
# handout_result = pd.DataFrame({'Id': id_list, 'Category': pred_list})
# handout_result.to_csv('C:/Users/user/PycharmProjects/dl/csv/do8-gridsearch.csv', index=False)
#
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#
# classes = ['1~10', '11~20', '21~30', '31~40', '41~']
# classes_cm = [0, 1, 2, 3, 4]
# test_loss = 0
# y_pred = []
# y_true = []
# test_acc = 0
# with torch.no_grad():
#     model.eval()
#     for data, labels, other in valid_loader:
#         data, labels = data.to(device), labels.to(device)
#
#         other_gender = other[1].to(device)
#         other_race = other[2].to(device)
#         logits = model(data, other_gender, other_race)
#         loss = criterion(logits, labels)
#         test_loss += loss.item()
#
#         top_p, top_class = logits.topk(1, dim=1)
#         y_pred.extend(top_class.data.cpu().numpy())
#         y_true.extend(labels.data.cpu().numpy())
#         equals = top_class == labels.reshape(top_class.shape)
#         test_acc += torch.sum(equals.type(torch.float)).detach().cpu()
#
#     test_acc /= len(valid_loader.dataset)
#     test_acc *= 100
#
# cm = confusion_matrix(y_true, y_pred, labels=classes_cm, normalize='true')
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
# disp.plot()
# plt.show()
# print('Test accuracy : {}'.format(test_acc))

'''
Kfold, Kfold ensemble 구현
'''

# fold_train_losses = []
# fold_val_losses = []
#
# valid_loss_min = np.Inf
#
# train_loss = torch.zeros(n_epochs)
# valid_loss = torch.zeros(n_epochs)
#
# train_acc = torch.zeros(n_epochs)
# valid_acc = torch.zeros(n_epochs)
#
# from sklearn.model_selection import KFold
# from torch.utils.data import SubsetRandomSampler
# kf = KFold(n_splits=5, shuffle=True)
#
# def reset_weights(m): # 가중치 재설정 함수
#   '''
#     Try resetting model weights to avoid
#     weight leakage.
#   '''
#   for layer in m.children():
#    if hasattr(layer, 'reset_parameters'):
#     print(f'Reset trainable parameters of layer = {layer}')
#     layer.reset_parameters()
#
#
# for fold, (train_ind, valid_ind) in enumerate(kf.split(train_dataset)):
#     print('=========================Starting fold = ', fold)
#
#     train_sampler_kfold = SubsetRandomSampler(train_ind) # 해당 폴드에 대한 훈련 데이터 샘플러(Sampler) 생성
#     valid_sampler_kfold = SubsetRandomSampler(valid_ind) # 해당 폴드에 대한 검증 데이터 샘플러(Sampler) 생성
#
#     # 데이터를 배치 단위로 로드하고, 주어진 배치 크기(batch size)에 따라 데이터를 분할
#     train_loader_kfold = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler_kfold)
#     valid_loader_kfold = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler_kfold)
#
#     # cost, optimizer, scheduler 재정의
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
#
#     model.apply(reset_weights) # 모델의 가중치 재설정
#
#     valid_loss_min = np.Inf # 손실 값, 정확도 값의 초기화
#
#     train_loss = torch.zeros(n_epochs)
#     valid_loss = torch.zeros(n_epochs)
#
#     train_acc = torch.zeros(n_epochs)
#     valid_acc = torch.zeros(n_epochs)
#
#     # 훈련을 진행하는 반복문으로 위와 같은 주석은 생략하였습니다.
#     for e in np.arange(n_epochs):
#         model.train()
#         for image, label, other in train_loader_kfold:
#             data = image.to(device)
#             label = label.to(device)
#             other_gender = other[1].to(device)
#             other_race = other[2].to(device)
#
#             optimizer.zero_grad()
#             logits = model(data, other_gender, other_race)
#
#             loss = criterion(logits, label)
#             loss.backward()
#             optimizer.step()
#
#             train_loss[e] += loss.item()
#
#             ps = F.softmax(logits, dim=1)
#             top_p, top_class = ps.topk(1, dim=1)
#             equals = top_class == label.reshape(top_class.shape)
#             train_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()
#
#         scheduler.step()
#         train_loss[e] /= len(train_loader)
#         train_acc[e] /= len(train_loader)
#         # 검증을 위한 반복문으로 위와 같은 주석은 생략하였습니다.
#         with torch.no_grad():
#             model.eval()
#             for data, label, other in valid_loader_kfold:
#                 data = data.to(device)
#                 label = label.to(device)
#                 other_gender = other[1].to(device)
#                 other_race = other[2].to(device)
#
#                 logits = model(data, other_gender, other_race)
#                 loss = criterion(logits, label)
#                 valid_loss[e] += loss.item()
#
#                 ps = F.softmax(logits, dim=1)
#                 top_p, top_class = ps.topk(1, dim=1)
#                 equals = top_class == label.reshape(top_class.shape)
#                 valid_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()
#
#         valid_loss[e] /= len(valid_loader_kfold)
#         valid_acc[e] /= len(valid_loader_kfold)
#
#         print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
#             e, train_loss[e], valid_loss[e]))
#
#         print('Epoch: {} \tTraining accuracy: {:.6f} \tValidation accuracy: {:.6f}'.format(
#             e, train_acc[e], valid_acc[e]))
#
#         if valid_loss[e] <= valid_loss_min:
#             print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
#                 valid_loss_min,
#                 valid_loss[e]))
#             torch.save(model.state_dict(), 'C:/Users/user/PycharmProjects/dl/model/bogosetest.pt')
#             valid_loss_min = valid_loss[e]
#             # kfold ensemble code
#             # if fold == 0:
#             #     torch.save(model.state_dict(), 'C:/Users/user/PycharmProjects/dl/model/redo2_fold0.pt')
#             # elif fold == 1:
#             #     torch.save(model.state_dict(), 'C:/Users/user/PycharmProjects/dl/model/redo2_fold1.pt')
#             # elif fold == 2:
#             #     torch.save(model.state_dict(), 'C:/Users/user/PycharmProjects/dl/model/redo2_fold2.pt')
#             # elif fold == 3:
#             #     torch.save(model.state_dict(), 'C:/Users/user/PycharmProjects/dl/model/redo2_fold3.pt')
#             # else:
#             #     torch.save(model.state_dict(), 'C:/Users/user/PycharmProjects/dl/model/redo2_fold4.pt')
#             valid_acc_max = valid_acc[e]
#
#
# model.load_state_dict(torch.load('C:/Users/user/PycharmProjects/dl/model/bogosetest.pt'))
#
# import matplotlib.pyplot as plt
#
# plt.plot(train_loss, label='training loss')
# plt.plot(valid_loss, label='validation loss')
# plt.legend()
#
# id_list = []
# pred_list = []
# with torch.no_grad():
#     model.eval()
#     for data, file_name, other in test_loader:
#         data = data.to(device)
#
#         other_gender = other[0].to(device)
#         other_race = other[1].to(device)
#
#         logits = model(data, other_gender, other_race)
#
#         ps = F.softmax(logits, dim=1)
#         top_p, top_class = ps.topk(1, dim=1)
#
#         id_list += list(file_name)
#         pred_list += top_class.T.tolist()[0]
#
# handout_result = pd.DataFrame({'Id': id_list, 'Category': pred_list})
# handout_result.to_csv('C:/Users/user/PycharmProjects/dl/csv/testing.csv', index=False)
#
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#
# classes = ['1~10', '11~20', '21~30', '31~40', '41~']
# classes_cm = [0, 1, 2, 3, 4]
# test_loss = 0
# y_pred = []
# y_true = []
# test_acc = 0
# with torch.no_grad():
#     model.eval()
#     for data, labels, other in valid_loader:
#         data, labels = data.to(device), labels.to(device)
#
#         other_gender = other[1].to(device)
#         other_race = other[2].to(device)
#         logits = model(data, other_gender, other_race)
#         loss = criterion(logits, labels)
#         test_loss += loss.item()
#
#         top_p, top_class = logits.topk(1, dim=1)
#         y_pred.extend(top_class.data.cpu().numpy())
#         y_true.extend(labels.data.cpu().numpy())
#         equals = top_class == labels.reshape(top_class.shape)
#         test_acc += torch.sum(equals.type(torch.float)).detach().cpu()
#
#     test_acc /= len(valid_loader.dataset)
#     test_acc *= 100
#
# cm = confusion_matrix(y_true, y_pred, labels=classes_cm, normalize='true')
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
# disp.plot()
# plt.show()
# print('Test accuracy : {}'.format(test_acc))