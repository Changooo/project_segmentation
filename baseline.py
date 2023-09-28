import os
import cv2
from PIL import Image
import pandas as pd
import numpy as np

import random as rand

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, csv_file_src=False, csv_file_trg=False, transform=None, infer=False):
        if infer:
            self.data = pd.read_csv(csv_file_trg)
            self.infer = infer
            self.transform = transform
        else:
            src = pd.read_csv(csv_file_src)
            trg = pd.read_csv(csv_file_trg)
            trg['gt_path'] = 0
            self.data = pd.concat([src, trg], axis=0)
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            self.transform = transform
            self.infer = infer

    def __len__(self):
        return len(self.data)

    # target의 경우 (image, NULL, False)
    # source의 경우 (image, mask, True) 의 값을 갖는다.
    # 즉 dataloader = (input, label, domain)
    def __getitem__(self, idx):
        isTrg = False
        img_path = self.data.iloc[idx, 1] 
        mask_path = self.data.iloc[idx, 2] if self.infer==False else 0
        if mask_path == 0:
            isTrg = True
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer or isTrg:
            if self.transform:
                image = self.transform(image=image)['image']
            return image, torch.empty([224, 224]), False

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12 #배경을 픽셀값 12로 간주

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask, True

transform = A.Compose(
    [   
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)

dataset = CustomDataset(csv_file_src='./train_source.csv', csv_file_trg='./train_target.csv', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# U-Net의 기본 구성 요소인 Double Convolution Block을 정의합니다.
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

# 간단한 U-Net 모델 정의
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 13, 1) # 12개 class + 1 background

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)
        featuremap = x.clone().detach().requires_grad_(True) # [16,512,28,28]
        # torch.tensor(x, requires_grad=True)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x) 
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out, featuremap

class GradReverse(torch.autograd.Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output): # 역전파 시에 gradient에 음수를 취함
        return (grad_output * -1)

class domain_classifier(nn.Module):
    def __init__(self):
        super(domain_classifier, self).__init__()
        # torch.Size([16, 512, 28, 28])
        self.fc1 = nn.Linear(512*28*28, 10)
        self.fc2 = nn.Linear(10, 1) # mnist = 0, svhn = 1 회귀 가정

    def forward(self, x):
        x = GradReverse.apply(x) # gradient reverse
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)


class DANN_UNET(nn.Module):
    def __init__(self, UNet):
        super(DANN_UNET, self).__init__()
        self.label_classifier = UNet().to(device) # CNN 구조 모델 받아오기
        self.domain_classifier = domain_classifier() # 도메인 분류 layer

    def forward(self, img):
        # 현상황 문제: domain classifier를 만드려면 이 classifier에 피처익스트랙터의 피처를 넣어줘야되는데
        #  Unet이 피처 익스트랙터+라벨 클래시파이어라... Unet을 분해해야하나?? 고민중
        label_logits, featuremap = self.label_classifier(img)
        domain_logits =  self.domain_classifier(featuremap)

        return label_logits, domain_logits

class DANN_Loss(nn.Module):
    def __init__(self):
        super(DANN_Loss, self).__init__()

        self.CE = nn.CrossEntropyLoss() # 0~9 class 분류용
        self.BCE = nn.BCELoss() # 도메인 분류용
        
    # result : DANN_CNN에서 반환된 값
    # label : 숫자 0 ~ 9에 대한 라벨
    # domain_num : 0(mnist) or 1(svhn)
    def forward(self, result, label, domain_num, alpha = 1):
        domain_logits, label_logits = result # DANN_CNN의 결과

        batch_size = domain_logits.shape[0]

        domain_target = torch.FloatTensor([domain_num] * batch_size).unsqueeze(1).to(device)

        domain_loss = self.BCE(domain_logits, domain_target) # domain 분류 loss

        target_loss = self.CE(label_logits, label) # class 분류 loss

        loss = target_loss + alpha * domain_loss

        return loss
    


    




# model 초기화
model = DANN_UNET(UNet).to(device)

# loss function과 optimizer 정의
criterion_lb = torch.nn.CrossEntropyLoss()
criterion_dm = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training loop
for epoch in range(1):  # 20 에폭 동안 학습합니다.
    model.train()
    epoch_loss = 0
    for images, label, domain in tqdm(dataloader):

        images = images.float().to(device)
        label = label.long().to(device) #masks
        domain = domain.int().to(device)
        domain_bool = [(True if i==1 else False) for i in domain.tolist()]
        predict_lb, predict_dm = model(images)

        # lable classifer는 source data만 갖고 학습한다.
        predict_lb = predict_lb[domain_bool]
        label = label[domain_bool]

        loss_lb = criterion_lb(predict_lb, label.squeeze(1))
        loss_dm = criterion_dm(predict_dm.squeeze(1), domain.float())
        loss = loss_lb + loss_dm

        # 일케 해도 되나? 진행시켜? 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')


test_dataset = CustomDataset(csv_file_trg='./test.csv', transform=transform, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)


def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


with torch.no_grad():
    model.eval()
    result = []
    for images, _, __ in tqdm(test_dataloader):
        images = images.float().to(device)
        outputs, ___ = model(images)
        outputs = torch.softmax(outputs, dim=1).cpu()
        outputs = torch.argmax(outputs, dim=1).numpy()
        # batch에 존재하는 각 이미지에 대해서 반복
        for pred in outputs:
            pred = pred.astype(np.uint8)
            pred = Image.fromarray(pred) # 이미지로 변환
            pred = pred.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
            pred = np.array(pred) # 다시 수치로 변환
            # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
            for class_id in range(12):
                class_mask = (pred == class_id).astype(np.uint8)
                if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
                    mask_rle = rle_encode(class_mask)
                    result.append(mask_rle)
                else: # 마스크가 존재하지 않는 경우 -1
                    result.append(-1)

submit = pd.read_csv('./sample_submission.csv')
submit['mask_rle'] = result

submit.to_csv('./baseline_submit.csv', index=False)





