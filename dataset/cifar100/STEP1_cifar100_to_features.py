import torchvision
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import resnet34, ResNet34_Weights
from torch import nn
import torch

device = 'cuda'

DATA_MEANS = np.array([0.485, 0.456, 0.406])
DATA_STD = np.array([0.229, 0.224, 0.225])

transforms = Compose([
    ToTensor(),
    Resize(size=(224, 224)), 
    Normalize(mean=DATA_MEANS, std=DATA_STD)])

cifar100_train = torchvision.datasets.CIFAR100(root='D:/implement/dl/data/cifar-100', train=True, download=True,transform=transforms)
cifar100_test = torchvision.datasets.CIFAR100(root='D:/implement/dl/data/cifar-100', train=False, download=True,transform=transforms)


train_loader = DataLoader(dataset=cifar100_train, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=cifar100_test, batch_size=32, shuffle=False, num_workers=0)

model = torchvision.models.resnet34(weights='IMAGENET1K_V1')

model.fc = nn.Sequential()

model.to(device)

train_features = []
test_features = []

root = 'D:/implement/dl/data/cifar-100'
ff = torch.load(f'{root}/test_resnet_feat.tar')

model.eval()
with torch.no_grad():
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        features = model(imgs)
        train_features.append(features)
    
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        features = model(imgs[0:1])
        # features1 = model(imgs[0:2])
        # tt = model(cifar100_test[0][0].unsqueeze(0).to(device))
        # vv = model(imgs[0:1])
        # ww = model(imgs[0:2])
        # zz = model(imgs[0:3])
        test_features.append(features)

train_features = torch.cat(train_features, dim=0)
torch.save(train_features.cpu(), 'D:/implement/dl/data/cifar-100/train_resnet_feat.tar')

test_features = torch.cat(test_features, dim=0)
torch.save(test_features.cpu(), 'D:/implement/dl/data/cifar-100/test_resnet_feat.tar')



