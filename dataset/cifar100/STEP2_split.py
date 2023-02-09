
import torchvision
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import resnet34, ResNet34_Weights
from torch import nn
import torch
import json

DATA_MEANS = np.array([0.485, 0.456, 0.406])
DATA_STD = np.array([0.229, 0.224, 0.225])

np.random.seed(42)

transforms = Compose([
    ToTensor(),
    Resize(size=(224, 224)), 
    Normalize(mean=DATA_MEANS, std=DATA_STD)])

cifar100_train = torchvision.datasets.CIFAR100(root='D:/implement/dl/data/cifar-100', train=True, download=True,transform=transforms)
cifar100_test = torchvision.datasets.CIFAR100(root='D:/implement/dl/data/cifar-100', train=False, download=True,transform=transforms)

datasets = {}

labels = torch.LongTensor(cifar100_train.targets)
num_labels = labels.max() + 1

indices = torch.argsort(labels).reshape(num_labels, -1)
num_val = indices.shape[1] // 10

train_ind = indices[:, num_val:].reshape(-1)
val_ind = indices[:, :num_val].reshape(-1)

test_labels = torch.LongTensor(cifar100_test.targets)
test_ind = torch.argsort(test_labels)

dataset = {
    'train': {'index': train_ind.numpy().tolist(), 'labels': labels[train_ind].numpy().tolist()},
    'val': {'index': val_ind.numpy().tolist(), 'labels': labels[val_ind].numpy().tolist()},
    'test': {'index': test_ind.numpy().tolist(), 'labels': test_labels[test_ind].numpy().tolist()}
}

json.dump(dataset, open('D:/implement/dl/data/cifar-100/feature_dataset.json', 'w'))
