
import os
import sys
import matplotlib.pyplot as plt

sys.path.append('D:/implement/dl/pytorch')

from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.tiny_imagenet import TinyImagenet

train_transform = transforms.Compose([
    transforms.Resize((56, 56)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((56, 56), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.ToTensor()
])

dataset = TinyImagenet('D:/implement/dl/data/tiny-imagenet-200',
                       split='val', transforms=train_transform)
loader = DataLoader(dataset, batch_size=10, shuffle=True)


for img, label, names in loader:
    plt.imshow(img[0].numpy().transpose((1, 2, 0)))
    print(names[0])
    plt.show()
    # print(label.argmax(axis=1))
    print(1)
