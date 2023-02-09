from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class TinyImagenet(Dataset):

    def __init__(self, root, split, transforms):
        self.root = root
        self.transforms = transforms
        self.img_files = []
        self.labels = []
        self.encodings = {}

        with open(f'{root}/wnids.txt', 'r') as f:
            lines = f.readlines()
            for i, l in enumerate(lines):
                self.encodings[l.replace('\n', '')] = i

        if split == 'train':
            for fd in os.listdir(f'{root}/train'):
                for img in os.listdir(f'{root}/train/{fd}/images'):
                    self.img_files.append(f'{root}/train/{fd}/images/{img}')
                    self.labels.append(self.encodings[fd])
        
        if split == 'val':
            label_map = {}
            with open(f'{root}/val/val_annotations.txt', 'r') as f:
                lines = f.readlines()
                for l in lines:
                    t = l.split('\t')
                    label_map[t[0]] = t[1]
            for img in os.listdir(f'{root}/val/images'):
                self.img_files.append(f'{root}/val/images/{img}')
                # self.img_files.append('D:/implement/dl/data/tiny-imagenet-200/train/n02481823/images/n02481823_25.JPEG')
                self.labels.append(self.encodings[label_map[img]])
        
        if split == 'debug':
            label_map = {}
            with open(f'{root}/val/val_annotations.txt', 'r') as f:
                lines = f.readlines()
                for l in lines:
                    t = l.split('\t')
                    label_map[t[0]] = t[1]
            for img in os.listdir(f'{root}/val/images'):
                self.img_files.append(f'{root}/val/images/{img}')
                # self.img_files.append('D:/implement/dl/data/tiny-imagenet-200/train/n02481823/images/n02481823_25.JPEG')
                self.labels.append(self.encodings[label_map[img]])
                break
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        im = Image.open(self.img_files[index])
        if self.transforms is not None:
            x = self.transforms(im)
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
        else:
            x = np.asarray(im)
        # gray-scale image
            if len(x.shape) == 2:
                x = np.repeat(x[:, :, None], 3, axis=2)
            x = x.transpose((2, 0, 1))

        y = self.labels[index]

        return x, y, self.img_files[index]
            
            
    
