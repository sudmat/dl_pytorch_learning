from torch.utils.data import Dataset
import torch
import json
import torchvision
import numpy as np
from torchvision.transforms import Resize, Normalize, ToTensor, Compose

class Cifar100Features(Dataset):

    def __init__(self, root, phase) -> None:
        super().__init__()
        dataset = json.load(open(f'{root}/feature_dataset.json'))

        DATA_MEANS = np.array([0.485, 0.456, 0.406])
        DATA_STD = np.array([0.229, 0.224, 0.225])

        transforms = Compose([
            ToTensor(),
            Resize(size=(224, 224)), 
            Normalize(mean=DATA_MEANS, std=DATA_STD)])

        if phase == 'train' or phase == 'val':
            images_ds = torchvision.datasets.CIFAR100(root=root, train=True, download=True,transform=transforms)
            features = torch.load(f'{root}/train_resnet_feat.tar')
            indices = dataset[phase]['index']
            labels = dataset[phase]['labels']
            features = features[indices, :]
        else:
            assert phase == 'test'
            images_ds = torchvision.datasets.CIFAR100(root=root, train=False, download=True,transform=transforms)
            features = torch.load(f'{root}/test_resnet_feat.tar')
            indices = dataset[phase]['index']
            labels = dataset[phase]['labels']
        self.features = features
        self.labels = labels
        self.image_ds = images_ds
        self.indices = indices

    def __getitem__(self, index):
        return self.image_ds[self.indices[index]], self.features[index, :], self.labels[index]

class SetAnomalyDataset(Dataset):

    def __init__(self, root, phase, set_size) -> None:
        super().__init__()
        dataset = json.load(open(f'{root}/feature_dataset.json'))

        self.dataset = dataset
        self.set_size = set_size

        self.phase = phase

        if phase == 'train' or phase == 'val':
            features = torch.load(f'{root}/train_resnet_feat.tar')
            indices = np.array(dataset[phase]['index'])
            labels = np.array(dataset[phase]['labels'])
        else:
            assert phase == 'test'
            features = torch.load(f'{root}/test_resnet_feat.tar')
            indices = np.array(dataset[phase]['index'])
            labels = np.array(dataset[phase]['labels'])
        self.label_index_map = indices.reshape(labels.max() + 1, -1)
        self.features = features
        self.labels = labels
        self.indices = indices

        if phase == 'test':
            self.test_data_indices, self.test_normal_labels = self._create_test_indices()
    
    def _create_test_indices(self):
        np.random.seed(2023)
        test_data_indices = []
        test_normal_labels = []
        for i, anomaly_label in enumerate(self.labels):
            normal_label = np.random.randint(0, self.labels.max())
            if normal_label == anomaly_label:
                normal_label += 1
            normal_indices = np.random.choice(self.label_index_map.shape[1], self.set_size - 1)
            test_data_indices.append(self.label_index_map[normal_label, normal_indices])
            test_normal_labels.append(np.array([normal_label] * (self.set_size - 1)))
        test_data_indices = np.stack(test_data_indices, axis=0)
        test_normal_labels = np.stack(test_normal_labels, axis=0)
        return test_data_indices, test_normal_labels
    
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        if self.phase == 'test':
            indices = np.concatenate([self.test_data_indices[index], np.array([self.indices[index]])])
            labels = np.concatenate([self.test_normal_labels[index], np.array([self.labels[index]])])
        else:
            anomaly_label = self.labels[index]
            normal_label = np.random.randint(0, self.labels.max())
            if normal_label == anomaly_label:
                normal_label += 1
            normal_indices = np.random.choice(self.label_index_map.shape[1], self.set_size - 1)
            indices = np.concatenate([self.label_index_map[normal_label, normal_indices], np.array([self.indices[index]])])
            labels = np.array([normal_label] * (self.set_size - 1) + [anomaly_label])
        features = self.features[indices, :]
        return torch.from_numpy(indices), features, torch.from_numpy(labels)

if __name__ == '__main__':

    from torch import nn

    import matplotlib.pyplot as plt
    
    root = 'D:/implement/dl/data/cifar-100'

    DATA_MEANS = np.array([0.485, 0.456, 0.406])
    DATA_STD = np.array([0.229, 0.224, 0.225])

    transforms = Compose([
        ToTensor(),
        Resize(size=(224, 224)), 
        Normalize(mean=DATA_MEANS, std=DATA_STD)])

    test_ds = SetAnomalyDataset(root, 'test', 10)
    test_imgs = torchvision.datasets.CIFAR100(root=root, train=False, download=True,transform=transforms)

    val_ds = SetAnomalyDataset(root, 'val', 10)
    val_imgs = torchvision.datasets.CIFAR100(root=root, train=True, download=True,transform=transforms)

    train_ds = SetAnomalyDataset(root, 'train', 10)
    train_imgs = torchvision.datasets.CIFAR100(root=root, train=True, download=True,transform=transforms)

    model = torchvision.models.resnet34(weights='IMAGENET1K_V1')

    model.fc = nn.Sequential()
    model.eval()

    fig, ax = plt.subplots(9, 10)
    for i in range(3):
        indices, features, labels = test_ds[i]
        for ii, j in enumerate(indices):
            assert labels[ii] == test_imgs[j][1]
            img = test_imgs[j][0].numpy()
            # check the saved features correspond to the correct images.
            with torch.no_grad():
                feature_compare = model(test_imgs[j][0].unsqueeze(0))
            assert abs(feature_compare - features[ii]).mean() < 1e-3
            ax[i][ii].imshow(img.transpose((1, 2, 0)))

    for i in range(3, 6):
        indices, features, labels = val_ds[i]
        for ii, j in enumerate(indices):
            assert labels[ii] == val_imgs[j][1]
            img = val_imgs[j][0].numpy()
            with torch.no_grad():
                feature_compare = model(val_imgs[j][0].unsqueeze(0))
            assert abs(feature_compare - features[ii]).mean() < 1e-3
            ax[i][ii].imshow(img.transpose((1, 2, 0)))
    
    for i in range(6, 9):
        indices, features, labels = train_ds[i]
        for ii, j in enumerate(indices):
            assert labels[ii] == train_imgs[j][1]
            img = train_imgs[j][0].numpy()
            with torch.no_grad():
                feature_compare = model(train_imgs[j][0].unsqueeze(0))
            assert abs(feature_compare - features[ii]).mean() < 1e-3
            ax[i][ii].imshow(img.transpose((1, 2, 0)))

    plt.show()

    print(1)