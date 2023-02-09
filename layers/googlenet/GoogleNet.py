from .inception_block import InceptionBlock
from torch import nn
import torch


class GoogleNet(nn.Module):

    def __init__(self):
        super().__init__(in_channels, out_channels)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=192,
                      kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(4),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inception_3a = InceptionBlock(
            c_in=192, c_out_1x1=64, c_out_3x3_dr=96,
            c_out_3x3=128, c_out_5x5_dr=16, c_out_5x5=32, c_out_mp=32, act_fn=nn.ReLU
        )
        self.inception_3b = InceptionBlock(
            c_in=256, c_out_1x1=128, c_out_3x3_dr=128,
            c_out_3x3=192, c_out_5x5_dr=32, c_out_5x5=96, c_out_mp=64, act_fn=nn.ReLU
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_4a = InceptionBlock(
            c_in=480, c_out_1x1=192, c_out_3x3_dr=96,
            c_out_3x3=208, c_out_5x5_dr=16, c_out_5x5=48, c_out_mp=64, act_fn=nn.ReLU
        )
        self.inception_4b = InceptionBlock(
            c_in=512, c_out_1x1=160, c_out_3x3_dr=112,
            c_out_3x3=224, c_out_5x5_dr=24, c_out_5x5=64, c_out_mp=64, act_fn=nn.ReLU
        )
        self.inception_4c = InceptionBlock(
            c_in=512, c_out_1x1=128, c_out_3x3_dr=128,
            c_out_3x3=256, c_out_5x5_dr=24, c_out_5x5=64, c_out_mp=64, act_fn=nn.ReLU
        )
        self.inception_4d = InceptionBlock(
            c_in=512, c_out_1x1=112, c_out_3x3_dr=144,
            c_out_3x3=288, c_out_5x5_dr=32, c_out_5x5=64, c_out_mp=64, act_fn=nn.ReLU
        )
        self.inception_4e = InceptionBlock(
            c_in=528, c_out_1x1=256, c_out_3x3_dr=160,
            c_out_3x3=320, c_out_5x5_dr=32, c_out_5x5=128, c_out_mp=128, act_fn=nn.ReLU
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_5a = InceptionBlock(
            c_in=832, c_out_1x1=256, c_out_3x3_dr=160,
            c_out_3x3=320, c_out_5x5_dr=32, c_out_5x5=128, c_out_mp=128, act_fn=nn.ReLU
        )
        self.inception_5b = InceptionBlock(
            c_in=832, c_out_1x1=384, c_out_3x3_dr=192,
            c_out_3x3=384, c_out_5x5_dr=48, c_out_5x5=128, c_out_mp=128, act_fn=nn.ReLU
        )
        self.avg_pool5x5 = nn.AvgPool2d(kernel_size=5, stride=3, padding=0)
        self.avg_pool7x7 = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)

        self.dropout = nn.Dropout2d(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=1000)

        self.softmax = nn.Softmax(dim=1)

        # auxilary layers
        self.conv_aug_1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.fc_aug_1a = nn.Linear(in_features=128*4*4, out_features=1024)
        self.fc_aug_1b = nn.Linear(in_features=1024, out_features=1000)
        self.dropout_aug_1 = nn.Dropout2d(p=0.7)

        self.conv_aug_2 = nn.Conv2d(in_channels=528, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.fc_aug_2a = nn.Linear(in_features=128*4*4, out_features=1024)
        self.fc_aug_2b = nn.Linear(in_features=1024, out_features=out_channels)
        self.dropout_aug_2 = nn.Dropout2d(p=0.7)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.pool1(x)
        x = self.inception_4a(x)
        x = self.inception_4b(x)

        x_aug1 = self.avg_pool5x5(x)
        x_aug1 = self.conv_aug_1(x_aug1)
        x_aug1 = x_aug1.reshape(x_aug1.shape[0], -1)
        x_aug1 = self.fc_aug_1a(x_aug1)
        x_aug1 = self.dropout_aug_1(x_aug1)
        x_aug1 = self.fc_aug_1b(x_aug1)
        y_aug1 = self.softmax(x_aug1)


        x = self.inception_4c(x)
        x = self.inception_4d(x)

        x_aug2 = self.avg_pool5x5(x)
        x_aug2 = self.conv_aug_2(x_aug2)
        x_aug2 = x_aug2.reshape(x_aug2.shape[0], -1)
        x_aug2 = self.fc_aug_2a(x_aug2)
        x_aug2 = self.dropout_aug_2(x_aug2)
        x_aug2 = self.fc_aug_2b(x_aug2)
        y_aug2 = self.softmax(x_aug2)

        x = self.inception_4e(x)
        x = self.pool2(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avg_pool7x7(x)
        x = self.dropout(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        y = self.softmax(x)

        return y, y_aug1, y_aug2

class TinyGoogleNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64,
                      kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=192,
                      kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(4),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inception_3a = InceptionBlock(
            c_in=192, c_out_1x1=64, c_out_3x3_dr=96,
            c_out_3x3=128, c_out_5x5_dr=16, c_out_5x5=32, c_out_mp=32, act_fn=nn.ReLU
        )
        self.inception_3b = InceptionBlock(
            c_in=256, c_out_1x1=128, c_out_3x3_dr=128,
            c_out_3x3=192, c_out_5x5_dr=32, c_out_5x5=96, c_out_mp=64, act_fn=nn.ReLU
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_4a = InceptionBlock(
            c_in=480, c_out_1x1=192, c_out_3x3_dr=96,
            c_out_3x3=208, c_out_5x5_dr=16, c_out_5x5=48, c_out_mp=64, act_fn=nn.ReLU
        )
        self.inception_4b = InceptionBlock(
            c_in=512, c_out_1x1=160, c_out_3x3_dr=112,
            c_out_3x3=224, c_out_5x5_dr=24, c_out_5x5=64, c_out_mp=64, act_fn=nn.ReLU
        )
        self.inception_4c = InceptionBlock(
            c_in=512, c_out_1x1=128, c_out_3x3_dr=128,
            c_out_3x3=256, c_out_5x5_dr=24, c_out_5x5=64, c_out_mp=64, act_fn=nn.ReLU
        )
        self.inception_4d = InceptionBlock(
            c_in=512, c_out_1x1=112, c_out_3x3_dr=144,
            c_out_3x3=288, c_out_5x5_dr=32, c_out_5x5=64, c_out_mp=64, act_fn=nn.ReLU
        )
        self.inception_4e = InceptionBlock(
            c_in=528, c_out_1x1=256, c_out_3x3_dr=160,
            c_out_3x3=320, c_out_5x5_dr=32, c_out_5x5=128, c_out_mp=128, act_fn=nn.ReLU
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_5a = InceptionBlock(
            c_in=832, c_out_1x1=256, c_out_3x3_dr=160,
            c_out_3x3=320, c_out_5x5_dr=32, c_out_5x5=128, c_out_mp=128, act_fn=nn.ReLU
        )
        self.inception_5b = InceptionBlock(
            c_in=832, c_out_1x1=384, c_out_3x3_dr=192,
            c_out_3x3=384, c_out_5x5_dr=48, c_out_5x5=128, c_out_mp=128, act_fn=nn.ReLU
        )
        self.avg_pool5x5 = nn.AvgPool2d(kernel_size=5, stride=3, padding=0)
        self.avg_pool7x7 = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)

        self.dropout = nn.Dropout2d(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=out_channels)

        self.softmax = nn.Softmax(dim=1)

        # auxilary layers
        self.conv_aug_1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.fc_aug_1a = nn.Linear(in_features=128*4*4, out_features=1024)
        self.fc_aug_1b = nn.Linear(in_features=1024, out_features=out_channels)
        self.dropout_aug_1 = nn.Dropout2d(p=0.7)

        self.conv_aug_2 = nn.Conv2d(in_channels=528, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.fc_aug_2a = nn.Linear(in_features=128*4*4, out_features=1024)
        self.fc_aug_2b = nn.Linear(in_features=1024, out_features=out_channels)
        self.dropout_aug_2 = nn.Dropout2d(p=0.7)

    def forward(self, x):

        x = self.conv2(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.pool1(x)
        x = self.inception_4a(x)
        x = self.inception_4b(x)

        x_aug1 = self.avg_pool5x5(x)
        x_aug1 = self.conv_aug_1(x_aug1)
        x_aug1 = x_aug1.reshape(x_aug1.shape[0], -1)
        x_aug1 = self.fc_aug_1a(x_aug1)
        x_aug1 = self.dropout_aug_1(x_aug1)
        x_aug1 = self.fc_aug_1b(x_aug1)
        y_aug1 = x_aug1


        x = self.inception_4c(x)
        x = self.inception_4d(x)

        x_aug2 = self.avg_pool5x5(x)
        x_aug2 = self.conv_aug_2(x_aug2)
        x_aug2 = x_aug2.reshape(x_aug2.shape[0], -1)
        x_aug2 = self.fc_aug_2a(x_aug2)
        x_aug2 = self.dropout_aug_2(x_aug2)
        x_aug2 = self.fc_aug_2b(x_aug2)
        y_aug2 = x_aug2

        x = self.inception_4e(x)
        x = self.pool2(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avg_pool7x7(x)
        x = self.dropout(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        y = x

        return y, y_aug1, y_aug2