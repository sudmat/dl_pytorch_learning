from torch import nn
import torch


class InceptionBlock(nn.Module):

    def __init__(self,
                 c_in,
                 c_out_1x1,
                 c_out_3x3_dr,
                 c_out_3x3,
                 c_out_5x5_dr,
                 c_out_5x5,
                 c_out_mp,
                 act_fn
                 ):

        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out_1x1, kernel_size=1, padding='same'),
            nn.BatchNorm2d(c_out_1x1),
            act_fn()
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_out_3x3_dr, kernel_size=1, padding='same'),
            nn.BatchNorm2d(c_out_3x3_dr),
            act_fn(),
            nn.Conv2d(c_out_3x3_dr, c_out_3x3, kernel_size=3, padding='same'),
            nn.BatchNorm2d(c_out_3x3),
            act_fn()
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_out_5x5_dr, kernel_size=1, padding='same'),
            nn.BatchNorm2d(c_out_5x5_dr),
            act_fn(),
            nn.Conv2d(c_out_5x5_dr, c_out_5x5, kernel_size=5, padding='same'),
            nn.BatchNorm2d(c_out_5x5),
            act_fn()
        )
        self.conv_mp = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c_in, c_out_mp, kernel_size=1, padding='same'),
            nn.BatchNorm2d(c_out_mp),
            act_fn()
        )

    def forward(self, x):

        conv1x1_out = self.conv1x1(x)
        conv3x3_out = self.conv3x3(x)
        conv5x5_out = self.conv5x5(x)
        conv_mp_out = self.conv_mp(x)

        y = torch.cat([conv1x1_out, conv3x3_out,
                      conv5x5_out, conv_mp_out], axis=1)

        return y


class InceptionBlockV1(nn.Module):

    def __init__(self, c_in, c_red : dict, c_out : dict, act_fn):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            nn.BatchNorm2d(c_out["1x1"]),
            act_fn()
        )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn()
        )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn()
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn()
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out