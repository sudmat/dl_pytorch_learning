from .multihead_attention import MultiheadAttention
from torch import nn
from torch.nn import functional as F
import torch
import math
import matplotlib.pyplot as plt

class EncoderNotClever(nn.Module):

    def __init__(self, in_dim, hidden_fw_dim, num_heads, num_blocks) -> None:

        super().__init__()

        self.num_blocks = num_blocks

        self.MHA = []
        self.MHA_LayerNorm = []
        self.FW = []
        self.FW_LayerNorm = []
        for i in range(num_blocks):
            self.MHA.append(MultiheadAttention(in_dim=in_dim, out_dim=in_dim, num_head=num_heads))
            self.MHA_LayerNorm.append(nn.LayerNorm(in_dim))
            self.FW.append(nn.Sequential(*[
                nn.Linear(in_features=in_dim, out_features=hidden_fw_dim),
                nn.Linear(in_features=hidden_fw_dim, out_features=in_dim),
                nn.ReLU()
            ]))
            self.FW_LayerNorm.append(nn.LayerNorm(in_dim))

    def forward(self, x):

        for i in range(self.num_blocks):
            xa = self.MHA[i](x)
            x += xa
            x = self.MHA_LayerNorm[i](x)
            xf = self.FW[i](x)
            x += xf
            x = self.FW_LayerNorm[i](x)
        
        return x


class EncodingBlock(nn.Module):

    def __init__(self, in_dim, hidden_fw_dim, num_heads, dropout=0.0) -> None:
        super().__init__()

        self.mha = MultiheadAttention(in_dim, in_dim, num_heads)
        self.fw = nn.Sequential(
            nn.Linear(in_dim, hidden_fw_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_fw_dim, in_dim),
            nn.ReLU(inplace=True)
        )
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, mask=None):
        x_atten = self.mha(x, mask)
        x = x + self.dropout(x_atten)
        x = self.norm1(x)
        
        x_fw = self.fw(x)
        x = x + self.dropout(x_fw)
        x = self.norm2(x)
        return x

class PE(nn.Module):

    def __init__(self, maxl, d) -> None:
        super().__init__()
        encoding = torch.zeros((maxl, d))
        position = torch.arange(0, maxl, step=1).unsqueeze(-1)
        dim_pos = torch.arange(0, d, step=2)

        encoding[:, 0::2] = torch.sin(position * torch.exp(-(dim_pos * math.log(10000)/d)))
        encoding[:, 1::2] = torch.cos(position * torch.exp(-(dim_pos * math.log(10000)/d)))

        # plt.imshow(encoding.numpy())
        # plt.show()

        self.register_buffer('pe', encoding.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]

class Encoder(nn.Module):

    def __init__(self, in_dim, hidden_fw_dim, num_heads, num_blocks, dropout=0.0) -> None:
        super().__init__()
        self.pe = PE(maxl=500, d=in_dim)
        self.blocks = nn.ModuleList([EncodingBlock(in_dim, hidden_fw_dim, num_heads, dropout) for i in range(num_blocks)])
    
    def forward(self, x, mask=None):
        y = x
        for block in self.blocks:
            y = block(x, mask)
        return y

    def get_attn_maps(self, x, mask=None):

        maps = []
        for layer in self.blocks:
            _, m = layer.mha(x, mask, return_attn=True)
            maps.append(m)
            x = layer(x)
        
        return maps
