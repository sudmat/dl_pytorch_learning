from torch import nn
import torch

class Test(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        dummy = torch.rand((10, 10))
        self.register_buffer('dummy', dummy)


if __name__ == '__main__':

    t = Test()
    t.to('cuda')
    print(t.dummy.device)