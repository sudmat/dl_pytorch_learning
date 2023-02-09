import torch


A = torch.randn((10, 8, 32, 63))
B = torch.randn((10, 8, 64, 128))

C = A @ B
print(C.shape)

C = A.matmul(B)
print(C.shape)