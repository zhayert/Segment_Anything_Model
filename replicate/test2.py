import torch

x = torch.randn(3,7,6)
b,n,c = x.shape
x = x.reshape(b,n,2,c//2)
print(x.shape)
print(x.transpose(1,0).shape)
