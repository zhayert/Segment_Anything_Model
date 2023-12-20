import torch
import torch.nn as nn

class Test(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        # 初始化权重和偏置
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# 创建一个输入张量
batch_size = 32
num_features = 64
input_tensor = torch.randn((batch_size, num_features))

# 创建你的归一化模块
your_normalization_layer = Test(num_features)

# 使用forward方法进行测试
output_tensor = your_normalization_layer(input_tensor)

# 打印输入和输出张量的信息
print("Input Tensor:")
print(input_tensor)
print("Output Tensor:")
print(output_tensor)
