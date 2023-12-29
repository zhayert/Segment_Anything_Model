import torch
a = torch.randn(5, 3, 256, 256)

# 指定保存路径
save_path = "../a_tensor.pt"

# 保存张量到本地
torch.save(a, save_path)

# 加载保存的张量（可选）
loaded_a = torch.load(save_path)



