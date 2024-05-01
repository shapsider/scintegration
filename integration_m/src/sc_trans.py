import torch.nn as nn
import torch

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
x = torch.rand(10, 2000, 512)
for _ in range(2):
    x = encoder_layer(x)

print(x.size())