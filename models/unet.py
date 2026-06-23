import torch.nn as nn
import torch
import math

class TimeEmbedding(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj1 = nn.Linear(d_in, d_out)
        self.silu = nn.SiLU()
        self.proj2 = nn.Linear(d_out, d_out)
        self.d_in = d_in

    def forward(self, t):
        time_embedding = torch.zeros((t.shape[0], self.d_in), device=t.device)
        for i in range(0, self.d_in, 2):
            time_embedding[:, i] = math.sin(t / 10000 ** (i / self.d_in))
        for i in range(1, self.d_in, 2):
            time_embedding[:, i] = math.cos(t / 10000 ** (i / self.d_in))
        
        return self.proj2(self.silu(self.proj1(time_embedding)))

class ConvBlock(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.d_in = d_in
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1)
        self.norm1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)
        self.norm2 = nn.GroupNorm(8, 64)
        self.x = nn.Linear(28, 7)
        self.silu1 = nn.SiLU()
    
    def forward(self, image, time):
        image = self.norm1(self.conv1(image))
        image = self.norm2(self.conv2(image))
        image = self.silu1(self.x(image))

        time_embedding = TimeEmbedding(28, 7).forward(time)

        image = image + time_embedding
        return image


        


