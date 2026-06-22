import torch.nn as nn
import torch
import math

class TimeEmbedding(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj1 = nn.Linear(d_in, d_out)
        self.relu = nn.ReLU()
        self.proj2 = nn.Linear(d_out, d_out)
        self.d_in = d_in

    def forward(self, t):
        time_embedding = torch.zeros((t.shape[0], self.d_in), device=t.device)
        for i in range(0, self.d_in, 2):
            time_embedding[:, i] = math.sin(t / 10000 ** (i / self.d_in))
        for i in range(1, self.d_in, 2):
            time_embedding[:, i] = math.cos(t / 10000 ** (i / self.d_in))
        
        return self.proj2(self.relu(self.proj1(time_embedding)))

class ConvBlock(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.d_in = d_in
        self.conv1 = nn.Conv2d(d_in, d_in // 2, 3, stride=2)
        self.norm1 = nn.BatchNorm2d(d_in // 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(d_in // 2, d_in // 4, 3, stride=2)
        self.norm2 = nn.BatchNorm2d(d_in // 4)
        self.relu2 = nn.ReLU()
    
    def forward(self, image):
        
        


