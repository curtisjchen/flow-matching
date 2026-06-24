from typing import Any

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
    def __init__(self, in_channels, out_channels, time_emb_dim, downsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2 if downsample else 1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.silu1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.silu2 = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
    
    def forward(self, image, time):
        image = self.silu1(self.norm1(self.conv1(image)))
        t = self.time_proj(time)[:, :, None, None]
        image = image + t
        image = self.silu2(self.norm2(self.conv2(image)))
        return image

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.convblock = ConvBlock(in_channels=in_channels, out_channels=out_channels, time_emb_dim=time_emb_dim)
        self.upsample = nn.Upsample()

    def forward(self, image, skip, time):


        


