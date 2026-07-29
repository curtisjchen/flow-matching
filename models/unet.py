from typing import Any

import torch.nn as nn
import torch
from modules import TimeEmbedding, ClassEmbedding, WEmbedding

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1)
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
        self.convblock = ConvBlock(in_channels=2*in_channels, out_channels=out_channels, time_emb_dim=time_emb_dim)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, image, skip, time):
        image = self.upsample(image)
        image = torch.cat([image, skip], dim=1)
        image = self.convblock(image, time)
        return image


class UNet(nn.Module):
    def __init__(self,
                 w_min=1.0,
                 w_max=5.0,
                 down_in_1=1, 
                 down_out_1=32, 
                 down_in_2=32, 
                 down_out_2=64,
                 prefinal=16,
                 time_in=128,
                 time_out=256,
                 num_classes=0
                 ):
        super().__init__()
        self.down1 = ConvBlock(in_channels=down_in_1, 
                               out_channels=down_out_1, 
                               time_emb_dim=time_out
                               )
        self.down2 = ConvBlock(in_channels=down_in_2, 
                               out_channels=down_out_2,
                               time_emb_dim=time_out
                               )
        self.bottleneck = ConvBlock(in_channels=down_out_2, 
                                    out_channels=down_out_2,
                                    time_emb_dim=time_out)
        self.up1 = UpBlock(in_channels=down_out_2, 
                           out_channels=down_in_2,
                           time_emb_dim=time_out
                           )
        self.up2 = UpBlock(in_channels=down_in_2, 
                           out_channels=prefinal,
                           time_emb_dim=time_out)
        self.final = nn.Conv2d(in_channels=prefinal, 
                               out_channels=down_in_1, 
                               kernel_size=1, 
                               stride=1, 
                               padding=0)
        self.time_emb = TimeEmbedding(time_in, time_out)
        self.downsample1 = nn.Conv2d(in_channels=down_out_1, out_channels=down_out_1, kernel_size=3, padding=1, stride=2)
        self.downsample2 = nn.Conv2d(in_channels=down_out_2, out_channels=down_out_2, kernel_size=3, padding=1, stride=2)
        self.null_class_idx = num_classes
        self.class_emb = ClassEmbedding(num_classes=num_classes, embedding_dim=time_out)
        self.w_emb = WEmbedding(d_in=time_in, d_out=time_out, w_min=w_min, w_max=w_max)
    
    def forward(self, image, r: torch.Tensor, t: torch.Tensor, w=None, class_labels=None):
        stack = []
        cond = self.time_emb(r, t) + self.w_emb(w) + self.class_emb(class_labels)
        image = self.down1(image, cond)
        stack.append(image)
        image = self.downsample1(image)
        image = self.down2(image, cond)
        stack.append(image)
        image = self.downsample2(image)
        image = self.bottleneck(image, cond)
        image = self.up1(image, stack.pop(), cond)
        image = self.up2(image, stack.pop(), cond)
        image = self.final(image)
        return image

if __name__ == "__main__":
    model = UNet(num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    r = torch.zeros(4)
    t = torch.rand(4)
    w = torch.tensor([1.0, 2.0, 3.0, 4.0])
    labels = torch.tensor([0, 1, 2, model.null_class_idx])
    
    out = model(x, r, t, w, labels)
    print(out.shape)





