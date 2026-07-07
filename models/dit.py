import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size.shape[0])
    
    def forward(self, image):
        return self.conv1(image)

