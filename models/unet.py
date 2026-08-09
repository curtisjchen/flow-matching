import torch
import torch.nn as nn
from modules import TimeEmbedding, ClassEmbedding, WEmbedding

# [ConvBlock and UpBlock remain exactly the same as your code]
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

# [Refactored Dynamic UNet]
class UNet(nn.Module):
    def __init__(self,
                 w_min=1.0,
                 w_max=5.0,
                 in_channels=1,
                 channels=(64, 256),  # Replaces down_in/down_out. Pass (64, 128, 256) for 3-layers!
                 prefinal=32,
                 time_in=128,
                 time_out=256,
                 num_classes=0
                 ):
        super().__init__()
        self.null_class_idx = num_classes
        
        # Embeddings
        self.time_emb = TimeEmbedding(time_in, time_out)
        self.class_emb = ClassEmbedding(num_classes=num_classes, embedding_dim=time_out)
        self.w_emb = WEmbedding(d_in=time_in, d_out=time_out, w_min=w_min, w_max=w_max)

        # Dynamic Downsampling Layers
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        curr_channels = in_channels
        for out_channels in channels:
            self.downs.append(ConvBlock(in_channels=curr_channels, out_channels=out_channels, time_emb_dim=time_out))
            self.downsamples.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2))
            curr_channels = out_channels
            
        # Bottleneck
        self.bottleneck = ConvBlock(in_channels=curr_channels, out_channels=curr_channels, time_emb_dim=time_out)
        
        # Dynamic Upsampling Layers
        self.ups = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        
        for i in range(len(reversed_channels)):
            up_in = reversed_channels[i]
            # The last upblock connects to the prefinal layer instead of another up_out
            up_out = reversed_channels[i+1] if i + 1 < len(reversed_channels) else prefinal
            self.ups.append(UpBlock(in_channels=up_in, out_channels=up_out, time_emb_dim=time_out))
            
        self.final = nn.Conv2d(in_channels=prefinal, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, image, r: torch.Tensor, t: torch.Tensor, w=None, class_labels=None):
        cond = self.time_emb(r, t) + self.w_emb(w) + self.class_emb(class_labels)
        stack = []
        
        # Down pass
        for down, downsample in zip(self.downs, self.downsamples):
            image = down(image, cond)
            stack.append(image)
            image = downsample(image)
            
        # Bottleneck
        image = self.bottleneck(image, cond)
        
        # Up pass
        for up in self.ups:
            skip = stack.pop()
            image = up(image, skip, cond)
            
        image = self.final(image)
        return image
    

if __name__ == "__main__":
    # Option A
    model = UNet(channels=(64, 296), prefinal=32, time_in=128, time_out=256, num_classes=10)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params:,}")