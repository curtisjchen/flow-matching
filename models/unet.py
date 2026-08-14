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
    def __init__(self, in_channels, out_channels, time_emb_dim, do_upsample=True):
        super().__init__()
        self.do_upsample = do_upsample
        self.convblock = ConvBlock(in_channels=2*in_channels, out_channels=out_channels, time_emb_dim=time_emb_dim)
        
        # Only initialize Upsample if this block is meant to change resolution
        if self.do_upsample:
            self.upsample = nn.Upsample(scale_factor=2)
        else:
            self.upsample = nn.Identity()

    def forward(self, image, skip, time):
        image = self.upsample(image)
        image = torch.cat([image, skip], dim=1)
        image = self.convblock(image, time)
        return image

class UNet(nn.Module):
    def __init__(self,
                 w_min=1.0,
                 w_max=5.0,
                 in_channels=1,
                 channels=(64, 128, 256, 512), 
                 downsample_flags=(True, True, False, False), # NEW: Control resolution per block
                 prefinal=32,
                 time_in=128,
                 time_out=256,
                 num_classes=0
                 ):
        super().__init__()
        self.null_class_idx = num_classes
        
        if len(channels) != len(downsample_flags):
            raise ValueError("downsample_flags must be the exact same length as channels")
        
        self.time_emb = TimeEmbedding(time_in, time_out)
        self.class_emb = ClassEmbedding(num_classes=num_classes, embedding_dim=time_out)
        self.w_emb = WEmbedding(d_in=time_in, d_out=time_out, w_min=w_min, w_max=w_max)

        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        curr_channels = in_channels
        for out_channels, do_downsample in zip(channels, downsample_flags):
            self.downs.append(ConvBlock(in_channels=curr_channels, out_channels=out_channels, time_emb_dim=time_out))
            
            if do_downsample:
                self.downsamples.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2))
            else:
                self.downsamples.append(nn.Identity())
                
            curr_channels = out_channels
            
        self.bottleneck = ConvBlock(in_channels=curr_channels, out_channels=curr_channels, time_emb_dim=time_out)
        
        self.ups = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        reversed_flags = list(reversed(downsample_flags))
        
        for i in range(len(reversed_channels)):
            up_in = reversed_channels[i]
            up_out = reversed_channels[i+1] if i + 1 < len(reversed_channels) else prefinal
            do_up = reversed_flags[i]
            
            self.ups.append(UpBlock(
                in_channels=up_in, 
                out_channels=up_out, 
                time_emb_dim=time_out, 
                do_upsample=do_up
            ))
            
        self.final = nn.Conv2d(in_channels=prefinal, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, image, r: torch.Tensor, t: torch.Tensor, w=None, class_labels=None):
        cond = self.time_emb(r, t) + self.w_emb(w) + self.class_emb(class_labels)
        stack = []
        
        for down, downsample in zip(self.downs, self.downsamples):
            image = down(image, cond)
            stack.append(image)
            image = downsample(image)
            
        image = self.bottleneck(image, cond)
        
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