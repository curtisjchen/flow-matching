import torch
import torch.nn as nn
from modules import TimeEmbedding, WEmbedding, ClassEmbedding
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, h, w, patch_size, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=hidden_dim, 
                               kernel_size=patch_size,
                               stride=patch_size)
        self.grid_h = h // patch_size
        self.grid_w = w // patch_size
        self.num_patches = self.grid_h * self.grid_w
        
        self.pos_embed = nn.Embedding(num_embeddings=self.num_patches, embedding_dim=hidden_dim)
    
    def forward(self, image: torch.Tensor):
        positions = torch.arange(self.num_patches, device=image.device)
        pos_emb = self.pos_embed(positions)
        image = self.conv1(image)

        image = image.permute(0, 2, 3, 1)
        image = image.flatten(1, 2)
        image = image + pos_emb
        return image

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.swiglu = nn.SiLU()

        self.adaLN = nn.Linear(hidden_dim, hidden_dim * 6)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, image, t):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(t).chunk(6, dim=-1)
        shift1 = shift1[:, None, :]
        scale1 = scale1[:, None, :]
        gate1 = gate1[:, None, :]
        shift2 = shift2[:, None, :]
        scale2 = scale2[:, None, :]
        gate2 = gate2[:, None, :]

        h = self.ln1(image)
        h = h * (1 + scale1) + shift1
        image = image + gate1 * self.attention(h, h, h, need_weights=False)[0]

        h = self.ln2(image)
        h = h * (1 + scale2) + shift2
        image = image + gate2 * self.down_proj(self.swiglu(self.up_proj(h)))
        return image

class DiT(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, patch_size, in_channels, h, w, num_classes, w_min=1.0, w_max=5.0):
        super().__init__()
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        self.grid_h = h // patch_size
        self.grid_w = w // patch_size
        
        self.patch_embed = PatchEmbed(
            in_channels=in_channels, 
            h=h, 
            w=w, 
            patch_size=patch_size, 
            hidden_dim=hidden_dim
        )
        
        self.ditblocks = nn.ModuleList(DiTBlock(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers))
        self.time_embed = TimeEmbedding(hidden_dim, hidden_dim)
        self.final_layer = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)
        self.null_class_idx = num_classes
        self.w_embed = WEmbedding(hidden_dim, hidden_dim)
        self.class_embed = ClassEmbedding(num_classes, hidden_dim)

    def forward(self, image, r: torch.Tensor, t: torch.Tensor, w, class_labels):
        b, c, img_h, img_w = image.shape
        image = self.patch_embed(image) 
        cond = self.time_embed(r, t) + self.w_embed(w) + self.class_embed(class_labels)
        
        for block in self.ditblocks:
            image = block(image, cond)
            
        image = self.final_layer(image)
        
        image = image.reshape(b, self.grid_h, self.grid_w, self.patch_size, self.patch_size, self.in_channels)
        image = image.permute(0, 5, 1, 3, 2, 4)
        image = image.reshape(b, c, img_h, img_w)

        return image

if __name__ == "__main__":
    model = DiT(hidden_dim=256, num_heads=4, num_layers=6, patch_size=4, in_channels=1, h=28, w=28, num_classes=10)
    
    x = torch.randn(4, 1, 28, 28)
    r = torch.zeros(4) 
    t = torch.rand(4)
    w = torch.tensor([1.0, 2.0, 3.0, 4.0])
    labels = torch.tensor([0, 1, 2, model.null_class_idx])
    
    out = model(x, r, t, w, labels) 
    print(out.shape)





