import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patchsize, hidden_dims, image_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=hidden_dims, 
                               kernel_size=(patchsize, patchsize),
                               stride=patchsize)
        self.num_patches = (image_size // patchsize) ** 2
        self.pos_embed = nn.Embedding(num_embeddings=self.num_patches, embedding_dim=hidden_dims)
    
    def forward(self, image: torch.Tensor):
        positions = torch.arange(self.num_patches, device=image.device)  # [0, 1, ..., 48]
        pos_emb = self.pos_embed(positions)  # shape (49, hidden_dims)
        image = self.conv1(image)
        image = image.permute(0, -2, -1, -3)
        image = image.flatten(1, 2)
        image = image + pos_emb
        return image

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
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



