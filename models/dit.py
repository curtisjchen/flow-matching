import torch
import torch.nn as nn
from modules import TimeEmbedding, WEmbedding, ClassEmbedding

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
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)

        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.swiglu = nn.SiLU()

        self.adaLN = nn.Linear(hidden_dim, hidden_dim * 6)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def attention(self, h):
        B, N, C = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: (B, num_heads, N, head_dim)

        q = self.q_norm(q)  # bounds Q magnitude before QK^T
        k = self.k_norm(k)  # bounds K magnitude before QK^T

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.attn_proj(out)

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
        image = image + gate1 * self.attention(h)

        h = self.ln2(image)
        h = h * (1 + scale2) + shift2
        image = image + gate2 * self.down_proj(self.swiglu(self.up_proj(h)))
        return image

class DiT(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, patch_size, in_channels, image_size, num_classes, w_min=1.0, w_max=5.0):
        super().__init__()
        self.num_layers = num_layers
        self.patchsize = patch_size
        self.in_channels = in_channels
        self.grid_size = image_size // patch_size
        self.patch_embed = PatchEmbed(in_channels=in_channels, patchsize=patch_size, hidden_dims=hidden_dim, image_size=image_size)
        self.ditblocks = nn.ModuleList(DiTBlock(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers))
        self.time_embed = TimeEmbedding(hidden_dim, hidden_dim)
        self.final_layer = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)
        self.null_class_idx = num_classes
        self.w_embed = WEmbedding(hidden_dim, hidden_dim)
        self.class_embed = ClassEmbedding(num_classes, hidden_dim)


    def forward(self, image, r: torch.Tensor, t: torch.Tensor, w, class_labels):
        b, c, h, img_w = image.shape
        image = self.patch_embed(image) 
        cond = self.time_embed(r, t) + self.w_embed(w) + self.class_embed(class_labels)
        for block in self.ditblocks:
            image = block(image, cond)
        image = self.final_layer(image)
        image = image.reshape(b, self.grid_size, self.grid_size, self.patchsize, self.patchsize, self.in_channels)
        image = image.permute(0, 5, 1, 3, 2, 4)
        image = image.reshape(b, c, h, img_w)

        return image

if __name__ == "__main__":
    model = DiT(hidden_dim=256, num_heads=4, num_layers=6, patch_size=4, in_channels=1, image_size=28, num_classes=10)
    
    x = torch.randn(4, 1, 28, 28)
    r = torch.zeros(4) # Standard flow matching uses r=0
    t = torch.rand(4)
    w = torch.tensor([1.0, 2.0, 3.0, 4.0])
    labels = torch.tensor([0, 1, 2, model.null_class_idx])
    
    # FIX: Pass all required arguments to the forward pass
    out = model(x, r, t, w, labels) 
    print(out.shape)





