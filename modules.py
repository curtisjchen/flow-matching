import torch.nn as nn
import torch

def sinusoidal_embed(t: torch.Tensor, d_in) -> torch.Tensor:
    i = torch.arange(0, d_in, 2, device=t.device)  # [0, 2, 4, ..., d_in-2]
    freqs = 10000 ** (i / d_in)                     # shape (d_in/2,)
    args = t[:, None] / freqs[None, :]                   # shape (batch, d_in/2)
    
    embedding = torch.zeros((t.shape[0], d_in), device=t.device)
    embedding[:, 0::2] = torch.sin(args)
    embedding[:, 1::2] = torch.cos(args)
    return embedding

class TimeEmbedding(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.proj1 = nn.Linear(2 * d_in, d_out)
        self.silu = nn.SiLU()
        self.proj2 = nn.Linear(d_out, d_out)
        self.d_in = d_in
    
    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb_r = sinusoidal_embed(r, self.d_in)
        emb_t = sinusoidal_embed(t, self.d_in)
        
        emb_joint = torch.cat([emb_r, emb_t], dim=-1)

        return self.proj2(self.silu(self.proj1(emb_joint)))

class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes + 1, embedding_dim=embedding_dim)
    
    def forward(self, label):
         return self.embedding(label)

class WEmbedding(nn.Module):
    def __init__(self, d_in, d_out, w_max = 5.0):
        super().__init__()
        self.w_max = w_max
        self.d_in = d_in
        self.proj1 = nn.Linear(d_in, d_out)
        self.silu = nn.SiLU()
        self.proj2 = nn.Linear(d_out, d_out)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        w_norm = w / self.w_max
        return self.proj2(self.silu(self.proj1(sinusoidal_embed(w_norm, self.d_in))))


if __name__ == "__main__":
    ce = ClassEmbedding(num_classes=10, embedding_dim=16)

    labels = torch.tensor([0, 3, 9])
    print("normal labels output shape:", ce(labels).shape)  # expect (3, 16)

    null_label = torch.tensor([10])
    print("null label output shape:", ce(null_label).shape)  # expect (1, 16)

    print("params found:", [n for n, _ in ce.named_parameters()])