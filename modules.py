import torch.nn as nn
import torch

class TimeEmbedding(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj1 = nn.Linear(d_in, d_out)
        self.silu = nn.SiLU()
        self.proj2 = nn.Linear(d_out, d_out)
        self.d_in = d_in

    def forward(self, t):
        i = torch.arange(0, self.d_in, 2, device=t.device)  # [0, 2, 4, ..., d_in-2]
        freqs = 10000 ** (i / self.d_in)                     # shape (d_in/2,)
        args = t[:, None] / freqs[None, :]                   # shape (batch, d_in/2)
        
        time_embedding = torch.zeros((t.shape[0], self.d_in), device=t.device)
        time_embedding[:, 0::2] = torch.sin(args)
        time_embedding[:, 1::2] = torch.cos(args)
        
        return self.proj2(self.silu(self.proj1(time_embedding)))