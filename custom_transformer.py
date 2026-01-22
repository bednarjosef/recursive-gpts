import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.g


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, device):
        t = torch.arange(max_seq_len, device = device, dtype = self.inv_freq.dtype)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim = -1)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


class BidirectionalAttention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        assert dim % heads == 0, f"Dimension {dim} must be divisible by heads {heads}"
        
        dim_head = dim // heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = RMSNorm(dim)
        
        # Projection to Q, K, V
        # inner_dim is simply dim * 3 because (heads * dim_head) = dim
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, pos_emb = None):
        h = self.heads
        
        x = self.norm(x)
        
        # Split qkv
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        
        # Rearrange to (batch, heads, seq, head_dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # Apply RoPE
        if pos_emb is not None:
            q = apply_rotary_pos_emb(pos_emb, q)
            k = apply_rotary_pos_emb(pos_emb, k)

        # Attention
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # Merge heads back
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class RecursiveTransformerBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        heads = 8, 
        ff_mult = 4, 
        dropout = 0.
    ):
        super().__init__()
        dim_head = dim // heads
        self.rotary_emb = RotaryEmbedding(dim_head)
        
        self.attn = BidirectionalAttention(dim, heads=heads, dropout=dropout)
        self.ff = FeedForward(dim, expansion_factor=ff_mult, dropout=dropout)
        
        self.ff_norm = RMSNorm(dim)
        self.final_norm = RMSNorm(dim)

    def forward(self, x):
        b, n, d = x.shape
        
        # Positional Embeddings
        pos_emb = self.rotary_emb(n, x.device)
        
        # Attention
        attn_out = self.attn(x, pos_emb = pos_emb)
        x = x + attn_out
        
        # Feed Forward
        ff_out = self.ff(self.ff_norm(x))
        x = x + ff_out
        
        # Normalize the output before passing it back to the recursive loop
        return self.final_norm(x)