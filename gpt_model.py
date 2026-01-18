from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_embd: int = 128
    n_head: int = 8
    n_layer: int = 4
    n_recursion: int = 4
    effective_depth: int = n_layer * n_recursion
    bias: bool = False


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        d = x.shape[-1]
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos.unsqueeze(0).unsqueeze(2)) + (rotate_half(x) * sin.unsqueeze(0).unsqueeze(2))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        
        # merged QKV
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # RoPE (0 Params)
        self.rotary = RotaryPositionalEmbeddings(self.head_dim)
        
        # causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # apply RoPE
        cos, sin = self.rotary(q)
        q = apply_rotary_pos_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_pos_emb(k, cos, sin).transpose(1, 2) 

        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # final projection
        y = self.c_proj(y)
        return y
    

class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
        )

    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # only token embeddings (no positional for RoPE)
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)

        # blocks
        self.transformer_blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        # weight tying for embedding/unembedding
        self.token_embedding_table.weight = self.lm_head.weight

    def forward(self, idx, targets=None, prompt_injection=False):
        B, T = idx.shape
        
        # embed tokens (positions embedded via RoPE in attention)
        prompt_embeddings = self.token_embedding_table(idx)
        x = prompt_embeddings

        for _ in range(self.config.n_recursion):
            for block in self.transformer_blocks:
                if prompt_injection:
                    x = block(prompt_embeddings + x)
                else:
                    x = block(x)

        # LayerNorm and classifier
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def generate(self, idx, max_new_tokens=16, greedy=True):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if greedy:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
