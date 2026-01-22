from dataclasses import dataclass
import torch, math
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
    halt_loss_weight: float = 1
    num_latent_refinements: int = 4
    num_output_refinements: int = 4
    num_register_tokens: int = 0


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


# class CausalSelfAttention(nn.Module):
#     def __init__(self, config: GPTConfig):
#         super().__init__()
#         assert config.n_embd % config.n_head == 0
        
#         self.head_dim = config.n_embd // config.n_head
#         self.n_head = config.n_head
        
#         # merged QKV
#         self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
#         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
#         # RoPE (0 Params)
#         self.rotary = RotaryPositionalEmbeddings(self.head_dim)
        
#         # causal mask
#         self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
#                                      .view(1, 1, config.block_size, config.block_size))

#     def forward(self, x):
#         B, T, C = x.shape
#         qkv = self.c_attn(x)
#         q, k, v = qkv.split(C, dim=2)
        
#         k = k.view(B, T, self.n_head, self.head_dim)
#         q = q.view(B, T, self.n_head, self.head_dim)
#         v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

#         # apply RoPE
#         cos, sin = self.rotary(q)
#         q = apply_rotary_pos_emb(q, cos, sin).transpose(1, 2)
#         k = apply_rotary_pos_emb(k, cos, sin).transpose(1, 2) 

#         # flash attention
#         y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
#         y = y.transpose(1, 2).contiguous().view(B, T, C)

#         # final projection
#         y = self.c_proj(y)
#         return y
    

# class FeedForward(nn.Module):
#     def __init__(self, config: GPTConfig):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
#             nn.GELU(),
#             nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
#         )

#     def forward(self, x):
#         return self.mlp(x)


# class Block(nn.Module):
#     def __init__(self, config: GPTConfig):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
#         self.attn = CausalSelfAttention(config)
#         self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
#         self.ffwd = FeedForward(config)

#     def forward(self, x):
#         x = x + self.attn(self.ln1(x))
#         x = x + self.ffwd(self.ln2(x))
#         return x
    

# class ReasoningModule(nn.Module):
#     def __init__(self, config: GPTConfig):
#         super().__init__()
#         self.transformer_blocks = nn.ModuleList([
#             Block(config) for _ in range(config.n_layer)
#         ])

#         self.norm = nn.LayerNorm(config.n_embd, bias=config.bias)

#     def forward(self, x):
#         for block in self.transformer_blocks:
#             x = block(x)
#         return self.norm(x)

from functools import partial


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim, bias = False)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, dim_hidden, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim_hidden, dim),
        nn.Dropout(dropout)
    )

def MLPMixer1D(*, dim, depth, seq_len, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(seq_len, int(expansion_factor * dim), dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, int(expansion_factor_token * dim), dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim, bias = False)
    )
    

class HaltHead(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.head = nn.Linear(config.n_embd, 1, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, x):
        x = x.mean(dim=1)         
        x = self.head(x)
        return x.squeeze(-1)
    

# class HaltHead(nn.Module):
#     def __init__(self, config: GPTConfig):
#         super().__init__()
#         self.head = nn.Linear(config.n_embd, 1, bias=False)

#     def forward(self, x):
#         # Use the first token (Register 0) as the decision maker
#         return self.head(x[:, 0, :]).squeeze(-1)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # only token embeddings (no positional for RoPE)
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)

        # blocks
        # self.network = ReasoningModule(config)

        mixer_seq_len = (config.block_size - 1) + config.num_register_tokens
        self.network = MLPMixer1D(
            dim = config.n_embd,
            depth = 1,
            seq_len = mixer_seq_len
        )

        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        # weight tying for embedding/unembedding
        self.token_embedding_table.weight = self.lm_head.weight

        # random first latents, outputs
        self.latent_init_embed = nn.Parameter(torch.randn(config.n_embd) * 1e-2)
        self.output_init_embed = nn.Parameter(torch.randn(config.n_embd) * 1e-2)

        # register (thought) tokens
        self.register_tokens = nn.Parameter(torch.randn(config.num_register_tokens, config.n_embd) * 1e-2)

        # halt classifier
        self.halt_head = HaltHead(config)

    def get_initial_state(self):
        return self.latent_init_embed.clone(), self.output_init_embed.clone()

    # refine 'thoughts'
    def refine_latents(self, inputs, latents, outputs):
        for _ in range(self.config.num_latent_refinements):
            latents = self.network(inputs + latents + outputs)
        outputs = self.network(latents + outputs)
        return latents, outputs
    
    # refine the current output based on refined 'thoughts'
    def refine_output(self, inputs, latents, outputs):
        with torch.no_grad():
            for _ in range(self.config.num_output_refinements - 1):
                latents, outputs = self.refine_latents(inputs, latents, outputs)
        
        latents, outputs = self.refine_latents(inputs, latents, outputs)
        return latents, outputs


    def forward(self, idx, latents, outputs, targets=None):
        B, T = idx.shape

        # scale = math.sqrt(self.config.n_embd)

        # embed tokens (positions embedded via RoPE in attention)
        inputs = self.token_embedding_table(idx) # * scale

        # append register tokens
        registers = self.register_tokens.expand(B, -1, -1)
        inputs = torch.cat([registers, inputs], dim=1)

        # recursive reasoning - this is the magic
        latents, outputs = self.refine_output(inputs, latents, outputs)

        # remove register tokens for loss etc. -> they are only intended for the thinking process
        outputs_for_pred = outputs[:, self.config.num_register_tokens:, :]

        # LayerNorm and classifier
        x = self.ln_f(outputs_for_pred)  # layer norm maybe not in TRM?
        logits = self.lm_head(x)
        halt_logits = self.halt_head(outputs)
        halt_prob = halt_logits.sigmoid()
        latents, outputs = latents.detach(), outputs.detach()

        # no loss calculation during inference
        if targets is None:
            return latents, outputs, logits, halt_prob

        # compute losses
        loss = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction='none')
        loss = loss.mean(dim=1)

        is_all_correct = (logits.argmax(dim=-1) == targets).all(dim=1)

        halt_loss = F.binary_cross_entropy_with_logits(halt_logits, is_all_correct.float(), reduction='none')
        total_loss = loss + (halt_loss * self.config.halt_loss_weight)

        return total_loss.sum(), (loss, halt_loss), latents, outputs, logits, halt_prob
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=10, halt_threshold=0.5, max_outer_steps=12):
        for _ in range(max_new_tokens):
            preds, _ = self.predict(
                idx, 
                halt_threshold=halt_threshold, 
                max_deep_refinement_steps=max_outer_steps
            )
            
            next_token_pred = preds[:, -1].unsqueeze(-1)
            
            idx = torch.cat((idx, next_token_pred), dim=1)
        return idx

    @torch.no_grad()
    def predict(self, idx, halt_threshold=0.5, max_deep_refinement_steps=12):
        B, T = idx.shape
        device = idx.device
        num_regs = self.config.num_register_tokens

        inputs_embed = self.token_embedding_table(idx)
        registers = self.register_tokens.expand(B, -1, -1)
        
        curr_inputs = torch.cat([registers, inputs_embed], dim=1)
        seq_len = curr_inputs.shape[1]

        init_l, init_o = self.get_initial_state()
        curr_latents = init_l.view(1, 1, -1).expand(B, seq_len, -1).clone()
        curr_outputs = init_o.view(1, 1, -1).expand(B, seq_len, -1).clone()

        active_indices = torch.arange(B, device=device)
        
        finished_preds = []
        finished_steps = []
        finished_batch_indices = []

        for step in range(max_deep_refinement_steps):
            is_last = (step == max_deep_refinement_steps - 1)

            curr_latents, curr_outputs = self.refine_output(curr_inputs, curr_latents, curr_outputs)

            outputs_clean = curr_outputs[:, num_regs:, :]
            
            halt_logits = self.halt_head(outputs_clean)
            halt_probs = halt_logits.sigmoid()
            should_halt = (halt_probs >= halt_threshold) | is_last

            if should_halt.any():
                halting_outputs = outputs_clean[should_halt]
                
                x = self.ln_f(halting_outputs)
                logits = self.lm_head(x)
                preds = logits.argmax(dim=-1)

                finished_preds.append(preds)
                finished_steps.append(torch.full((should_halt.sum(),), step + 1, device=device))
                finished_batch_indices.append(active_indices[should_halt])

            keep_mask = ~should_halt
            
            if not keep_mask.any():
                break

            curr_inputs = curr_inputs[keep_mask]
            curr_latents = curr_latents[keep_mask]
            curr_outputs = curr_outputs[keep_mask]
            active_indices = active_indices[keep_mask]

        all_preds = torch.cat(finished_preds, dim=0)
        all_steps = torch.cat(finished_steps, dim=0)
        all_indices = torch.cat(finished_batch_indices, dim=0)

        sort_order = torch.argsort(all_indices)

        return all_preds[sort_order], all_steps[sort_order]