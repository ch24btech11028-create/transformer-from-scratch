"""
model.py — a GPT-style decoder-only transformer, written from scratch in raw PyTorch.

Every module here is derived and explained step by step in the notebooks:
  notebooks/02_attention.ipynb              -> CausalSelfAttention
  notebooks/03_transformer_architecture.ipynb -> MLP, Block, GPT
  notebooks/07_kv_cache_and_upgrades.ipynb  -> the kv_cache path through forward()

Design choices (the "why", in one place):
  * Attention is written out manually (matmul -> scale -> mask -> softmax -> matmul)
    instead of calling F.scaled_dot_product_attention, so every step is visible.
    Production code should use the fused kernel (FlashAttention); it computes the
    exact same function, just faster and with less memory traffic.
  * Pre-norm (LayerNorm *before* attention/MLP, inside the residual branch).
    Post-norm (the 2017 paper's layout) puts LayerNorm on the main path, which
    breaks the clean identity path for gradients and needs careful warmup to
    train at depth. Pre-norm is what GPT-2 and essentially everything since uses.
  * Learned absolute positional embeddings (simple, what GPT-2 used). RoPE is
    implemented as an upgrade in notebook 07.
  * Weight tying between the token embedding and the LM head: both map between
    token identity and the model's feature space, so sharing the matrix saves
    vocab_size*n_embd params and acts as a regularizer.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65      # size of the token alphabet
    block_size: int = 256     # maximum context length the model can attend over
    n_layer: int = 4          # number of transformer blocks stacked
    n_head: int = 4           # attention heads per block (must divide n_embd)
    n_embd: int = 128         # width of the residual stream
    dropout: float = 0.0      # 0 for small datasets we barely finish one epoch on
    bias: bool = False        # False: slightly better and faster, like modern GPTs


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Each token emits a query ("what am I looking for?"), a key ("what do I
    contain?") and a value ("what do I hand over if attended to"). A token's
    output is the value-vectors of all *previous* tokens (and itself), averaged
    with weights softmax(q·k / sqrt(head_dim)).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # One fused linear layer computes Q, K and V for all heads at once.
        # Three separate Linears would be mathematically identical; one big
        # matmul is simply faster on GPU.
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection: after the heads are concatenated back together,
        # this lets the model mix information *across* heads.
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Causal mask: a lower-triangular matrix of ones. Registered as a
        # buffer (moves with .to(device), saved in state_dict, but not a
        # parameter — nothing to learn here).
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        # Debugging/visualization hooks (used by notebook 06):
        self.record_attention = False   # set True to stash attention weights
        self.att_weights = None         # (B, n_head, T, T) after a forward pass

    def forward(self, x, kv_cache=None, layer_idx=None):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Project to q, k, v and split heads: (B, T, C) -> (B, n_head, T, head_dim).
        # Heads are just a reshape — each head gets its own C/n_head-dim slice
        # and attends independently, letting different heads track different
        # relations (syntax, previous-token, punctuation, ...).
        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        # KV cache: during generation we only feed the *new* token(s); keys and
        # values of past tokens were already computed on earlier steps, so we
        # concatenate them instead of recomputing. Queries for past positions
        # are NOT needed — we only want the next-token prediction at the end.
        if kv_cache is not None and kv_cache[layer_idx] is not None:
            past_k, past_v = kv_cache[layer_idx]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        if kv_cache is not None:
            kv_cache[layer_idx] = (k, v)

        T_total = k.size(2)  # past tokens + current tokens

        # Attention scores: (B, nh, T, hd) @ (B, nh, hd, T_total) -> (B, nh, T, T_total).
        # Divide by sqrt(head_dim): dot products of two random hd-dim vectors
        # have variance ~hd, and softmax of large-magnitude logits saturates to
        # one-hot, killing gradients. Scaling keeps variance ~1 at init.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)

        # Causal mask: position i may only attend to positions <= i. We slice
        # the precomputed triangular mask so it also works when q covers only
        # the last T positions of a longer cached sequence. Masked positions
        # are set to -inf so softmax gives them exactly zero weight (and the
        # remaining weights renormalize over the allowed positions).
        att = att.masked_fill(self.mask[:, :, T_total - T:T_total, :T_total] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        if self.record_attention:
            self.att_weights = att.detach()

        y = att @ v  # weighted average of value vectors: (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concat heads back
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network.

    Attention moves information *between* positions; this MLP does the actual
    per-token computation on what was gathered ("attention communicates, MLP
    computes"). The 4x expansion is the transformer-paper convention — a wider
    hidden layer gives the block capacity; 4x is empirical, not derived.
    GELU instead of ReLU: smooth, non-zero gradient for small negative inputs,
    and it is what GPT-2/BERT validated at scale.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.down_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(self.gelu(self.up_proj(x))))


class Block(nn.Module):
    """One transformer block, pre-norm residual layout:

        x = x + attn(ln(x))
        x = x + mlp(ln(x))

    The residual stream `x` is never normalized or transformed on the main
    path — sublayers read from it (through LayerNorm) and write additive
    updates back. That identity path is why 10s-100s of layers train: the
    gradient always has a direct route to every layer below.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None, layer_idx=None):
        x = x + self.attn(self.attn_norm(x), kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)   # token embeddings
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)   # learned positions
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd, bias=config.bias)   # final pre-head norm
        self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the output head reuses the input embedding matrix.
        self.output_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # Residual projections get their init scaled down by sqrt(2*n_layer):
        # every block *adds* two contributions to the residual stream, so the
        # stream's variance grows with depth unless each contribution shrinks.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        """Parameter count, excluding positional embeddings — position tables
        don't represent learned "knowledge" the way token/attention weights do,
        so scaling-law comparisons conventionally exclude them. output_head is
        weight-tied to token_embedding so it's counted once automatically."""
        n = sum(p.numel() for p in self.parameters())
        return n - self.position_embedding.weight.numel()

    def forward(self, idx, targets=None, kv_cache=None):
        """idx: (B, T) token ids. If kv_cache (a list of per-layer (k,v) or
        None entries) is given, idx should contain only tokens not yet in the
        cache; the list is updated in place."""
        B, T = idx.shape
        past_len = 0
        if kv_cache is not None and kv_cache[0] is not None:
            past_len = kv_cache[0][0].size(2)
        assert past_len + T <= self.config.block_size, \
            f"sequence of length {past_len + T} exceeds block_size {self.config.block_size}"

        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos))  # (B, T, n_embd)
        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=kv_cache, layer_idx=i)
        x = self.final_norm(x)

        if targets is not None:
            logits = self.output_head(x)
            # Cross-entropy over the vocab at every position. view(-1, ...)
            # flattens (B, T) into B*T independent classification problems.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # Inference: only the last position's logits are needed to sample
            # the next token, so skip the output_head matmul for all others.
            logits = self.output_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 use_cache=False):
        """Autoregressive sampling. All strategies are implemented in one
        place: greedy is temperature=0 (argmax), top-k and top-p truncate the
        distribution before sampling. See notebooks/05_sampling.ipynb.
        """
        self.eval()
        block = self.config.block_size
        kv_cache = [None] * self.config.n_layer if use_cache else None
        input_ids = idx[:, -block:]  # prefill; crop a prompt longer than the context
        for _ in range(max_new_tokens):
            if not use_cache:
                # Without a cache we must re-feed the whole (cropped) sequence
                # and recompute attention for every position, every step.
                input_ids = idx[:, -block:]
            elif kv_cache[0] is not None and kv_cache[0][0].size(2) >= block:
                # Cache full: we use learned ABSOLUTE positional embeddings, which
                # only exist for positions < block_size, so the cache can't grow
                # past that. Evict the oldest entries to make room (a sliding
                # window). Note this makes cached/uncached outputs diverge *beyond*
                # block_size — the fundamental limit of absolute positions that
                # RoPE (notebook 07) removes. Within block_size they're identical.
                for l in range(len(kv_cache)):
                    k, v = kv_cache[l]
                    kv_cache[l] = (k[:, :, -(block - 1):, :], v[:, :, -(block - 1):, :])
            logits, _ = self(input_ids, kv_cache=kv_cache)
            logits = logits[:, -1, :]  # (B, vocab)

            if temperature == 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)  # greedy
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, top_k).values[:, [-1]]
                    logits[logits < kth] = float("-inf")
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum = torch.cumsum(probs, dim=-1)
                    # Keep the smallest set of tokens with cumulative prob >= top_p.
                    # shift right so the first token crossing the threshold survives.
                    cutoff = cum - probs > top_p
                    sorted_logits[cutoff] = float("-inf")
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            if use_cache:
                input_ids = next_id  # cache holds the past; feed only the new token
        return idx
