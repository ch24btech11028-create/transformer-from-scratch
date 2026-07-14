"""
train.py — training loop from scratch, no Trainer classes, no Hugging Face.

Every piece is derived and explained in notebooks/04_training.ipynb; this file
is the consolidated script version so you can run long jobs outside Jupyter:

    python train.py                          # char-level TinyShakespeare
    python train.py --dataset tinystories    # BPE TinyStories (see notebook 08)

What the loop contains and why (short version — the notebook has the long one):
  * cross-entropy on next-token prediction — the LM objective
  * AdamW with weight decay applied ONLY to matmul weights (2D params);
    LayerNorm gains and biases are scale parameters, decaying them hurts
  * cosine LR schedule with linear warmup — Adam's moment estimates are
    garbage for the first few hundred steps, warmup keeps them from
    launching the weights somewhere bad; cosine decay is the standard
    gentle landing
  * gradient clipping — one weird batch can produce a huge gradient and
    torpedo training; clip the global norm to 1.0
  * gradient accumulation — simulate large batches on small GPUs by summing
    gradients over micro-batches before stepping
  * mixed precision — bf16/fp16 matmuls on CUDA (~2x faster, half memory);
    fp16 needs a GradScaler because fp16 underflows small gradients
"""

import argparse
import math
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch

from model import GPT, GPTConfig

# ----------------------------------------------------------------- defaults
# (TinyShakespeare char-level; ~0.8M params, minutes on any GPU / Apple MPS)
cfg = dict(
    dataset="shakespeare_char",   # expects data/<dataset>/{train,val}.bin + meta.pkl
    out_dir="checkpoints",
    block_size=256, n_layer=4, n_head=4, n_embd=128, dropout=0.1,
    batch_size=64, grad_accum_steps=1,
    max_iters=3000, eval_interval=250, eval_iters=100,
    learning_rate=1e-3, min_lr=1e-4, warmup_iters=100, weight_decay=0.1,
    grad_clip=1.0, seed=1337, compile=False,
)

def parse_args():
    p = argparse.ArgumentParser()
    for k, v in cfg.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    return vars(p.parse_args())


def get_lr(it, *, warmup_iters, max_iters, learning_rate, min_lr):
    """Linear warmup to learning_rate, then cosine decay to min_lr."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return min_lr + coeff * (learning_rate - min_lr)


def configure_optimizer(model, weight_decay, learning_rate):
    """AdamW with decay only on >=2D tensors (matmul weights, embeddings).
    Weight decay is a prior that weights should be small — sensible for
    directions in a linear map, harmful for LayerNorm scales and biases."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95))


def main():
    c = parse_args()
    torch.manual_seed(c["seed"])

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    # bf16 has fp32's range (no GradScaler needed) but only ~3 significant
    # digits; fp16 has more precision but tiny range -> needs loss scaling.
    # T4-class GPUs lack bf16, so pick per hardware.
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    autocast = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    print(f"device={device}, amp={'off' if not use_amp else amp_dtype}")

    # data: uint16 token streams prepared by the notebooks (memmap: the whole
    # corpus never has to fit in RAM — matters for TinyStories' ~500M tokens)
    data_dir = os.path.join("data", c["dataset"])
    train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - c["block_size"] - 1, (c["batch_size"],))
        x = torch.stack([torch.from_numpy(data[i:i + c["block_size"]].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + c["block_size"]].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    model = GPT(GPTConfig(vocab_size=meta["vocab_size"], block_size=c["block_size"],
                          n_layer=c["n_layer"], n_head=c["n_head"], n_embd=c["n_embd"],
                          dropout=c["dropout"])).to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M parameters")
    if c["compile"]:
        model = torch.compile(model)
    opt = configure_optimizer(model, c["weight_decay"], c["learning_rate"])

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(c["eval_iters"])
            for k in range(c["eval_iters"]):
                x, y = get_batch(split)
                with autocast:
                    _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    best_val, t0 = float("inf"), time.time()
    history = []
    for it in range(c["max_iters"] + 1):
        lr = get_lr(it, warmup_iters=c["warmup_iters"], max_iters=c["max_iters"],
                    learning_rate=c["learning_rate"], min_lr=c["min_lr"])
        for g in opt.param_groups:
            g["lr"] = lr

        if it % c["eval_interval"] == 0 or it == c["max_iters"]:
            losses = estimate_loss()
            history.append((it, losses["train"], losses["val"]))
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {lr:.2e} | {time.time() - t0:.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                os.makedirs(c["out_dir"], exist_ok=True)
                torch.save({"model": model.state_dict(), "config": model.config,
                            "meta": meta, "iter": it, "val_loss": best_val, "history": history},
                           os.path.join(c["out_dir"], f"{c['dataset']}.pt"))
        if it == c["max_iters"]:
            break

        # gradient accumulation: grads sum across backward() calls until
        # zero_grad, so dividing each micro-batch loss by the number of
        # micro-batches yields exactly the large-batch average gradient
        opt.zero_grad(set_to_none=True)
        for _ in range(c["grad_accum_steps"]):
            x, y = get_batch("train")
            with autocast:
                _, loss = model(x, y)
                loss = loss / c["grad_accum_steps"]
            scaler.scale(loss).backward()
        scaler.unscale_(opt)  # clip real gradients, not scaled ones
        torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
        scaler.step(opt)
        scaler.update()

    print(f"done. best val loss {best_val:.4f}")


if __name__ == "__main__":
    main()
