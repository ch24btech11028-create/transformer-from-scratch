"""
sample.py — generate text from a trained checkpoint.

    python sample.py --ckpt checkpoints/shakespeare_char.pt --prompt "ROMEO:" \
        --max_new_tokens 400 --temperature 0.8 --top_k 40

The sampling strategies (greedy / temperature / top-k / top-p) are implemented
inside GPT.generate (model.py) and derived one by one, with plots of what they
do to the next-token distribution, in notebooks/05_sampling.ipynb.
"""

import argparse
import pickle

import torch

from model import GPT
from bpe import BPETokenizer


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["meta"]


def make_codec(meta):
    """The checkpoint's meta dict records which tokenizer produced the ids."""
    if meta["kind"] == "char":
        stoi, itos = meta["stoi"], meta["itos"]
        return (lambda s: [stoi[c] for c in s]), (lambda ids: "".join(itos[i] for i in ids))
    tok = BPETokenizer.load(meta["tokenizer_path"])
    return tok.encode, tok.decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/shakespeare_char.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max_new_tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)  # 0 = greedy
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no_cache", action="store_true", help="disable the KV cache")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    model, meta = load_model(args.ckpt, device)
    encode, decode = make_codec(meta)

    x = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    for i in range(args.num_samples):
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature,
                           top_k=args.top_k, top_p=args.top_p,
                           use_cache=not args.no_cache)
        print(decode(y[0].tolist()))
        print("-" * 60)


if __name__ == "__main__":
    main()
