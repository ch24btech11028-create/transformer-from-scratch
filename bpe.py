"""
bpe.py — byte-level Byte Pair Encoding, implemented from scratch.

Derived and explained in notebooks/01_tokenizer.ipynb. Two trainers live here:

  * naive_bpe_train: the textbook algorithm on the raw byte stream. Simple,
    correct, and O(corpus_len) per merge — too slow beyond toy sizes, and it
    happily merges across word boundaries ("e t" -> one token), which wastes
    vocab on artifacts of adjacency rather than meaningful subwords.

  * BPETokenizer.train: the practical version. The text is first split into
    chunks with the GPT-2 regex (words, numbers, punctuation, whitespace
    runs), merges are counted over *unique chunks weighted by frequency*
    (Shakespeare has ~1M characters but only ~30k distinct chunks), and pair
    counts are updated incrementally — after a merge, only chunks containing
    that pair are touched. This is the same idea as Sennrich et al.'s
    original subword-nmt implementation.

Byte-level means the base alphabet is the 256 byte values, so ANY string
(emoji, Chinese, typos) encodes without an <unk> token — same trick as GPT-2.
"""

import json
from collections import Counter, defaultdict

import regex  # `pip install regex` — supports \p{L} unicode classes that re doesn't

# GPT-2's pre-tokenization pattern. Reading left to right it peels off:
# common English contractions ('s, 'll, ...), a letter-run with optional
# leading space, a digit-run, a punctuation-run, and whitespace. The leading
# space rides along with the word (" the" not "the"), so the tokenizer never
# has to spend merges gluing spaces to words at encode time.
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_stats(ids, counts=None):
    """Count adjacent pairs in a list of ids. e.g. [1,2,3,1,2] -> {(1,2):2, (2,3):1, (3,1):1}"""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def naive_bpe_train(text, num_merges, verbose=False):
    """Textbook BPE on the raw byte stream. Returns the list of merges."""
    ids = list(text.encode("utf-8"))
    merges = {}  # (id, id) -> new id
    for i in range(num_merges):
        stats = get_stats(ids)
        # deterministic tie-break: highest count, then lowest pair ids
        pair = max(stats, key=lambda p: (stats[p], (-p[0], -p[1])))
        new_id = 256 + i
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        if verbose:
            print(f"merge {i + 1}: {pair} -> {new_id} (count {stats[pair]})")
    return merges


class BPETokenizer:
    """Byte-level BPE with GPT-2-style regex pre-splitting and a fast,
    incrementally-updated trainer."""

    def __init__(self):
        self.merges = {}            # (id, id) -> merged id, in training order
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> raw bytes
        self.pattern = GPT2_SPLIT_PATTERN

    # ------------------------------------------------------------- training
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 1) Pre-split into chunks; merges never cross chunk boundaries.
        #    Count each distinct chunk once — this is the whole speed trick:
        #    "the" appearing 20k times is processed as ONE sequence, weight 20k.
        chunk_freqs = Counter(regex.findall(self.pattern, text))
        seqs = [list(chunk.encode("utf-8")) for chunk in chunk_freqs]
        freqs = list(chunk_freqs.values())

        # 2) Initial pair counts, plus an index: pair -> set of seq indices
        #    that contain it, so each merge only touches affected chunks.
        pair_counts = defaultdict(int)
        pair_to_seqs = defaultdict(set)
        for si, (seq, f) in enumerate(zip(seqs, freqs)):
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += f
                pair_to_seqs[pair].add(si)

        for i in range(num_merges):
            if not pair_counts:
                print(f"ran out of pairs after {i} merges")
                break
            pair = max(pair_counts, key=lambda p: (pair_counts[p], (-p[0], -p[1])))
            new_id = 256 + i
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            # 3) Incremental update: rewrite only the chunks containing `pair`,
            #    subtracting their old pair counts and adding the new ones.
            for si in list(pair_to_seqs[pair]):
                seq, f = seqs[si], freqs[si]
                for p in zip(seq, seq[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    pair_to_seqs[p].discard(si)
                seq = merge(seq, pair, new_id)
                seqs[si] = seq
                for p in zip(seq, seq[1:]):
                    pair_counts[p] += f
                    pair_to_seqs[p].add(si)

            if verbose and (i + 1) % 100 == 0:
                print(f"merge {i + 1}/{num_merges}: {self.vocab[new_id]!r}")

    # ------------------------------------------------------------ encoding
    def _encode_chunk(self, chunk_bytes):
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            # Always apply the *earliest-learned* applicable merge first —
            # must replay merges in training order or results diverge.
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        ids = []
        for chunk in regex.findall(self.pattern, text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids)
        # errors="replace": a sampled id sequence can split a multi-byte
        # utf-8 character; better to show U+FFFD than crash.
        return data.decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    # --------------------------------------------------------- persistence
    def save(self, path):
        obj = {"pattern": self.pattern,
               "merges": [[a, b, v] for (a, b), v in self.merges.items()]}
        with open(path, "w") as f:
            json.dump(obj, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            obj = json.load(f)
        tok = cls()
        tok.pattern = obj["pattern"]
        for a, b, v in obj["merges"]:  # replay in order to rebuild vocab
            tok.merges[(a, b)] = v
            tok.vocab[v] = tok.vocab[a] + tok.vocab[b]
        return tok


class CharTokenizer:
    """Character-level tokenizer: the 5-minute baseline. Tiny vocab, zero
    unknowns on its training text, but sequences are ~4x longer than BPE."""

    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self):
        return len(self.stoi)
