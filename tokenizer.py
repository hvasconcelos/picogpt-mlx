
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------
class CharTokenizer:
    def __init__(self, text: str):
        self.vocab = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for i, c in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)