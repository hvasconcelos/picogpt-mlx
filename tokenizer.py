from pathlib import Path

from tokenizers import ByteLevelBPETokenizer, Tokenizer as HFTokenizer

DEFAULT_PATH = "pt-bpe.json"


class Tokenizer:
    """Byte-level BPE tokenizer trained on the local corpus."""

    def __init__(self, path: str = DEFAULT_PATH):
        self._tok = HFTokenizer.from_file(path)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, s: str) -> list[int]:
        return self._tok.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    @classmethod
    def train(cls, files: list[str], vocab_size: int = 16000,
              save_path: str = DEFAULT_PATH) -> "Tokenizer":
        bpe = ByteLevelBPETokenizer()
        bpe.train(files=files, vocab_size=vocab_size, min_frequency=2,
                  special_tokens=["<|endoftext|>"])
        bpe.save(save_path)
        return cls(save_path)

    @classmethod
    def load_or_train(cls, corpus_path: str, vocab_size: int = 16000,
                      save_path: str = DEFAULT_PATH) -> "Tokenizer":
        if Path(save_path).exists():
            return cls(save_path)
        print(f"Training {vocab_size}-vocab BPE on {corpus_path}...")
        return cls.train([corpus_path], vocab_size=vocab_size,
                         save_path=save_path)
