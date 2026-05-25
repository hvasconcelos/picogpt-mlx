import tiktoken

# Pretrained GPT-2 byte-level BPE (50,257 tokens). "<|endoftext|>" is id 50256.
_ENCODING = "gpt2"


class Tokenizer:
    """Pretrained GPT-2 byte-level BPE (via tiktoken)."""

    def __init__(self) -> None:
        self._tok = tiktoken.get_encoding(_ENCODING)

    @property
    def vocab_size(self) -> int:
        return self._tok.n_vocab

    def encode(self, s: str) -> list[int]:
        # allowed_special="all" so a literal "<|endoftext|>" encodes to its
        # special id instead of raising. Safe for a single-corpus run.
        return self._tok.encode(s, allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)
