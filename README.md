# tiny-gpt-mlx ⚡️

A minimal, from-scratch GPT implementation in [MLX](https://github.com/ml-explore/mlx) — Apple's array framework for Apple Silicon. The whole model, tokenizer, and training loop live in **under 400 lines of Python** across five files, and the model trains on Tiny Shakespeare on a MacBook in a few minutes.

The goal is pedagogical: a readable, runnable Transformer that's small enough to fit in one sitting but real enough to actually generate coherent-ish text.

## Sample output

After ~3,000 steps on Tiny Shakespeare, prompting with `"ROMEO:"`:

```
ROMEO:
What, mistress! the matter? wherefore is thy heart
So heavy on thy tongue? speak, gentle Juliet, —
...
```

(Quality depends on your training budget; this isn't GPT-4.)

## Inspiration

Heavily inspired by Andrej Karpathy's [**nanoGPT**](https://github.com/karpathy/nanoGPT) and his [*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture — same architecture, same Tiny Shakespeare dataset, same "make it small enough to read" philosophy. The twist here is **MLX instead of PyTorch**, so the whole thing runs natively on Apple Silicon's unified memory with `mx.compile` graph fusion and bf16 mixed precision.

## Architecture

A textbook decoder-only Transformer:

| Component               | Choice                                          |
| ----------------------- | ----------------------------------------------- |
| Tokenizer               | GPT-2 byte-level BPE (`tiktoken`, 50,257 vocab) |
| Context window          | 256 tokens                                      |
| Layers                  | 6                                               |
| Attention heads         | 8                                               |
| Model dim (`d_model`)   | 384                                             |
| FFN expansion           | 4× (GELU)                                       |
| Embeddings              | Learned token + learned positional              |
| LM head                 | **Tied** to input embeddings                    |
| Precision               | bf16 weights/grads, fp32 loss                   |

Implementation lives in three files:

- `attention.py` — `CausalSelfAttention` (fused QKV, multi-head, causal mask)
- `gpt.py` — `FeedForward`, `Block` (pre-LN + residual), `GPT` (embeddings + stack + tied head + `generate`)
- `tokenizer.py` — thin wrapper over `tiktoken`'s pretrained GPT-2 BPE

## Training

- **Dataset:** Tiny Shakespeare (`input.txt`, ~1 MB) — 90% train / 10% val split.
- **Optimizer:** AdamW with **linear warmup → cosine decay** (200 warmup steps, decays to 10% of peak LR).
- **Loss:** cross-entropy on next-token prediction, computed in fp32 for numerical stability.
- **Compile:** the training step is wrapped in `mx.compile` to fuse the forward + backward + optimizer update into one graph.
- **Batching:** random crops of `block_size` tokens, `(x, y)` shifted by 1.

Defaults: `batch_size=32`, `lr=3e-4`, `steps=3000`.

## Stack

- **[MLX](https://github.com/ml-explore/mlx)** (`>=0.31.2`) — array + autograd + optimizers on Apple Silicon
- **[tiktoken](https://github.com/openai/tiktoken)** — pretrained GPT-2 BPE
- **numpy** — incidental
- **[uv](https://github.com/astral-sh/uv)** — package + venv management (Python `>=3.11`)

No PyTorch, no CUDA, no GPU drivers — just MLX talking to the Metal Performance Shaders runtime on your Mac.

## Run it

```bash
# 1. Install deps (uv recommended)
uv sync

# 2. Grab Tiny Shakespeare (gitignored — bring your own corpus or curl this one)
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 3. Train + sample
uv run python main.py
```

You'll see something like:

```
Hello from tiny-gpt-mlx .... lets go ⚡️
vocab size: 50257, train tokens: 301966
params: 29,945,856  dtype: mlx.core.bfloat16
Training the model...
step    1  train 10.9... val 10.9... lr ...
step  200  train  5.1... val  5.0... lr 3.00e-04
...
```

## Project layout

```
.
├── attention.py     # CausalSelfAttention
├── gpt.py           # FeedForward, Block, GPT
├── tokenizer.py     # tiktoken GPT-2 BPE wrapper
├── train.py         # batching, loss, training loop (mx.compile + LR schedule)
├── main.py          # entry point: load → init → train → sample
├── input.txt        # training corpus (gitignored)
├── LLM.md           # architecture notes
└── pyproject.toml
```

## License

Personal learning project — do whatever you want with it.
