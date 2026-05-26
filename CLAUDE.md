# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repo.

## What this is

A minimal, pedagogical decoder-only GPT implemented from scratch in [MLX](https://github.com/ml-explore/mlx), trained on Tiny Shakespeare. Inspired by Karpathy's nanoGPT, but targeting Apple Silicon via MLX instead of PyTorch/CUDA. The entire codebase is under 400 lines across five Python files — keep it that way.

## Commands

This project uses **uv** for dependency and venv management. Python `>=3.11`.

```bash
uv sync                          # install deps into .venv
uv run python main.py            # train + sample (3000 steps by default)
```

There are **no tests, no linter config, no formatter config**. Don't add one unless the user asks.

`input.txt` (Tiny Shakespeare, ~1 MB) is gitignored. If it's missing:

```bash
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Architecture

Decoder-only Transformer. Five files, each one job:

| File            | Contents                                                         |
| --------------- | ---------------------------------------------------------------- |
| `tokenizer.py`  | Thin wrapper over `tiktoken`'s pretrained GPT-2 BPE (50,257 vocab) |
| `attention.py`  | `CausalSelfAttention` — fused QKV linear, multi-head, causal mask  |
| `gpt.py`        | `FeedForward`, `Block` (pre-LN + residual), `GPT` (embeddings, block stack, **tied LM head** via `tok_emb.as_linear`, `generate`) |
| `train.py`      | `get_batch`, `loss_fn` (fp32 cross-entropy), `train` (AdamW + linear warmup → cosine decay, wrapped in `mx.compile`) |
| `main.py`       | Entry point: load data → init model → bf16 cast → train → sample  |

Default hyperparams (in `gpt.py:GPT.__init__` and `main.py`):
`block_size=256`, `n_layer=6`, `n_head=8`, `d_model=384`, `batch_size=32`, `lr=3e-4`, `steps=3000`.

## MLX-specific things to know

These bite if you're coming from PyTorch:

- **Lazy evaluation.** MLX builds a graph; nothing runs until `mx.eval(...)` is called (or a value is materialized via `.item()` / `.tolist()`). `main.py` calls `mx.eval(model.parameters())` right after construction to force lazy weight allocation before dtype casting.
- **`mx.compile`.** The training step in `train.py` is wrapped in `mx.compile(step, inputs=state, outputs=state)` to fuse forward + backward + optimizer update. If you change what the step touches (model/optimizer state), make sure `state` reflects it or you'll get stale values.
- **bf16 mixed precision.** `model.set_dtype(mx.bfloat16)` is called *after* parameters exist. Loss is computed in fp32 inside `loss_fn` (cast logits before `cross_entropy`) for numerical stability of the log-softmax.
- **Tied embeddings.** There is no separate LM head — `GPT.__call__` ends with `self.tok_emb.as_linear(x)` to project hidden states back to vocab using the input embedding's weights.
- **`tree_flatten`.** Use `mlx.utils.tree_flatten(model.parameters())` to count params or iterate over them.

## Conventions

- **No type stubs, no test scaffolding, no over-abstraction.** Match the existing style: every class has a docstring, every non-trivial line has a `# comment` explaining intent. It's verbose on purpose — this repo is read more than it's run.
- Shape annotations in comments (`# [B, T, d]`) are part of the style. Add them when you add tensor ops.
- Keep files small and single-purpose. Don't merge them; don't split them further.
- The Tokenizer is intentionally a thin wrapper — don't expand it unless adding a new tokenizer family.

## Repo notes

- Git remote: `hvasconcelos/picogpt-mlx` on GitHub.
- Default branch is `master` (not `main`).
- `LLM.md` is a prose architecture overview written for humans; it does not need to stay in sync with code changes line-by-line, but keep the high-level flow accurate.
