import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
from gpt import GPT

# ------------------------------------------------------------
# Batching
# ------------------------------------------------------------


def get_batch(data: mx.array, batch_size: int, block_size: int):
    """
    Samples random batches of (input, target) sequences from data.

    Each batch consists of `batch_size` rows.
    Each row contains a block of `block_size` tokens as input,
    and the next block_size tokens as target (shifted by 1).

    Args:
        data (mx.array): Flat array of token ids (e.g., train or val).
        batch_size (int): Number of rows in the batch.
        block_size (int): Length of the input/target sequence for each row.

    Returns:
        tuple: (xs, ys)
            - xs: [batch_size, block_size] input tokens
            - ys: [batch_size, block_size] target tokens (next-token prediction)
    """
    # Number of possible sequence start positions.
    n = data.size - block_size - 1
    # Randomly sample starting indices for each batch element.
    ix = mx.random.randint(0, n, shape=(batch_size,))
    # Gather input blocks from data for each starting index.
    xs = mx.stack([data[i : i + block_size] for i in ix.tolist()])
    # Gather target blocks, shifted by +1 relative to input.
    ys = mx.stack([data[i + 1 : i + 1 + block_size] for i in ix.tolist()])
    return xs, ys


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def loss_fn(model: GPT, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)                                        # [B, T, V]
    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
        reduction="mean",
    )


def train(model: GPT, train_ids: mx.array, val_ids: mx.array,
          steps: int = 3000, batch_size: int = 32, lr: float = 3e-4,
          warmup_steps: int = 200, min_lr_ratio: float = 0.1):
    # Linear warmup from 0 → lr, then cosine decay from lr → lr * min_lr_ratio.
    warmup = optim.linear_schedule(0.0, lr, steps=warmup_steps)
    cosine = optim.cosine_decay(lr, decay_steps=max(steps - warmup_steps, 1),
                                end=lr * min_lr_ratio)
    schedule = optim.join_schedules([warmup, cosine], [warmup_steps])
    optimizer = optim.AdamW(learning_rate=schedule)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    state = [model.state, optimizer.state]

    # mx.compile speeds up the training step by fusing the graph.
    def step(x, y):
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    step = mx.compile(step, inputs=state, outputs=state)

    tic = time.perf_counter()
    for it in range(1, steps + 1):
        x, y = get_batch(train_ids, batch_size, model.block_size)
        loss = step(x, y)

        if it % 200 == 0 or it == 1:
            mx.eval(state)
            xv, yv = get_batch(val_ids, batch_size, model.block_size)
            val = loss_fn(model, xv, yv)
            mx.eval(val)
            elapsed = time.perf_counter() - tic
            current_lr = optimizer.learning_rate.item()
            print(f"step {it:>4}  train {loss.item():.4f}  "
                  f"val {val.item():.4f}  lr {current_lr:.2e}  "
                  f"({elapsed:.1f}s)")