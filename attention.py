import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import math
# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, h: int):
        """
        Initializes a CausalSelfAttention module.

        CausalSelfAttention is a self-attention module that implements the
        causal self-attention mechanism. It is used to model the dependencies
        between tokens in a sequence.

        Args:
            d_model (int): The dimension of the model.
            h (int): The number of attention heads (H).
        """
        super().__init__()
        # Check that the model dimension is divisible by the number of heads.
        assert d_model % h == 0
        # Initialize the number of heads and the head dimension.
        self.h = h
        # The head dimension is the dimension of each head.
        self.head_dim = d_model // h
        # Initialize the linear layers for the query, key, and value.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        # Initialize the linear layer for the projection.
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass for the CausalSelfAttention module.

        Args:
            x (mx.array): The input tensor of shape [batch_size, sequence_length, d_model].

        Returns:
            mx.array: The output tensor of shape [batch_size, sequence_length, d_model].
        """
        # Get the batch size, sequence length, and model dimension.
        B, T, C = x.shape
        # Apply the linear layer to the input.
        qkv = self.qkv(x)                      # [B, T, 3C]
        # Split the output into query, key, and value.
        q, k, v = mx.split(qkv, 3, axis=-1)    # each [B, T, C]
        # Reshape the query, key, and value to [B, h, T, head_dim].
        # → [B, h, T, head_dim]
        q = q.reshape(B, T, self.h, self.head_dim).transpose(0, 2, 1, 3)
        # Reshape the key to [B, h, T, head_dim].
        k = k.reshape(B, T, self.h, self.head_dim).transpose(0, 2, 1, 3)
        # Reshape the value to [B, h, T, head_dim].
        v = v.reshape(B, T, self.h, self.head_dim).transpose(0, 2, 1, 3)

        # Calculate the scale for the attention.
        scale = 1.0 / math.sqrt(self.head_dim)
        # Calculate the attention.
        att = (q @ k.transpose(0, 1, 3, 2)) * scale         # [B, H, T, T]
        # causal mask: positions can only see themselves and the past
        # Create a causal mask.
        mask = mx.triu(mx.ones((T, T)), k=1).astype(mx.bool_)
        # Apply the causal mask to the attention.
        att = mx.where(mask, mx.array(-mx.inf), att)
        # Apply the softmax to the attention.
        att = mx.softmax(att, axis=-1)
        # Calculate the output.
        y = att @ v                                          # [B, H, T, head_dim]
        # Reshape the output to [B, T, C].
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        # Apply the projection layer to the output.
        return self.proj(y)