
import mlx.core as mx
import mlx.nn as nn
from attention import CausalSelfAttention

class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        """
        Initializes a FeedForward module.

        FeedForward is a feedforward network that implements the
        feedforward network mechanism. It is used to model the dependencies
        between tokens in a sequence.

        Args:
            d_model (int): The dimension of the model.
        """
        super().__init__()
        # Initialize the first linear layer.
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        # Initialize the second linear layer.
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass for the FeedForward module.

        Args:
            x (mx.array): The input tensor of shape [batch_size, sequence_length, d_model].

        Returns:
            mx.array: The output tensor of shape [batch_size, sequence_length, d_model].
        """
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        """
        Initializes a Block module.

        Block is a block that implements the block mechanism. It is used to model the dependencies
        between tokens in a sequence.

        Args:
            d_model (int): The dimension of the model.
            n_heads (int): The number of attention heads (H).
        """
        super().__init__()
        # Initialize the layer normalization for the attention.
        self.ln1 = nn.LayerNorm(d_model)
        # Initialize the causal self-attention.
        self.attn = CausalSelfAttention(d_model, n_heads)
        # Initialize the layer normalization for the feedforward.
        self.ln2 = nn.LayerNorm(d_model)
        # Initialize the feedforward.
        self.ff = FeedForward(d_model)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass for the Block module.

        Args:
            x (mx.array): The input tensor of shape [batch_size, sequence_length, d_model].

        Returns:
            mx.array: The output tensor of shape [batch_size, sequence_length, d_model].
        """
        # Apply the causal self-attention.
        x = x + self.attn(self.ln1(x))
        # Apply the feedforward.
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int = 256,
                 n_layer: int = 6, n_head: int = 8, d_model: int = 384):
        """
        Initializes a GPT module.

        The `block_size` parameter here specifies the context window size, i.e., the maximum number of tokens
        the model can consider at once during both training and generation. It determines how many previous tokens
        the model can use to predict the next token.

        The `n_layer` parameter determines the number of transformer blocks stacked in the model.
        Each block consists of a layer of self-attention and a feed-forward network. Increasing `n_layer`
        typically allows the model to capture more complex relationships in the data, as the input is processed
        through more layers of nonlinear transformations and attention mechanisms.

        Args:
            vocab_size (int): The size of the vocabulary.
            block_size (int): The context window size (maximum number of tokens per sequence).
            n_layer (int): The number of layers.
            n_head (int): The number of attention heads (H).
            d_model (int): The dimension of the model.
        """
        super().__init__()
        # Initialize the block size.
        self.block_size = block_size
        # Initialize the token embedding.
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Initialize the position embedding.
        self.pos_emb = nn.Embedding(block_size, d_model)
        # Initialize the blocks.
        self.blocks = [Block(d_model, n_head) for _ in range(n_layer)]
        # Initialize the layer normalization for the final output.
        self.ln_f = nn.LayerNorm(d_model)
        # No separate LM head: weights are tied to tok_emb via as_linear (see __call__).

    def __call__(self, idx: mx.array) -> mx.array:
        """
        Forward pass for the GPT module.

        Args:
            idx (mx.array): The input tensor of shape [batch_size, sequence_length].

        Returns:
            mx.array: The output tensor of shape [batch_size, sequence_length, vocab_size].
        """
        _, T = idx.shape
        # Initialize the position embedding.
        pos = mx.arange(T)
        x = self.tok_emb(idx) + self.pos_emb(pos)            # [B, T, d]
        # Apply the blocks.
        for blk in self.blocks:
            x = blk(x)
        # Apply the layer normalization for the final output.
        x = self.ln_f(x)
        # Tied LM head: project back to vocab using the input embedding's weights.
        return self.tok_emb.as_linear(x)                     # [B, T, vocab]

    def generate(self, idx: mx.array, max_new: int, temperature: float = 1.0):
        """
        Generates a sequence of tokens from the GPT module.

        Args:
            idx (mx.array): The input tensor of shape [batch_size, sequence_length].
            max_new (int): The maximum number of new tokens to generate.
            temperature (float): The temperature for the categorical distribution.
        """
        for _ in range(max_new):
            cropped = idx[:, -self.block_size:]
            logits = self(cropped)[:, -1, :] / temperature   # [B, V]
            next_id = mx.random.categorical(logits)          # [B]
            idx = mx.concatenate([idx, next_id[:, None]], axis=1)
        return idx

    def num_parameters(self) -> int:
        """
        Returns the number of trainable parameters (weights) in the model.

        Returns:
            int: Number of weights in the model.
        """
        from mlx.utils import tree_flatten
        params = tree_flatten(self.parameters())
        return sum(p.size for _, p in params)