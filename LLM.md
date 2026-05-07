Detailed LLM Architecture Overview: GPT/Transformer-based Model
--------------------------------------------------------------

This file implements a large language model (LLM) based on the Transformer architecture, following the GPT (Generative Pre-trained Transformer) paradigm.
The model is composed of several core components organized into modular classes: `FeedForward`, `Block`, and `GPT`.
The architecture supports autoregressive language modeling and can be trained or used for token generation.

High-Level Flow:

1. **Embedding Layers**:
   - **Token Embedding**: Maps each input token ID to a high-dimensional continuous vector using a learned embedding matrix.
   - **Position Embedding**: Adds information about token positions via another embedding, allowing the model to be sensitive to order and context length.

2. **Stacked Transformer Blocks**:
   - The input embeddings are fed sequentially through a stack of `Block` modules. Each block contains:
     - **LayerNorm**: Applied before each sublayer to stabilize and accelerate training.
     - **Causal Self-Attention**: Multi-head masked attention where each token can only "see" leftward/context tokens (not future tokens) to preserve causality for autoregressive generation.
     - **FeedForward Network**: Position-wise MLP with non-linearity, expanding and then projecting the hidden dimension.
   - Residual connections add each sublayer's output to its input, preserving gradient flow and enabling deeper models.

3. **Output Processing**:
   - After passing through all blocks, a final LayerNorm is applied.
   - The result is projected to vocabulary size by a linear layer (head), producing logits for each token position.

4. **Autoregressive Generation**:
   - Generation proceeds step-by-step: For each new token to generate, it feeds the context (up to a maximum `block_size`) and predicts the next token, sampling from the softmax distribution (with optional temperature scaling).

Class Roles:

- `FeedForward`: Implements the position-wise two-layer MLP with GELU nonlinearity, applied independently to each sequence position.
- `Block`: Wraps LayerNorm, CausalSelfAttention, FeedForward, and residual connections to form the basic Transformer block.
- `GPT`: Manages input embeddings, stacking of blocks, output normalization and projection, and implements both the forward pass and generation loop.

Key Hyperparameters:

- `vocab_size`: Number of unique tokens in the vocabulary.
- `block_size`: Maximum context window (sequence length) the model can consider.
- `n_layer`: Number of stacked Transformer blocks.
- `n_head`: Number of attention heads in each block.
- `d_model`: Hidden model dimension.

This modular structure enables both flexibility and scalability, and closely follows the GPT/Transformer best practices for LLMs.
