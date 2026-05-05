import mlx.core as mx
from mlx.utils import tree_flatten
from pathlib import Path
from tokenizer import CharTokenizer
from gpt import GPT
from train import train

# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------


def load_data(path: str = "input.txt") -> tuple[CharTokenizer, mx.array, mx.array]:
    """
    Loads text data from file, instantiates CharTokenizer, and encodes data.

    Args:
        path (str): Path to the input text file (default: "input.txt").

    Returns:
        tuple:
            - CharTokenizer object (tokenizer fitted to the text)
            - mx.array of token ids for training (first 90%)
            - mx.array of token ids for validation (last 10%)
    """
    # Read the entire text file as a single string.
    text = Path(path).read_text(encoding="utf-8")
    # Build a character-level tokenizer from the text.
    tok = CharTokenizer(text)
    # Encode the text into a list of integer token IDs, then into an MX array.
    ids = mx.array(tok.encode(text), dtype=mx.int32)
    # Split 90% train / 10% validation.
    n = int(0.9 * ids.size)
    return tok, ids[:n], ids[n:]


def main():
    """
    Main entry point: loads data, prints summary statistics, and samples a batch.
    """
    print("Hello from tiny-gpt-mlx .... lets go ⚡️")
    tok, train_ids, val_ids = load_data("input.txt")
    print(f"vocab size: {tok.vocab_size}, train tokens: {train_ids.size}")

    model = GPT(vocab_size=tok.vocab_size)
    mx.eval(model.parameters())   # force lazy init to actually allocate

    n_params = sum(p.size for _, p in
                   tree_flatten(model.parameters()))
    print(f"params: {n_params:,}")

    # Train the model.
    print("Training the model...")
    train(model, train_ids, val_ids, steps=3000)
    print("Training complete.")

    # Generate a sample
    print("\n--- sample ---")
    seed = mx.array([[tok.stoi["\n"]]])
    out = model.generate(seed, max_new=500)
    mx.eval(out)
    ids = out[0].tolist()
    print(f"Generated text: {tok.decode(ids)}")

if __name__ == "__main__":
    main()
