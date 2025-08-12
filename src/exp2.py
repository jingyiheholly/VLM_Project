from transformers import AutoTokenizer

# Load the tokenizer (match it to your LLaVA base model)
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")  # or your own path/checkpoint

# Get the full vocabulary as a dict: {token: id}
vocab = tokenizer.get_vocab()

# To get a list of tokens
vocab_tokens = list(vocab.keys())

# Optional: Save to a file
with open("llava_vocab.txt", "w", encoding="utf-8") as f:
    for token in vocab_tokens:
        f.write(f"{token}\n")

print(f"Vocabulary size: {len(vocab_tokens)}")
