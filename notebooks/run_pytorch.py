import time
from pprint import pprint
from pathlib import Path

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

import torch
import torch.nn as nn
from torch.utils.data import  Dataset, DataLoader

vocab_size = 10_000
embed_dim = 768
maxlen = 512
num_heads = 8
d_ff = embed_dim * 4
num_transformer_blocks = 4
num_epochs = 1
batch_size = 64

device = torch.device("cuda")
print("Using device: ", device)

ds = load_dataset("karpathy/tinystories-gpt4-clean", split="train")
test = ds.select(range(0, 300_000))

TOKENIZER_PATH = "./data/tokenizer.json"
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
pad_id = tokenizer.token_to_id("[PAD]")
tokenizer.enable_padding(length=maxlen, pad_id = pad_id, pad_token="[PAD]")
tokenizer.enable_truncation(max_length=maxlen)

class TorchDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, pad_id = 0):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.pad_id = pad_id
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw_text = self.dataset["text"][idx]
        formatted_text = f"<|startoftext|> {raw_text} <|endoftext|>"
        token_ids = torch.tensor(tokenizer.encode(formatted_text).ids)
        x = token_ids
        y = torch.cat([token_ids[1:], torch.tensor([self.pad_id])])
        return x, y

train_dataset = TorchDataset(test, tokenizer)

dataloader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    drop_last = True,
    num_workers = 0,
    pin_memory = torch.cuda.is_available()
)

class EmbeddingBlock(nn.Module):
    def __init__(self, vocab_size: int, maxlen: int, embed_dim: int):
        super().__init__()
        self.token_embed = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_dim)
        self.pos_embed = nn.Embedding(num_embeddings = maxlen, embedding_dim = embed_dim)

    def forward(self, x):
        position = torch.arange(x.shape[1], device = device).view(1, -1) # creating a 2D array
        position_embedding = self.pos_embed(position)
        token_embedding = self.token_embed(x)
        return token_embedding + position_embedding


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, d_ff: int, rate: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.dropout1 = nn.Dropout(p = rate)
        self.layer_norm1 = nn.LayerNorm(normalized_shape = embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, embed_dim),
            nn.Dropout(p = rate)
        )
        self.layer_norm2 = nn.LayerNorm(normalized_shape = embed_dim)

    def forward(self, x):
        seq_len = x.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device = x.device)
        atten_output, _ = self.mha(x, x, x, attn_mask = causal_mask, is_causal = True, need_weights = False)
        attention_output = self.dropout1(atten_output)
        normalized = self.layer_norm1(x + attention_output)
        ffn_output = self.mlp(normalized)
        return self.layer_norm2(normalized + ffn_output)

class GPT(nn.Module):
    def __init__(self, vocab_size, maxlen, embed_dim, num_heads, d_ff, num_transformer_blocks):
        super().__init__()
        self.embedding_block = EmbeddingBlock(vocab_size, maxlen, embed_dim)
        self.transformer_block = nn.Sequential(
            *[AttentionBlock(embed_dim, num_heads, d_ff) for _ in range(num_transformer_blocks)]
        )
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding_block(x)
        x = self.transformer_block(x)
        return self.output_layer(x)

model = GPT(
    vocab_size = vocab_size,
    maxlen = maxlen,
    embed_dim = embed_dim,
    num_heads = num_heads,
    d_ff = d_ff,
    num_transformer_blocks = num_transformer_blocks
).to(device)

lr = 1e-4
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

@torch.no_grad()
def generate_text(model, start_prompt, tokenizer, new_max_tokens, maxlen):
    model.eval()
    pad_id = tokenizer.token_to_id("[PAD]")
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    
    # Strip padding tokens so we predict directly after the prompt text
    encoded = [tid for tid in tokenizer.encode(start_prompt).ids if tid != pad_id]
    input_ids = torch.tensor([encoded], device=device)

    for _ in range(new_max_tokens):
        context = input_ids[:, -maxlen:]
        logits = model(context)
        next_token_logit = logits[:, -1, :]
        next_token = torch.argmax(next_token_logit, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == eos_id:
            break
    model.train()
    return tokenizer.decode(input_ids[0].tolist())

num_epochs = 1
start_prompt = "<|startoftext|> once upon a time"
metric_history = {
    "step": [],
    "train_loss": [], 
    "train_perplexity": [],
    "total_time_sec": 0.0
}
model.train()
step = 0
running_loss = 0.0
total_start_time = time.time()

for epoch in range(num_epochs):
    start_time = time.time()

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % 100 == 0:
            avg_loss = running_loss / 100
            perplexity = torch.exp(torch.tensor(avg_loss)).item()

            metric_history["train_loss"].append(avg_loss)
            metric_history["train_perplexity"].append(perplexity)
            metric_history["step"].append(step + 1)

            elapsed_time = time.time() - start_time
            print(f"\nStep {step + 1} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f} | Time: {elapsed_time:.2f}s")

            print("generated text:")
            print(generate_text(model, start_prompt, tokenizer, new_max_tokens = 50, maxlen = maxlen))

            running_loss = 0.0
            start_time = time.time()
        
        step += 1

total_time = time.time() - total_start_time
metric_history["total_time_sec"] = total_time
print(f"\n[PyTorch] Total Training Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

# Save metrics JSON
import json
out_dir = Path(__file__).parent
with open(out_dir / "pytorch_metrics.json", "w") as f:
    json.dump(metric_history, f, indent=2)
print("Saved pytorch_metrics.json")

# Plot Loss & Perplexity
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(metric_history["step"], metric_history["train_loss"], color="red", label="Train Loss")
ax1.set_title(f"PyTorch Loss Curve (Total Time: {total_time:.1f}s)")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Loss")
ax1.grid(True)
ax1.legend()

ax2.plot(metric_history["step"], metric_history["train_perplexity"], color="blue", label="Perplexity")
ax2.set_title("PyTorch Perplexity Curve")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Perplexity")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig(out_dir / "pytorch_curves.png")
print("Saved pytorch_curves.png")

    
