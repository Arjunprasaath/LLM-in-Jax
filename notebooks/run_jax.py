import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "0.20"

import time
from pprint import pprint

import grain.python as grain

import jax
import jax.numpy as jnp

import flax.nnx as nnx
import optax

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

vocab_size = 10_000
embed_dim = 768
maxlen = 512
num_heads = 8
d_ff = embed_dim * 4
num_transformer_blocks = 4
batch_size = 64

num_epochs = 1
rngs = nnx.Rngs(0)

ds = load_dataset("karpathy/tinystories-gpt4-clean", split="train")
test = ds.select(range(0, 300_000))

tokenizer = Tokenizer.from_file("data/tokenizer.json")
pad_id = tokenizer.token_to_id("[PAD]")
tokenizer.enable_padding(length=maxlen, pad_id = pad_id, pad_token="[PAD]")
tokenizer.enable_truncation(max_length=maxlen)

# Have a look at : https://google-grain.readthedocs.io/en/latest/tutorials/data_loader_tutorial.html#transformations
class ProcessStory(grain.MapTransform):
    def map(self, element):
        text = "<|startoftext|> " + element['text'] + " <|endoftext|>"
        tokenized_text = tokenizer.encode(text).ids
        return jnp.asarray(tokenized_text)

sampler = grain.IndexSampler(
    num_records = len(test),
    num_epochs = num_epochs,
    shuffle = True,
    seed = 40,
    shard_options = grain.NoSharding()
)

dataloader = grain.DataLoader(
    data_source = test,
    sampler = sampler,
    operations = [ProcessStory(),
     grain.Batch(batch_size = batch_size, drop_remainder=True)],
     worker_count = 0
)

class EmbeddingBlock(nnx.Module):
    def __init__(self, vocab_size: int, maxlen: int, embed_dim: int, rngs: nnx.Rngs):
        self.token_embed = nnx.Embed(num_embeddings = vocab_size, features = embed_dim, rngs = rngs)
        self.pos_embed = nnx.Embed(num_embeddings = maxlen, features = embed_dim, rngs = rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        poisitons = jnp.arange(0, x.shape[1])[None, :] # create a 2D array
        position_embedding = self.pos_embed(poisitons)
        token_embedding = self.token_embed(x)
        return token_embedding + position_embedding

# use this link: https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/nn/attention.html#flax.nnx.MultiHeadAttention
class AttentionBlock(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, d_ff: int, rngs: nnx.Rngs, rate: float = 0.1):
        self.mha = nnx.MultiHeadAttention(num_heads = num_heads, in_features = embed_dim, rngs = rngs)
        self.dropout1 = nnx.Dropout(rate = rate, rngs = rngs)
        self.layer_norm1 = nnx.LayerNorm(epsilon = 1e-6, num_features = embed_dim, rngs = rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(in_features = embed_dim, out_features = d_ff, rngs = rngs),
            nnx.relu,
            nnx.Linear(in_features = d_ff, out_features = embed_dim, rngs = rngs),
            nnx.Dropout(rate = rate, rngs = rngs))
        self.layer_norm2 = nnx.LayerNorm(epsilon = 1e-6, num_features = embed_dim, rngs = rngs)
    
    def __call__(self, inputs):
        causal_mask = nnx.make_causal_mask(inputs[..., 0], dtype=bool)
        attention_output = self.dropout1(self.mha(inputs_q = inputs, mask = causal_mask, decode = False))
        normalized = self.layer_norm1(inputs + attention_output)
        ffn_output = self.mlp(normalized)
        return self.layer_norm2(normalized + ffn_output)

        
class GPT(nnx.Module):
    def __init__(self, vocab_size : int, maxlen : int, embed_dim: int, num_heads: int, d_ff: int, num_transformer_blocks: int, rngs: nnx.Rngs):
        self.embedding_block = EmbeddingBlock(vocab_size = vocab_size, maxlen = maxlen, embed_dim = embed_dim, rngs = rngs)
        self.transformer_blocks = nnx.Sequential(
            *[AttentionBlock(embed_dim = embed_dim, num_heads = num_heads, d_ff = d_ff, rngs = rngs)  for _ in range(num_transformer_blocks)]
        )
        self.output_layer = nnx.Linear(in_features = embed_dim, out_features = vocab_size, rngs = rngs)
    
    def __call__(self, inputs):
        x = self.embedding_block(inputs)
        x = self.transformer_blocks(x)
        return self.output_layer(x)
    
    def sample_from(self, logits, top_k = 10):
        logits, indices = jax.lax.top_k(logits, k = top_k) # https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html
        logits = jax.nn.softmax(logits) # https://docs.jax.dev/en/latest/_autosummary/jax.nn.softmax.html
        return jax.random.choice(jax.random.PRNGKey(1), indices, p = logits) # https://docs.jax.dev/en/latest/_autosummary/jax.random.choice.html

    @nnx.jit(donate_argnums=(1, ))
    def generate_step(self, padded_token, sample_index):
        logits = self(padded_token)
        next_token = self.sample_from(logits[0][sample_index])
        return next_token
        
    def generate_text(self, max_tokens, padded_tokens, sample_index, tokenizer):
        generated_tokens = []
        for i in range(max_tokens):
            sample_index += 1
            # print(f"Sample index: {sample_index}")
            next_token = self.generate_step(padded_tokens, sample_index)
            if next_token == tokenizer.encode("<|endoftext|>").ids[0]:
                break
            generated_tokens.append(int(next_token))
        return tokenizer.decode(generated_tokens)
    
def loss_fun(model, batch):
    logits = model(batch[0]).astype(jnp.float32)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits = logits, labels = batch[1])
    mask = (batch[1] != pad_id).astype(jnp.float32)
    masked_loss = (loss * mask).sum() / mask.sum()
    perplexity = jnp.exp(masked_loss)
    return masked_loss, (perplexity, logits)

@nnx.jit(donate_argnums=(0, 1, 3))
def train_step(model: GPT, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True) # https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/transforms.html#flax.nnx.value_and_grad
    (loss, (perplexity, logits)), grad = grad_fun(model, batch)
    metrics.update(loss = loss, perplexity = perplexity, logits = logits, labels = batch[1]) # https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/training/metrics.html
    optimizer.update(model, grad) # https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/training/optimizer.html

model = GPT(vocab_size, maxlen, embed_dim, num_heads, d_ff, num_transformer_blocks, rngs)
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)
metrics = nnx.MultiMetric(loss = nnx.metrics.Average("loss"), perplexity = nnx.metrics.Average("perplexity"))
rng = jax.random.PRNGKey(0)

start_prompt = "<|startoftext|> once upon a time"
tok_prompt = tokenizer.encode(start_prompt)
start_token = jnp.asarray(tok_prompt.ids)[None, :]
sample_index = sum(tok_prompt.attention_mask)

metric_history = {
    "step": [],
    "train_loss": [], 
    "train_perplexity": [],
    "total_time_sec": 0.0
}

# https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html
prep_target_batch = jax.vmap(
    lambda token: jnp.concatenate((token[1:], jnp.array([0]))) # we are creating target by shifting one token to the right, adding [0] for shape consistency
) # and using vmap cause we want to parallelize this lambda function over the batch dimension.

# training steps
step = 0
total_start_time = time.time()
for epoch in range(num_epochs):
    start_time = time.time()
    for batch in dataloader:
        # print(batch.shape)
        train_step(model, optimizer, metrics, (batch, prep_target_batch(batch)))
        if (step + 1) % 100 == 0:
            for metric, value in metrics.compute().items():
                metric_history[f"train_{metric}"].append(float(value))
            metric_history["step"].append(step + 1)
            metrics.reset()
            elapsed_time = time.time() - start_time
            print(f"\n Step {step + 1}, Loss: {metric_history['train_loss'][-1]:.4f}, Perplexity: {metric_history['train_perplexity'][-1]:.4f}, Spent Time: {elapsed_time:.2f} seconds")
            start_time = time.time()
            print("Generated text:")
            print(model.generate_text(100, start_token, sample_index, tokenizer))
        step += 1

total_time = time.time() - total_start_time
metric_history["total_time_sec"] = total_time
print(f"\n[JAX] Total Training Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

# Save metrics JSON
import json
from pathlib import Path
out_dir = Path(__file__).parent
with open(out_dir / "jax_metrics.json", "w") as f:
    json.dump(metric_history, f, indent=2)
print("Saved jax_metrics.json")

# Plot Loss & Perplexity
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(metric_history["step"], metric_history["train_loss"], color="red", label="Train Loss")
ax1.set_title(f"JAX Loss Curve (Total Time: {total_time:.1f}s)")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Loss")
ax1.grid(True)
ax1.legend()

ax2.plot(metric_history["step"], metric_history["train_perplexity"], color="blue", label="Perplexity")
ax2.set_title("JAX Perplexity Curve")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Perplexity")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig(out_dir / "jax_curves.png")
print("Saved jax_curves.png")