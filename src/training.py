# import statements
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = ".20"

import time
from pathlib import Path
from pprint import pprint

import jax
import jax.numpy as jnp

import flax.nnx as nnx

import optax
import grain.python as grain
import orbax.checkpoint as orbax    

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset

from model.gpt import GPT

import matplotlib.pyplot as plt

# hyperparemeters
vocab_size = 10_000
embed_dim = 768
maxlen = 512
num_heads = 8
d_ff = embed_dim * 4
num_transformer_blocks = 4

batch_size = 512
num_epochs = 1
lr = 1e-4

rngs = nnx.Rngs(0)

version = "v1"

start_prompt = ["<|startoftext|> once up on a time"]

# load the dataset
ds = load_dataset("karpathy/tinystories-gpt4-clean", split="train")

# load the tokenizer
tokenizer = Tokenizer.from_file("data/tokenizer.json")
pad_id = tokenizer.token_to_id("[PAD]")
tokenizer.enable_padding(length=maxlen, pad_id = pad_id, pad_token="[PAD]")
tokenizer.enable_truncation(max_length=maxlen)

tokenizer_start_prompt = tokenizer.encode(start_prompt[0])
start_token, sample_index = jnp.asarray(tokenizer_start_prompt.ids)[None, :], sum(tokenizer_start_prompt.attention_mask)

class ProcessStory(grain.MapTransform):
    def map(self, element):
        text = "<|startoftext|> " + element['text'] + " <|endoftext|>"
        tokenized_text = tokenizer.encode(text).ids
        return jnp.asarray(tokenized_text)

sampler = grain.IndexSampler(
    num_records = len(ds),
    num_epochs = num_epochs,
    shuffle = True,
    seed = 40,
    shard_options = grain.NoSharding()
)

dataloader = grain.DataLoader(
    data_source = ds,
    sampler = sampler,
    operations = [ProcessStory(),
     grain.Batch(batch_size = batch_size, drop_remainder = True)],
     worker_count = 0
)

# initializing model,  optimizer adn metrics
model = GPT(vocab_size, maxlen, embed_dim, num_heads, d_ff, num_transformer_blocks, rngs)
optimizer = nnx.Optimizer(model, optax.adam(lr), wrt = nnx.Param)
metrics = nnx.MultiMetric(loss = nnx.metrics.Average("loss"))

metric_history = {
    "train_loss": []
}


# https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html
prep_target_batch = jax.vmap(
    lambda token: jnp.concatenate((token[1:], jnp.array([0]))) # we are creating target by shifting one token to the right, adding [0] for shape consistency
) # and using vmap cause we want to parallelize this lambda function over the batch dimension.

def loss_fun(model, batch):
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits = logits, labels = batch[1])
    mask = (batch[1] != pad_id).astype(jnp.float32)
    masked_loss = (loss * mask).sum() / mask.sum()
    print(loss, mask, masked_loss)
    return masked_loss, logits

@nnx.jit(donate_argnums=(0, 1, 3))
def train_step(model: GPT, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True) # https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/transforms.html#flax.nnx.value_and_grad
    (loss, logits), grad = grad_fun(model, batch)
    metrics.update(loss = loss, logits = logits, labels = batch[1]) # https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/training/metrics.html
    optimizer.update(model, grad) # https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/training/optimizer.html

# training steps
step = 0
for epoch in range(num_epochs):
    start_time = time.time()
    for batch in dataloader:
        # print(batch.shape)
        train_step(model, optimizer, metrics, (batch, prep_target_batch(batch)))
        if (step + 1) % 100 == 0:
            for metric, value in metrics.compute().items():
                metric_history[f"train_{metric}"].append(value)
            metrics.reset()
            elapsed_time = time.time() - start_time
            print(f"\n Step {step + 1}, Loss: {metric_history["train_loss"][-1]}, Spent Time: {elapsed_time:.2f} seconds")
            start_time = time.time()
            print("Generated text:")
            print(model.generate_text(100, start_token, sample_index, tokenizer))
        step += 1

# saving the model
state = nnx.state(model)
checkpoint_path = Path(f"checkpoint_{batch_size}_{version}").resolve()
checkpointer = orbax.PyTreeCheckpointer()
checkpointer.save(checkpoint_path, args = orbax.args.PyTreeSave(state), force=True)

plt.plot(metric_history['train_loss'])
plt.title(f"Training loss checkpoint_{batch_size}_{version}")
plt.xlabel("Steps % 100")
plt.ylabel("Loss")
plt.savefig(f"train_loss_checkpoint_{batch_size}_{version}.png")
