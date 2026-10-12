# imports
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx

import orbax.checkpoint as orbax

from tokenizers import Tokenizer
from model.gpt import GPT

# hyperparemeters
vocab_size = 10_000
embed_dim = 768
maxlen = 512
num_heads = 8
d_ff = embed_dim * 4
num_transformer_blocks = 6

batch_size = 512
num_epochs = 1
lr = 1e-4

rngs = nnx.Rngs(0)

version = "v1"

start_prompt = ["<|startoftext|> once up on a time"]

# model initialization
model = GPT(vocab_size, maxlen, embed_dim, num_heads, d_ff, num_transformer_blocks, rngs)

# restore checkpoints
checkpoint_path = Path(f"checkpoint_{batch_size}_{version}").resolve()
checkpointer = orbax.PyTreeCheckpointer()
abstract_state = nnx.state(model)
restored_state = checkpointer.restore(checkpoint_path, args = orbax.args.PyTreeRestore(item = abstract_state))
nnx.update(model, restored_state)

# load tokenizer
tokenizer = Tokenizer.from_file("data/tokenizer.json")
pad_id = tokenizer.token_to_id("[PAD]")
tokenizer.enable_padding(length=maxlen, pad_id = pad_id, pad_token="[PAD]")
tokenizer.enable_truncation(max_length=maxlen)

start_prompt = ["<|startoftext|> once up on a time"]
tokenizer_start_prompt = tokenizer.encode(start_prompt[0])
start_token, sample_index = jnp.asarray(tokenizer_start_prompt.ids)[None, :], sum(tokenizer_start_prompt.attention_mask) - 1

# generate
output_text = model.generate_text(max_tokens = maxlen, padded_tokens = start_token, sample_index = sample_index, tokenizer = tokenizer)
print(f"Generated text: {start_prompt[0]} {output_text}")
