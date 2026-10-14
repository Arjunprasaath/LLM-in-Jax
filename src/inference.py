# imports
import yaml
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx

import orbax.checkpoint as orbax

from tokenizers import Tokenizer
from model.gpt import GPT

# hyperparemeters
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

vocab_size = config["model_training"]["vocab_size"]
embed_dim = config["model_training"]["embed_dim"]
maxlen = config["model_training"]["maxlen"]
num_heads = config["model_training"]["num_heads"]
d_ff = config["model_training"]["d_ff"]
num_transformer_blocks = config["model_training"]["num_transformer_blocks"]

batch_size = config["model_training"]["batch_size"]
num_epochs = config["model_training"]["num_epochs"]
lr = config["model_training"]["lr"]
version = config["model_training"]["version"]

rngs = nnx.Rngs(0)


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
