# imports
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from embedding.embedding import EmbeddingBlock
from attention.attention import AttentionBlock

# basic GPT model
class GPT(nnx.Module):
    def __init__(self, vocab_size : int, maxlen : int, embed_dim: int, num_heads: int, d_ff: int, num_transformer_blocks: int, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.bfloat16):
        self.embedding_block = EmbeddingBlock(vocab_size = vocab_size, maxlen = maxlen, embed_dim = embed_dim, rngs = rngs)
        self.transformer_blocks = nnx.Sequential(
            *[AttentionBlock(embed_dim = embed_dim, num_heads = num_heads, d_ff = d_ff, rngs = rngs)  for _ in range(num_transformer_blocks)]
        )
        self.output_layer = nnx.Linear(in_features = embed_dim, out_features = vocab_size, rngs = rngs, dtype = dtype, param_dtype = dtype)
    
    def __call__(self, inputs):
        x = self.embedding_block(inputs)
        x = self.transformer_blocks(x)
        return self.output_layer(x)
    
    def sample_from(self, logits, rng_key, top_k = 10):
        logits, indices = jax.lax.top_k(logits, k = top_k) # https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html
        logits = jax.nn.softmax(logits) # https://docs.jax.dev/en/latest/_autosummary/jax.nn.softmax.html
        return jax.random.choice(rng_key, indices, p = logits) # https://docs.jax.dev/en/latest/_autosummary/jax.random.choice.html

    @nnx.jit(donate_argnums=(1, ))
    def generate_step(self, padded_token, sample_index, rng_key):
        logits = self(padded_token)
        next_token = self.sample_from(logits[0][sample_index], rng_key)
        return next_token
        
    def generate_text(self, max_tokens, padded_tokens, sample_index, tokenizer):
        generated_tokens = []
        rng_key = jax.random.PRNGKey(42)
        for i in range(max_tokens):
            # print(padded_tokens)
            rng_key, step_key = jax.random.split(rng_key)
            next_token = self.generate_step(padded_tokens, sample_index, step_key)
            if next_token == tokenizer.encode("<|endoftext|>").ids[0]:
                break
            generated_tokens.append(next_token)
            padded_tokens = padded_tokens.at[0, sample_index].set(next_token)
            sample_index += 1
        return tokenizer.decode(generated_tokens)
    