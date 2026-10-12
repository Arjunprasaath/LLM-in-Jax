# imports
import jax
import jax.numpy as jnp
import flax.nnx as nnx

# basic embedding block
class EmbeddingBlock(nnx.Module):
    def __init__(self, vocab_size: int, maxlen: int, embed_dim: int, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.bfloat16):
        self.token_embed = nnx.Embed(num_embeddings = vocab_size, features = embed_dim, rngs = rngs, dtype = dtype, param_dtype = dtype)
        self.pos_embed = nnx.Embed(num_embeddings = maxlen, features = embed_dim, rngs = rngs, dtype = dtype, param_dtype = dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        poisitons = jnp.arange(0, x.shape[1])[None, :] # create a 2D array
        position_embedding = self.pos_embed(poisitons)
        token_embedding = self.token_embed(x)
        return token_embedding + position_embedding