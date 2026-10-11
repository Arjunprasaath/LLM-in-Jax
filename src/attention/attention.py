# use this link: https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/nn/attention.html#flax.nnx.MultiHeadAttention

# imports
import jax
import jax.numpy as jnp
import flax.nnx as nnx

# basic attention block
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

        