# use this link: https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/nn/attention.html#flax.nnx.MultiHeadAttention

# imports
import jax
import jax.numpy as jnp
import flax.nnx as nnx

def cudnn_causal_attention(query, key, value, **kwargs):
    return jax.nn.dot_product_attention(query, key, value, is_causal = True, implementation = "cudnn")

# basic attention block
class AttentionIsAllYouNeed(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, d_ff: int, rngs: nnx.Rngs, rate: float = 0.1, dtype: jnp.dtype = jnp.bfloat16):
        self.param_dtype = jnp.float32
        self.mha = nnx.MultiHeadAttention(num_heads = num_heads, in_features = embed_dim, rngs = rngs, dtype = dtype, param_dtype = self.param_dtype)
        self.dropout1 = nnx.Dropout(rate = rate, rngs = rngs)
        self.layer_norm1 = nnx.LayerNorm(epsilon = 1e-6, num_features = embed_dim, rngs = rngs, dtype = dtype, param_dtype = self.param_dtype)
        self.mlp = nnx.Sequential(
            nnx.Linear(in_features = embed_dim, out_features = d_ff, rngs = rngs, dtype = dtype, param_dtype = self.param_dtype),
            nnx.relu,
            nnx.Linear(in_features = d_ff, out_features = embed_dim, rngs = rngs, dtype = dtype, param_dtype = self.param_dtype),
            nnx.Dropout(rate = rate, rngs = rngs))
        self.layer_norm2 = nnx.LayerNorm(epsilon = 1e-6, num_features = embed_dim, rngs = rngs, dtype = dtype, param_dtype = self.param_dtype)
    
    def __call__(self, inputs):
        causal_mask = nnx.make_causal_mask(inputs[..., 0], dtype=bool)
        attention_output = self.dropout1(self.mha(inputs_q = inputs, mask = causal_mask, decode = False))
        normalized = self.layer_norm1(inputs + attention_output)
        ffn_output = self.mlp(normalized)
        return self.layer_norm2(normalized + ffn_output)

class PreNormAttentionBlock(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, d_ff: int, rngs: nnx.Rngs, rate: float = 0.1, dtype: jnp.dtype = jnp.bfloat16):
        self.param_dtype = jnp.float32
        self.mha = nnx.MultiHeadAttention(num_heads = num_heads, in_features = embed_dim, rngs = rngs, dtype = dtype, param_dtype = self.param_dtype, attention_fn = cudnn_causal_attention)
        self.dropout1 = nnx.Dropout(rate = rate, rngs = rngs)
        self.layer_norm1 = nnx.LayerNorm(epsilon = 1e-6, num_features = embed_dim, rngs = rngs, dtype = dtype, param_dtype = self.param_dtype)
        self.mlp = nnx.Sequential(
            nnx.Linear(in_features = embed_dim, out_features = d_ff, rngs = rngs, dtype = dtype, param_dtype = self.param_dtype),
            nnx.relu,
            nnx.Linear(in_features = d_ff, out_features = embed_dim, rngs = rngs, dtype = dtype, param_dtype = self.param_dtype),
            nnx.Dropout(rate = rate, rngs = rngs))
        self.layer_norm2 = nnx.LayerNorm(epsilon = 1e-6, num_features = embed_dim, rngs = rngs, dtype = dtype, param_dtype = self.param_dtype)

    def __call__(self, inputs):
        # causal_mask = nnx.make_causal_mask(inputs[..., 0], dtype=bool)
        prenorm_inputs = self.layer_norm1(inputs)
        attention_output = self.dropout1(self.mha(inputs_q = prenorm_inputs, decode = False))
        attention_residual = inputs + attention_output
        ffn_input = self.layer_norm2(attention_residual)
        ffn_output = self.mlp(ffn_input)
        return attention_residual + ffn_output