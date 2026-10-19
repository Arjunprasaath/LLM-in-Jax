import math
import jax
import optax
import jax.numpy as jnp
import flax.nnx as nnx
from datasets import load_dataset
from model.gpt import GPT

@nnx.jit
def eval_step(model: GPT, inputs, targets, pad_id):
    logits = model(inputs).astype(jnp.float32)
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits = logits, labels = targets)
    mask = (targets != pad_id).astype(jnp.float32)
    batch_loss_nats = (ce_loss * mask).sum()
    batch_tokens = mask.sum()
    return batch_loss_nats, batch_tokens

def evaluate_ppl_bpb(model: GPT, dataset, tokenizer, maxlen, pad_id, batch_size = 32, max_samples = 100):
    total_loss_nats = 0.0
    total_tokens = 0
    total_bytes = 0

    eval_data = dataset.select(range(min(len(dataset), max_samples)))
    for i in range(0, len(eval_data), batch_size):
        batch_texts = eval_data[i : i + batch_size]['text']

        for text in batch_texts:
            total_bytes += len(text.encode("utf-8"))
        
        formatted_texts = ["<|startoftext>" + t + "<|endoftext|>" for t in batch_texts]
        encoded = [tokenizer.encode(t).ids for t in formatted_texts]
        inputs = jnp.asarray(encoded)
        targets = jnp.concatenate([inputs[:, 1:], jnp.full((inputs.shape[0], 1), pad_id)], axis = 1)
        loss_nats, num_tokens = eval_step(model, inputs, targets, pad_id)

        total_loss_nats += float(loss_nats)
        total_tokens += int(num_tokens)

    avg_loss_per_token = total_loss_nats / total_tokens
    perplexity = math.exp(avg_loss_per_token)

    total_bits = total_loss_nats / math.log(2)
    bpb = total_bits / total_bytes
    
    print(f"\n--- Evaluation Results ({min(len(eval_data), max_samples)} samples) ---")
    print(f"Per-token Loss : {avg_loss_per_token:.4f} nats")
    print(f"Perplexity     : {perplexity:.2f}")
    print(f"Bits-Per-Byte  : {bpb:.4f} bpb")
    return {"loss": avg_loss_per_token, "ppl": perplexity, "bpb": bpb}


