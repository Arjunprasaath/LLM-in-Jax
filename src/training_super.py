"""
TODO:
    [ x ] Add multi-dataset training and evaluation
    [ x ] Add in-place checkpointing
    [ X ] Add Parallel Model Training
    [ x ] Add live tensorboard support
    [ x ] Bring in the BPB evaluation code here
    [ x ] Also add light eval evaluation
    [ x ] loss and training tokens used log-log graph
    [ x ] Split the test eval for each domain (math / web + wiki)
    [ x ] Used cuDNN attention
    [ x ] Use gradient accumulation
    [ x ] Cosine learning rate scheduler with warmup
    [ x ] Introduce packing
"""


# import statements
import os
import gc
import math
import glob
import json
import yaml
import time
from pathlib import Path
from pprint import pprint
from itertools import islice
import matplotlib.pyplot as plt

import jax
jax.random.key(0).block_until_ready()
import numpy as np
import jax.numpy as jnp
from tensorboard.summary import Writer
from torch.utils.tensorboard import SummaryWriter

import flax.nnx as nnx
import optax
import grain.python as grain
import orbax.checkpoint as orbax    

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset

import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.custom.custom_model import CustomModelConfig
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager

from model.gpt import GPT, GPT_v2_1
from utils.scheduler import warmup_stable_decay_scheduler

ROOT = Path(__file__).resolve().parent.parent   # repo root
# hyperparemeters
with open(Path(__file__).resolve().parent / "config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# model 1 configs
vocab_size = config["model_training_1"]["vocab_size"]
embed_dim = config["model_training_1"]["embed_dim"]
maxlen = config["model_training_1"]["maxlen"]
num_heads = config["model_training_1"]["num_heads"]
d_ff = config["model_training_1"]["d_ff"]
num_transformer_blocks = config["model_training_1"]["num_transformer_blocks"]

max_steps = config["model_training_1"]["max_steps"]
batch_size = config["model_training_1"]["batch_size"]
lr = float(config["model_training_1"]["lr"])
version = config["model_training_1"]["version"]

init_seed = config["model_training_1"]["init_seed"]
data_seed = config["model_training_1"]["data_seed"]
grad_accum = config["model_training_1"]["grad_accum"]
warmup_micro_steps = config["model_training_1"]["warmup_micro_steps"]
stable_micro_steps = config["model_training_1"]["stable_micro_steps"]
decay_micro_steps = config["model_training_1"]["decay_micro_steps"]
checkpoint_interval = config["model_training_1"]["checkpoint_interval"]
eval_interval = config["model_training_1"]["eval_interval"]
lighteval_interval = config["model_training_1"]["lighteval_interval"]
training_interval = config["model_training_1"]["training_interval"]
resume = config["model_training_1"]["resume"]

rngs_1 = nnx.Rngs(init_seed)
log_dir = f"logs/tensorboard_{batch_size}_{version}{"_resumed" if resume else ""}"
writer = Writer(log_dir)
fig_writer = SummaryWriter(log_dir)

# load the tokenizer
print(f'Loading tokenizer ...')
start_prompt = ["<|startoftext|> once up on a time"]
tokenizer = Tokenizer.from_file(str(ROOT / "final_tokenizer_32k/tokenizer.json"))
pad_id = tokenizer.token_to_id("[PAD]")
# tokenizer.enable_padding(length=maxlen, pad_id = pad_id, pad_token="[PAD]")
# tokenizer.enable_truncation(max_length=maxlen)

tokenizer_start_prompt = tokenizer.encode(start_prompt[0])
start_token, sample_index = jnp.asarray(tokenizer_start_prompt.ids)[None, :], sum(tokenizer_start_prompt.attention_mask)
padded = jnp.zeros((1, maxlen), dtype = jnp.int32)
padded = padded.at[:, :start_token.shape[-1]].set(start_token)

# getting the token bytes for Bit-Per-Bytes
special_ids = {pad_id, tokenizer.token_to_id("<|startoftext|>"), tokenizer.token_to_id("<|endoftext|>")}
token_bytes = np.zeros(vocab_size, np.float32)

for token_str, idx in tokenizer.get_vocab().items():
    if idx not in special_ids:
        token_bytes[idx] = len(tokenizer.decode([idx]).encode("utf-8"))
token_bytes = jnp.asarray(token_bytes)

# load the dataset
class PackedMemmapDataset(grain.RandomAccessDataSource):
    def __init__(self, bin_path, maxlen):
        self.maxlen = maxlen
        self.data = np.memmap(bin_path, dtype = np.uint16, mode="r")
        self.num_sequences = (len(self.data) - 1) // maxlen

    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx * self.maxlen
        chunk = self.data[start : start + self.maxlen + 1].astype(np.int32)
        return {
            "input_ids": chunk[:-1],
            "target_ids": chunk[1:],
        }

def make_data_fun(source, seed = 100, repeat = True):
    return (
        grain.MapDataset.source(source)
        .shuffle(seed = seed)
        .repeat()
        .to_iter_dataset()
    )

print(f"Loading datasets ...")
WEB_WIKI_TRAIN_PATH = ROOT / "data_bin/web_wiki_train.bin"
WEB_WIKI_TEST_PATH = ROOT / "data_bin/web_wiki_test.bin"
MATH_TRAIN_PATH = ROOT / "data_bin/math_train.bin"
MATH_TEST_PATH = ROOT / "data_bin/math_test.bin"


web_wiki_train = make_data_fun(PackedMemmapDataset(WEB_WIKI_TRAIN_PATH, maxlen = maxlen), seed = data_seed)
web_wiki_test = make_data_fun(PackedMemmapDataset(WEB_WIKI_TEST_PATH, maxlen = maxlen), seed = 121)
math_train = make_data_fun(PackedMemmapDataset(MATH_TRAIN_PATH, maxlen = maxlen), seed = data_seed + 2)
math_test = make_data_fun(PackedMemmapDataset(MATH_TEST_PATH, maxlen = maxlen), seed = 123)

train_dataset = grain.IterDataset.mix([web_wiki_train, math_train], weights = [1, 0.8])

train_pipeline = (
    train_dataset
    .batch(batch_size = batch_size, drop_remainder = True)
)
train_iterator = iter(train_pipeline)

test_pipeline_web_wiki = (
    web_wiki_test
    .batch(batch_size = batch_size, drop_remainder = True)
)
test_pipeline_math = (
    math_test
    .batch(batch_size = batch_size, drop_remainder = True)
)

# initializing model,  optimizer adn metrics
print(f'Loading models, optimizer and metrics ...')
model_1 = GPT_v2_1(vocab_size, maxlen, embed_dim, num_heads, d_ff, num_transformer_blocks, rngs_1)
print(nnx.tabulate(model_1, jnp.ones((1, maxlen), dtype = jnp.int32)))

# LR scheduler
scheduler = warmup_stable_decay_scheduler(
    init_value = 0.0,
    peak_value = lr,
    warmup_steps = warmup_micro_steps // grad_accum,
    stable_steps = stable_micro_steps // grad_accum,
    decay_steps = decay_micro_steps // grad_accum,
    end_value = lr / 100,
    decay_type = "linear",
)

optimizer_1 = nnx.Optimizer(
    model_1, 
    optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(scheduler),
            ),
        every_k_schedule = grad_accum
    ),
    wrt = nnx.Param,
)

# Check for existing checkpoint and restore model, optimizer, and iterator state
step = 0
checkpointer = orbax.PyTreeCheckpointer()
checkpoint_path_1 = Path(f"checkpoint_{batch_size}_{version}").resolve()

if resume:
    load_checkpoint_path = checkpoint_path_1
    if load_checkpoint_path.exists():
        # Restore Model and Optimizer
        abstract_state = nnx.state((model_1, optimizer_1))
        restored_state = checkpointer.restore(load_checkpoint_path, args=orbax.args.PyTreeRestore(item=abstract_state))
        nnx.update((model_1, optimizer_1), restored_state)
        print(f"Successfully restored model and optimizer from {load_checkpoint_path}")

        # Restore Iterator state and global step
        iter_state_path = load_checkpoint_path / "iterator_state.json"
        if iter_state_path.exists():
            with open(iter_state_path, "r") as f:
                saved_info = json.load(f)
            train_iterator.set_state(saved_info["iter_state"])
            step = saved_info.get("step", 0)
            print(f"Restored iterator state from step: {step}")
        else:
            print(f"Warning: No iterator_state.json found in {load_checkpoint_path}, starting data from step 0")
    else:
        print(f"Warning: Checkpoint not found at {load_checkpoint_path}, training from scratch with step: {step}")
else:
    print(f"Fresh run starting with step: {step}")


train_1_metrics = nnx.MultiMetric(loss = nnx.metrics.Average("loss"), perplexity = nnx.metrics.Average("perplexity"))
eval_1_metrics = nnx.MultiMetric(loss = nnx.metrics.Average("loss"), perplexity = nnx.metrics.Average("perplexity"))

history_path = checkpoint_path_1 / "metric_history.json"

if resume and history_path.exists():
    with open(history_path ,'r') as f:
        metric_history = json.load(f)
    print(f"Loaded existing metric histroy with {len(metric_history['train_1_loss'])} and {len(metric_history['eval_1_loss'])}\n")
else:
    metric_history = {"train_1_loss": [], 
    "train_1_perplexity": [],
    "eval_1_loss": [],
    "eval_1_perplexity": [],
    "eval_1_bpb": [],
    "train_2_loss": [], 
    "train_2_perplexity": [],
    "eval_2_loss": [],
    "eval_2_perplexity": [],
    "eval_2_bpb": []}

# Set up Lighteval
from lightevaluation import JaxEvalModel
lighteval_evaluation_tracker = EvaluationTracker(
    output_dir=f"eval_results/{version}",
    save_details=True
)

# Configure the pipeline
lighteval_pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.CUSTOM,
    max_samples=500, # TO BE CHANGED
    custom_tasks_directory = str(ROOT / "src/custom_benchmarks.py"),
)

# Create and run the pipeline
lighteval_pipeline = Pipeline(
    tasks=",".join(["hellaswag_cloze", "arc:easy", "piqa_cloze", "sciq", "lambada_openai_acc", "openbookqa_cloze"]), # no spaces: lighteval doesn't strip names
    pipeline_parameters=lighteval_pipeline_params,
    evaluation_tracker=lighteval_evaluation_tracker,
    model=JaxEvalModel(model_1, ROOT / "final_tokenizer_32k/tokenizer.json", maxlen)
)

# https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html
prep_target_batch = jax.vmap(
    lambda token: jnp.concatenate((token[1:],jnp.array([pad_id]))) # we are creating target by shifting one token to the right, adding [0] for shape consistency
) # and using vmap cause we want to parallelize this lambda function over the batch dimension.

def loss_fun(model, input_batch, target_batch):
    logits = model(input_batch).astype(jnp.float32)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits = logits, labels = target_batch)
    mean_loss = loss.mean()
    perplexity = jnp.exp(mean_loss)
    return mean_loss, (perplexity, target_batch.size)

@nnx.jit(donate_argnums=(0, 1, 2))
def train_step(model: GPT_v2_1, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, input_batch, target_batch):
    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True) # https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/transforms.html#flax.nnx.value_and_grad
    (loss, (perplexity, num_tokens)), grad = grad_fun(model, input_batch, target_batch)
    metrics.update(loss = loss, perplexity = perplexity, labels = target_batch) # https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/training/metrics.html
    optimizer.update(model, grad) # https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/training/optimizer.html
    return loss, perplexity, num_tokens

@nnx.jit
def eval_step(model: GPT_v2_1, metrics: nnx.MultiMetric, input_batch, target_batch, token_bytes):
    (loss, (perplexity, num_tokens)) = loss_fun(model, input_batch, target_batch)
    metrics.update(loss = loss, perplexity = perplexity, labels = target_batch)
    mask = (target_batch != pad_id).astype(jnp.float32)
    return loss * mask.sum(), (token_bytes[target_batch] * mask).sum()

# training steps
tokens_per_step = batch_size * maxlen
total_tokens = step * tokens_per_step
window_tokens = 0
num_eval_batches = 100
loss_vs_tokens = []
eval_dataset = {
    "web_wiki": list(islice(test_pipeline_web_wiki, num_eval_batches)),
    "math": list(islice(test_pipeline_math, num_eval_batches)),
}

window_start = time.perf_counter()

print("Training starts ...")
for batch in train_iterator:
    model_1.train()
    input_batch = batch["input_ids"]
    target_batch = batch["target_ids"]
    if step % checkpoint_interval == 0:
        print(f"\n Data check step {step + 1}: {input_batch[0,:8].tolist()}")

    loss_1, perplexity_1, num_tokens_1 = train_step(model_1, optimizer_1, train_1_metrics, input_batch, target_batch)

    total_tokens += tokens_per_step
    window_tokens += tokens_per_step

    if (step + 1) % training_interval == 0:
        loss_val = float(loss_1)
        elapsed = time.perf_counter() - window_start
        tokens_per_sec = window_tokens / elapsed
        gradient_step = int(optimizer_1.opt_state.gradient_step[...])
        lr_now = float(scheduler(max(gradient_step - 1, 0)))

        writer.add_scalar("train_1/tokens_per_sec", tokens_per_sec, step = step)
        writer.add_scalar("train_1/total_tokens", total_tokens, step = step)
        writer.add_scalar("train_1/loss", loss_val, step = step)
        writer.add_scalar("train_1/perplexity", float(perplexity_1), step = step)
        writer.add_scalar("train_1/gradient_step", gradient_step, step = step) # should always be = (step + 1) // grad_accum
        writer.add_scalar("train_1/lr", lr_now, step = step)
        loss_vs_tokens.append((int(total_tokens), loss_val))

        for metric, value in train_1_metrics.compute().items():
            metric_history[f"train_1_{metric}"].append(float(value))
        train_1_metrics.reset()
        writer.add_scalar("train_1/loss_avg", metric_history["train_1_loss"][-1], step = step)
        print(f"\n Training Loss 1: {metric_history["train_1_loss"][-1]}, Perplexity 1: {metric_history["train_1_perplexity"][-1]}, Step {step + 1}")
        window_tokens = 0
        window_start = time.perf_counter()

    if (step + 1) % eval_interval == 0:
        eval_start = time.perf_counter()
        model_1.eval()
        for name, batches in eval_dataset.items():
            eval_1_metrics.reset()
            total_nats = total_bytes = 0.0
            for eval_batch in batches:
                eval_input = eval_batch["input_ids"]
                eval_target = prep_target_batch(eval_input)
                nats, n_bytes = eval_step(model_1, eval_1_metrics, eval_input, eval_target, token_bytes)
                total_nats += float(nats); total_bytes += float(n_bytes)
        
            eval_bpb = total_nats / math.log(2) / total_bytes
            computed_eval_1 = eval_1_metrics.compute()
            
            metric_history.setdefault(f"eval_1_{name}_bpb", []).append(eval_bpb)
            for metric, value in computed_eval_1.items():
                metric_history.setdefault(f"eval_1_{name}_{metric}", []).append(float(value))

            writer.add_scalar(f"eval_1/{name}_loss", float(computed_eval_1["loss"]), step = step)
            writer.add_scalar(f"eval_1/{name}_perplexity", float(computed_eval_1["perplexity"]), step = step)
            writer.add_scalar(f"eval_1/{name}_bpb", float(eval_bpb), step = step)
            print(f"\n Eval {name} Loss: {computed_eval_1["loss"]:.4f}, Perplexity: {computed_eval_1["perplexity"]:.4f}, BPB: {eval_bpb:.4f}, Step {step + 1}")

        # Scaling law plot
        if loss_vs_tokens:
            toks, losses = zip(*loss_vs_tokens)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.loglog(toks, losses, label="Train Loss", color="royalblue")
            ax.set_xlabel("Training Tokens")
            ax.set_ylabel("Train Loss")
            ax.set_title("Scaling Law: Loss vs Tokens")
            ax.grid(True, which="both", linestyle="--", alpha=0.3)
            ax.legend()
            
            fig_writer.add_figure("scaling/loss_vs_tokens", fig, global_step=step)
            plt.close(fig)
        
        writer.add_scalar("eval_1/eval_seconds", time.perf_counter() - eval_start, step = step)
        writer.flush()
        window_tokens = 0
        window_start = time.perf_counter()

    if (step + 1) % lighteval_interval == 0:
        lighteval_start = time.perf_counter()
        model_1.eval()
        # clearing loggers so eval messuers only the current model performance.
        lighteval_pipeline.evaluation_tracker.metrics_logger.metrics_values.clear()
        lighteval_pipeline.evaluation_tracker.metrics_logger.metric_aggregated.clear()
        if hasattr(lighteval_pipeline.evaluation_tracker, "details_logger"):
            lighteval_pipeline.evaluation_tracker.details_logger.details.clear()
        lighteval_pipeline.evaluate()

        for task, scores in lighteval_pipeline.get_results()["results"].items():
            if task == "all": continue
            for metric, value in scores.items():
                writer.add_scalar(f"lighteval/{task}/{metric}", float(value), step = step)
                print(f"\n Lighteval {task}: {metric}: {float(value):.4f}, Step {step + 1}")

        writer.add_scalar("lighteval/eval_seconds", time.perf_counter() - lighteval_start, step = step)
        print("Generated text Model 1:")
        print(model_1.generate_text(100, padded, sample_index, tokenizer))
        window_tokens = 0
        window_start = time.perf_counter()

    # checkpoint every inplace_checkpoint epochs or at the end of training
    if (step + 1) % checkpoint_interval == 0 or (step + 1) >= max_steps:
        state = nnx.state((model_1, optimizer_1))
        checkpointer.save(checkpoint_path_1, args = orbax.args.PyTreeSave(state), force = True)

        with open(checkpoint_path_1 / "iterator_state.json", "w") as f:
            json.dump({"step": step + 1, "iter_state": train_iterator.get_state()}, f)

        with open (history_path, "w") as f:
            json.dump(metric_history, f)

        window_tokens = 0
        window_start = time.perf_counter()

    step += 1
    if step >= max_steps: break

train_iterator.close()
del train_iterator
gc.collect()
writer.close()
fig_writer.close()
