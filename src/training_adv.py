"""
TODO:
    [ x ] Add multi-dataset training and evaluation
    [] Add in-place checkpointing
    [ X ] Add Parallel Model Training
    [ x ] Add live tensorboard support
    [ x ] Bring in the BPB evaluation code here
    [ x ] Also add light eval evaluation
    [ x ] loss and training tokens used log-log graph
    [ x ] Split the test eval for each domain (math / web + wiki)
    [ x ] Used cuDNN attention
    [ x ] Use gradient accumulation
    [ x ] Cosine learning rate scheduler with warmup
    [  ] Introduce packing


"""


# import statements
import os
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = "platform"

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

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.custom.custom_model import CustomModelConfig
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager

from model.gpt import GPT, GPT_v2_1

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

# model 2 config
num_heads_2 = config["model_training_2"]["num_heads"]
num_transformer_blocks_2 = config["model_training_2"]["num_transformer_blocks"]
version_2 = config["model_training_2"]["version"]

shuffle_seed = 40
rngs_1 = nnx.Rngs(1)
rngs_2 = nnx.Rngs(2)
writer = Writer(f"logs/tensorboard_{batch_size}_{version}")
fig_writer = SummaryWriter(f"logs/tensorboard_{batch_size}_{version}")

# load the tokenizer
print(f'Loading tokenizer ...')
start_prompt = ["<|startoftext|> once up on a time"]
tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer_32k/tokenizer.json"))
pad_id = tokenizer.token_to_id("[PAD]")
tokenizer.enable_padding(length=maxlen, pad_id = pad_id, pad_token="[PAD]")
tokenizer.enable_truncation(max_length=maxlen)

tokenizer_start_prompt = tokenizer.encode(start_prompt[0])
start_token, sample_index = jnp.asarray(tokenizer_start_prompt.ids)[None, :], sum(tokenizer_start_prompt.attention_mask)

# getting the token bytes for Bit-Per-Bytes
special_ids = {pad_id, tokenizer.token_to_id("<|startoftext|>"), tokenizer.token_to_id("<|endoftext|>")}
token_bytes = np.zeros(vocab_size, np.float32)

for token_str, idx in tokenizer.get_vocab().items():
    if idx not in special_ids:
        token_bytes[idx] = len(token_str)
token_bytes = jnp.asarray(token_bytes)

# load the dataset
class NormalizeText(grain.MapTransform):
    def map(self, element):
        return {"text": str(element["text"])}

class ProcessData(grain.MapTransform):
    def map(self, element):
        text = "<|startoftext|> " + element['text'] + " <|endoftext|>"
        tokenized = tokenizer.encode(text)
        return {
            "input_ids": np.asarray(tokenized.ids, dtype = np.int32),
            "attention_mask": np.asarray(tokenized.attention_mask, dtype = np.int32)
        }

def make_data_fun(files, seed = 100, repeat = True):
    ds = grain.experimental.ParquetIterDataset(files)
    ds = ds.map(NormalizeText())
    ds = grain.experimental.WindowShuffleIterDataset(ds, window_size = 10_000, seed = seed)
    if repeat:
        ds = grain.experimental.RepeatIterDataset(ds, num_epochs = None)
    return ds

print(f"Loading datasets ...")
WEB_WIKI_TRAIN_PATH = ROOT / "datasets/preprocessed_combined/train_split/"
WEB_WIKI_TEST_PATH = ROOT / "datasets/preprocessed_combined/test_split/"
MATH_TRAIN_PATH = ROOT / "datasets/preprocessed_finemath/train_split/"
MATH_TEST_PATH = ROOT / "datasets/preprocessed_finemath/test_split/"

web_wiki_train_parquet_files = sorted(glob.glob(str(WEB_WIKI_TRAIN_PATH / "*.parquet")))
math_train_parquet_files = sorted(glob.glob(str(MATH_TRAIN_PATH / "*.parquet")))
web_wiki_test_parquet_files = sorted(glob.glob(str(WEB_WIKI_TEST_PATH / "*.parquet")))
math_test_parquet_files = sorted(glob.glob(str(MATH_TEST_PATH / "*.parquet")))


web_wiki_train_aligned = make_data_fun(web_wiki_train_parquet_files, seed = 10)
web_wiki_test_aligned = make_data_fun(web_wiki_test_parquet_files, seed = 11)
math_train_aligned = make_data_fun(math_train_parquet_files, seed = 12)
math_test_aligned = make_data_fun(math_test_parquet_files, seed = 13)

train_dataset = grain.IterDataset.mix([web_wiki_train_aligned, math_train_aligned], weights = [1, 0.5])
# test_dataset = grain.IterDataset.mix([web_wiki_test_aligned, math_test_aligned], weights = [1, 0.5])

train_pipeline = (
    train_dataset
    .map(ProcessData())
    .batch(batch_size = batch_size, drop_remainder = True)
)

test_pipeline_web_wiki = (
    web_wiki_test_aligned
    .map(ProcessData())
    .batch(batch_size = batch_size, drop_remainder = True)
)
test_pipeline_math = (
    math_test_aligned
    .map(ProcessData())
    .batch(batch_size = batch_size, drop_remainder = True)
)
train_iterator = iter(train_pipeline)

# initializing model,  optimizer adn metrics
print(f'Loading models, optimizer and metrics ...')
model_1 = GPT_v2_1(vocab_size, maxlen, embed_dim, num_heads, d_ff, num_transformer_blocks, rngs_1)
# model_2 = GPT(vocab_size, maxlen, embed_dim, num_heads_2, d_ff, num_transformer_blocks_2, rngs_2)
print(nnx.tabulate(model_1, jnp.ones((1, maxlen), dtype = jnp.int32)))
# print(nnx.tabulate(model_2, jnp.ones((1, maxlen), dtype = jnp.int32)))

# LR scheduler
scheduler = optax.warmup_cosine_decay_scheduler(
    init_value = 0.0,
    peak_value = lr,
    warmup_steps = max_steps * 0.15,
    decay_steps = max_steps * 0.65,
    end_value = lr // 100,
)

optimizer_1 = nnx.Optimizer(
    model_1, 
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.MultiSteps(
            optax.adamw(scheduler),
            every_k_schedule = 4
            ), 
        ),
    wrt = nnx.Param,

# optimizer_2 = nnx.Optimizer(model_2, optax.adamw(lr), wrt = nnx.Param)

train_1_metrics = nnx.MultiMetric(loss = nnx.metrics.Average("loss"), perplexity = nnx.metrics.Average("perplexity"))
eval_1_metrics = nnx.MultiMetric(loss = nnx.metrics.Average("loss"), perplexity = nnx.metrics.Average("perplexity"))
# train_2_metrics = nnx.MultiMetric(loss = nnx.metrics.Average("loss"), perplexity = nnx.metrics.Average("perplexity"))
# eval_2_metrics = nnx.MultiMetric(loss = nnx.metrics.Average("loss"), perplexity = nnx.metrics.Average("perplexity"))

checkpoint_path_1 = Path(f"checkpoint_{batch_size}_{version}").resolve()
# checkpoint_path_2 = Path(f"checkpoint_{batch_size}_{version_2}").resolve()

history_path = checkpoint_path_1 / "metric_history.json"

if history_path.exists():
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
    output_dir="eval_results",
    save_details=True
)

# Configure the pipeline
lighteval_pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.CUSTOM,
    max_samples=128, # TO BE CHANGED
)

# Create and run the pipeline
lighteval_pipeline = Pipeline(
    tasks="swag", # TO BE CHANGED
    pipeline_parameters=lighteval_pipeline_params,
    evaluation_tracker=lighteval_evaluation_tracker,
    model=JaxEvalModel(model_1, ROOT / "tokenizer_32k/tokenizer.json", maxlen)
)

# https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html
prep_target_batch = jax.vmap(
    lambda token: jnp.concatenate((token[1:], jnp.array([0]))) # we are creating target by shifting one token to the right, adding [0] for shape consistency
) # and using vmap cause we want to parallelize this lambda function over the batch dimension.

def loss_fun(model, input_batch, target_batch):
    logits = model(input_batch).astype(jnp.float32)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits = logits, labels = target_batch)
    mask = (target_batch != pad_id).astype(jnp.float32)
    masked_loss = (loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    perplexity = jnp.exp(masked_loss)
    return masked_loss, (perplexity, mask.sum())

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
step = 0
total_tokens = 0
training_interval = 10
eval_interval = 10
lighteval_interval = 10
num_eval_batches = 50
loss_vs_tokens = []
start_time = time.time()
eval_dataset = {
    "web_wiki": list(islice(test_pipeline_web_wiki, num_eval_batches)),
    "math": list(islice(test_pipeline_math, num_eval_batches)),
}

print("Training starts ...")
for batch in train_iterator:
    model_1.train()
    input_batch = batch['input_ids']
    target_batch = prep_target_batch(input_batch)

    loss_1, perplexity_1, num_tokens_1 = train_step(model_1, optimizer_1, train_1_metrics, input_batch, target_batch)
    # loss_2, perplexity_2 = train_step(model_2, optimizer_2, train_2_metrics, input_batch, target_batch)
    
    writer.add_scalar("train_1/loss", float(loss_1), step = step)
    writer.add_scalar("train_1/perplexity", float(perplexity_1), step = step)
    total_tokens += num_tokens_1
    loss_vs_tokens.append((int(total_tokens), float(loss_1)))
    # writer.add_scalar("train_2/loss", float(loss_2), step = step)
    # writer.add_scalar("train_2/perplexity", float(perplexity_2), step = step)

    if (step + 1) % training_interval == 0:
        for metric, value in train_1_metrics.compute().items():
            metric_history[f"train_1_{metric}"].append(float(value))
        train_1_metrics.reset()
        # for metric, value in train_2_metrics.compute().items():
        #     metric_history[f"train_2_{metric}"].append(float(value))
        # train_2_metrics.reset()
        spent_time = time.time() - start_time

        print(f"\n Training Loss 1: {metric_history["train_1_loss"][-1]}, Perplexity 1: {metric_history["train_1_perplexity"][-1]}, Step {step + 1}, Spent Time: {spent_time:.2f} seconds")
        # print(f"\n Training Loss 2: {metric_history["train_2_loss"][-1]}, Perplexity 2: {metric_history["train_2_perplexity"][-1]}, Step {step + 1}, Spent Time: {spent_time:.2f} seconds")

    if (step + 1) % eval_interval == 0:
        model_1.eval()
        for name, batches in eval_dataset.items():
            eval_1_metrics.reset()
            # eval_2_metrics.reset()
            total_nats = total_bytes = 0.0
            for eval_batch in batches:
                eval_input = eval_batch['input_ids']
                eval_target = prep_target_batch(eval_input)
                nats, n_bytes = eval_step(model_1, eval_1_metrics, eval_input, eval_target, token_bytes)
                # eval_step(model_2, eval_2_metrics, eval_input, eval_target)
                total_nats += float(nats); total_bytes += float(n_bytes)
        
            eval_bpb = total_nats / math.log(2) / total_bytes
            computed_eval_1 = eval_1_metrics.compute()
            
            metric_history.setdefault("eval_1_{name}_bpb", []).append(eval_bpb)
            for metric, value in computed_eval_1.items():
                metric_history.setdefault(f"eval_1_{name}_{metric}", []).append(float(value))

            # computed_eval_2 = eval_2_metrics.compute()
            # for metric, value in computed_eval_2.items():
            #     metric_history[f"eval_2_{metric}"].append(float(value))

            writer.add_scalar(f"eval_1/{name}_loss", float(computed_eval_1["loss"]), step = step)
            writer.add_scalar(f"eval_1/{name}_perplexity", float(computed_eval_1["perplexity"]), step = step)
            writer.add_scalar(f"eval_1/{name}_bpb", float(eval_bpb), step = step)
            # writer.add_scalar("eval_2/loss", float(computed_eval_2["loss"]), step = step)
            # writer.add_scalar("eval_2/perplexity", float(computed_eval_2["perplexity"]), step = step)

            print(f"\n Eval {name} Loss: {computed_eval_1["loss"]:.4f}, Perplexity: {computed_eval_1["perplexity"]:.4f}, BPB: {eval_bpb:.4f}, Step {step + 1}")
            # print(f"\n Eval Loss: {computed_eval_2["loss"]:.4f}, Perplexity: {computed_eval_2["perplexity"]:.4f}, Step {step + 1}")

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
        
        writer.flush()

    if (step + 1) % lighteval_interval == 0:
        model_1.eval()
        lighteval_pipeline.evaluate()
        for task, scores in lighteval_pipeline.get_results()["results"].items():
            if task != "all":
                writer.add_scalar(f"lighteval/{task}_acc", float(scores["acc"]), step = step)
                print(f"\n Lighteval {task}: acc {float(scores['acc']):.4f}, Step {step + 1}")

        start_time = time.time()
        print("Generated text Model 1:")
        print(model_1.generate_text(100, start_token, sample_index, tokenizer))
        # print("Generated text Model 2:")
        # print(model_2.generate_text(100, start_token, sample_index, tokenizer))

    step += 1
    if step >= max_steps: break

train_iterator.close()
del train_iterator
gc.collect()
writer.close()
fig_writer.close()

# saving the model, optimizer and train_loss metric
checkpointer = orbax.PyTreeCheckpointer()
state = nnx.state((model_1, optimizer_1))
checkpointer.save(checkpoint_path_1, args = orbax.args.PyTreeSave(state), force=True)

# checkpointer = orbax.PyTreeCheckpointer()
# state = nnx.state((model_2, optimizer_2))
# checkpointer.save(checkpoint_path_2, args = orbax.args.PyTreeSave(state), force=True)


# with open(history_path, 'w') as f:
#     json.dump(metric_history, f)
# print(f"Saved checkpoint and updated metric successfully!\n")

# plt.plot(metric_history['train_loss'])
# plt.title(f"Training loss checkpoint_{batch_size}_{version}")
# plt.xlabel("Steps % 100")
# plt.ylabel("Loss")
# plt.savefig(f"train_loss_checkpoint_{batch_size}_{version}.png")
