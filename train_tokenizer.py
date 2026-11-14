import os
import glob
import itertools
import random

import pyarrow.parquet as pq

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers, decoders

seed = 101
VOCAB_SIZE = 32000
CHUNK = 10_000
DOCS_PER_SOURCE = 1_000_000
WEB_WIKI_PATH = "./datasets/preprocessed_combined/train_split/"
MATH_PATH = "./datasets/preprocessed_finemath/train_split/"


def text_chunks(files, max_docs):
    n = 0
    random.Random(seed).shuffle(files)
    for f in files:
        for rb in pq.ParquetFile(f).iter_batches(batch_size = CHUNK, columns=["text"]):
            texts = rb.column("text").to_pylist()[: max_docs - n]
            n += len(texts)
            yield texts
            if n >= max_docs:
                return
def iterator():
    yield from itertools.chain(
        text_chunks(web_wiki_parquet_files, DOCS_PER_SOURCE),
        text_chunks(math_parquet_files, DOCS_PER_SOURCE)
    )

web_wiki_parquet_files = sorted(glob.glob(WEB_WIKI_PATH + "*.parquet")) # returns a list of all the parquet files in that directory
math_parquet_files = sorted(glob.glob(MATH_PATH + "*.parquet"))

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
tokenizer.decoder = decoders.ByteLevel()
trainer = BpeTrainer(special_tokens=["[PAD]", "<|startoftext|>", "<|endoftext|>"], 
    vocab_size = VOCAB_SIZE,
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
)
tokenizer.train_from_iterator(iterator(), trainer = trainer, length = 2 * DOCS_PER_SOURCE)
os.makedirs("final_tokenizer_32k", exist_ok = True)
tokenizer.save("final_tokenizer_32k/tokenizer.json")
