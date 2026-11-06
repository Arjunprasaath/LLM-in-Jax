import os
import glob
import itertools

import grain.python as grain

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers, decoders

MAX_BATCHES = 20000
VOCAB_SIZE = 32000
WEB_WIKI_PATH = "./datasets/preprocessed_combined/train_split/"
MATH_PATH = "./datasets/preprocessed_finemath/train_split/"

class NormalizeWikiWeb(grain.MapTransform):
    def map(self, element):
        return {"text": str(element["text"])}

class NormalizeMath(grain.MapTransform):
    def map(self, element):
        return {"text": str(element["text"])}

def iterator():
    for batch in itertools.islice(batched_dataset, MAX_BATCHES):
        yield batch['text'].tolist()

# Load parquet dataset in grain: https://google-grain.readthedocs.io/en/latest/tutorials/data_sources/parquet_dataset_tutorial.html#load-dataset
web_wiki_parquet_files = glob.glob(WEB_WIKI_PATH + "*.parquet") # returns a list of all the parquet files in that directory
math_parquet_files = glob.glob(MATH_PATH + "*.parquet")

wiki_web_source = grain.experimental.ParquetIterDataset(web_wiki_parquet_files)
math_source = grain.experimental.ParquetIterDataset(math_parquet_files)

wiki_web_aligned = wiki_web_source.map(NormalizeWikiWeb())
math_algined = math_source.map(NormalizeMath())

mixed_dataset = grain.IterDataset.mix([wiki_web_aligned, math_algined], weights = [1, 0.5])
batched_dataset = mixed_dataset.batch(batch_size = 2048, drop_remainder = True)

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
tokenizer.decoder = decoders.ByteLevel()
trainer = BpeTrainer(special_tokens=["[PAD]", "<|startoftext|>", "<|endoftext|>"], vocab_size = VOCAB_SIZE, initial_alphabet = pre_tokenizers.ByteLevel.alphabet())
tokenizer.train_from_iterator(iterator(), trainer = trainer)
os.makedirs("tokenizer_32k", exist_ok = True)
tokenizer.save("tokenizer_32k/tokenizer.json")
