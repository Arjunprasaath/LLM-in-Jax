import os
import glob
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
print(ROOT)
# 1. Load tokenizer (disable truncation/padding for raw tokenization)
tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer_32k/tokenizer.json"))
tokenizer.no_padding()
tokenizer.no_truncation()

bos_id = tokenizer.token_to_id("<|startoftext|>")
eos_id = tokenizer.token_to_id("<|endoftext|>")

def process_split(parquet_files, output_bin_path, batch_size=4096):
    """Tokenizes parquet files and writes token IDs directly to a .bin file."""
    output_bin_path = Path(output_bin_path)
    output_bin_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {len(parquet_files)} files -> {output_bin_path}")
    total_tokens = 0
    
    with open(output_bin_path, "wb") as f_out:
        for pfile in tqdm(parquet_files, desc="Parquet Files"):
            table = pq.read_table(pfile, columns=["text"])
            texts = table["text"].to_pylist()
            
            # Batch tokenize for fast Rust multithreading
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                # Filter empty texts
                batch_texts = [t for t in batch_texts if t and t.strip()]
                if not batch_texts:
                    continue
                
                encodings = tokenizer.encode_batch(batch_texts)
                
                # Flatten documents separated by [BOS] and [EOS]
                batch_tokens = []
                for enc in encodings:
                    batch_tokens.append(bos_id)
                    batch_tokens.extend(enc.ids)
                    batch_tokens.append(eos_id)
                
                # Write uint16 to disk
                arr = np.array(batch_tokens, dtype=np.uint16)
                arr.tofile(f_out)
                total_tokens += len(arr)

    print(f"Finished {output_bin_path.name}: {total_tokens:,} tokens ({output_bin_path.stat().st_size / (1024**2):.2f} MB)")
    return total_tokens

if __name__ == "__main__":
    WEB_WIKI_TRAIN = sorted(glob.glob(str(ROOT / "datasets/preprocessed_combined/train_split/*.parquet")))
    WEB_WIKI_TEST = sorted(glob.glob(str(ROOT / "datasets/preprocessed_combined/test_split/*.parquet")))
    MATH_TRAIN = sorted(glob.glob(str(ROOT / "datasets/preprocessed_finemath/train_split/*.parquet")))
    MATH_TEST = sorted(glob.glob(str(ROOT / "datasets/preprocessed_finemath/test_split/*.parquet")))

    DATA_DIR = ROOT / "data_bin"
    process_split(WEB_WIKI_TRAIN, DATA_DIR / "web_wiki_train.bin")
    process_split(WEB_WIKI_TEST, DATA_DIR / "web_wiki_test.bin")
    process_split(MATH_TRAIN, DATA_DIR / "math_train.bin")
    process_split(MATH_TEST, DATA_DIR / "math_test.bin")