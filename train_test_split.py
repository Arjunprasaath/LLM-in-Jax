import pyarrow as pa
from pathlib import Path
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.writers import ParquetWriter

TASK = 14
NUM_WORKERS = 14

OUTPUT_SCHEMA = pa.schema([
    ("text", pa.string()),
    ("id", pa.string()),
    ("metadata", pa.struct([
        ("dump", pa.string()),
        ("url", pa.string()),
        ("file_path", pa.string()),
        ("language", pa.string()),
        ("language_score", pa.float64()),
        ("token_count", pa.int64()),
        ("score", pa.float64()),
        ("int_score", pa.int64()),
        ("duplicate_count", pa.int64()),
        ("source", pa.string()),
    ])),
])

STAGE4_SCHEMA = pa.schema([
    OUTPUT_SCHEMA.field("text"),
    OUTPUT_SCHEMA.field("id"),
    pa.field("metadata", pa.struct(
        list(OUTPUT_SCHEMA.field("metadata").type) + [
            pa.field("fqdn", pa.string()),
            pa.field("suffix", pa.string()),
            pa.field("ccnet_perplexity_wikipedia_en", pa.float64()),
            pa.field("length", pa.int64()),
            pa.field("white_space_ratio", pa.float64()),
            pa.field("non_alpha_digit_ratio", pa.float64()),
            pa.field("digit_ratio", pa.float64()),
            pa.field("uppercase_ratio", pa.float64()),
            pa.field("elipsis_ratio", pa.float64()),
            pa.field("punctuation_ratio", pa.float64()),
        ]
    )),
])

def main():
    INPUT_DIR = "datasets/preprocessed_combined/final_clean_with_stats/"
    VAL_DIR = "datasets/preprocessed_combined/test_split/"
    TRAIN_DIR = "datasets/preprocessed_combined/train_split/"
    executor = LocalPipelineExecutor(
                pipeline = [
                    ParquetReader(data_folder=str(Path(INPUT_DIR).resolve())),
                    SamplerFilter(rate = 0.005, seed = 20, exclusion_writer = ParquetWriter(output_folder = TRAIN_DIR, schema = STAGE4_SCHEMA)),
                    ParquetWriter(output_folder=VAL_DIR, schema=STAGE4_SCHEMA)
                    ],
                tasks = TASK,
                workers = NUM_WORKERS,
                logging_dir = "logs/fine_web_wiki/train_test_split/",
                ).run()

if __name__ == "__main__":
    main()