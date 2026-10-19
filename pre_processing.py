import os
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter

# Filters
from datatrove.pipeline.filters import (
    GopherQualityFilter,
    GopherRepetitionFilter,
)

# Deduplication
from datatrove.pipeline.dedup.exact_dedup import (
    ExactDedupConfig,
    ExactDedupSignature,
    ExactFindDedups,
    ExactDedupFilter,
)
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupSignature,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.utils.hashing import HashConfig
from datatrove.pipeline.dedup.sentence_dedup import (
    SentDedupConfig,
    SentenceDedupSignature,
    SentenceFindDedups,
    SentenceDedupFilter,
)

# Stats & Tokens
from datatrove.pipeline.tokens.counter import TokensCounter
from datatrove.pipeline.stats.doc_stats import DocStats
from datatrove.pipeline.stats.perplexity_stats import CCNetPerplexityStats


# =====================================================================
# Helper: Pretty Print Stage & Block Drop Stats
# =====================================================================
def print_stage_summary(stage_name: str, pipeline_stats):
    """Parses DataTrove PipelineStats and prints kept/dropped metrics & reasons."""
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY FOR: {stage_name}")
    print(f"{'='*70}")

    if pipeline_stats is None:
        print(f"ℹ️ No stats available for {stage_name} (skipped or previously completed).")
        return None

    last_forwarded = None

    for block_stat in pipeline_stats.stats:
        metrics = block_stat.stats
        if not metrics:
            continue

        # Look for standard DataTrove counters
        total = int(metrics["total"].total) if "total" in metrics else None
        forwarded = int(metrics["forwarded"].total) if "forwarded" in metrics else None
        dropped = int(metrics["dropped"].total) if "dropped" in metrics else 0

        if forwarded is not None:
            last_forwarded = forwarded

        if total is not None:
            drop_pct = (dropped / total * 100) if total > 0 else 0
            keep_pct = (forwarded / total * 100) if (forwarded is not None and total > 0) else 0
            forwarded_str = f"{forwarded:>10,d} ({keep_pct:6.2f}%)" if forwarded is not None else "       N/A"
            print(f"\n🔹 Block: {block_stat.name}")
            print(f"   ├─ Input Documents:     {total:>10,d}")
            print(f"   ├─ Kept (Forwarded):   {forwarded_str}")
            print(f"   └─ Dropped / Filtered: {dropped:>10,d} ({drop_pct:6.2f}%)")

            # Print granular drop reasons if available (e.g. sub-rules)
            reasons = {
                k: int(v.total)
                for k, v in metrics.items()
                if k.startswith("dropped_") and v.total > 0
            }
            if reasons:
                print("      Drop Reason Breakdown:")
                for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                    clean_reason = reason.replace("dropped_", "")
                    pct = (count / total * 100) if total > 0 else 0
                    print(f"        • {clean_reason:<30}: {count:>8,d} ({pct:5.2f}%)")

    # Also display DataTrove's native formatted summary
    print(pipeline_stats.get_repr(text=stage_name))
    return last_forwarded


# ==========================================
# Configurations & Paths
# ==========================================
INPUT_DATA = "datasets/datasets/data/enwiki"
BASE_WORKING_DIR = "datasets/preprocessed_finewiki/pipeline_workspace"
BASE_LOG_DIR = "logs/finewiki"

TASKS = 14
WORKERS = 14


def main():
    os.makedirs(BASE_WORKING_DIR, exist_ok=True)
    overall_counts = {}

    # =====================================================================
    # STAGE 1: Gopher Quality & Repetition Filtering
    # =====================================================================
    print("\n>>> STARTING STAGE 1: Gopher Quality & Repetition Filters <<<")
    STAGE1_OUT = f"{BASE_WORKING_DIR}/stage1_filtered"

    stage1_stats = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=INPUT_DATA),
            GopherQualityFilter(),
            GopherRepetitionFilter(),
            ParquetWriter(output_folder=STAGE1_OUT),
        ],
        tasks=TASKS,
        workers=WORKERS,
        logging_dir=f"{BASE_LOG_DIR}/stage1_gopher/",
    ).run()

    overall_counts["Stage 1: After Gopher Filters"] = print_stage_summary("Stage 1 (Gopher)", stage1_stats) # 8_716_438

    # =====================================================================
    # STAGE 2: Exact Deduplication
    # =====================================================================
    print("\n>>> STARTING STAGE 2: Exact Deduplication <<<")
    EXACT_DIR = f"{BASE_WORKING_DIR}/exact_dedup"
    STAGE2_OUT = f"{BASE_WORKING_DIR}/stage2_exact_deduped"

    # Signatures
    exact_dedup_config = ExactDedupConfig(
        content_getter=lambda doc: doc.text.encode("utf-8"),
    )

    LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=STAGE1_OUT),
            ExactDedupSignature(
                output_folder=f"{EXACT_DIR}/signatures",
                config=exact_dedup_config,
                finder_workers=TASKS,
            ),
        ],
        tasks=TASKS,
        workers=WORKERS,
        logging_dir=f"{BASE_LOG_DIR}/exact_1_sig/",
    ).run()

    # Find Duplicates
    LocalPipelineExecutor(
        pipeline=[
            ExactFindDedups(
                data_folder=f"{EXACT_DIR}/signatures",
                output_folder=f"{EXACT_DIR}/duplicates",
                config=exact_dedup_config,
            ),
        ],
        tasks=TASKS,
        workers=WORKERS,
        logging_dir=f"{BASE_LOG_DIR}/exact_2_find/",
    ).run()

    # Filter Duplicates
    stage2_stats = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=STAGE1_OUT),
            ExactDedupFilter(
                data_folder=f"{EXACT_DIR}/duplicates",
                config=exact_dedup_config,
            ),
            ParquetWriter(output_folder=STAGE2_OUT),
        ],
        tasks=TASKS,
        workers=WORKERS,
        logging_dir=f"{BASE_LOG_DIR}/exact_3_filter/",
    ).run()

    overall_counts["Stage 2: After Exact Dedup"] = print_stage_summary("Stage 2 (Exact Dedup)", stage2_stats)

    # =====================================================================
    # STAGE 3: Fuzzy Deduplication (MinHash LSH)
    # =====================================================================
    print("\n>>> STARTING STAGE 3: Fuzzy / MinHash Deduplication <<<")
    MINHASH_DIR = f"{BASE_WORKING_DIR}/minhash"
    STAGE3_OUT = f"{BASE_WORKING_DIR}/stage3_fuzzy_deduped"

    minhash_config = MinhashConfig(
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5,
        hash_config=HashConfig(hash_fc="sha1"),
    )

    # 3a: Signatures
    LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=STAGE2_OUT),
            MinhashDedupSignature(output_folder=f"{MINHASH_DIR}/signatures", config=minhash_config),
        ],
        tasks=TASKS,
        workers=WORKERS,
        logging_dir=f"{BASE_LOG_DIR}/minhash_1_sig/",
    ).run()

    # 3b: Buckets
    LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{MINHASH_DIR}/signatures",
                output_folder=f"{MINHASH_DIR}/buckets",
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        workers=WORKERS,
        logging_dir=f"{BASE_LOG_DIR}/minhash_2_buckets/",
    ).run()

    # 3c: Cluster
    LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{MINHASH_DIR}/buckets",
                output_folder=f"{MINHASH_DIR}/clusters",
                config=minhash_config,
            ),
        ],
        tasks=1,
        workers=1,
        logging_dir=f"{BASE_LOG_DIR}/minhash_3_cluster/",
    ).run()

    # 3d: Filter
    stage3_stats = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=STAGE2_OUT),
            MinhashDedupFilter(input_folder=f"{MINHASH_DIR}/clusters"),
            ParquetWriter(output_folder=STAGE3_OUT),
        ],
        tasks=TASKS,
        workers=WORKERS,
        logging_dir=f"{BASE_LOG_DIR}/minhash_4_filter/",
    ).run()

    overall_counts["Stage 3: After MinHash Dedup"] = print_stage_summary("Stage 3 (MinHash Dedup)", stage3_stats)

    # =====================================================================
    # STAGE 4: Stats, CCNet Perplexity & Token Counts
    # =====================================================================
    print("\n>>> STARTING STAGE 4: Perplexity & Token Counting <<<")
    FINAL_OUT = "datasets/final_clean_with_stats/"
    STATS_METRICS_DIR = "datasets/stats_metrics/"

    stage4_stats = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=STAGE3_OUT),
            TokensCounter(tokenizer_name_or_path="gpt2", count_eos_token=False),
            CCNetPerplexityStats(
                output_folder=f"{STATS_METRICS_DIR}/perplexity",
                model_dataset="wikipedia",
                language="en",
            ),
            DocStats(output_folder=f"{STATS_METRICS_DIR}/doc_stats"),
            ParquetWriter(output_folder=FINAL_OUT),
        ],
        tasks=TASKS,
        workers=WORKERS,
        logging_dir=f"{BASE_LOG_DIR}/stage4_stats/",
    ).run()

    print_stage_summary("Stage 4 (Stats & Perplexity)", stage4_stats)

    # =====================================================================
    # FINAL ATTRITION / RETENTION REPORT
    # =====================================================================
    print("\n" + "="*70)
    print("🏁 OVERALL DATASET RETENTION ACROSS ALL STAGES")
    print("="*70)

    base_count = None
    prev_count = None

    for stage_name, count in overall_counts.items():
        if count is None:
            continue
        if base_count is None:
            base_count = count
            prev_count = count
            print(f"{stage_name:<35}: {count:>10,d} docs (100.0% of filtered base)")
            continue

        retained_from_base = (count / base_count * 100) if base_count > 0 else 0
        dropped_in_stage = prev_count - count
        dropped_stage_pct = (dropped_in_stage / prev_count * 100) if prev_count > 0 else 0

        print(
            f"{stage_name:<35}: {count:>10,d} docs "
            f"({retained_from_base:5.1f}% retained, -{dropped_in_stage:,d} dropped / -{dropped_stage_pct:4.1f}%)"
        )
        prev_count = count
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
