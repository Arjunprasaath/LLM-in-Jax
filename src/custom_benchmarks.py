import re
import numpy as np
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import LoglikelihoodAcc
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod

acc_norm = SampleLevelMetric(
    metric_name="acc_norm",
    sample_level_fn=LoglikelihoodAcc(logprob_normalization=LogProbCharNorm()),
    category=SamplingMethod.LOGPROBS, corpus_level_fn=np.mean, higher_is_better=True)

def _clean(t):  # standard HellaSwag preprocessing
    t = t.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", t).replace("  ", " ")

def hellaswag_cloze(line, task_name=None):
    ctx = line["ctx_a"] + " " + line["ctx_b"].capitalize()
    return Doc(task_name=task_name, query=_clean(line["activity_label"] + ": " + ctx),
               choices=[" " + _clean(e) for e in line["endings"]], gold_index=int(line["label"]))

def piqa_cloze(line, task_name=None):
    return Doc(task_name=task_name, query=f"Question: {line['goal']}\nAnswer:",
               choices=[" " + line["sol1"], " " + line["sol2"]], gold_index=int(line["label"]))

def obqa_cloze(line, task_name=None):
    return Doc(task_name=task_name, query=line["question_stem"],
               choices=[" " + c for c in line["choices"]["text"]],
               gold_index=line["choices"]["label"].index(line["answerKey"].strip()))

def lambada_doc(line, task_name=None):
    ctx, last = line["text"].rsplit(" ", 1)
    return Doc(task_name=task_name, query=ctx, choices=[" " + last], gold_index=0)

def cfg(name, fn, repo, subset, splits, eval_split, metrics):
    return LightevalTaskConfig(name=name, prompt_function=fn, hf_repo=repo, hf_subset=subset,
        hf_avail_splits=splits, evaluation_splits=[eval_split], few_shots_split=None,
        few_shots_select=None, generation_size=-1, metrics=metrics, stop_sequence=["\n"], version=0)

TASKS_TABLE = [
    cfg("hellaswag_cloze", hellaswag_cloze, "Rowan/hellaswag", "default", ["train", "validation"], "validation", [Metrics.loglikelihood_acc, acc_norm]),
    cfg("piqa_cloze", piqa_cloze, "baber/piqa", "default", ["train", "validation"], "validation", [Metrics.loglikelihood_acc, acc_norm]),
    cfg("openbookqa_cloze", obqa_cloze, "allenai/openbookqa", "main", ["train", "validation", "test"], "test", [Metrics.loglikelihood_acc, acc_norm]),
    cfg("lambada_openai_acc", lambada_doc, "cimec/lambada", "plain_text", ["train", "validation", "test"], "test",
        [Metrics.acc_golds_likelihood, Metrics.target_perplexity]),
]
