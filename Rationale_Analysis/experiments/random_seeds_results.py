import os, json
import numpy as np
import pandas as pd
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--lei", dest="lei", action="store_true")
parser.add_argument("--kuma", action="store_true")

defaults = json.load(open("Rationale_Analysis/default_values.json"))


def main_lei(args):
    datasets = ["SST", "agnews", "multirc", "evinf", "movies"]

    seeds = [1000, 2000, 3000, 4000, 5000]
    rationale = ["top_k", "contiguous"]
    values = []
    for d, r, seed in product(datasets, rationale, seeds):
        path = os.path.join(
            args.output_dir,
            d,
            "bert_encoder_generator" if args.lei else "bert_kuma_encoder_generator",
            str(defaults[d]["MAX_LENGTH_RATIO"]),
            "random_seed_variance",
            "RANDOM_SEED=" + str(seed),
            r + "_thresholder",
            str(defaults[d]["MAX_LENGTH_RATIO"]),
        )

        metrics_file_direct = os.path.join(path, "test_metrics.json")
        if os.path.isfile(metrics_file_direct):
            metrics = json.load(open(metrics_file_direct))["validation_metric"]
            values.append({"dataset": d, "thresholder": r, "seed": seed, "value": metrics})
        else :
            print(metrics_file_direct)

    values = pd.DataFrame(values)
    values_g = values.groupby(["dataset", "thresholder"]).agg(
        lambda x: "{:0.2f}".format(np.median(x))
        + " ("
        + "{:0.2f}".format(np.min(x))
        + "-"
        + "{:0.2f}".format(np.max(x))
        + ")"
    )

    print(values_g)


def main_ours(args):
    datasets = ["SST", "agnews", "multirc", "evinf", "movies"]
    saliency = ["wrapper", "simple_gradient"]
    rationale = ["top_k", "contiguous"]

    seeds = [1000, 2000, 3000, 4000, 5000]
    values = []

    for d, seed in product(datasets, seeds):
        path = os.path.join(args.output_dir, d, "bert_classification", "random_seed_variance", "RANDOM_SEED=" + str(seed))
        metrics_file_direct = os.path.join(path, "metrics.json")
        if os.path.isfile(metrics_file_direct):
            metrics = json.load(open(metrics_file_direct))["test_validation_metric"]
            values.append(
                {
                    "dataset": d,
                    "saliency": "Base",
                    "rationale": "Base",
                    "seed": seed,
                    "value": metrics,
                }
            )
        else:
            print("Not found", metrics_file_direct)

    for d, s, r, seed in product(datasets, saliency, rationale, seeds):
        path = os.path.join(
            args.output_dir,
            d,
            "bert_classification",
            "random_seed_variance",
            "RANDOM_SEED=" + str(seed),
            s + "_saliency",
            r + "_thresholder",
            str(defaults[d]['MAX_LENGTH_RATIO']),
        )

        metrics_file_direct = os.path.join(path, "model_b", "metrics.json")
        if os.path.isfile(metrics_file_direct):
            metrics = json.load(open(metrics_file_direct))["test_validation_metric"]
            values.append(
                {
                    "dataset": d,
                    "saliency": s,
                    "rationale": r,
                    "seed": seed,
                    "value": metrics,
                }
            )

    values = pd.DataFrame(values)

    values_g = values.groupby(["dataset", "saliency", "rationale"]).agg(
        lambda x: "{:0.3f}".format(np.median(x))
        + " ("
        + "{:0.3f}".format(np.min(x))
        + "-"
        + "{:0.3f}".format(np.max(x))
        + ")"
    )

    print(values_g)


from scipy.stats import ttest_rel


def analyse_globality(values):
    m = {"top_k": "top_k", "contiguous": "contiguous", "global_top_k": "top_k", "global_contig": "contiguous"}

    values = values[values.extraction == "direct"]
    values["global"] = values["rationale"].apply(lambda x: "global" in x)
    values["rationale"] = values["rationale"].apply(lambda x: m[x])

    def compute_t_stat(x):
        if "global" in x and len(x[x["global"] == True]) == len(x[x["global"] == False]["value"]):
            stat, pval = ttest_rel(x[x["global"] == True]["value"], x[x["global"] == False]["value"])
            diff = x[x["global"] == True]["value"].mean() - x[x["global"] == False]["value"].mean()
            return pd.Series({"delta": diff, "stat": stat, "pval": pval})
        return pd.Series({"delta": -1, "stat": -1, "pval": -1})

    values = values.groupby(["dataset", "saliency", "rationale"]).apply(compute_t_stat)
    print(values)
    print(values.to_latex(float_format="{:0.4f}".format))


if __name__ == "__main__":
    args = parser.parse_args()
    if args.lei or args.kuma:
        main_lei(args)
    else:
        main_ours(args)
