import os, json
import numpy as np
import pandas as pd
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir")
parser.add_argument("--lei", dest="lei", action="store_true")
parser.add_argument("--kuma", action="store_true")


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
            "direct",
            "RANDOM_SEED=" + str(seed),
            r + "_rationale",
            "direct",
        )

        metrics_file_direct = os.path.join(path, "test_metrics.json")
        if os.path.isfile(metrics_file_direct):
            metrics = json.load(open(metrics_file_direct))['validation_metric']
            values.append(
                {
                    "dataset": d,
                    "rationale": r,
                    "saliency": "-",
                    "extraction": "-",
                    "seed": seed,
                    "value": metrics
                }
            )

    values = pd.DataFrame(values)
    idx = values.groupby(["dataset", "saliency", "rationale", "extraction"])["value"].transform(max) == values["value"]
    print(values[idx])

    values_g = values.groupby(["dataset", "saliency", "rationale", "extraction"]).agg(
        lambda x: "{:0.2f}".format(np.median(x))
        + " ("
        + "{:0.2f}".format(np.min(x))
        + "-"
        + "{:0.2f}".format(np.max(x))
        + ")"
    )

    print(values_g)
    print(values_g["value"].unstack(level=0).to_latex())

    breakpoint()


def main_ours(args):
    datasets = ["SST", "agnews", "multirc", "evinf", "movies"]
    saliency = ["wrapper", "simple_gradient"]
    rationale = ["top_k", "contiguous"]

    seeds = [1000, 2000, 3000, 4000, 5000]
    values = []

    for d, seed in product(datasets, seeds):
        path = os.path.join(args.output_dir, d, "bert_classification",  "direct", "RANDOM_SEED=" + str(seed))
        metrics_file_direct = os.path.join(path, "metrics.json")
        if os.path.isfile(metrics_file_direct):
            metrics = json.load(open(metrics_file_direct))['test_validation_metric']
            values.append(
                {
                    "dataset": d,
                    "saliency": "Base",
                    "rationale": "Base",
                    "extraction": "Base",
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
            "direct",
            "RANDOM_SEED=" + str(seed),
            s + "_saliency",
            r + "_rationale",
            "direct",
        )

        metrics_file_direct = os.path.join(path, "model_b", "metrics.json")
        if os.path.isfile(metrics_file_direct):
            metrics = json.load(open(metrics_file_direct))['test_validation_metric']
            values.append(
                {
                    "dataset": d,
                    "saliency": s,
                    "rationale": r,
                    "extraction": "direct",
                    "seed": seed,
                    "value": metrics,
                }
            )

        # if r.startswith("global"):
        #     continue

        # metrics_file_direct = os.path.join(path, "bert_generator_saliency", "direct", "model_b", "metrics.json")
        # if os.path.isfile(metrics_file_direct):
        #     metrics = json.load(open(metrics_file_direct))
        #     metrics = {k: v for k, v in metrics.items() if k.startswith("test_fscore") or k.startswith("test__fscore")}
        #     values.append(
        #         {
        #             "dataset": d,
        #             "saliency": s,
        #             "rationale": r,
        #             "extraction": "crf",
        #             "seed": seed,
        #             "value": np.mean(list(metrics.values())),
        #         }
        #     )

    values = pd.DataFrame(values)
    idx = values.groupby(["dataset", "saliency", "rationale", "extraction"])["value"].transform(max) == values["value"]
    print(values[idx])

    values_g = (
        values
        .groupby(["dataset", "saliency", "rationale", "extraction"])
        .agg(
            lambda x: "{:0.3f}".format(np.median(x))
            + " ("
            + "{:0.3f}".format(np.min(x))
            + "-"
            + "{:0.3f}".format(np.max(x))
            + ")"
        )
    )

    print(values_g)
    breakpoint()


    # values_g = (
    #     values[values.rationale.apply(lambda x: "global" not in x) & values.extraction.isin(['direct'])]
    #     .groupby(["dataset", "saliency", "rationale", "extraction"])
    #     .agg(
    #         lambda x: "{:0.2f}".format(np.median(x))
    #         + " ("
    #         + "{:0.2f}".format(np.min(x))
    #         + "-"
    #         + "{:0.2f}".format(np.max(x))
    #         + ")"
    #     )
    # )

    # print(values_g)
    # print(values_g["value"].unstack(level=0).to_latex())

    # analyse_globality(values)
    # analyse_crf(values)
    return values


from scipy.stats import ttest_ind, ttest_rel


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

