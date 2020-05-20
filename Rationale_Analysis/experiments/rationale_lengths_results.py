import argparse
import json
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str)
parser.add_argument("--min-scale", type=float)
parser.add_argument("--max-scale", type=float)


datasets = {"SST": "SST", "agnews": "AGNews", "evinf": "Ev. Inf.", "movies": "Movies", "multirc": "MultiRC"}
cut_point_thresh = {
    "SST": [0.1, 0.2],
    "agnews": [0.1, 0.2],
    "evinf": [0.05, 0.1],
    "movies": [0.15, 0.3],
    "multirc": [0.1, 0.2],
}

lei_best = {
    'SST' : 'top_k',
    'agnews' : 'top_k',
    'evinf' : 'contiguous',
    'movies' : 'top_k',
    'multirc' : 'top_k'
}

fresh_best = {
    'SST' : ['wrapper', 'top_k'],
    'agnews' : ['wrapper', 'top_k'],
    'evinf' : ['wrapper', 'contiguous'],
    'movies' : ['wrapper', 'top_k'],
    'multirc' : ['simple_gradient', 'contiguous']
}


def get_metrics(args, dataset, seed):
    c1, c2 = cut_point_thresh[dataset]
    lei_dir_c1 = os.path.join(
        args.output_dir,
        f"{dataset}/bert_encoder_generator/{c1}/random_seed_variance/RANDOM_SEED={seed}/{lei_best[dataset]}_thresholder/{c1}/test_metrics.json",
    )
    lei_dir_c2 = os.path.join(
        args.output_dir,
        f"{dataset}/bert_encoder_generator/{c2}/random_seed_variance/RANDOM_SEED={seed}/{lei_best[dataset]}_thresholder/{c2}/test_metrics.json",
    )

    s, t = fresh_best[dataset]

    fresh_dir_c1 = os.path.join(
        args.output_dir,
        f"{dataset}/bert_classification/random_seed_variance/RANDOM_SEED={seed}/{s}_saliency/{t}_thresholder/{c1}/model_b/metrics.json",
    )
    fresh_dir_c2 = os.path.join(
        args.output_dir,
        f"{dataset}/bert_classification/random_seed_variance/RANDOM_SEED={seed}/{s}_saliency/{t}_thresholder/{c2}/model_b/metrics.json",
    )

    def get_validation_metric(file):
        try:
            metrics = json.load(open(file))
            if "test_validation_metric" in metrics:
                m = metrics["test_validation_metric"]
            else:
                m = metrics["validation_metric"]
            return max(0, m)
        except FileNotFoundError:
            print(file)
            return None

    return [get_validation_metric(file) for file in [lei_dir_c1, lei_dir_c2, fresh_dir_c1, fresh_dir_c2]]


def results(args):
    data = []
    for dataset, dataset_name in datasets.items():
        for seed in [1000, 2000, 3000, 4000, 5000]:
            metrics = get_metrics(args, dataset, seed)
            lei_1, lei_2, fresh_1, fresh_2 = metrics
            if lei_1 is not None:
                data.append(
                    {
                        "Dataset": dataset_name,
                        "Model": "Lei et al",
                        "seed": seed,
                        "cut_point": str(cut_point_thresh[dataset][0]),
                        "Macro F1": lei_1,
                    }
                )

            if lei_2 is not None:
                data.append(
                    {
                        "Dataset": dataset_name,
                        "Model": "Lei et al",
                        "seed": seed,
                        "cut_point": str(cut_point_thresh[dataset][1]),
                        "Macro F1": lei_2,
                    }
                )

            if fresh_1 is not None:
                data.append(
                    {
                        "Dataset": dataset_name,
                        "Model": "FRESH",
                        "seed": seed,
                        "cut_point": str(cut_point_thresh[dataset][0]),
                        "Macro F1": fresh_1,
                    }
                )

            if fresh_2 is not None:
                data.append(
                    {
                        "Dataset": dataset_name,
                        "Model": "FRESH",
                        "seed": seed,
                        "cut_point": str(cut_point_thresh[dataset][1]),
                        "Macro F1": fresh_2,
                    }
                )

    sns.set_context("talk")
    sns.set(style="white", rc={"lines.linewidth": 1.7}, font_scale=1.5)
    data = pd.DataFrame(data)

    f, axes = plt.subplots(1, 5, figsize=(12, 5), sharey=True)
    for i, (x, ax) in enumerate(zip(datasets.keys(), axes)):
        sns.pointplot(
            x="cut_point",
            y="Macro F1",
            hue="Model",
            data=data[data.Dataset == datasets[x]],
            ax=ax,
            dodge=True,
            join=True,
            palette=["blue", "red"],
            markers=["o", "D"],
            ci="sd",
            estimator=np.median,
        )
        ax.set_xlabel("")
        ax.set_title(datasets[x])
        ax.legend_ = None
        if i != 0:
            ax.set_ylabel("")

    plt.ylim(args.min_scale, args.max_scale)
    plt.legend().remove()
    sns.despine()
    f.tight_layout()
    f.savefig("cut-point.pdf", bbox_inches="tight")


if __name__ == "__main__":
    args = parser.parse_args()
    results(args)
