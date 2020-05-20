import os
import json

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str)
parser.add_argument("--output-dir", type=str)
parser.add_argument("--min-scale", type=float)
parser.add_argument("--max-scale", type=float)


import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

datasets = {"multirc" : "MultiRC" , "evinf": "Ev. Inf."}
saliency = {"multirc" : "simple_gradient", "evinf" : "wrapper"}
max_lengths = {'multirc' : 0.2, 'evinf' : 0.1}
lengths = [0.0, 0.2, 0.5, 1.0]

def get_new_type(args, dataset, seed, hp):
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

    c1 = max_lengths[dataset]
    lei_dir = os.path.join(
        args.output_dir,
        f"{dataset}/bert_encoder_generator_human/{c1}/random_seed_variance/RANDOM_SEED={seed}/human_supervision={hp}/top_k_thresholder/{c1}/test_metrics.json",
    )

    fresh_dir = os.path.join(
        args.output_dir,
        f"{dataset}/bert_classification/random_seed_variance/RANDOM_SEED={seed}/{saliency[dataset]}_saliency/max_length_thresholder/{c1}/human_supervision={hp}/model_b/metrics.json",
    )

    return get_validation_metric(lei_dir), get_validation_metric(fresh_dir)

def results(args):
    data = []
    for seed in [1000, 2000, 3000, 4000, 5000]:
        for hp in lengths :
            lei, fresh = get_new_type(args, args.dataset, seed, hp)
            if lei is not None:
                data.append(
                    {
                        "Dataset": datasets[args.dataset],
                        "Model": "lei",
                        "hp": hp,
                        "Macro F1": lei,
                    }
                )

            if fresh is not None:
                data.append(
                    {
                        "Dataset": datasets[args.dataset],
                        "Model": "fresh",
                        "hp": hp,
                        "Macro F1": fresh,
                    }
                )

    sns.set_context("talk")
    sns.set(style="white", rc={"lines.linewidth": 1.7}, font_scale=1.5)
    data = pd.DataFrame(data)

    breakpoint()

    f, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    sns.pointplot(
        x="hp",
        y="Macro F1",
        hue="Model",
        data=data[data.Dataset == datasets[args.dataset]],
        ax=ax,
        dodge=True,
        join=True,
        hue_order=["lei", "fresh"],
        palette=["blue", "red"],
        markers=["o", "D"],
        ci="sd",
        estimator=np.median,
    )
    ax.set_xlabel("")
    ax.set_title(datasets[args.dataset])
    ax.legend_ = None

    plt.ylim(args.min_scale, args.max_scale)
    plt.legend().remove()
    sns.despine()
    f.tight_layout()
    f.savefig(f"human_prob_{args.dataset}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    args = parser.parse_args()
    results(args)