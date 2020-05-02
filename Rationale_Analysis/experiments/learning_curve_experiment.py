import subprocess
import os

search_space = {"KEEP_PROB": [0.2, 0.4, 0.6, 0.8, 1.0], "RANDOM_SEED": [1000, 2000, 3000, 4000, 5000]}

import json

default_values = json.load(open("Rationale_Analysis/default_values.json"))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--script-type", type=str, required=True)
parser.add_argument("--dry-run", dest="dry_run", action="store_true")
parser.add_argument("--run-one", dest="run_one", action="store_true")
parser.add_argument("--cluster", dest="cluster", action="store_true")
parser.add_argument("--total-data", type=float, required=True)


parser.add_argument("--output-dir", type=str)
parser.add_argument("--min-scale", type=float)
parser.add_argument("--max-scale", type=float)

exp_default = {"MU": 0.0}


def main(args):
    new_env = os.environ.copy()
    dataset = new_env["DATASET_NAME"]
    new_env.update({k: str(v) for k, v in default_values[dataset].items()})
    new_env.update({k: str(v) for k, v in exp_default.items()})

    search_space["KEEP_PROB"] = [x / args.total_data for x in search_space["KEEP_PROB"]]

    cmd = (
        [
            "python",
            "Rationale_Analysis/experiments/model_a_experiments.py",
            "--exp-name",
            "learning_curve",
            "--search-space",
            json.dumps(search_space),
            "--script-type",
            args.script_type,
        ]
        + (["--dry-run"] if args.dry_run else [])
        + (["--run-one"] if args.run_one else [])
        + (["--cluster"] if args.cluster else [])
    )

    print(new_env)
    subprocess.run(cmd, check=True, env=new_env)


from itertools import product
import pandas as pd
import seaborn as sns
import matplotlib

# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

datasets = {"SST": "SST", "agnews": "AGNews", "evinf": "Ev. Inf."}
total_data = [1.0, 2.5, 1.0]

def results(args):
    names = ["Lei et al", "[CLS] Attention + Top K"]
    data = []
    for c, (dataset, dataset_name) in enumerate(datasets.items()) :
        dataset_search_space = deepcopy(search_space)
        dataset_search_space["KEEP_PROB"] = [x / total_data[c] for x in dataset_search_space["KEEP_PROB"]]
        keys, values = list(zip(*dataset_search_space.items()))
        output_dirs = [
            os.path.join(
                args.output_dir,
                "bert_encoder_generator",
                dataset,
                "learning_curve",
                "EXP_NAME_HERE",
                "top_k_rationale",
                "direct",
                "test_metrics.json",
            ),
            os.path.join(
                args.output_dir,
                "bert_classification",
                dataset,
                "learning_curve",
                "EXP_NAME_HERE",
                "wrapper_saliency",
                "top_k_rationale",
                "direct",
                "model_b",
                "metrics.json",
            ),
        ]

        for name, output_dir in zip(names, output_dirs):
            for prod in product(*values):
                exp_dict = {"Model": name, "Dataset": dataset_name}
                exp_name = []
                for k, v in zip(keys, prod):
                    exp_name.append(k + "=" + str(v))
                    exp_dict[k] = v if k != 'KEEP_PROB' else (v*total_data[c])

                try:
                    metrics = json.load(open(output_dir.replace("EXP_NAME_HERE", ":".join(exp_name))))
                    metrics = {
                        k: v
                        for k, v in metrics.items()
                        if k.startswith("test_fscore")
                        or k.startswith("test__fscore")
                        or k.startswith("_fscore")
                        or k.startswith("fscore")
                    }
                    m = np.mean(list(metrics.values()))
                    exp_dict["Macro F1"] = max(0, m)
                except FileNotFoundError as e:
                    print(name, output_dir, exp_name)
                    # exp_dict['Macro F1'] = 0.0

                data.append(exp_dict)

    sns.set_context("talk")
    sns.set(style="white", rc={"lines.linewidth": 1.7}, font_scale=1.5)
    data = pd.DataFrame(data)
    fig = plt.figure(figsize=(4, 3))
    ax = sns.catplot(
        y="Macro F1",
        x="KEEP_PROB",
        hue="Model",
        ci="sd",
        aspect=1,
        data=data,
        estimator=np.median,
        markers=["o", "D"],
        kind="point",
        col="Dataset",
        legend=False,
        palette=["blue", "red"],
        dodge=True,
        join=True,
        sharex=False,
    )

    for c, (_, n) in enumerate(datasets.items()) :
        thresh = total_data[c]
        ax.axes[0, c].set_xticklabels(labels=[x/thresh for x in [0.2, 0.4, 0.6, 0.8, 1.0]])
        if c > 0 :
            ax.axes[0, c].set_xlabel("")
        ax.axes[0, c].set_title(n)

    ax.axes[0, 0].set_xlabel("Training Set Proportion")

    plt.ylim(args.min_scale, args.max_scale)
    plt.tight_layout()
    plt.legend().remove()
    sns.despine()
    plt.savefig("learning-curve.pdf", bbox_inches='tight')


if __name__ == "__main__":
    args = parser.parse_args()
    if args.script_type == "results":
        results(args)
    else:
        main(args)
