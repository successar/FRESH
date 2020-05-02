import argparse
import os
import json
from itertools import product
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--exp-folder", type=str, required=True)
parser.add_argument("--exp-name", type=str, required=True)
parser.add_argument("--search-space", type=str, required=False)

parser.add_argument("--value", type=str, required=True)
parser.add_argument("--metric", type=str, required=True)
parser.add_argument("--deviation", type=str, required=True)


def main(args):
    global_exp_name = args.exp_name
    global_exp_folder = args.exp_folder

    x_axis_field = args.value
    y_axis_field = args.metric
    deviation_field = args.deviation

    metrics = []

    exp_dicts = []
    exp_names = []

    if args.search_space is not None:
        search_space = json.loads(args.search_space)
        keys, values = zip(*search_space.items())

        for prod in product(*values):
            exp_dict = dict(zip(keys, prod))
            exp_name = []
            for k, v in zip(keys, prod):
                exp_name.append(k + "=" + str(v))

            exp_dicts.append(exp_dict)
            exp_names.append(":".join(exp_name))
    else :
        #globy glob
        dirs = [f.name for f in os.scandir(os.path.join(global_exp_folder, global_exp_name)) if f.is_dir()]
        for d in dirs :
            exp_dict = [x.split('=') for x in d.strip().split(':')]
            exp_dicts.append(dict([(x, float(y)) for x, y in exp_dict]))
            exp_names.append(d)

    logging.info("Experiments :")

    for exp_name, exp_dict in zip(exp_names, exp_dicts):
        metrics_file = json.load(open(os.path.join(global_exp_folder, global_exp_name, exp_name, "metrics.json")))
        metric = metrics_file[args.metric]

        metrics.append(
            {x_axis_field: exp_dict[x_axis_field], y_axis_field: metric, deviation_field: exp_dict[deviation_field]}
        )

        logging.info(exp_name)

    metrics = pd.DataFrame(metrics)
    sns.pointplot(x=x_axis_field, y=y_axis_field, data=metrics, ci='sd')
    sns.swarmplot(x=x_axis_field, y=y_axis_field, data=metrics)
    plt.tight_layout()
    plt.savefig(
        os.path.join(global_exp_folder, global_exp_name, x_axis_field + "_vs_" + y_axis_field + ".pdf"),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
