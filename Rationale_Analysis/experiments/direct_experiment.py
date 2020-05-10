import subprocess
import os

import json

default_values = json.load(open("Rationale_Analysis/default_values.json"))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--script-type", type=str, required=True)
parser.add_argument("--dry-run", dest="dry_run", action="store_true")
parser.add_argument("--all-data", dest="all_data", action="store_true")

def main(args):
    if args.all_data :
        datasets = default_values.keys()
    else :
        datasets = [os.environ["DATASET_NAME"]]

    for dataset in datasets :
        new_env = os.environ.copy()
        new_env.update({k:str(v) for k, v in default_values[dataset].items()})
        new_env['KEEP_PROB'] = str(1.0)
        new_env['DATASET_NAME'] = dataset

        ith_search_space = {}
        ith_search_space['RANDOM_SEED'] = [1000, 2000, 3000, 4000, 5000]

        cmd = (
            [
                "python",
                "Rationale_Analysis/experiments/run_experiments.py",
                "--exp-name",
                "direct",
                "--search-space",
                json.dumps(ith_search_space),
                "--script-type", 
                args.script_type
            ]
            + (["--dry-run"] if args.dry_run else [])
        )

        print(default_values[dataset])
        print(ith_search_space)
        subprocess.run(cmd, check=True, env=new_env)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)