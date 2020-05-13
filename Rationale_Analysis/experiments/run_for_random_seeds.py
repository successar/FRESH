import argparse
import json
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--script-type", type=str, required=True)
parser.add_argument('--defaults-file', type=str, default="Rationale_Analysis/default_values.json")
parser.add_argument("--dry-run", dest="dry_run", action="store_true")
parser.add_argument("--all-data", dest="all_data", action="store_true")


def main(args):
    default_values = json.load(open(args.defaults_file))
    if args.all_data:
        datasets = default_values.keys()
    else:
        datasets = [os.environ["DATASET_NAME"]]

    for dataset in datasets:
        new_env = os.environ.copy()
        new_env.update({k: str(v) for k, v in default_values[dataset].items()})
        new_env["DATASET_NAME"] = dataset

        ith_search_space = {}
        ith_search_space["RANDOM_SEED"] = [1000, 2000, 3000, 4000, 5000]

        cmd = [
            "python",
            "Rationale_Analysis/experiments/run_experiments.py",
            "--exp-name",
            "random_seed_variance",
            "--search-space",
            json.dumps(ith_search_space),
            "--script-type",
            args.script_type,
        ] + (["--dry-run"] if args.dry_run else [])

        print(f'Default values for {dataset} is {default_values[dataset]}')
        print(f'Search over {ith_search_space}')
        subprocess.run(cmd, check=True, env=new_env)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
