import subprocess
import os

search_space = {"MAX_LENGTH_RATIO": [0.05, 0.1, 0.25, 0.5]}

import json

default_values = json.load(open("Rationale_Analysis/default_values.json"))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--script-type", type=str, required=True, choices=['model_a', 'saliency'])
parser.add_argument("--dry-run", dest="dry_run", action="store_true")
parser.add_argument("--run-one", dest="run_one", action="store_true")
parser.add_argument("--cluster", dest="cluster", action="store_true")


def main(args):
    new_env = os.environ.copy()
    dataset = new_env["DATASET_NAME"]
    new_env.update({k:str(v) for k, v in default_values[dataset].items()})
    new_env['KEEP_PROB'] = 1.0

    cmd = (
        [
            "python",
            "Rationale_Analysis/experiments/model_a_experiments.py",
            "--exp-name",
            "learning_curve",
            "--search-space",
            json.dumps(search_space),
            "--script-type", 
            args.script_type
        ]
        + (["--dry-run"] if args.dry_run else [])
        + (["--run-one"] if args.run_one else [])
        + (["--cluster"] if args.cluster else [])
    )

    print(new_env)
    subprocess.run(cmd, check=True, env=new_env)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
