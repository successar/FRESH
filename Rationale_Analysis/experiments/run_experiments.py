import argparse
import os
import json
from itertools import product
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--script-type", type=str, required=True)
parser.add_argument("--exp-name", type=str, required=True)
parser.add_argument("--search-space", type=str, required=True)
parser.add_argument("--dry-run", dest="dry_run", action="store_true")

def main(args):
    '''
    We perform grid search over search space.
    Results are stored in output_dir/exp_name/hyperparameter_string for each hyperparameter setting.
    search space is specified as json string {'P1' : list of values, 'P2' : list of values, ...}
    '''
    global_exp_name = args.exp_name
    search_space = json.loads(args.search_space)
    keys, values = zip(*search_space.items())

    for prod in product(*values):
        new_env = os.environ.copy()
        exp_name = []
        for k, v in zip(keys, prod):
            new_env[k] = str(v)
            exp_name.append(k + "=" + str(v))

        exp_name = os.path.join(global_exp_name, ":".join(exp_name))
        new_env["EXP_NAME"] = exp_name
        cmd = ["bash", "Rationale_Analysis/commands/" + args.script_type]
        print("Running ", cmd, " with exp name ", exp_name)
        if not args.dry_run:
            subprocess.run(cmd, check=False, env=new_env)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
