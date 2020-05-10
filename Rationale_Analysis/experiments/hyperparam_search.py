import argparse
import os
import json
import subprocess
import hyperopt
from hyperopt import hp

import numpy as np
np.exp = lambda x : 10**x

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, required=True)
parser.add_argument("--search-space-file", type=str, required=True)
parser.add_argument("--dry-run", dest="dry_run", action="store_true")
parser.add_argument('--num-searches', type=int, required=True)


def main(args):
    global_exp_name = args.exp_name
    search_space_config = json.load(open(args.search_space_file))
    hyperparam_space = {k:eval(v['type'])(k, **v['options']) for k, v in search_space_config.items()}

    for i in range(args.num_searches) :
        new_env = os.environ.copy()
        hyperparam_vals = hyperopt.pyll.stochastic.sample(hyperparam_space)
        for k, v in hyperparam_vals.items():
            new_env[k] = str(v)

        print(hyperparam_vals)
        exp_name = os.path.join(global_exp_name, "search_" + str(i))
        new_env["EXP_NAME"] = exp_name
        cmd = ["bash", "Rationale_Analysis/commands/model_a_train_script.sh"]
        print("Running ", cmd, " with exp name ", exp_name)
        if not args.dry_run:
            subprocess.run(cmd, check=True, env=new_env)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
