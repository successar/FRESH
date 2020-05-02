import argparse
import os
import json
from itertools import product
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--rationale', type=str, required=True)
parser.add_argument('--search-space', type=str, required=True)
parser.add_argument('--dry-run', dest='dry_run', action='store_true')

def main(args) :
    rationale_extractor = args.rationale
    search_space = json.loads(args.search_space)

    keys = list(search_space.keys())
    values = list(search_space.values())

    os.environ['RATIONALE'] = rationale_extractor
    for prod in product(*values) :
        exp_name = []
        for k, v in zip(keys, prod) :
            os.environ[k] = str(v)
            exp_name.append(k + '=' + str(v))

        exp_name = ":".join(exp_name)
        os.environ['RATIONALE_EXP_NAME'] = exp_name
        cmd = ['bash', 'Rationale_Analysis/commands/rationale_and_model_b_script.sh']
        print("Running ", cmd, ' with exp name ', exp_name)
        if not args.dry_run :
            subprocess.run(cmd, check=True)


if __name__ == '__main__' :
    args = parser.parse_args()
    main(args)