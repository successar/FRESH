import json
import csv
import os
import sys
import pandas as pd

""" Usage:
python json_to_amt.py [jsonl test file]

NOTE: all these input files should be TEST files.
"""

if __name__ == "__main__":

    infile = sys.argv[1]

    movies = "movies" in infile
    multirc = "multirc" in infile

    if movies and multirc:
        raise Exception("file naming issue")
    elif movies:
        desired_inxs = "human_amt/test_inxs_movies.txt"
    elif multirc:
        desired_inxs = "human_amt/test_inxs_multirc.txt"
        original_data = [json.loads(line) for line in open('Datasets/multirc/data/test.jsonl')]
        original_data = {x['annotation_id']:x for x in original_data}

    # load ids to include
    with open(desired_inxs, "r") as f:
        ids = [f.strip() for f in f.readlines()]
    assert len(ids) == 100

    outfile = os.path.join(
        os.path.split(infile)[0], os.path.splitext(os.path.basename(infile))[0] + ".csv"
    )

    docs = []
    found = set()
    for line in open(infile, "r"):
        item = json.loads(line)
        el_id = item["annotation_id"]
        if el_id in ids:
            docs.append(item)
            found.add(el_id)

    # ensure all 100 ids found
    assert len(docs) == 100
    assert found == set(ids)

    assert item["label"] == item["label"]

    # determine which format to use:
    if movies:

        with open(outfile, "w") as g:
            writer = csv.DictWriter(
                g, fieldnames=["annotation_id", "text", "pred_label", "label", "method"]
            )
            writer.writeheader()
            for i, item in enumerate(docs):

                try:
                    if item["predicted_label"] == "POS":
                        pred_label = 1
                    elif item["predicted_label"] == "NEG":
                        pred_label = 0
                    else:
                        raise Exception("labeling off")
                except (KeyError):
                    if item["metadata"]["predicted_label"] == "POS":
                        pred_label = 1
                    elif item["metadata"]["predicted_label"] == "NEG":
                        pred_label = 0
                    else:
                        raise Exception("labeling off")

                if item["label"] == "POS":
                    true_label = 1
                elif item["label"] == "NEG":
                    true_label = 0
                else:
                    raise Exception("labeling off")

                # remove query
                # subset tokens
                tokens = item["metadata"]["tokens"]
                t = tokens[0 : tokens.index("[SEP]")]

                token_level_rationale = [0] * len(tokens)
                spans = [x["span"] for x in item["rationale"]["spans"]]
                for s, e in spans:
                    for i in range(s, e):
                        token_level_rationale[i] = 1
                token_level_rationale = token_level_rationale[0 : len(t)]
                assert len(token_level_rationale) == len(t)

                rationale = " ".join(
                    [i for i, z in zip(t, token_level_rationale) if z == 1]
                )
                assert rationale in item["rationale"]["document"]

                # convert to dictionary & write to csv file
                writer.writerow(
                    {
                        "text": rationale,
                        "annotation_id": item["metadata"]["annotation_id"],
                        "pred_label": pred_label,
                        "label": true_label,
                        "method": os.path.splitext(os.path.basename(infile))[0],
                    }
                )

    elif multirc:

        with open(outfile, "w") as g:
            writer = csv.DictWriter(
                g,
                fieldnames=[
                    "annotation_id",
                    "text",
                    "label",
                    "method",
                    "question",
                    "answer",
                ],
            )
            writer.writeheader()
            for i, item in enumerate(docs):

                rationale_key = 'rationale' if 'rationale' in item else 'predicted_rationale'

                annotation_id = item['annotation_id']
                original_doc = original_data[annotation_id]

                # try:
                #     if item["predicted_label"] == "True":
                #         pred_label = 1
                #     elif item["predicted_label"] == "False":
                #         pred_label = 0
                #     else:
                #         raise Exception("labeling off")
                # except (KeyError):
                #     if item["metadata"]["predicted_label"] == "True":
                #         pred_label = 1
                #     elif item["metadata"]["predicted_label"] == "False":
                #         pred_label = 0
                #     else:
                        # raise Exception("labeling off")

                if item["label"] == "True":
                    true_label = 1
                elif item["label"] == "False":
                    true_label = 0
                else:
                    raise Exception("labeling off")

                # remove query
                # subset tokens
                tokens = original_doc['document'].split()

                token_level_rationale = [0]*len(tokens)
                spans = [x['span'] for x in item[rationale_key]['spans']]
                for s, e in spans:
                    for i in range(s, e):
                        token_level_rationale[i] = 1

                assert len(token_level_rationale) == len(tokens)

                rationale = ' '.join([i for i,z in zip(tokens, token_level_rationale) if z == 1])
                assert rationale == item[rationale_key]['document']

                # convert to dictionary & write to csv file
                writer.writerow(
                    {
                        "text": rationale.encode("utf-8").decode('ascii', errors='ignore'),
                        "annotation_id": item["annotation_id"],
                        "label": true_label,
                        "method": os.path.splitext(os.path.basename(infile))[0],
                        "question": original_doc["query"].split("||")[0].strip(),
                        "answer": original_doc["query"].split("||")[1].strip(),
                    }
                )

    else:
        raise Exception("invalid dataset")
