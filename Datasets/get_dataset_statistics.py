import pandas as pd
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
pd.set_option('display.max_colwidth', -1)



def main(args):
    keys = ["train", "dev", "test"]
    dfs = []
    for k in keys :
        df = pd.read_json(os.path.join(args.dataset, "data", k + ".jsonl"), lines=True)
        df['split'] = k
        dfs.append(df)

    dfs = pd.concat(dfs).reset_index(drop=True)
    if 'annotation_id' in dfs.columns :
        dfs = dfs.drop(columns=['annotation_id'])
    dfs['document_length'] = dfs['document'].apply(lambda x : len(x.split()))
    dfs = dfs.drop(columns=['document'])
    if 'query' in dfs.columns :
        dfs['query_length'] = dfs['query'].apply(lambda x : len(x.split()))
        dfs = dfs.drop(columns=['query'])
    else :
        dfs['query_length'] = 0.0

    if 'rationale' in dfs.columns :
        dfs['rationale_length'] = dfs["rationale"].apply(lambda x: sum([y[1] - y[0] for y in x])) / dfs['document_length']
        dfs = dfs.drop(columns=['rationale'])
    else :
        dfs['rationale_length'] = 0.0

    
    def aggregate(rows) :
        new_data = {}
        for col in rows.columns :
            if col.endswith("length") :
                desc = np.median(rows[col].values)
                print(col, np.max(rows[col].values))
                print(col, np.percentile(rows[col].values, 90))
                new_data[col] = desc

            elif col == 'label':
                label = rows[col].value_counts(normalize=True)[sorted(rows[col].unique())]
                new_data[col] = " / ".join([str(x) for x in label.round(2).values])
                new_data[col + '_'] = " / ".join([str(x) for x in label.round(2).index])

        new_data['N'] = len(rows)

        return pd.Series(new_data)

    agg = aggregate(dfs)
    print(agg)
    agg.name = args.dataset
    agg = pd.DataFrame(agg).T
    agg = agg.drop(columns=['label_'])
    print(agg.to_latex())


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

