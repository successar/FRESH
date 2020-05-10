bash Rationale_Analysis/commands/model_a_train_script.sh;

export PYTHONPATH=.

for saliency in wrapper simple_gradient;
    do 
    SALIENCY=$saliency bash Rationale_Analysis/commands/saliency_script.sh;
    for rationale in top_k max_length;
        do
        SALIENCY=$saliency \
        RATIONALE=$rationale \
        RATIONALE_EXP_NAME=direct \
        sbatch Cluster_scripts/gpu_sbatch.sh bash Rationale_Analysis/commands/rationale_and_model_b_script.sh;
        done;
    done;