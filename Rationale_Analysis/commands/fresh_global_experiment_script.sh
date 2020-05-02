for saliency in wrapper simple_gradient;
    do 
    SALIENCY=$saliency bash Rationale_Analysis/commands/saliency_script.sh;
    for rationale in global_top_k global_contig;
        do
        MIN_INST_PERCENT=10 CUDA_DEVICE=-1 BSIZE=100000 SALIENCY=$saliency RATIONALE=$rationale RATIONALE_EXP_NAME=direct bash Rationale_Analysis/commands/rationale_extractor_script.sh;

        SALIENCY=$saliency RATIONALE=$rationale RATIONALE_EXP_NAME=direct sbatch --time=$( t2sd ) Cluster_scripts/multi_gpu_sbatch.sh bash Rationale_Analysis/commands/model_b_train_script.sh;
        
        SALIENCY=$saliency RATIONALE=$rationale RATIONALE_EXP_NAME=direct \
        RATIONALE_CLASSIFIER=bert_generator_saliency RC_EXP_NAME=direct USE_CRF=true \
        sbatch --time=$( t2sd ) Cluster_scripts/multi_gpu_sbatch.sh bash Rationale_Analysis/commands/rationale_train_and_predict_script.sh;
        done;
    done;