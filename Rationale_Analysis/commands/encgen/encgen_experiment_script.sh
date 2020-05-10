echo ${CLASSIFIER:?"Set a Classifier"}

bash Rationale_Analysis/commands/model_a_train_script.sh;

for rationale in top_k contiguous;
    do
    RATIONALE=$rationale \
    RATIONALE_EXP_NAME=direct \
    bash Rationale_Analysis/commands/encgen_predict_script.sh;
    done;