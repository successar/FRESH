export EXP_NAME=${MAX_LENGTH_RATIO}/${EXP_NAME}/human_supervision=${HUMAN_PROB:?"Set human prob"}

CLASSIFIER=bert_encoder_generator_human bash Rationale_Analysis/commands/model_a_train_script.sh;

for rationale in top_k contiguous;
    do
    CLASSIFIER=bert_encoder_generator_human \
    RATIONALE=$rationale \
    RATIONALE_EXP_NAME=$MAX_LENGTH_RATIO \
    bash Rationale_Analysis/commands/encgen/predict_script.sh;
    done;