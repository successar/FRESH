export RATIONALE=max_length
export RATIONALE_EXP_NAME=direct
export CLASSIFIER=bert_classification

export PYTHONPATH=.

export RATIONALE_PATH=${OUTPUT_DIR:-outputs}/${DATASET_NAME:?"Set dataset name"}/bert_classification/${EXP_NAME}/${SALIENCY}_saliency/${RATIONALE}_rationale/$RATIONALE_EXP_NAME

bash Rationale_Analysis/commands/extractor_train_script.sh

KEEP_PROB=1.0 \
DATA_BASE_PATH=${RATIONALE_PATH}/human_supervision=${HUMAN_PROB:?"Set Human Prob"} \
EXP_NAME=${EXP_NAME}/${SALIENCY}_saliency/${RATIONALE}_rationale/$RATIONALE_EXP_NAME/human_supervision=${HUMAN_PROB:?"Set Human Prob"}/model_b \
bash Rationale_Analysis/commands/model_train_script.sh