bash Rationale_Analysis/commands/model_a_train_script.sh;

export PYTHONPATH=.

bash Rationale_Analysis/commands/fresh/saliency_script.sh;

export THRESHOLDER_EXP_NAME=$MAX_LENGTH_RATIO;

bash Rationale_Analysis/commands/fresh/thresholder_script.sh;

export RATIONALE_PATH=${OUTPUT_DIR:-outputs}/${DATASET_NAME:?"Set dataset name"}/$CLASSIFIER/${EXP_NAME}/${SALIENCY}_saliency/${THRESHOLDER}_thresholder/${THRESHOLDER_EXP_NAME}

bash Rationale_Analysis/commands/fresh/extractor_train_script.sh

KEEP_PROB=1.0 \
DATA_BASE_PATH=${RATIONALE_PATH}/human_supervision=${HUMAN_PROB:?"Set Human Prob"} \
EXP_NAME=${EXP_NAME}/${SALIENCY}_saliency/${THRESHOLDER}_thresholder/${THRESHOLDER_EXP_NAME}/human_supervision=${HUMAN_PROB:?"Set Human Prob"}/model_b \
bash Rationale_Analysis/commands/model_train_script.sh