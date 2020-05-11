export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs}/${DATASET_NAME:?"Set dataset name"}/${CLASSIFIER:?"Set classifier"}/${EXP_NAME:?"Set Exp name"}

export TRAIN_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY:?"Set Saliency scorer"}_saliency/train.jsonl
export DEV_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/dev.jsonl
export TEST_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/test.jsonl

export THRESHOLDER_CONFIG_FILE=Rationale_Analysis/training_config/thresholders/${THRESHOLDER:?"Set Thresholder"}.jsonnet
export THRESHOLDER_FOLDER_NAME=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/${THRESHOLDER}_thresholder/${THRESHOLDER_EXP_NAME}

mkdir -p $THRESHOLDER_FOLDER_NAME

function rationale {
    if [[ -f "$1"  && -z "$again" ]]; then 
        echo "$1 exists .. Not Predicting";
    else 
        echo "$1 do not exist ... Predicting";
        allennlp rationale \
        --output-file $1 \
        --batch-size ${BSIZE:-50} \
        --use-dataset-reader \
        --dataset-reader-choice validation \
        --predictor rationale_predictor \
        --include-package Rationale_Analysis \
        --silent --cuda-device ${CUDA_DEVICE:?"set cuda device"} \
        $THRESHOLDER_CONFIG_FILE $2;
    fi;
}

rationale $THRESHOLDER_FOLDER_NAME/train.jsonl $TRAIN_DATA_PATH
rationale $THRESHOLDER_FOLDER_NAME/dev.jsonl $DEV_DATA_PATH
rationale $THRESHOLDER_FOLDER_NAME/test.jsonl $TEST_DATA_PATH