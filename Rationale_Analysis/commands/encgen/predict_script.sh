export DATA_BASE_PATH=${DATASETS_FOLDER:-Datasets}/${DATASET_NAME:?"Set dataset name"}/data

export TRAIN_DATA_PATH=${DATA_BASE_PATH}/train.jsonl
export DEV_DATA_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_DATA_PATH=$DATA_BASE_PATH/test.jsonl

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs}/${DATASET_NAME:?"Set dataset name"}/${CLASSIFIER:?"Set classifier"}/${EXP_NAME:?"Set Exp name"}

export RATIONALE_FOLDER_NAME=$OUTPUT_BASE_PATH/${RATIONALE}_thresholder/${RATIONALE_EXP_NAME:?"Set Rationale Extractor experiment name. May use hyperparameter settings for naming"}

mkdir -p $RATIONALE_FOLDER_NAME

function rationale {
    if [[ -f "$1"  && -z "$again" ]]; then 
        echo "$1 exists .. Not Predicting";
    else 
        echo "$1 do not exist ... Predicting";
        allennlp predict \
        --output-file $1 \
        --batch-size ${BSIZE} \
        --use-dataset-reader \
        --dataset-reader-choice validation \
        --predictor rationale_predictor \
        --include-package Rationale_Analysis \
        --cuda-device ${CUDA_DEVICE:?"set cuda device"} \
        --overrides "{model: {rationale_extractor : {type : '${RATIONALE}', max_length_ratio: ${MAX_LENGTH_RATIO}}}}" \
        --silent \
        $OUTPUT_BASE_PATH/model.tar.gz $2;
    fi;
}


function evaluate {
    if [[ -f "$1"  && -z "$again" ]]; then 
        echo "$1 exists .. Not Predicting";
    else 
        echo "$1 do not exist ... Predicting";
        allennlp evaluate \
        --output-file $1 \
        --batch-size ${BSIZE} \
        --include-package Rationale_Analysis \
        --cuda-device ${CUDA_DEVICE:?"set cuda device"} \
        --overrides "{model: {rationale_extractor : {type : '${RATIONALE}', max_length_ratio: ${MAX_LENGTH_RATIO}}}}" \
        $OUTPUT_BASE_PATH/model.tar.gz $2;
    fi;
}

evaluate $RATIONALE_FOLDER_NAME/dev_metrics.json $DEV_DATA_PATH
evaluate $RATIONALE_FOLDER_NAME/test_metrics.json $TEST_DATA_PATH

rationale $RATIONALE_FOLDER_NAME/dev_rationales.jsonl $DEV_DATA_PATH
rationale $RATIONALE_FOLDER_NAME/test_rationales.jsonl $TEST_DATA_PATH