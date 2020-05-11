export CONFIG_FILE=Rationale_Analysis/training_config/extractors/bert_extractor.jsonnet
export CUDA_DEVICE=${CUDA_DEVICE:?"set cuda device"}

export TRAIN_DATA_PATH=${RATIONALE_PATH:?"Set thresholded rationale path"}/train.jsonl
export DEV_DATA_PATH=${RATIONALE_PATH}/dev.jsonl
export TEST_DATA_PATH=${RATIONALE_PATH}/test.jsonl

export OUTPUT_BASE_PATH=${RATIONALE_PATH}/human_supervision=${HUMAN_PROB:?"Set Human Prob"}

export SEED=${RANDOM_SEED:-100}
export EPOCHS=${EPOCHS:-2}

if [[ -f "${OUTPUT_BASE_PATH}/metrics.json" ]]; then
    echo "${OUTPUT_BASE_PATH}/metrics.json exists ... . Not running Training ";
else 
    echo "${OUTPUT_BASE_PATH}/metrics.json does not exists ... . TRAINING ";
    allennlp train -s $OUTPUT_BASE_PATH --include-package Rationale_Analysis --force $CONFIG_FILE
fi;

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
        --silent \
        $OUTPUT_BASE_PATH/model.tar.gz $2;
    fi;
}

rationale ${OUTPUT_BASE_PATH}/train.jsonl $TRAIN_DATA_PATH
rationale ${OUTPUT_BASE_PATH}/dev.jsonl $DEV_DATA_PATH
rationale ${OUTPUT_BASE_PATH}/test.jsonl $TEST_DATA_PATH