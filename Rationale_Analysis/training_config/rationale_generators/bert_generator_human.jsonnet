local bert_type = if std.findSubstr('evinf', std.extVar('TRAIN_DATA_PATH')) == [] then 'roberta-base' else 'allenai/scibert_scivocab_uncased';

local bert_model = {
  type: "supervised_bert_extractor",
  bert_model: bert_type,
  requires_grad: '10,11,pooler',
  dropout : 0.2,
  max_length_ratio: std.extVar('MAX_LENGTH_RATIO')
};

local indexer = "pretrained-simple";

{
  dataset_reader : {
    type : "extractor_reader",
    token_indexers : {
      bert : {
        type : indexer,
        model_name : bert_type,
      },
    },
    human_prob: std.extVar('HUMAN_PROB')
  },
  validation_dataset_reader: {
    type : "extractor_reader",
    token_indexers : {
      bert : {
        type : indexer,
        model_name : bert_type,
      },
    },
    human_prob: std.extVar('HUMAN_PROB')
  },
  train_data_path: std.extVar('TRAIN_DATA_PATH'),
  validation_data_path: std.extVar('DEV_DATA_PATH'),
  test_data_path: std.extVar('TEST_DATA_PATH'),
  model: bert_model,
  iterator: {
    type: "basic",
    batch_size : std.extVar('BSIZE')
  },
  trainer: {
    num_epochs: 20,
    patience: 10,
    grad_norm: 5.0,
    validation_metric: "+f1",
    checkpointer: {num_serialized_models_to_keep: 1,},
    
    cuda_device: std.extVar("CUDA_DEVICE"),
    optimizer: {
      type: "adamw",
      lr: 2e-5
    },
    should_log_learning_rate: true
  },
  random_seed:  std.parseInt(std.extVar("SEED")),
  pytorch_seed: std.parseInt(std.extVar("SEED")),
  numpy_seed: std.parseInt(std.extVar("SEED")),
  evaluate_on_test: true
}
