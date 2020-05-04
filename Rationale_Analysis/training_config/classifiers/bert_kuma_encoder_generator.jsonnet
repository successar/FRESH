local bert_type = std.extVar('BERT_TYPE');

local bert_model = {
  type: "bert_classifier",
  bert_model: bert_type,
  requires_grad: '10,11,pooler',
  dropout : 0.2,
};

local indexer = "pretrained-simple";

local bert_gen_model = {
  type: "kuma_bert_generator",
  bert_model: bert_type,
  requires_grad: '10,11,pooler',
  dropout : 0.2,
};

{
  dataset_reader : {
    type : "base_reader",
    token_indexers : {
      bert : {
        type : indexer,
        model_name : bert_type,
      },
    },
  },
  validation_dataset_reader: {
    type : "base_reader",
    token_indexers : {
      bert : {
        type : indexer,
        model_name : bert_type,
      },
    },
  },
  train_data_path: std.extVar('TRAIN_DATA_PATH'),
  validation_data_path: std.extVar('DEV_DATA_PATH'),
  test_data_path: std.extVar('TEST_DATA_PATH'),
  model: {
    type: "kuma_gen_enc_classifier",
    generator: bert_gen_model,
    encoder : bert_model,
    samples: 1,
    lambda_init: std.parseJson(std.extVar('LAMBDA_INIT')),
    desired_length: std.parseJson(std.extVar('MAX_LENGTH_RATIO'))
  },
  data_loader : {
    batch_size: std.parseInt(std.extVar('BSIZE')),
    shuffle: true,
  },
  validation_data_loader : {
    batch_size: std.parseInt(std.extVar('BSIZE')),
    shuffle: false,
  },
  trainer: {
    num_epochs: std.parseInt(std.extVar('EPOCHS')),
    patience: 10,
    grad_norm: 5.0,
    validation_metric: "+validation_metric",
    checkpointer: {num_serialized_models_to_keep: 1,},
    cuda_device: std.parseInt(std.extVar("CUDA_DEVICE")),
    optimizer: {
      type: "adamw",
      lr: 2e-5
    },
  },
  random_seed:  std.parseInt(std.extVar("SEED")),
  pytorch_seed: std.parseInt(std.extVar("SEED")),
  numpy_seed: std.parseInt(std.extVar("SEED")),
  evaluate_on_test: true
}
