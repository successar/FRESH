local berts = import '../berts.libsonnet'; 

{
  dataset_reader : {
    type : "base_reader",
    token_indexers : {
      bert : berts.indexer,
    }
  },
  validation_dataset_reader: {
    type : "base_reader",
    token_indexers : {
      bert : berts.indexer,
    },
  },
  train_data_path: std.extVar('TRAIN_DATA_PATH'),
  validation_data_path: std.extVar('DEV_DATA_PATH'),
  test_data_path: std.extVar('TEST_DATA_PATH'),
  model: berts.classifier,
  data_loader : {
    batch_size: std.extVar('BSIZE'),
    shuffle: true,
    batch_sampler: "random"
  },
  trainer: {
    num_epochs: std.extVar('EPOCHS'),
    patience: 10,
    grad_norm: 5.0,
    validation_metric: "+validation_metric",
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
