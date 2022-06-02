# StyleCLR: Style-transfer augmented constrastive representations for images 

## Project structure
```
SimCLRPL/
├─ conf/
│  ├─ config.yaml
│  ├─ StyleCLR_base_format.yaml
├─ data/
│  ├─ pretrained_model_adain/
│  │  ├─ decored.pth 
│  │  ├─ vgg_normalised.pth
│  ├─ style_augmentation_images/
│  │  ├─ 1.jpg
│  │  ├─ ...
│  │  ├─ 1000.jpg
│  ├─ train_info_1000.csv
├─ data_aug/
│  ├─ adain.py
│  ├─ style_transforms.py
├─ exceptions/
│  ├─ exceptions.py
├─ models/
│  ├─ resnet_simclr.py
├─ dataset.py
├─ lightning_modules.py
├─ pl_main.py
├─ README.md
├─ requirements.txt
```
## Hydra configuration files
Set configuration in `pl_main.py` with
```python
@hydra.main(config_path="conf", config_name="StyleCLR_base_format")
```
Can create a new configuration file folowing the base format available in `conf/`.
For detailed information on how to modify configuration files through CLI, check [Hydra Documentation](https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/).

#### Configuration file example.
```yaml
model_name: StyleCLR_base                                                  # Model name for loggin (neptune).
root_path: /ROOT/                                                          # Project main root.
seed: Seed                                                                 # Random seed in random, torch, pandas and cuda.
gpu_ids:                                                                   # GPU ID's as list.
    - 1
    - 2
    - 3
    - 4
train:                                                                     # PL trainer params. See PL trainer documentation.
  strategy: ddp
  precision: 32
  max_epochs: 100
  log_every_n_steps: 4
  check_val_every_n_epoch: 0
  amp_backend: native
  enable_checkpointing: True
model:
  base_model: resnet50                                                     # Backbone model for SimCLR.
  out_dim: 128                                                             # SimCLR output dimension.
  temperature: 0.07                                                        # InfoNCE temperature parameters.
  style_alpha: 0.8                                                         # AdaIn alpha (higher -> stronger style).
  style_vgg_path: /ROOT/data/pretrained_models_adain/vgg_normalised.pth    # Style transfer VGG backbone network pretrained weights' path.
  style_decoder_path: /ROOT/data/pretrained_models_adain/decoder.pth       # Style transfer decoder network pretrained weights' path.
dataset:
  style:
    recalculate_vgg_embeddings: False                                      # Whether to calculate pretrained embeddings for style images.
    data_path: /ROOT/data/                                                 # Style data path (metadata and images).
    image_path: style_augmentation_images/                                 # Style images folder name.
    metadata_filename: train_info_1000.csv                                 # Style dataset metadata CSV filename.
    pickle_filename: style_precalculated_features.pkl                      # Precalculated style images embeddings generated with frozen backbone VGG.
    image_size: 96                                                         # Style image warp size (square).
    device: 0                                                              # Device to calculate preprocessed style embeddings.
  dataset_name: stl10                                                      # Dataset to use (stl10/cifar10)
  batch_size: 512                                                          # Main batch size.
  num_workers: 4                                                           # Main dataloader number of workers.
  n_views: 2                                                               # Number of views to generate (currently supports 2)
  data_path: /ROOT/datasets                                                # Content images path.
  len_train_loader: set_at_runtime                                         # Holder for train dataloader size (used in scheduler).
optimizer:
  lr: 0.0003                                                               # Main Learning rate.
  weight_decay: 0.0001                                                     # Weight decay.
logger:
  project_name: example/StyleCLR                                           # Logger name for neptune.
  tags:
      - base
      - example_dataset_name
      - example_only_adain_augmentations                                   # Tags for neptune logger as list.
  api_key: NEPTUNE_API_TOKEN                                               # Neptune API token.
callbacks:
  monitor: val_loss                                                        # Torch Lightning callback parameters
  monitor_mode: min                                                        # Example - Checkpointer, early stopping, etc.
  save_top_k: 5
  patience: 1000
```