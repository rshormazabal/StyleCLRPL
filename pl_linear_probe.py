import hydra
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelSummary, LearningRateMonitor

from dataset import LinearProbeDataset
from lightning_modules import ClassificationModel
import pandas as pd


@hydra.main(config_path="conf", config_name="LinearProbe_base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main training class. All parameters are defined in the yaml Hydra configuration.
    :param cfg: Hydra format configuration. [omegaconf.dictconfig.DictConfig]
    :return: None.
    """
    # set seeds
    pl.seed_everything(cfg.seed)

    # base setup
    data = LinearProbeDataset(cfg)

    # pl model
    pl_model = ClassificationModel(cfg)

    # checkpoit info
    checkpoint_name = f'latest-epoch={cfg.checkpoint.epoch}.ckpt'
    model_checkpoint_path = f'{cfg.root_path}/{cfg.checkpoint.run_name}/{checkpoint_name}'

    # load resnet pretrained parameters and freeze first layers
    backbone_weights = {k.replace('model.', ''): v for k, v in torch.load(model_checkpoint_path)['state_dict'].items() if k.startswith('model.')}
    pl_model.model.load_state_dict(backbone_weights, strict=False)
    print(f'Weights loaded from run: {model_checkpoint_path}')
    pl_model.model.freeze_conv_params()

    # trainer
    trainer = pl.Trainer(gpus=cfg.gpu_ids,
                         precision=cfg.train.precision,
                         log_every_n_steps=cfg.train.log_every_n_steps,
                         check_val_every_n_epoch=500,
                         amp_backend=cfg.train.amp_backend,
                         max_epochs=cfg.train.max_epochs,
                         callbacks=[ModelSummary(max_depth=1),
                                    LearningRateMonitor()])
    trainer.fit(pl_model, data)

    # validation and save results
    probe_test_results = trainer.validate(pl_model, data.test_dataloader())
    simclr_metadata = yaml.safe_load(open(f'{cfg.checkpoint.run_name}/config_file.yaml'))
    all_results = pd.read_csv(f'{cfg.root_path}/{cfg.results_path}')
    result = {'run': cfg.checkpoint.run_name.split('StyleCLRPL/')[-1],
              'checkpoint': checkpoint_name,
              'simclr_epochs': int(checkpoint_name.split('=')[-1].replace('.ckpt', '')),
              'simclr_outdim': simclr_metadata['model']['out_dim'],
              'simclr_temperature': simclr_metadata['model']['temperature'],
              'simclr_batchsize': simclr_metadata['dataset']['batch_size'],
              'simclr_optimizer': simclr_metadata['optimizer']['name'],
              'simclr_lr': simclr_metadata['optimizer']['lr'],
              'probe_epochs': simclr_metadata['probe']['max_epochs'],
              'probe_optimizer': simclr_metadata['probe']['optimizer_name'],
              'probe_lr': pl_model.configure_optimizers()[0].state_dict()['param_groups'][0]['lr'],
              'probe_batchsize': simclr_metadata['probe']['batch_size'],
              'probe_acc/top1': probe_test_results[0]['acc/top1'],
              'probe_acc/top5': probe_test_results[0]['acc/top5']}
    all_results.loc[len(all_results.index)] = result

    all_results.to_csv(cfg.results_path, index=False)


if __name__ == '__main__':
    main()
