import os
import pickle

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelSummary, LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset import PrecalculateSyleEmbeddings
from dataset import StyleCLRPLDataset
from lightning_modules import StyleCLRPLModel


def setup_neptune_logger(cfg: DictConfig, tags: list = None):
    """
    Nettune AI loger configuration. Needs API key.
    :param cfg: Configuration dictionary. [omegaconf.dictconfig.DictConfig]
    :param tags: List of tags to log a particular run. [list]
    :return:
    """
    meta_tags = [cfg.model_name]

    if tags is not None:
        meta_tags.extend(tags)

    # setup logger
    neptune_logger = NeptuneLogger(api_key=cfg.logger.api_key,
                                   project=cfg.logger.project_name,
                                   tags=meta_tags)

    neptune_logger.experiment["parameters/model"] = cfg.model
    neptune_logger.experiment["parameters/data"] = cfg.data
    neptune_logger.experiment["parameters/optimizer"] = cfg.optimizer
    neptune_logger.experiment["parameters/run"] = {k: v for k, v in cfg.items() if not isinstance(v, omegaconf.dictconfig.DictConfig)}

    return neptune_logger


@hydra.main(config_path="conf", config_name="StyleCLR_base_format")
def main(cfg: DictConfig) -> None:
    """
    Main training class. All parameters are defined in the yaml Hydra configuration.
    :param cfg: Hydra format configuration. [omegaconf.dictconfig.DictConfig]
    :return: None.
    """
    # setup logger
    # logger = setup_neptune_logger(cfg)

    # set seeds
    pl.seed_everything(cfg.seed)

    # base setup
    data = StyleCLRPLDataset(cfg.dataset)
    data.setup()

    # set dataloader len for schduler
    cfg.dataset['len_train_loader'] = len(data.train_dataloader())

    # profiler
    profiler = None

    # pl model
    model = StyleCLRPLModel(cfg)

    # precalculate VGG features for style and content images
    if cfg.dataset.style.recalculate_vgg_embeddings or not os.path.exists(f'{cfg.dataset.style.data_path}{cfg.dataset.style.pickle_filename}'):
        print('Recalculate style features set to True or file does not exist, calculating style embeddings.')
        # style images dataset and dataloader
        # TODO: create flag for the style loader batch size and style transform size
        style_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
        style_dataset = PrecalculateSyleEmbeddings(cfg.dataset.style,
                                                   style_transform=style_transform)
        style_loader = DataLoader(style_dataset,
                                  batch_size=128,
                                  shuffle=False,
                                  num_workers=cfg.dataset.num_workers,
                                  drop_last=False)

        # calculate features and dump
        # TODO: could do the same thing with all images and avoid loading VGG network (speedup?)
        device = f'cuda:{cfg.dataset.style.device}'
        model.style_vgg.to(device)
        precalculated_style_features = []
        for batch in tqdm(style_loader):
            with torch.no_grad():
                precalculated_style_features.append(model.style_vgg(batch.to(device)))

        # concatenate and dump
        dump_path = f'{cfg.dataset.style.data_path}{cfg.dataset.style.pickle_filename}'
        precalculated_style_features = torch.cat(precalculated_style_features)
        pickle.dump(precalculated_style_features.cpu(), open(dump_path, 'wb'))
        print(f'Precalculated embeddings dumped at {dump_path}')

    # create trainer and fit
    trainer = pl.Trainer(gpus=cfg.gpu_ids,
                         strategy=cfg.train.strategy,
                         precision=cfg.train.precision,
                         log_every_n_steps=cfg.train.log_every_n_steps,
                         check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
                         amp_backend=cfg.train.amp_backend,
                         max_epochs=cfg.train.max_epochs,
                         enable_checkpointing=cfg.train.enable_checkpointing,
                         callbacks=[ModelSummary(max_depth=1),
                                    LearningRateMonitor()],
                         profiler=profiler)
    trainer.fit(model, data)


if __name__ == '__main__':
    main()
