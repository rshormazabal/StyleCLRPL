#######
#      Taken from https://github.com/sthalles/SimCLR.
#######
from pathlib import Path

import omegaconf
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from datetime import datetime
from pytorch_lightning.utilities import rank_zero_only


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LastEpochCheckpoint(ModelCheckpoint):
    def __init__(self, dataset_name: str, base_model_name: str, every_k_epochs: int, max_epochs: int, *args, **kwargs):
        """
        Checkpointer callback that saves PL module weights every K epochs.
        :param base_model_name: Current backnonem model name. [str]
        :param every_k_epochs: Every how many epochs to save the checkpoint. [int]
        """
        super().__init__(*args, **kwargs)
        self.every_k_epochs = every_k_epochs
        self.max_epochs = max_epochs

        # save in the format /checkpoints/day/time/bas_model/
        self.dirpath = Path(self.dirpath) / dataset_name / base_model_name
        self.dirpath = self.dirpath / datetime.today().strftime('%Y-%m-%d') / datetime.today().strftime('%H-%M-%S')

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        # save pl module parameters (includes model parameters, scheduler config, etc)
        if (pl_module.current_epoch % self.every_k_epochs == 0) or (pl_module.current_epoch == self.max_epochs):
            assert self.dirpath is not None

            # get current path
            current = self.dirpath / f"latest-epoch={pl_module.current_epoch}.ckpt"

            # get previous checkpoint filename to delete
            # prev = (self.dirpath / f"latest-epoch={pl_module.current_epoch - self.every_k_epochs}.ckpt")
            trainer.save_checkpoint(current)

            # Beter to not delete previous to test after training.
            # prev.unlink(missing_ok=True)

        # save YAML config on the first epoch
        # TODO: fails if we start from a configuration file with epoch different to zero.
        if pl_module.current_epoch == 0:
            omegaconf.OmegaConf.save(pl_module.cfg, self.dirpath / "config_file.yaml")


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
