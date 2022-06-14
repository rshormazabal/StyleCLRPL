import os
import pickle

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelSummary, LearningRateMonitor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset import PrecalculateSyleEmbeddings
from dataset import StyleCLRPLDataset
from lightning_modules import StyleCLRPLModel
from utils import LastEpochCheckpoint


@hydra.main(config_path="conf", config_name="StyleCLR_test", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main training class. All parameters are defined in the yaml Hydra configuration.
    :param cfg: Hydra format configuration. [omegaconf.dictconfig.DictConfig]
    :return: None.
    """
    # set seeds
    pl.seed_everything(cfg.seed)

    # base setup
    data = StyleCLRPLDataset(cfg)

    # checkpoint callback, saves last model every k epochs. Better than implementing directly in the PL logic since
    # it can have problems on DDP if main process not set corectly.
    checkpoint_callback = LastEpochCheckpoint(dirpath=cfg.callbacks.checkpoints.dirpath,
                                              dataset_name=cfg.dataset.content.name,
                                              base_model_name=cfg.model.base_model,
                                              every_k_epochs=cfg.callbacks.checkpoints.every_k_epochs,
                                              max_epochs=cfg.train.max_epochs)

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
        precalculated_style_features = torch.cat(precalculated_style_features).detach().cpu()
        pickle.dump(precalculated_style_features, open(dump_path, 'wb'))

        # add style embeddings to train dataset
        data.train_dataset.style_embeddings = precalculated_style_features
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
                                    LearningRateMonitor(),
                                    checkpoint_callback])
    trainer.fit(model, data)


if __name__ == '__main__':
    main()