from abc import ABC
from inspect import classify_class_attrs

from typing import Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch.nn import functional as F, Sequential

from flash.core.optimizers import LARS

from dataset import ContentImageDataset
from omegaconf import DictConfig
from data_aug.adain import vgg, decoder
from data_aug.transforms import adain, background, dali
from models.resnet_simclr import ResNetSimCLR, ResNetDownStream
from utils import accuracy


class StyleCLRPLModel(pl.LightningModule, ABC):
    def __init__(self, cfg: DictConfig):
        """
        Main Torch lightning module.
        :param cfg: Hydra format CFG configuration object. [omegaconf.dictconfig.DictConfig]
        """
        super().__init__()
        # config files
        self.cfg = cfg

        # parameters for style transfer
        self.alpha = cfg.model.style_alpha

        # SimCLR model and style networks
        self.model = ResNetSimCLR(model_cfg=self.cfg.model)
        self.style_vgg = vgg
        self.style_decoder = decoder

        # load VGG pretrained weights
        print('Loading VGG and decoder weights.')
        self.style_vgg.load_state_dict(torch.load(self.cfg.model.style_vgg_path))
        self.style_decoder.load_state_dict(torch.load(self.cfg.model.style_decoder_path))

        # drop last backbone layers
        self.style_vgg = Sequential(*list(self.style_vgg.children())[:31])

        # set to eval
        self.style_vgg = self.style_vgg.eval()
        self.style_decoder = self.style_decoder.eval()

        self.adain = adain(self.style_vgg, self.style_decoder, self.alpha)
        self.background = background()
        self.dali = dali(self.cfg, self.cfg.augment.crop, self.cfg.augment.color, self.cfg.augment.blur)

        ##
        # linear predictor
        ##

        train_data, valid_data, _ = ContentImageDataset(self.cfg).get_dataset_for_linear_probe()

        self.probe_train_dataloader = torch.utils.data.DataLoader(train_data,
                        batch_size=self.cfg.dataset.batch_size,
                        shuffle=True,
                        num_workers=self.cfg.dataset.num_workers,
                        pin_memory=True,
                        drop_last=True)

        self.probe_valid_dataloader = torch.utils.data.DataLoader(valid_data,
                        batch_size=self.cfg.dataset.batch_size,
                        shuffle=False,
                        num_workers=self.cfg.dataset.num_workers,
                        pin_memory=True,
                        drop_last=True)


    def forward(self, images: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Model forward pass.
        :param images: Batch of images (batch_dim, channels, weidth, height). [torch.Tensor]
        :param args: None.
        :param kwargs: None.
        :return:
        """
        return self.model(images)


    def training_step(self, batch: list) -> STEP_OUTPUT:
        """
        Compute and return the training loss and some additional metrics for e.g. the progress bar or logger.
        See PL documentation.
        :param batch: Batch from self.train_dataloader. [list]
        :return: Dictionary with per-batch loss and metrics. [dict]
        """
        # get content images and create two views
        content_images, transparency, style_feats1, style_feats2 = batch[0]

        content_images = torch.cat([content_images for _ in range(2)], dim=0)
        style_feats = torch.cat([style_feats1, style_feats2], dim=0)

        ######################
        # Augment the images #
        ######################

        if self.cfg.augment.adain:
            styled_images = self.adain(content_images, style_feats)
        else:
            styled_images = content_images
            
        if self.cfg.augment.background_replacer:
            styled_images = self.background(content_images, styled_images, transparency)

        if self.cfg.augment.crop or self.cfg.augment.color or self.cfg.augment.blur:
            styled_images = self.dali(styled_images)


        # augmented views contrastive setup
        features = self.model(styled_images)
        logits, labels = self.info_nce_loss(features)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        # get top 1 and top 5 accuracies
        top1, top5 = accuracy(logits, labels, topk=(1, 5))
        return {'logits': logits.detach(),
                'labels': labels.detach(),
                'loss': loss,
                'nce/top1': top1[0].detach(),
                'nce/top5': top5[0].detach()}


    


    def training_step_end(self, step_outputs: STEP_OUTPUT) -> STEP_OUTPUT:
        """
        Called right after training step. Useful for NCE-loss if using dp/ddp2 since batches are split across GPUS.
        See PL documentation.
        :param step_outputs: output of training_step. [dict]
        :return:
        """
        return step_outputs


    def training_epoch_end(self, outputs: EPOCH_OUTPUT):
        """
        Called at the end of the training epoch with the outputs of all training steps as a list.
        See PL documentation
        :param outputs: Concatenated list of batch outputs. [list[dicts]]
        :return:
        """
        top1_epoch_avg = torch.stack([o['nce/top1'] for o in outputs]).mean()
        top5_epoch_avg = torch.stack([o['nce/top5'] for o in outputs]).mean()

        # get mean across all batches
        self.log("nce/top1", top1_epoch_avg, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("nce/top5", top5_epoch_avg, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.current_epoch % self.cfg.probe.run_every_n_epoch == 0:
            result = self.run_linear_predictor()            
            self.log_dict(result, on_epoch=True, prog_bar=True, sync_dist=True)

    
    def run_linear_predictor(self):

        trainer = pl.Trainer(gpus=self.cfg.gpu_ids,
            strategy=self.cfg.train.strategy,
            precision=self.cfg.train.precision,
            log_every_n_steps=1,
            check_val_every_n_epoch=0,
            amp_backend=self.cfg.train.amp_backend,
            max_epochs=self.cfg.probe.epochs,
        )

        downstream_model = ClassificationModel(self.cfg)
        downstream_model.model.get_params_from_resnetsimclr(self.model)
        downstream_model.model.freeze_conv_params()
        
        trainer.fit(downstream_model, self.probe_train_dataloader)
        return trainer.validate(downstream_model, self.probe_valid_dataloader)[0]


    def info_nce_loss(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Noise-Contrastive Estimation loss, contrastive loss used for SSl.
        :param features: Augmented views features, vstacked (2 * batch_size, backbone_output_dim). [torch.Tensor]
        :return: logits tensor for all views with labels (zeros). [torch.Tensor, torch.Tensor]
        """
        # create labels different views (positive pairs are augmented views of same image)
        labels = [torch.arange(self.cfg.dataset.batch_size, device=features.device) for _ in range(self.cfg.dataset.n_views)]
        labels = torch.cat(labels, dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # pairwise similarity
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # concatenate logits and divide by temperature parameters
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)

        logits = logits / self.cfg.model.temperature
        return logits, labels


    def configure_optimizers(self):
        """
        Choose optimizers, schedulers and learning-rate schedulers to use. Automatically called by PL.
        See PL documentation.
        :return:
        """

        optim_dict = {
            "LARS": LARS,
            "Adam": torch.optim.Adam
        }

        optim_name = optim_dict[self.cfg.optimizer.name]

        optimizer = optim_name(self.model.parameters(),
                            self.cfg.optimizer.lr,
                            weight_decay=self.cfg.optimizer.weight_decay)

        if self.cfg.optimizer.use_cosine_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.cfg.dataset.len_train_loader,
                                                               eta_min=0,
                                                               last_epoch=-1)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 1)

        return [optimizer] , [{"scheduler": scheduler}]





class ClassificationModel(pl.LightningModule, ABC):

    def __init__(self, cfg: DictConfig):
        """
        Main Torch lightning module.
        :param cfg: Hydra format CFG configuration object. [omegaconf.dictconfig.DictConfig]
        """
        super().__init__()
        # config files
        self.cfg = cfg
        
        # Downstream Classification Model
        self.model = ResNetDownStream(model_cfg=self.cfg.model)

        self.dali = dali(self.cfg, do_crop = True, do_color = False, do_blur = False)


    def forward(self, images: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Model forward pass.
        :param images: Batch of images (batch_dim, channels, weidth, height). [torch.Tensor]
        :param args: None.
        :param kwargs: None.
        :return:
        """
        return self.model(images)


    def training_step(self, batch: list) -> STEP_OUTPUT:
        """
        Compute and return the training loss and some additional metrics for e.g. the progress bar or logger.
        See PL documentation.
        :param batch: Batch from self.train_dataloader. [list]
        :return: Dictionary with per-batch loss and metrics. [dict]
        """
      
        content_images, labels = batch

        # Cropping and flipping
        content_images = self.dali(content_images)

        features = self.model(content_images)
        loss = torch.nn.CrossEntropyLoss()(features, labels)

        # get top 1 and top 5 accuracies
        top1, top5 = accuracy(features, labels, topk=(1, 5))
        return {'logits': features.detach(),
                'labels': labels.detach(),
                'loss': loss,
                'trn/top1': top1[0].detach(),
                'trn/top5': top5[0].detach()}




    def training_epoch_end(self, outputs: EPOCH_OUTPUT):
        """
        Called at the end of the training epoch with the outputs of all training steps as a list.
        See PL documentation
        :param outputs: Concatenated list of batch outputs. [list[dicts]]
        :return:
        """
        top1_epoch_avg = torch.stack([o['trn/top1'] for o in outputs]).mean()
        top5_epoch_avg = torch.stack([o['trn/top5'] for o in outputs]).mean()

        # get mean across all batches
        self.log("trn/top1", top1_epoch_avg, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("trn/top5", top5_epoch_avg, on_epoch=True, prog_bar=True, sync_dist=True)


    def validation_step(self, batch: list, batch_idx) -> STEP_OUTPUT:

        content_images, labels = batch

        features = self.model(content_images)

        # get top 1 and top 5 accuracies
        top1, top5 = accuracy(features, labels, topk=(1, 5))
        return {
            'acc/top1': top1, 
            'acc/top5': top5
        }


    def validation_epoch_end(self, outputs: EPOCH_OUTPUT):
        """
        Called at the end of the training epoch with the outputs of all training steps as a list.
        See PL documentation
        :param outputs: Concatenated list of batch outputs. [list[dicts]]
        :return:
        """
        top1_epoch_avg = torch.stack([o['acc/top1'] for o in outputs]).mean()
        top5_epoch_avg = torch.stack([o['acc/top5'] for o in outputs]).mean()

        # get mean across all batches
        self.log("acc/top1", top1_epoch_avg, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("acc/top5", top5_epoch_avg, on_epoch=True, prog_bar=True, sync_dist=True)
        

    def configure_optimizers(self):
        """
        Choose optimizers, schedulers and learning-rate schedulers to use. Automatically called by PL.
        See PL documentation.
        :return:
        """

        opt_dict = {
            "Adam": torch.optim.Adam(self.model.parameters(),
                                     self.cfg.optimizer.lr,
                                     weight_decay=self.cfg.optimizer.weight_decay),
            "Nesterov": torch.optim.SGD(self.model.parameters(),
                                    momentum = 0.9,
                                    nesterov = True,
                                    lr = self.cfg.probe.lr)
        }

        optimizer = opt_dict[self.cfg.probe.optimizer_name]

        return [optimizer]
