from abc import ABC
from typing import Tuple

import pytorch_lightning as pl
import torch
from flash.core.optimizers import LARS
from omegaconf import DictConfig
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch.nn import functional as F, Sequential

from nvidia.dali.plugin.pytorch import feed_ndarray
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from data_aug.adain import vgg, decoder
from data_aug.transforms import adain, background, dali_augmentation, ExternalInputGPUIterator
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

        # self.adain = adain(self.style_vgg, self.style_decoder, self.alpha)

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
        # content_images, transparency, style_feats1, style_feats2 = batch[0]
        content_images = torch.cat([batch[0], batch[0]], dim=0)

        if self.cfg.augment.crop or self.cfg.augment.color or self.cfg.augment.blur:
            styled_images = self.dali_augmentation(content_images)
        else:
            styled_images = content_images

        # import matplotlib.pyplot as plt
        # f, axarr = plt.subplots(2, 2)
        # axarr[0, 0].imshow(content_images[2].permute(1, 2, 0).detach().cpu().numpy())
        # axarr[0, 1].imshow(styled_images[2].permute(1, 2, 0).detach().cpu().numpy())
        # axarr[1, 0].imshow(content_images[3].permute(1, 2, 0).detach().cpu().numpy())
        # axarr[1, 1].imshow(styled_images[3].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()
        # content_images = torch.cat([content_images for _ in range(2)], dim=0)
        # style_feats = torch.cat([style_feats1, style_feats2], dim=0)
        #
        # if self.cfg.augment.adain:
        #     styled_images = self.adain(content_images, style_feats)
        # else:
        #     styled_images = content_images
        #
        # if self.cfg.augment.background_replacer:
        #     styled_images = self.background(content_images, styled_images, transparency)

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

        # select only the negatives
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

        optim_dict = {"LARS": LARS,
                      "Adam": torch.optim.Adam}

        optim_name = optim_dict[self.cfg.optimizer.name]

        optimizer = optim_name(self.model.parameters(),
                               self.cfg.optimizer.lr,
                               weight_decay=self.cfg.optimizer.weight_decay)

        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=self.cfg.scheduler.warmup_epochs,
                                                  max_epochs=self.cfg.train.max_epochs)

        return [optimizer], [{"scheduler": scheduler}]

    @pipeline_def
    def augment_pipeline(self, cuda_images):
        images = fn.external_source(source=cuda_images, device='gpu', batch=True, cuda_stream=0, dtype=types.UINT8)
        images = fn.random_resized_crop(images, size=self.cfg.augment.size) if self.cfg.augment.crop else images
        images = fn.flip(images, horizontal=1, vertical=0) if torch.rand(1) < 0.5 and self.cfg.augment.crop else images
        b, c, s = torch.distributions.uniform.Uniform(1 - 0.8, 1 + 0.8).sample([3, ])
        h = torch.distributions.uniform.Uniform(-0.2, 0.2).sample([1, ])
        images = fn.color_twist(images, brightness=b, contrast=c, saturation=s, hue=h) if torch.rand(1) < 0.8 and self.cfg.augment.color else images  # only accept hwc
        images = fn.color_space_conversion(images, image_type=types.RGB, output_type=types.GRAY) if torch.rand(1) < 0.2 and self.cfg.augment.color else images  # only accept hwc, uint8
        images = fn.gaussian_blur(images, window_size=int(0.1 * self.cfg.augment.size)) if self.cfg.augment.blur else images
        return images

    def dali_augmentation(self, images):
        input_dtype = images.dtype

        eii = ExternalInputGPUIterator(images)
        augment = self.augment_pipeline(eii, batch_size=self.cfg.dataset.batch_size * 2, num_threads=8, device_id=self.local_rank)
        augment.build()
        images = augment.run()

        images = images[0].as_tensor()  # type : TensorGPU

        augmented_images = torch.zeros(images.shape(), dtype=torch.uint8).cuda()
        feed_ndarray(images, augmented_images)
        augmented_images = augmented_images.permute(0, 3, 1, 2).type(input_dtype) / 255.
        c = augmented_images.shape[1]
        if c == 1:
            augmented_images = augmented_images.repeat(1, 3, 1, 1)

        return augmented_images


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
        self.model = ResNetDownStream(cfg=self.cfg)

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
        return {'acc/top1': top1,
                'acc/top5': top5}

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
        opt_dict = {"Adam": torch.optim.Adam(self.model.parameters(),
                                             self.cfg.optimizer.lr,
                                             weight_decay=self.cfg.optimizer.weight_decay),
                    "Nesterov": torch.optim.SGD(self.model.parameters(),
                                                momentum=0.9,
                                                nesterov=True,
                                                lr=self.cfg.dataset.batch_size / 256 * 0.05)}

        optimizer = opt_dict[self.cfg.optimizer.name]

        return [optimizer]
