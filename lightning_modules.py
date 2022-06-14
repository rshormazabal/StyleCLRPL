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
from nvidia.dali import Pipeline
from data_aug.adain import vgg, decoder
from data_aug.transforms import adain, background, dali_augmentation, ExternalInputGPUIterator
from models.resnet_simclr import ResNetSimCLR, ResNetDownStream
from utils import accuracy
from nvidia.dali.pipeline.experimental import pipeline_def


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

        self.adain = adain(self.style_vgg, self.style_decoder, self.cfg.model.style_alpha)

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
        # if adain, apply style and then augmnetations
        # self.adain = adain(self.style_vgg, self.style_decoder, 0.4)
        if self.cfg.augment.adain:
            content_images, transparency, style_feats1, style_feats2 = batch[0]
            # style_feats = torch.cat([style_feats1, style_feats2], dim=0)
            # content_images = torch.cat([content_images, content_images], dim=0)
            styled_image1 = self.adain(content_images, style_feats1)
            styled_image2 = self.adain(content_images, style_feats2)
            styled_images = torch.cat([styled_image1, styled_image2], dim=0)

            augmented_styled_images = self.dali_augmentations(styled_images)

            if self.cfg.augment.crop or self.cfg.augment.color or self.cfg.augment.blur:
                augmented_styled_images = self.dali_augmentations(styled_images)
        else:
            augmented_styled_images = torch.cat([self.dali_augmentations(im) for im in [batch[0], batch[0]]], dim=0)

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.axes_grid1 import ImageGrid
        # fig = plt.figure(figsize=(6., 6.))
        # grid = ImageGrid(fig, 111,  # similar to subplot(111)
        #                  nrows_ncols=(4, 4),  # creates 2x2 grid of axes
        #                  axes_pad=0.1)  # pad between axes in inch.
        #
        # for ax, im in zip(grid, augmented_styled_images[:16].permute(0, 2, 3, 1).detach().cpu().numpy()):
        #     ax.imshow(im)
        #     ax.set_axis_off()
        # plt.show()
        #
        #
        # for ax, im in zip(grid, styled_images[self.cfg.dataset.batch_size:16+self.cfg.dataset.batch_size].permute(0, 2, 3, 1).detach().cpu().numpy()):
        #     ax.set_axis_off()
        #     ax.imshow(im)
        # plt.show()
        #
        # for ax, im in zip(grid, styled_images[:16].permute(0, 2, 3, 1).detach().cpu().numpy()):
        #     ax.set_axis_off()
        #     ax.imshow(im)
        # plt.show()
        #
        # if self.cfg.augment.background_replacer:
        #     styled_images = self.background(content_images, styled_images, transparency)

        # augmented views contrastive setup
        features = self.model(augmented_styled_images)
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

    @pipeline_def()
    def augmentations_pipeline(self, cuda_images):
        # get current pipeline information
        pipe = Pipeline.current()
        images = fn.external_source(source=cuda_images, device='gpu', batch=True, cuda_stream=0, dtype=types.FLOAT)

        # crop, resize and flip
        if self.cfg.augment.crop:
            images = fn.random_resized_crop(images, size=self.cfg.augment.size) if self.cfg.augment.crop else images
            images = fn.flip(images, horizontal=1, vertical=0) if torch.rand(1) < 0.5 and self.cfg.augment.crop else images

        # sample color jitter transformations. brightness, constrast, saturation and hue, with an certain probability.
        if self.cfg.augment.color:
            # TODO: move the range parameters to the config file. [0.8, 0.8, 0.8, 0.2]
            bcs_range, h_range, augment_prob = 0.8, 0.2, 0.8
            color_jitter_parameters = torch.ones((4, pipe.max_batch_size), device=pipe.device_id)
            bcsh_mask = torch.cuda.FloatTensor(pipe.max_batch_size).uniform_() < augment_prob  # sample cases to augment
            color_jitter_parameters[:3, bcsh_mask] = torch.distributions.uniform.Uniform(1 - bcs_range, 1 + bcs_range).sample([3, bcsh_mask.sum()]).to(pipe.device_id)

            # set non-sampled images to hue to zero.
            color_jitter_parameters[3, bcsh_mask] = torch.distributions.uniform.Uniform(-h_range, h_range).sample((bcsh_mask.sum(),)).to(pipe.device_id)
            color_jitter_parameters[3, ~bcsh_mask] = 0

            print(images, images.shape)
            # need to sample random mask to emulate the probability of not doing anything.
            images = fn.color_twist(images,
                                    brightness=color_jitter_parameters,
                                    contrast=0.8,
                                    saturation=0.8,
                                    hue=0.2,
                                    # brightness=color_jitter_parameters[0].unsqueeze(1),
                                    # contrast=color_jitter_parameters[1].unsqueeze(1),
                                    # saturation=color_jitter_parameters[2].unsqueeze(1),
                                    # hue=color_jitter_parameters[3].unsqueeze(1))
                                    )

            # to emulate randomGrayScale we need to use HSV and coin flip
            grayscale_prob = 0.2
            saturate = fn.random.coin_flip(probability=1 - grayscale_prob)
            saturate = fn.cast(saturate, dtype=types.FLOAT)
            images = fn.hsv(images, saturation=saturate)

        # blur
        if self.cfg.augment.blur:
            blur_prob = 0.5
            sigma_range = [0.1, 2]
            blur_parameters = torch.zeros((pipe.max_batch_size, 1), device=pipe.device_id)
            sigma_mask = torch.cuda.FloatTensor(pipe.max_batch_size).uniform_() < blur_prob

            # sample sigmas
            blur_parameters[sigma_mask, 0] = torch.distributions.uniform.Uniform(sigma_range[0], sigma_range[1]).sample((sigma_mask.sum(),)).to(pipe.device_id)

            print(blur_parameters, blur_parameters.shape)
            images = fn.gaussian_blur(images, sigma=0.5, window_size=int(0.1 * self.cfg.augment.size))
        return images

    def dali_augmentations(self, images):
        batch_size, c, h, w = images.shape
        # iterator to feed to pipeline
        eii = ExternalInputGPUIterator(images)

        # build augmentation pipeline
        augment = self.augmentations_pipeline(eii, batch_size=batch_size, num_threads=8, device_id=self.trainer.root_gpu)
        augment.build()

        # run augmentation and reshape output to original shape
        output_images = augment.run()[0]
        output_images = output_images.as_tensor()

        # populate vector with augmented images
        augmented_images = torch.zeros((batch_size, h, w, c), dtype=torch.float, device=images.device)
        feed_ndarray(output_images, augmented_images)

        # to [0, 1] range
        augmented_images = augmented_images.permute(0, 3, 1, 2) / 255.

        # repeat channel for BW images
        if augmented_images.shape[1] == 1:
            augmented_images = augmented_images.repeat(1, 3, 1, 1)
        #
        # ############## DEBUGGING##############
        # import matplotlib.gridspec as gridspec
        # import matplotlib.pyplot as plt
        #
        # def show_images(image_batch):
        #     columns = 4
        #     rows = (4)
        #     fig = plt.figure(figsize=(24, (24 // columns) * rows))
        #     gs = gridspec.GridSpec(rows, columns)
        #     for j in range(rows * columns):
        #         plt.subplot(gs[j])
        #         plt.axis("off")
        #         plt.imshow(image_batch[j])
        #     plt.show()
        #
        # show_images(augmented_images[:16].detach().cpu().permute(0, 2, 3, 1).numpy())

        # plt.imshow((augmented_images[1].permute(1, 2, 0)).cpu().detach())
        # plt.show()
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
