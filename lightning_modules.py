from abc import ABC
from inspect import classify_class_attrs

from typing import Tuple

import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import feed_ndarray
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch.nn import functional as F, Sequential

from dataset import ContentImageDataset
from omegaconf import DictConfig
from data_aug import adain
from data_aug.adain import adaptive_instance_normalization
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
        self.style_vgg = adain.vgg
        self.style_decoder = adain.decoder

        # load VGG pretrained weights
        print('Loading VGG and decoder weights.')
        self.style_vgg.load_state_dict(torch.load(self.cfg.model.style_vgg_path))
        self.style_decoder.load_state_dict(torch.load(self.cfg.model.style_decoder_path))

        # drop last backbone layers
        self.style_vgg = Sequential(*list(self.style_vgg.children())[:31])

        # set to eval
        self.style_vgg = self.style_vgg.eval()
        self.style_decoder = self.style_decoder.eval()

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

        styled_images = content_images

        ######################
        # Augment the images #
        ######################

        if self.cfg.augment.adain:
            with torch.no_grad():
                assert (0.0 <= self.alpha <= 1.0)
                content_feats = self.style_vgg(styled_images)

                styled_images = adaptive_instance_normalization(content_feats, style_feats)
                styled_images = styled_images * self.alpha + content_feats * (1 - self.alpha)

                styled_images = self.style_decoder(styled_images)

        if self.cfg.augment.background_remover:
            with torch.no_grad():
                transparency = torch.cat([transparency for _ in range(2)], dim=0)
                styled_images = transparency * content_images + (1 - transparency) * styled_images

        if self.cfg.augment.crop or self.cfg.augment.color or self.cfg.augment.blur:
            styled_images = self.data_augmentation(styled_images)


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


    def data_augmentation(self, styled_images):
        class ExternalInputGPUIterator(object):
            def __init__(self, images):
                self.images = 255*images.permute(0,2,3,1).contiguous() 

            def __iter__(self):
                self.i=0
                self.n=self.images.shape[0]
                return self

            def __next__(self):
                return [self.images[i,:,:,:].type(torch.uint8) for i in range(self.n)]

        eii = ExternalInputGPUIterator(styled_images)
        pipe = Pipeline(batch_size=self.cfg.dataset.batch_size*2, num_threads=1, device_id=self.cfg.dataset.style.device)
        with pipe:
            styled_image = fn.external_source(source=eii, device='gpu', batch=True, cuda_stream=0, dtype=types.UINT8)
            styled_image = fn.random_resized_crop(styled_image, size=self.cfg.augment.size) if self.cfg.augment.crop else styled_image
            styled_image = fn.flip(styled_image, horizontal=1, vertical=0) if torch.rand(1)<0.5 and self.cfg.augment.crop else styled_image
            b,c,s = torch.distributions.uniform.Uniform(1-0.8, 1+0.8).sample([3,])
            h = torch.distributions.uniform.Uniform(-0.2, 0.2).sample([1,])
            styled_image = fn.color_twist(styled_image, brightness=b, contrast=c, saturation=s, hue=h) if torch.rand(1)<0.8 and self.cfg.augment.color else styled_image #only accept hwc
            styled_image = fn.color_space_conversion(styled_image, image_type=types.RGB, output_type=types.GRAY) if torch.rand(1)<0.2 and self.cfg.augment.color else styled_image #only accept hwc, uint8
            styled_image = fn.gaussian_blur(styled_image, window_size=int(0.1*self.cfg.augment.size)) if self.cfg.augment.blur else styled_image
            pipe.set_outputs(styled_image)
        pipe.build()
        styled_image=pipe.run()
        
        styled_image = styled_image[0].as_tensor() # type:TensorGPU
        
        styled_augmented_images = torch.zeros(styled_image.shape(), dtype=torch.uint8).cuda()
        feed_ndarray(styled_image, styled_augmented_images)
        styled_augmented_images = styled_augmented_images.permute(0,3,1,2).type(styled_images.dtype)/225.
        c=styled_augmented_images.shape[1]
        if c==1:
            styled_augmented_images = styled_augmented_images.repeat(1,3,1,1)

        return styled_augmented_images


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
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     self.cfg.optimizer.lr,
                                     weight_decay=self.cfg.optimizer.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.cfg.dataset.len_train_loader,
                                                               eta_min=0,
                                                               last_epoch=-1)
        return [optimizer], [{"scheduler": scheduler}]





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
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     self.cfg.optimizer.lr,
                                     weight_decay=self.cfg.optimizer.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.cfg.dataset.len_train_loader,
                                                               eta_min=0,
                                                               last_epoch=-1)
        return [optimizer], [{"scheduler": scheduler}]
