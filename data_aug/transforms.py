import torch

import nvidia.dali.fn as fn
import torchvision
from nvidia.dali.plugin.pytorch import feed_ndarray
from nvidia.dali import pipeline_def
import nvidia.dali.types as types
from torch import nn
import numpy as np

from data_aug.adain import adaptive_instance_normalization


def adain(vgg, decoder, alpha):
    assert (0.0 <= alpha <= 1.0)

    def return_function(content_images, style_feats):
        with torch.no_grad():
            content_feats = vgg(content_images)

            styled_images = adaptive_instance_normalization(content_feats, style_feats)
            styled_images = styled_images * alpha + content_feats * (1 - alpha)

            return decoder(styled_images)

    return return_function


def background():
    def return_function(content_images, styled_images, transparency):
        with torch.no_grad():
            transparency = torch.cat([transparency for _ in range(2)], dim=0)
            return transparency * content_images + (1 - transparency) * styled_images

    return return_function


class ExternalInputGPUIterator(object):
    def __init__(self, images):
        self.images = 255 * images.permute(0, 2, 3, 1).contiguous()

    def __iter__(self):
        self.i = 0
        self.n = self.images.shape[0]
        return self

    def __next__(self):
        return [self.images[i, :, :, :].type(torch.uint8) for i in range(self.n)]


def dali(cfg, do_crop, do_color, do_blur):
    @pipeline_def(batch_size=cfg.dataset.batch_size * 2, num_threads=1, device_id=cfg.dataset.style.device)
    def augment_pipeline(cuda_images):
        images = fn.external_source(source=cuda_images, device='gpu', batch=True, cuda_stream=0, dtype=types.UINT8)
        images = fn.random_resized_crop(images, size=cfg.augment.size) if do_crop else images
        images = fn.flip(images, horizontal=1, vertical=0) if torch.rand(1) < 0.5 and do_crop else images
        b, c, s = torch.distributions.uniform.Uniform(1 - 0.8, 1 + 0.8).sample([3, ])
        h = torch.distributions.uniform.Uniform(-0.2, 0.2).sample([1, ])
        images = fn.color_twist(images, brightness=b, contrast=c, saturation=s, hue=h) if torch.rand(1) < 0.8 and do_color else images  # only accept hwc
        images = fn.color_space_conversion(images, image_type=types.RGB, output_type=types.GRAY) if torch.rand(
            1) < 0.2 and do_color else images  # only accept hwc, uint8
        images = fn.gaussian_blur(images, window_size=int(0.1 * cfg.augment.size)) if do_blur else images
        return images

    def return_function(images):
        input_dtype = images.dtype

        eii = ExternalInputGPUIterator(images)
        augment = augment_pipeline(eii)
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

    return return_function


class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(size=size),
                                                               torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                                                               torchvision.transforms.RandomApply([color_jitter], p=0.8),
                                                               torchvision.transforms.RandomGrayscale(p=0.2),
                                                               GaussianBlur(kernel_size=int(0.1 * size)),
                                                               torchvision.transforms.ToTensor()])

        # do we need to also add the ratios as in the paper?
        self.test_transform = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(size=size),
                                                              torchvision.transforms.RandomHorizontalFlip(),
                                                              torchvision.transforms.ToTensor()])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1), stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size), stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = torchvision.transforms.ToTensor()
        self.tensor_to_pil = torchvision.transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img
