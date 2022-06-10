import torch

import torch.nn as nn

# decoder architecure for Style transfer (AdaIn)
decoder = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(512, 256, (3, 3)),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 256, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 256, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 256, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 128, (3, 3)),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(128, 128, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(128, 64, (3, 3)),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(64, 64, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(64, 3, (3, 3)))

# VGG architecure for Style transfer (AdaIn)
vgg = nn.Sequential(nn.Conv2d(3, 3, (1, 1)),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(3, 64, (3, 3)),
                    nn.ReLU(),  # relu1-1
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64, 64, (3, 3)),
                    nn.ReLU(),  # relu1-2
                    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64, 128, (3, 3)),
                    nn.ReLU(),  # relu2-1
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128, 128, (3, 3)),
                    nn.ReLU(),  # relu2-2
                    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128, 256, (3, 3)),
                    nn.ReLU(),  # relu3-1
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 256, (3, 3)),
                    nn.ReLU(),  # relu3-2
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 256, (3, 3)),
                    nn.ReLU(),  # relu3-3
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 256, (3, 3)),
                    nn.ReLU(),  # relu3-4
                    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 512, (3, 3)),
                    nn.ReLU(),  # relu4-1, this is the last layer used
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512, 512, (3, 3)),
                    nn.ReLU(),  # relu4-2
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512, 512, (3, 3)),
                    nn.ReLU(),  # relu4-3
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512, 512, (3, 3)),
                    nn.ReLU(),  # relu4-4
                    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512, 512, (3, 3)),
                    nn.ReLU(),  # relu5-1
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512, 512, (3, 3)),
                    nn.ReLU(),  # relu5-2
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512, 512, (3, 3)),
                    nn.ReLU(),  # relu5-3
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512, 512, (3, 3)),
                    nn.ReLU())  # relu5-4


# TODO: this file needs docstrings.
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    u, d, v = torch.svd(x)
    return torch.mm(torch.mm(u, d.pow(0.5).diag()), v.t())
