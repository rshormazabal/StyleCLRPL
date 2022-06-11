#######
#      Taken from https://github.com/sthalles/SimCLR.
#######
import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):
    def __init__(self, model_cfg):
        super(ResNetSimCLR, self).__init__()
        self.model_cfg = model_cfg
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=model_cfg.out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=model_cfg.out_dim)}

        self.backbone = self._get_basemodel(model_cfg.base_model)

        if model_cfg.smaller_base:
            # See simclr paper about cifar-10 decrease in size
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.backbone.maxpool = nn.Identity()

        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


class ResNetDownStream(nn.Module):
    def __init__(self, cfg):
        super(ResNetDownStream, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=cfg.probe.num_classes),
                            "resnet50": models.resnet50(pretrained=False, num_classes=cfg.probe.num_classes)}

        self.backbone = self._get_basemodel(cfg.model.base_model)

        if cfg.model.smaller_base:
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=False)
            self.backbone.maxpool = nn.Identity()

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)

    def get_params_from_resnetsimclr(self, model: ResNetSimCLR):

        self.load_state_dict(model.state_dict(), strict=False)

    def freeze_conv_params(self):

        for name, param in self.named_parameters():
            if name not in ['backbone.fc.weight', 'backbone.fc.bias']:
                param.requires_grad = False