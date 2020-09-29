import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import MULRESNET

import constants as c
import efficientnet as mish_efficientnet


def init_cnn(m, leak=0):
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    if getattr(m, 'bias', None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=leak)
    for l in m.children():
        init_cnn(l)


class EfficientNet(nn.Module):
    def __init__(self, model_name, n_class, norm_layer, relu_fn):
        super(EfficientNet, self).__init__()
        # mishefficientnet-b0 to mishefficientnet-b7
        assert model_name.startswith('mishefficientnet') or model_name.startswith('efficientnet')
        if model_name.startswith('mishefficientnet'):
            model_name = model_name[4:]
        self.model = mish_efficientnet.EfficientNet.from_pretrained(model_name, norm_layer, relu_fn,
                                                                    num_classes=n_class)

        _conv_stem = self.model._conv_stem
        Conv2d = mish_efficientnet.utils.get_same_padding_conv2d(image_size=256)
        self.model._conv_stem = Conv2d(6, _conv_stem.out_channels,
                                       kernel_size=_conv_stem.kernel_size,
                                       stride=_conv_stem.stride,
                                       bias=_conv_stem.bias)
        self.model._conv_stem.weight.data[:, :3, :, :] = _conv_stem.weight.data
        self.model._conv_stem.weight.data[:, 3:6, :, :] = _conv_stem.weight.data

        num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(num_ftrs, n_class)

    def forward(self, x):
        return self.model(x)


class ArcEfficientNet(EfficientNet):
    def __init__(self, model_name, n_class, norm_layer, relu_fn):
        super(ArcEfficientNet, self).__init__(model_name, n_class, norm_layer, relu_fn)
        # efficientnet-b0 to efficientnet-b7
        num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(num_ftrs, 512)
        # self.model._bn2 = nn.BatchNorm1d(512)
        self.arc_margin_product = ArcMarginProduct(512, n_class)

    def forward(self, x, extract_feature=False):
        features = self.model(x)
        # self.model._bn2(features)
        cosine = self.arc_margin_product(features)
        if extract_feature:
            return cosine, features
        else:
            return cosine


class ArcMarginProduct(nn.Module):
    # https://github.com/wujiyang/Face_Pytorch/blob/master/margin/ArcMarginProduct.py
    def __init__(self, in_feature=512, out_feature=c.N_CLASS):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = nn.Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        return cosine


class Resnet(nn.Module):

    def __init__(self, layers=34, n_class=c.N_CLASS):
        super(Resnet, self).__init__()
        if layers == 18:
            self.model = models.resnet18(pretrained=True)
        elif layers == 34:
            self.model = models.resnet34(pretrained=True)
        elif layers == 50:
            self.model = models.resnet50(pretrained=True)
        elif layers == 101:
            self.model = models.resnet101(pretrained=True)
        elif layers == 152:
            self.model = models.resnet152(pretrained=True)
        else:
            raise Exception('No such model resnet%d' % layers)

        conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(in_channels=6,
                                     out_channels=conv1.out_channels,
                                     kernel_size=conv1.kernel_size,
                                     stride=conv1.stride,
                                     padding=conv1.padding,
                                     bias=conv1.bias)

        # copy pretrained weights
        self.model.conv1.weight.data[:, :3, :, :] = conv1.weight.data
        self.model.conv1.weight.data[:, 3:6, :, :] = conv1.weight.data
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_class)

    def forward(self, x):
        return self.model.forward(x)

    def activation_func(self):
        def get_activation(out_ftr):
            return out_ftr.reshape(out_ftr.size(0), -1).data
        return get_activation

    def register_feature_hook(self, hook):
        self.model.avgpool.register_forward_hook(hook)

    def fc_weight(self):
        return self.model.fc.weight.data, self.model.fc.bias.data


class Resnet2in2out(nn.Module):

    def __init__(self, layers=34, n_class=c.N_CLASS):
        super(Resnet2in2out, self).__init__()
        if layers == 18:
            self.model = MULRESNET.resnet18(pretrained=True)
        elif layers == 34:
            self.model = MULRESNET.resnet34(pretrained=True)
        elif layers == 50:
            self.model = MULRESNET.resnet50(pretrained=True)
        elif layers == 101:
            self.model = MULRESNET.resnet101(pretrained=True)
        elif layers == 152:
            self.model = MULRESNET.resnet152(pretrained=True)
        else:
            raise Exception('No such model resnet%d' % layers)

        conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(in_channels=6,
                                     out_channels=conv1.out_channels,
                                     kernel_size=conv1.kernel_size,
                                     stride=conv1.stride,
                                     padding=conv1.padding,
                                     bias=conv1.bias)

        # copy pretrained weights
        self.model.conv1.weight.data[:, :3, :, :] = conv1.weight.data
        self.model.conv1.weight.data[:, 3:6, :, :] = conv1.weight.data
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_class)
        self.model.fc2 = nn.Linear(num_ftrs, c.N_CLASS_CONTROL)

    def forward(self, x, x2):
        return self.model.forward(x, x2)


class Densenet(nn.Module):

    def __init__(self, layers=121, n_class=c.N_CLASS):
        super(Densenet, self).__init__()
        if layers == 121:
            self.model = models.densenet121(pretrained=True)
        elif layers == 161:
            self.model = models.densenet161(pretrained=True)
        elif layers == 169:
            self.model = models.densenet169(pretrained=True)
        elif layers == 201:
            self.model = models.densenet201(pretrained=True)
        else:
            raise Exception('No such model densenet%d' % layers)

        conv0 = self.model.features.conv0
        self.model.features.conv0 = nn.Conv2d(in_channels=6,
                                              out_channels=conv0.out_channels,
                                              kernel_size=conv0.kernel_size,
                                              stride=conv0.stride,
                                              padding=conv0.padding,
                                              bias=conv0.bias)

        # copy pretrained weights
        self.model.features.conv0.weight.data[:, :3, :, :] = conv0.weight.data
        self.model.features.conv0.weight.data[:, 3:6, :, :] = conv0.weight.data
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, n_class)

    def forward(self, x):
        return self.model.forward(x)


class ResNeXt(nn.Module):

    def __init__(self, layers=50, n_class=c.N_CLASS):
        super(ResNeXt, self).__init__()
        if layers == 50:
            self.model = models.resnext50_32x4d(pretrained=True)
        elif layers == 101:
            self.model = models.resnext101_32x8d(pretrained=True)
        else:
            raise Exception('No such model resnext%d' % layers)

        conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(in_channels=6,
                                     out_channels=conv1.out_channels,
                                     kernel_size=conv1.kernel_size,
                                     stride=conv1.stride,
                                     padding=conv1.padding,
                                     bias=conv1.bias)

        # copy pretrained weights
        self.model.conv1.weight.data[:, :3, :, :] = conv1.weight.data
        self.model.conv1.weight.data[:, 3:6, :, :] = conv1.weight.data
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_class)

    def forward(self, x):
        return self.model.forward(x)


class FCN(nn.Module):

    def __init__(self, size, dims):
        super(FCN, self).__init__()
        self.layers = nn.ModuleList()
        if len(dims) > 0:
            self.layers.append(nn.Linear(size, dims[0]).double())
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i + 1]).double())
            self.layers.append(nn.Linear(dims[-1], size).double())
        else:
            self.layers.append(nn.Linear(size, size).double())

    def forward(self, mean_x, x):
        for i in range(len(self.layers)):
            mean_x = self.layers[i](mean_x)
        return torch.mm(x.double(), torch.t(mean_x))
