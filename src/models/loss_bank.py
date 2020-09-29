import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from functools import partial
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import constants as c
import util


def get_criterion(name):
    if name in ('cross_entropy', 'ce'):
        return nn.CrossEntropyLoss()
    elif name in ('plate_cross_entropy', 'pce'):
        return PlateCrossEntropyLoss()
    elif name in ('arc_face_loss', 'afl'):
        return ArcFaceLoss()
    elif name in ('focal_loss', 'fl'):
        return partial(focal_loss, alpha=0.5)
    elif name in ('label_smoothing', 'ls'):
        return LabelSmoothingLoss(label_smoothing=0.025)


def focal_loss(logits, labels, gamma=2.0, alpha=0.5):
    epsilon = 1.e-9
    gamma = torch.tensor(gamma).to(logits.device)

    input_soft = F.softmax(logits, dim=1) + epsilon
    target_one_hot = torch.zeros_like(logits, dtype=logits.dtype)
    target_one_hot[torch.arange(labels.size(0)), labels] = 1.0

    weight = torch.pow(torch.tensor(1.) - input_soft, gamma.to(logits.dtype))
    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)
    loss = torch.mean(loss_tmp)
    return loss


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=32.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss = nn.CrossEntropyLoss()(output, labels)
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        output1, output2 = output
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        margin_distance = torch.clamp(self.margin - euclidean_distance, min=0.0)
        loss_contrastive = torch.mean(
            (1 - label).float() * torch.pow(euclidean_distance, 2) + label.float() * torch.pow(margin_distance, 2))
        return loss_contrastive


class LabelSmoothingLoss(nn.Module):
    """
    Modified from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py#L33
    With label smoothing,
    """
    def __init__(self, label_smoothing):
        # label_smoothing=0.1 from bag of tricks from image classification with CNN
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        # smooth by plate
        self.smoothing_value = label_smoothing / (c.N_CLASS // 4 - 1)
        self.confidence = 1.0 - label_smoothing
        self.g2rna, self.masks = util.get_g2rna()

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        log_output = F.log_softmax(output, dim=1)
        target = target.contiguous().view(-1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * self.confidence + (1 - one_hot) * self.smoothing_value

        for i, val in enumerate(target.cpu().numpy()):
            group = -1
            for j in range(4):
                if val in self.g2rna[j]:
                    group = j
                    break
            one_hot[i, self.masks[group]] = 0.0

        loss = -(one_hot * log_output).sum(dim=1)
        return torch.mean(loss)


class PlateCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PlateCrossEntropyLoss, self).__init__()
        self.g2rna, _ = util.get_g2rna()
        self.masks = []
        for val in self.g2rna:
            self.masks.append(torch.tensor(sorted(val)))

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        target = target.contiguous().view(-1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)

        new_output = []
        new_one_hot = []
        for i, val in enumerate(target.cpu().numpy()):
            group = -1
            for j in range(4):
                if val in self.g2rna[j]:
                    group = j
                    break
            new_output.append(output[i, self.masks[group]])
            new_one_hot.append(one_hot[i, self.masks[group]])

        new_output = torch.stack(new_output)
        new_one_hot = torch.stack(new_one_hot)
        log_output = F.log_softmax(new_output, dim=1)
        loss = -(new_one_hot * log_output).sum(dim=1)
        return torch.mean(loss)


class GradualWarmupScheduler(_LRScheduler):
    """
    Copy from: https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)

    Testing code:
    v = torch.zeros(10)
    optimizer = torch.optim.Adam([v], lr=0.001)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-6)
    scheduler_warmup = lb.GradualWarmupScheduler(optimizer, multiplier=5, total_epoch=5, after_scheduler=scheduler_cosine)

    for epoch in range(1, 50):
        print(epoch, optimizer.param_groups[0]['lr'])
        scheduler_warmup.step(epoch)
    print(epoch, optimizer.param_groups[0]['lr'])
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.last_epoch = None
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
