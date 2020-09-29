import collections
import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.cuda.random as trandom
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from tqdm import tqdm

import constants as c
import loss_bank as lb
import model_bank as mb
import optimizer_bank as ob
import util

from dataset import CellDataset, PairDataset, SubimageDataset, SubimageControlDataset, DataPrefetcher

cudnn.benchmark = True
cudnn.deterministic = False


def _get_plate_group(g2rna, id_codes, prediction):
    plate_group = dict()
    for id_code, pred in zip(id_codes, prediction):
        exp = id_code.split('_')[0]
        plate = int(id_code.split('_')[1])
        key = (exp, plate)
        if key not in plate_group:
            plate_group[key] = [0, 0, 0, 0]
        sirna = np.argmax(pred)
        for i in range(4):
            if sirna in g2rna[i]:
                plate_group[key][i] += 1
                break
    return plate_group


def get_plate_postprocessing(id_codes, prediction):
    g2rna, masks = util.get_g2rna()
    plate_group = _get_plate_group(g2rna, id_codes, prediction)

    for i, id_code in enumerate(id_codes):
        exp = id_code.split('_')[0]
        plate = int(id_code.split('_')[1])
        key = (exp, plate)
        group = np.argmax(plate_group[key])
        prediction[i, masks[group]] = -np.inf
        if prediction.shape[1] > c.N_CLASS:
            prediction[:, c.N_CLASS:] = -np.inf
    return prediction


def _balancing_label(prob):
    idxs = np.dstack(np.unravel_index(np.argsort(prob.ravel()), prob.shape))[0][::-1]
    pred = -np.ones(prob.shape[0])
    used_idx = np.zeros(prob.shape[0])
    used_rna = np.zeros(prob.shape[1])
    for idx in idxs:
        if used_idx[idx[0]] == 0 and used_rna[idx[1]] == 0:
            pred[idx[0]] = idx[1]
            used_idx[idx[0]] = 1
            used_rna[idx[1]] = 1
    return pred


def balancing_class_prediction(id_codes, prediction):
    # at most 1 instance each class
    prediction = get_plate_postprocessing(id_codes, prediction)
    plates = set()
    for id_code in id_codes:
        plate = '_'.join(id_code.split('_')[:2])
        plates.add(plate)
    plates = sorted(plates)

    y_pred = np.zeros(len(id_codes))
    for plate in plates:
        idx = [i for i, x in enumerate(id_codes) if x.startswith(plate)]
        y_pred_i = _balancing_label(prediction[idx])
        y_pred[idx] = y_pred_i

    return y_pred


class Model:

    def __init__(self, model_name='resnet', ckpt_path=None, ckpt_epoch=None,
                 ckpt_full_path=None, output_ckpt_path=None, cell_type=None, criterion='cross_entropy',
                 train_transform=list(), progress_func=tqdm, lr=0.0001, load_optimizer=True,
                 freeze_eval=True, precision=16, plate_group=None, train_control=False, optimizer='adam',
                 training=True, gaussian_sigma=0):
        assert torch.cuda.is_available()
        torch.manual_seed(c.SEED)
        trandom.manual_seed_all(c.SEED)

        self.freeze_eval = freeze_eval
        self.device = torch.device('cuda')
        self.progress_func = progress_func
        self.train_transform = train_transform
        self.eval_transform = []
        self.cell_type = cell_type
        self.plate_group = plate_group
        self.criterion = criterion
        self.train_control = train_control
        self.gaussian_sigma = gaussian_sigma

        if train_control:
            n_class = c.N_CLASS + c.N_CLASS_CONTROL
        else:
            n_class = c.N_CLASS

        if model_name.startswith('resnet2in2out'):
            self.model = mb.Resnet2in2out(int(model_name[13:]), n_class)
        elif model_name.startswith('resnet'):
            self.model = mb.Resnet(int(model_name[6:]), n_class)
        elif model_name.startswith('arcresnet'):
            self.model = mb.Resnet(int(model_name[9:]), n_class)
        elif model_name.startswith('resnext'):
            self.model = mb.Resnet(int(model_name[7:]), n_class)
        elif model_name.startswith('densenet'):
            self.model = mb.Densenet(int(model_name[8:]), n_class)
        elif model_name.startswith('efficientnet'):
            if training:
                self.model = mb.EfficientNet(model_name, n_class, nn.BatchNorm2d, mb.mish_efficientnet.swish)
            else:
                self.model = mb.EfficientNet(model_name, n_class, mb.mish_efficientnet.MovingBatchNorm2d,
                                             mb.mish_efficientnet.swish)
        elif model_name.startswith('mishefficientnet'):
            if training:
                self.model = mb.EfficientNet(model_name, n_class, nn.BatchNorm2d, mb.mish_efficientnet.mish)
            else:
                self.model = mb.EfficientNet(model_name, n_class, mb.mish_efficientnet.MovingBatchNorm2d,
                                             mb.mish_efficientnet.mish)
        elif model_name.startswith('arcefficientnet'):
            self.model = mb.ArcEfficientNet(model_name[3:], n_class, nn.BatchNorm2d, mb.mish_efficientnet.swish)
        else:
            return

        self.model.cuda()
        # fixme: should be double - 64 bits, float - 32 bits, half - 16 bits
        if precision == 32:
            self.model.double()
        elif precision == 16:
            self.model.float()
        elif precision == 8:
            self.model.half()
        else:
            raise Exception('Precision %d not in (8, 16, 32)' % precision)
        self.precision = precision
        # training_params = []
        # for name, param in self.model.named_parameters():
        #     if 'fc' not in name:
        #         param.requires_grad = False
        #     training_params.append(param)
        if optimizer.lower().startswith('adam'):
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer.lower().startswith('ranger'):
            self.optimizer = ob.Ranger(self.model.parameters(), lr=lr)
        elif optimizer.lower().startswith('sgd'):
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=lr * 0.05, momentum=0.9)

        self.start_epoch = 0
        self.loss_history = {'train': [], 'valid': [], 'test': []}
        self.acc_history = {'train': [], 'valid': [], 'test': []}
        self.pp_acc_history = {'train': [], 'valid': []}
        self.ckpt_path = ckpt_path
        self.ckpt_full_path = ckpt_full_path
        self.ckpt_epoch = None
        self.output_ckpt_path = output_ckpt_path
        if output_ckpt_path:
            os.makedirs(output_ckpt_path, exist_ok=True)
        self._load_ckpt(ckpt_full_path, ckpt_path, ckpt_epoch, load_optimizer)
        if self.start_epoch == 0:
            logging.info('No checkpoint loaded.')

        if optimizer.endswith('swa'):
            self.optimizer = ob.StochasticWeightAverage(self.optimizer, swa_start=1, swa_freq=1, swa_lr=lr)

        g2rna, masks = util.get_g2rna()
        self.label2mask = []
        for i in range(c.N_CLASS):
            for j in range(4):
                if i in g2rna[j]:
                    self.label2mask.append(masks[j])
                    break
        assert len(self.label2mask) == c.N_CLASS

    def _load_ckpt(self, ckpt_full_path, ckpt_path, ckpt_epoch, load_optimizer=True):
        if ckpt_full_path is not None:
            path = ckpt_full_path
        elif ckpt_path is not None:
            cell_str = self.cell_type + '_' if self.cell_type else ''
            group_str = str(self.plate_group) + '_' if self.plate_group is not None else ''
            epoch_str = str(ckpt_epoch) if ckpt_epoch else 'best'
            path = os.path.join(ckpt_path, '%s%s%s.tar' % (cell_str, group_str, epoch_str))
            if not os.path.exists(path):
                path = os.path.join(ckpt_path, '%s%s.tar' % (cell_str, epoch_str))
        else:
            return False

        if os.path.exists(path):
            model_ckpt = torch.load(path)
            try:
                self.model.load_state_dict(model_ckpt['model'])
            except RuntimeError:
                weights = model_ckpt['model']
                new_weights = collections.OrderedDict()
                for k, v in weights.items():
                    new_weights['model.' + k] = v
                self.model.load_state_dict(new_weights)
            if load_optimizer:
                self.optimizer.load_state_dict(model_ckpt['optimizer'])
            self.start_epoch = model_ckpt['epoch'] + 1
            self.ckpt_epoch = model_ckpt['epoch']
            self.loss_history = model_ckpt['loss']
            self.acc_history = model_ckpt['acc']
            if 'pp_acc' in model_ckpt:
                self.pp_acc_history = model_ckpt['pp_acc']
            logging.info('Check point %s loaded', path)
            return True
        elif ckpt_path is not None:
            os.makedirs(ckpt_path, exist_ok=True)
        return False

    def _save_ckpt(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss_history,
            'acc': self.acc_history,
            'pp_acc': self.pp_acc_history,
        }, path)

    def _forward_batch(self, images, labels):
        if len(images.size()) == 5:
            B, T, C, H, W = images.size()
            outputs = self.model(images.view(-1, C, H, W))
            if labels is not None:
                labels = labels.view(-1)
        else:
            T = 1
            outputs = self.model(images)
        return outputs, labels, T

    def _predict_batch(self, images):
        if len(images.size()) == 5:
            B, T, C, H, W = images.size()
            outputs = self.model(images.view(-1, C, H, W))
            outputs = outputs.view(B, T, -1)
            outputs = outputs.mean(dim=1)
        else:
            outputs = self.model(images)
        return outputs

    def _train_epoch(self, dataloader, criterion):
        running_loss = 0.0
        running_corrects = 0

        self.model.train()
        prefetcher = DataPrefetcher(dataloader)
        images, labels = prefetcher.next()
        for _ in self.progress_func(range(len(dataloader))):
            # zero the parameter gradients
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs, labels, T = self._forward_batch(images, labels)

                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                loss.backward()
                self.optimizer.step()

            # statistics
            running_loss += loss.item() * labels.size(0) / T
            running_corrects += torch.sum(preds == labels).cpu().numpy() / T
            images, labels = prefetcher.next()
        assert images is None
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects / len(dataloader.dataset)
        return epoch_loss, epoch_acc

    def _eval_epoch(self, dataloader, criterion):
        running_loss = 0.0
        running_corrects = 0
        running_pp_corrects = 0

        if self.freeze_eval:
            self.model.eval()

        prefetcher = DataPrefetcher(dataloader)
        images, labels = prefetcher.next()
        for _ in self.progress_func(range(len(dataloader))):
            # forward
            with torch.set_grad_enabled(False):
                outputs = self._predict_batch(images)
                loss = criterion(outputs, labels)

            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            preds = np.argmax(outputs, axis=1)

            # statistics
            running_loss += loss.item() * labels.shape[0]
            running_corrects += np.sum(preds == labels)

            for i, l in enumerate(labels):
                # do not eval control data in eval_epoch for consistency
                if l < c.N_CLASS:
                    outputs[i, self.label2mask[l]] = -np.inf
            preds = np.argmax(outputs, axis=1)
            running_pp_corrects += np.sum(preds == labels)
            images, labels = prefetcher.next()
        assert images is None
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects / len(dataloader.dataset)
        epoch_pp_acc = running_pp_corrects / len(dataloader.dataset)
        return epoch_loss, epoch_acc, epoch_pp_acc

    def _eval_kld_epoch(self, dataloader, criterion):
        running_loss = 0.0
        running_kld_loss = 0.0
        running_corrects = 0
        running_pp_corrects = 0

        if self.freeze_eval:
            self.model.eval()
        for images1, images2, labels, masks in self.progress_func(dataloader):
            images1 = images1.to(self.device)
            images2 = images2.to(self.device)
            labels = labels.to(self.device)

            # forward
            with torch.set_grad_enabled(False):
                outputs1 = self._predict_batch(images1)
                outputs2 = self._predict_batch(images2)
                outputs = ((outputs1 + outputs2) / 2)
                loss = criterion(outputs, labels)

                for i, mask in enumerate(masks):
                    outputs1[i, mask] = -np.inf
                    outputs2[i, mask] = -np.inf
                kld_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs1, dim=1),
                                                               F.softmax(outputs2, dim=1))
                kld_loss += nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs2, dim=1),
                                                                F.softmax(outputs1, dim=1))
                kld_loss /= 2

            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            preds = np.argmax(outputs, axis=1)

            # statistics
            running_loss += loss.item() * labels.shape[0]
            running_kld_loss += kld_loss.item() * labels.shape[0]
            running_corrects += np.sum(preds == labels)

            for i, l in enumerate(labels):
                # do not eval control data in eval_epoch for consistency
                if l < c.N_CLASS:
                    outputs[i, self.label2mask[l]] = -np.inf
            preds = np.argmax(outputs, axis=1)
            running_pp_corrects += np.sum(preds == labels)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_kld_loss = running_kld_loss / len(dataloader.dataset)
        epoch_acc = running_corrects / len(dataloader.dataset)
        epoch_pp_acc = running_pp_corrects / len(dataloader.dataset)
        return epoch_loss, epoch_kld_loss, epoch_acc, epoch_pp_acc

    @staticmethod
    def _get_instance_weight(train_files):
        exp_count = dict()
        for f in train_files:
            exp_now = ''
            for exp in c.EXPS:
                if exp in f:
                    exp_now = exp
                    break
            if exp_now not in exp_count:
                exp_count[exp_now] = 0
            exp_count[exp_now] += 1

        weights = []
        for f in train_files:
            exp_now = ''
            for exp in c.EXPS:
                if exp in f:
                    exp_now = exp
                    break
            weights.append(1 / exp_count[exp_now])
        return weights

    def get_best_epoch(self, valid_exps):
        best_epoch = [-1] * len(valid_exps)
        best_loss = [np.inf] * len(valid_exps)
        for i, loss_dict in enumerate(self.loss_history['valid']):
            for j, exp in enumerate(valid_exps):
                if isinstance(loss_dict, dict):
                    loss = loss_dict[exp]
                else:
                    loss = loss_dict
                if loss < best_loss[j]:
                    best_loss[j] = loss
                    best_epoch[j] = i
        return best_loss, best_epoch

    def train(self, train_files, train_labels, train_stats, valid_files, valid_labels, valid_stats,
              test_files, test_labels, test_stats,
              epochs=10, patient=5, batch_size=32, num_workers=6, valid_exps=c.EXPS, dataset_class=CellDataset,
              balance_exp=False, eval_batch_size=32, eval_bn_batch_size=0, restore_loss=True):

        tw = 0
        for exp in valid_exps:
            tw += c.TEST_COUNT[exp]

        if restore_loss:
            best_loss, best_epoch = self.get_best_epoch(valid_exps)
        else:
            best_epoch = [-1] * len(valid_exps)
            best_loss = [np.inf] * len(valid_exps)

        train_dataset = dataset_class(train_files, train_labels, train_stats, self.train_transform, 'train',
                                      gaussian_sigma=self.gaussian_sigma)
        if balance_exp:
            sampler = WeightedRandomSampler(Model._get_instance_weight(train_dataset.files), len(train_dataset))
        else:
            sampler = RandomSampler(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
                                  pin_memory=True)
        valid_loaders = dict()
        test_loaders = dict()
        bn_loaders = dict()
        for exp in valid_exps:
            idx = util.get_exp_index(exp, valid_files)
            valid_loaders[exp] = DataLoader(
                dataset_class(valid_files[idx], valid_labels[idx], np.array(valid_stats)[idx], self.eval_transform,
                              'valid', gaussian_sigma=self.gaussian_sigma), batch_size=eval_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)
            if eval_bn_batch_size > 0:
                bn_loaders[exp] = DataLoader(
                    dataset_class(valid_files[idx], valid_labels[idx], np.array(valid_stats)[idx], self.eval_transform,
                                  'valid', gaussian_sigma=self.gaussian_sigma), batch_size=eval_bn_batch_size,
                    shuffle=False, num_workers=num_workers, pin_memory=True)

            idx = util.get_exp_index(exp, test_files)
            test_loaders[exp] = DataLoader(
                dataset_class(test_files[idx], test_labels[idx], np.array(test_stats)[idx], self.eval_transform,
                              'valid', gaussian_sigma=self.gaussian_sigma), batch_size=eval_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)

        # start swa after {swa_start_epoch} epochs
        swa_start_epoch = 5
        if isinstance(self.optimizer, ob.StochasticWeightAverage):
            epoch_steps = int(np.ceil(len(train_dataset) / batch_size))
            self.optimizer.set_swa_param(epoch_steps * swa_start_epoch, epoch_steps)
        criterion = lb.get_criterion(self.criterion)
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            train_loss, train_acc = self._train_epoch(train_loader, criterion)
            valid_loss_dict = dict()
            valid_acc_dict = dict()
            valid_pp_acc_dict = dict()
            test_loss_dict = dict()
            test_acc_dict = dict()

            if isinstance(self.optimizer, ob.StochasticWeightAverage) and epoch - self.start_epoch >= swa_start_epoch:
                logging.info('Update for SWA')
                self.optimizer.update_swa()
            if isinstance(self.optimizer, ob.StochasticWeightAverage) and self.optimizer.has_swa():
                logging.info('Swap SWA')
                self.optimizer.swap_swa_sgd()
                if self.freeze_eval:
                    logging.info('Update SWA BN params')
                    self.optimizer.bn_update(train_loader, self.model, device=self.device)

            for exp in valid_exps:
                valid_loss, valid_acc, valid_pp_acc = self._eval_epoch(valid_loaders[exp], criterion)
                valid_loss_dict[exp] = valid_loss
                valid_acc_dict[exp] = valid_acc
                valid_pp_acc_dict[exp] = valid_pp_acc
                if self.train_control:
                    self.model.eval()
                    test_loss, test_acc, test_pp_acc = self._eval_epoch(test_loaders[exp], criterion)
                    test_loss_dict[exp] = test_loss
                    test_acc_dict[exp] = test_acc

            valid_loss = np.sum([valid_loss_dict[exp] * c.TEST_COUNT[exp] / tw for exp in valid_exps])
            valid_acc = np.sum([valid_acc_dict[exp] * c.TEST_COUNT[exp] / tw for exp in valid_exps])
            valid_pp_acc = np.sum([valid_pp_acc_dict[exp] * c.TEST_COUNT[exp] / tw for exp in valid_exps])
            if self.train_control:
                test_loss = np.sum([test_loss_dict[exp] * c.TEST_COUNT[exp] / tw for exp in valid_exps])
                test_acc = np.sum([test_acc_dict[exp] * c.TEST_COUNT[exp] / tw for exp in valid_exps])
                logging.info('Epoch {} - Train / Valid / Test Loss: {:.6f} / {:.6f} / {:.6f}'.format(epoch, train_loss,
                                                                                                     valid_loss,
                                                                                                     test_loss))
                logging.info(
                    'Epoch {} - Train / Valid / Valid Plate / Test Acc: {:.4f}% / {:.4f}% / {:.4f}% / {:.4f}%'.format(
                        epoch, train_acc * 100, valid_acc * 100, valid_pp_acc * 100, test_acc * 100))
            else:
                logging.info('Epoch {} - Train / Valid Loss: {:.6f} / {:.6f}'.format(epoch, train_loss, valid_loss))
                logging.info(
                    'Epoch {} - Train / Valid / Valid Plate Acc: {:.4f}% / {:.4f}% / {:.4f}%'.format(
                        epoch, train_acc * 100, valid_acc * 100, valid_pp_acc * 100))

            for exp in valid_exps:
                logging.info('Epoch {} - {} Valid Loss / Acc / Plate Acc: {:.6f} / {:.4f}% / {:.4f}%'.format(epoch, exp,
                                                                                                             valid_loss_dict[
                                                                                                                 exp],
                                                                                                             valid_acc_dict[
                                                                                                                 exp] * 100,
                                                                                                             valid_pp_acc_dict[
                                                                                                                 exp] * 100))
            if self.train_control:
                for exp in valid_exps:
                    logging.info(
                        'Epoch {} - {} Test Loss / Acc: {:.6f} / {:.4f}%'.format(epoch, exp, test_loss_dict[exp],
                                                                                 test_acc_dict[exp] * 100))

            self.loss_history['train'].append(train_loss)
            self.acc_history['train'].append(train_acc)
            self.loss_history['valid'].append(valid_loss_dict)
            self.acc_history['valid'].append(valid_acc_dict)
            self.pp_acc_history['valid'].append(valid_pp_acc_dict)
            if self.train_control:
                self.loss_history['test'].append(test_loss_dict)
                self.acc_history['test'].append(test_acc_dict)

            # save best model
            if self.cell_type and self.output_ckpt_path:
                group_str = str(self.plate_group) + '_' if self.plate_group is not None else ''
                tar_name = os.path.join(self.output_ckpt_path, '%s_%s%d.tar' % (self.cell_type, group_str, epoch))
            else:
                tar_name = os.path.join(self.ckpt_path, '%d.tar' % epoch)
            self._save_ckpt(tar_name, epoch)
            best_loss_sum = np.sum([best_loss[i] * c.TEST_COUNT[exp] / tw for i, exp in enumerate(valid_exps)])
            if best_loss_sum > valid_loss:
                if self.cell_type and self.output_ckpt_path:
                    group_str = str(self.plate_group) + '_' if self.plate_group is not None else ''
                    tar_name = os.path.join(self.output_ckpt_path, '%s_%sbest.tar' % (self.cell_type, group_str))
                else:
                    tar_name = os.path.join(self.ckpt_path, 'best.tar')
                self._save_ckpt(tar_name, epoch)

            if isinstance(self.optimizer, ob.StochasticWeightAverage) and self.optimizer.has_swa():
                logging.info('Swap SWA')
                self.optimizer.swap_swa_sgd()
            if train_loss >= 2:
                _patient = patient * 2
            else:
                _patient = patient

            for i, exp in enumerate(valid_exps):
                if best_loss[i] > valid_loss_dict[exp]:
                    best_loss[i] = valid_loss_dict[exp]
                    best_epoch[i] = epoch
                    logging.info('%s best epoch %d loss %f', exp, epoch, valid_loss_dict[exp])
            if epoch - max(best_epoch) >= _patient:
                logging.info('Loss not improving for %d epochs! Break.', epoch - max(best_epoch))
                break
        return best_loss

    def get_swa_from_ckpts(self, train_files, train_labels, train_stats, valid_files, valid_labels, valid_stats,
                           ckpt_prefix, cell_type, first_epoch, last_epoch, epoch_step=5, batch_size=32, num_workers=6,
                           dataset_class=CellDataset, eval_batch_size=32):
        train_dataset = dataset_class(train_files, train_labels, train_stats, self.train_transform, 'train',
                                      gaussian_sigma=self.gaussian_sigma)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                  pin_memory=True)
        valid_loader = DataLoader(dataset_class(valid_files, valid_labels, np.array(valid_stats), self.eval_transform,
                                  'valid', gaussian_sigma=self.gaussian_sigma), batch_size=eval_batch_size,
                                  shuffle=False, num_workers=num_workers, pin_memory=True)

        best_loss = np.inf
        criterion = lb.get_criterion(self.criterion)
        swa_optimizer = ob.StochasticWeightAverage(self.optimizer, swa_start=1, swa_freq=1, swa_lr=1)
        for epoch in range(first_epoch, last_epoch + 1, epoch_step):
            ckpt_path = os.path.join('%s' % ckpt_prefix, '%s_%d.tar' % (cell_type, epoch))
            self._load_ckpt(ckpt_path, None, -1, False)
            swa_optimizer.update_swa()
            if self.freeze_eval:
                logging.info('Update SWA BN params')
                swa_optimizer.bn_update(train_loader, self.model, device=self.device)
            swa_optimizer.swap_swa_sgd()
            train_loss, train_acc, train_pp_acc = self._eval_epoch(train_loader, criterion)
            valid_loss, valid_acc, valid_pp_acc = self._eval_epoch(valid_loader, criterion)
            logging.info('Epoch {} - Train / Valid Loss: {:.6f} / {:.6f}'.format(epoch, train_loss, valid_loss))
            logging.info('Epoch {} - Train / Valid / Valid Plate Acc: {:.4f}% / {:.4f}% / {:.4f}%'.format(
                epoch, train_acc * 100, valid_acc * 100, valid_pp_acc * 100))

            self.loss_history['train'].append(train_loss)
            self.acc_history['train'].append(train_acc)
            self.loss_history['valid'].append({cell_type: valid_loss})
            self.acc_history['valid'].append({cell_type: valid_acc})
            self.pp_acc_history['valid'].append({cell_type: valid_pp_acc})

            if self.cell_type and self.output_ckpt_path:
                group_str = str(self.plate_group) + '_' if self.plate_group is not None else ''
                tar_name = os.path.join(self.output_ckpt_path, '%s_%s%d.tar' % (self.cell_type, group_str, epoch))
            else:
                tar_name = os.path.join(self.ckpt_path, '%d.tar' % epoch)
            self._save_ckpt(tar_name, epoch)
            if valid_loss < best_loss:
                best_loss = valid_loss
                if self.cell_type and self.output_ckpt_path:
                    group_str = str(self.plate_group) + '_' if self.plate_group is not None else ''
                    tar_name = os.path.join(self.output_ckpt_path, '%s_%sbest.tar' % (self.cell_type, group_str))
                else:
                    tar_name = os.path.join(self.ckpt_path, 'best.tar')
                self._save_ckpt(tar_name, epoch)
            swa_optimizer.swap_swa_sgd()
        return best_loss

    def eval_kld(self, files, stats, labels=None, batch_size=32):
        valid_loader = DataLoader(PairDataset(files, labels, stats, self.eval_transform, 'valid'),
                                  batch_size=batch_size, shuffle=False, num_workers=6)
        criterion = lb.get_criterion(self.criterion)
        loss, kld_loss, acc, pp_acc = self._eval_kld_epoch(valid_loader, criterion)
        logging.info('loss: %.6f', loss)
        logging.info('KLD loss: %.6f', kld_loss)
        logging.info('accuracy: %.4f%%', acc * 100)
        logging.info('pp accuracy: %.4f%%', pp_acc * 100)

    def predict(self, files, stats, labels=None, dataset='test', dataset_class=CellDataset, batch_size=32,
                eval_bn_batch_size=0, tta=(True, True, True, True)):
        dataloader = DataLoader(dataset_class(files, labels, stats, self.eval_transform, dataset, tta=tta),
                                batch_size=batch_size, shuffle=False, num_workers=6)
        if self.freeze_eval:
            self.model.eval()
        prediction = []
        labels = np.array([])
        for images, label in tqdm(dataloader):
            images = images.to(self.device)
            if self.precision == 8:
                images = images.half()
            with torch.set_grad_enabled(False):
                outputs = self._predict_batch(images)
            prediction.append(outputs.cpu().numpy())
            labels = np.concatenate((labels, label))
        return np.vstack(prediction), labels

    def saliency_map(self, files, stats, labels, output_dir, batch_size=32, dataset_class=SubimageDataset):
        dataloader = DataLoader(dataset_class(files, labels, stats, self.eval_transform, 'valid'),
                                batch_size=batch_size, shuffle=False, num_workers=6)

        i = 0
        corrects = 0.0
        self.model.eval()
        for images, label in tqdm(dataloader):
            images = images.to(self.device)
            label = label.to(self.device)
            if self.precision == 8:
                images = images.half()
            images.requires_grad = True
            B, T, C, H, W = images.size()
            images_flatten = images.view(-1, C, H, W)
            outputs = self.model(images_flatten)
            preds = outputs.detach().view(B, T, -1).mean(dim=1).cpu().numpy()
            raw_preds = outputs.detach().view(B, T, -1).cpu().numpy()

            label_flatten = label.repeat_interleave(T)
            outputs = outputs.gather(1, label_flatten.view(-1, 1)).squeeze()
            outputs.backward(torch.ones_like(outputs))

            for k, l in enumerate(label):
                # do not eval control data in eval_epoch for consistency
                if l < c.N_CLASS:
                    preds[k, self.label2mask[l]] = -np.inf
            preds = np.argmax(preds, axis=1)
            corrects += np.sum(preds == label.cpu().numpy())
            # save gradient image
            file_name = files[i:i + label.size(0)]
            # B, T, C, H, W to B, T, H, W, C
            grad_images = np.transpose(images.grad.cpu().numpy(), axes=(0, 1, 3, 4, 2))
            for j in range(label.size(0)):
                for idx in range(8):
                    path_i = file_name[j][idx // 4].replace('./data/train', output_dir).replace('//', '/')\
                        .replace('s%d' % (idx // 4 + 1), 's%d_%d' % (idx // 4 + 1, idx))
                    dir_i = path_i.rsplit('/', 1)[0]
                    os.makedirs(dir_i, exist_ok=True)
                    np.save(path_i, grad_images[j][idx])

                for idx in range(2):
                    path_i = file_name[j][idx].replace('./data/train', output_dir).replace('//', '/') \
                        .replace('s%d' % (idx + 1), 's%d_prob' % (idx + 1))
                    raw_pred_i = raw_preds[idx * 4:(idx + 1) * 4]
                    np.save(path_i, raw_pred_i)
            i += label.size(0)
        logging.info('Accuracy: %.4f%%', corrects / len(labels))

    def predict_proba_by_subimage(self, files, stats, labels, output_dir, batch_size=32, dataset_class=SubimageDataset):
        dataloader = DataLoader(dataset_class(files, labels, stats, self.eval_transform, 'valid'),
                                batch_size=batch_size, shuffle=False, num_workers=6)

        i = 0
        corrects = 0.0
        self.model.eval()
        for images, label in tqdm(dataloader):
            images = images.to(self.device)
            label = label.to(self.device)
            if self.precision == 8:
                images = images.half()
            with torch.set_grad_enabled(False):
                B, T, C, H, W = images.size()
                images_flatten = images.view(-1, C, H, W)
                outputs = self.model(images_flatten)
                preds = outputs.detach().view(B, T, -1).mean(dim=1).cpu().numpy()
                raw_preds = outputs.detach().view(B, T, -1).cpu().numpy()
            for k, l in enumerate(label):
                # do not eval control data in eval_epoch for consistency
                if l < c.N_CLASS:
                    preds[k, self.label2mask[l]] = -np.inf
            preds = np.argmax(preds, axis=1)
            corrects += np.sum(preds == label.cpu().numpy())
            # save gradient image
            file_name = files[i:i + label.size(0)]
            for j in range(label.size(0)):
                for idx in range(2):
                    path_i = file_name[j][idx].replace('./data/train', output_dir).replace('//', '/') \
                        .replace('s%d' % (idx + 1), 's%d_prob' % (idx + 1))
                    raw_pred_i = raw_preds[j, idx * 4:(idx + 1) * 4, :]
                    dir_i = path_i.rsplit('/', 1)[0]
                    os.makedirs(dir_i, exist_ok=True)
                    np.save(path_i, raw_pred_i)
            i += label.size(0)
        logging.info('Accuracy: %.4f%%', corrects / len(labels) * 100)

    def extract_feature(self, dataset_instance, batch_size=32):
        dataloader = DataLoader(dataset_instance, batch_size=batch_size, shuffle=False, num_workers=6)

        activation_func = self.model.activation_func()
        features = []

        def hook(module, in_ftr, out_ftr):
            features.append(activation_func(out_ftr).cpu().numpy())
            return None

        self.model.register_feature_hook(hook)

        if self.freeze_eval:
            self.model.eval()

        dl_labels = []
        for images, labels in tqdm(dataloader):
            images = images.to(self.device)
            with torch.set_grad_enabled(False):
                outputs, labels, _ = self._forward_batch(images, labels)
            dl_labels.append(labels.cpu().numpy())

        return np.vstack(features), np.concatenate(dl_labels)


class MulInOutModel(Model):
    def __init__(self, model_name='resnet', ckpt_path=None, ckpt_epoch=None,
                 ckpt_full_path=None, output_ckpt_path=None, cell_type=None, criterion='cross_entropy',
                 train_transform=list(), progress_func=tqdm, lr=0.0001, load_optimizer=True,
                 freeze_eval=True, labmda_control=0.1, precision=16):
        super(MulInOutModel, self).__init__(model_name=model_name, ckpt_path=ckpt_path, ckpt_epoch=ckpt_epoch,
                                            ckpt_full_path=ckpt_full_path, criterion=criterion,
                                            freeze_eval=freeze_eval, train_transform=train_transform,
                                            progress_func=progress_func, precision=precision,
                                            cell_type=cell_type, output_ckpt_path=output_ckpt_path)
        self.labmda_control = labmda_control
        self.loss_history['train2'] = []
        self.acc_history['train2'] = []

    def _train_epoch(self, dataloader, criterion):
        running_loss = 0.0
        running_corrects = 0
        running_loss2 = 0.0
        running_corrects2 = 0

        self.model.train()
        for images, labels, images2, labels2 in self.progress_func(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            images2 = images2.to(self.device)
            labels2 = labels2.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs, labels, T, outputs2, labels2, T2 = self._forward_batch(images, labels, images2, labels2)
                loss = criterion(outputs, labels)
                loss2 = criterion(outputs2, labels2)
                loss_total = loss + self.labmda_control * loss2
                _, preds = torch.max(outputs, 1)
                _, preds2 = torch.max(outputs2, 1)

                # backward + optimize only if in training phase
                loss_total.backward()
                self.optimizer.step()

            # statistics
            running_loss += loss.item() * labels.size(0) / T
            running_loss2 += loss2.item() * labels2.size(0) / T2
            running_corrects += torch.sum(preds == labels) / T
            running_corrects2 += torch.sum(preds2 == labels2) / T2

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_loss2 = running_loss2 / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        epoch_acc2 = running_corrects2.double() / len(dataloader.dataset)
        return epoch_loss, epoch_acc, epoch_loss2, epoch_acc2

    def _forward_batch(self, images, labels, images2, labels2):
        if len(images.size()) == 5:
            B, T, C, H, W = images.size()
            B2, T2, C2, H2, W2 = images2.size()
            outputs, outputs2 = self.model(images.view(-1, C, H, W), images2.view(-1, C2, H2, W2))
            labels = labels.view(-1)
            labels2 = labels2.view(-1)
        else:
            T = 1
            T2 = 1
            outputs, outputs2 = self.model(images, images2)
        return outputs, labels, T, outputs2, labels2, T2

    def _eval_epoch(self, dataloader, criterion):
        running_loss = 0.0
        running_corrects = 0
        running_loss2 = 0.0
        running_corrects2 = 0

        if self.freeze_eval:
            self.model.eval()
        for images, labels, images2, labels2 in self.progress_func(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            images2 = images2.to(self.device)
            labels2 = labels2.to(self.device)

            # forward
            with torch.set_grad_enabled(False):
                outputs, outputs2 = self._predict_batch(images, images2)
                loss = criterion(outputs, labels)
                loss2 = criterion(outputs2, labels2)
                loss_total = loss + self.labmda_control * loss2
                _, preds = torch.max(outputs, 1)
                _, preds2 = torch.max(outputs2, 1)

            # statistics
            running_loss += loss.item() * labels.size(0)
            running_corrects += torch.sum(preds == labels)
            running_loss2 += loss2.item() * labels2.size(0)
            running_corrects2 += torch.sum(preds2 == labels2)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        epoch_loss2 = running_loss2 / len(dataloader.dataset)
        epoch_acc2 = running_corrects2.double() / len(dataloader.dataset)
        return epoch_loss, epoch_acc, epoch_loss2, epoch_acc2

    def _predict_batch(self, images, images2):
        if len(images.size()) == 5:
            B, T, C, H, W = images.size()
            B2, T2, C2, H2, W2 = images2.size()
            outputs, outputs2 = self.model(images.view(-1, C, H, W), images2.view(-1, C2, H2, W2))
            outputs = outputs.view(B, T, -1)
            outputs = outputs.mean(dim=1)
            outputs2 = outputs2.view(B2, T2, -1)
            outputs2 = outputs2.mean(dim=1)
        else:
            outputs, outputs2 = self.model(images, images2)
        return outputs, outputs2

    def train(self, train_files, train_labels, train_stats, valid_files, valid_labels, valid_stats,
              train_ct_files, train_ct_labels, train_ct_stats, valid_ct_files, valid_ct_labels, valid_ct_stats,
              epochs=10, patient=5, batch_size=32, num_workers=6, valid_exps=c.EXPS,
              dataset_class=SubimageControlDataset,
              balance_exp=False, eval_batch_size=32, eval_bn_batch_size=0):

        tw = 0
        for exp in valid_exps:
            tw += c.TEST_COUNT[exp]

        best_epoch = -1
        best_loss = np.inf
        for i, loss_dict in enumerate(self.loss_history['valid']):
            if isinstance(loss_dict, dict):
                loss = np.sum([loss_dict[exp] * c.TEST_COUNT[exp] / tw for exp in valid_exps])
            else:
                loss = loss_dict
            if loss < best_loss:
                best_loss = loss
                best_epoch = i

        train_dataset = dataset_class(train_files, train_labels, train_stats,
                                      train_ct_files, train_ct_labels, train_ct_stats, self.train_transform, 'train')
        if balance_exp:
            sampler = WeightedRandomSampler(Model._get_instance_weight(train_dataset.files), len(train_dataset))
        else:
            sampler = RandomSampler(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
        valid_loaders = dict()
        bn_loaders = dict()
        for exp in valid_exps:
            idx = util.get_exp_index(exp, valid_files)
            idx_ct = util.get_exp_index(exp, valid_ct_files)
            valid_loaders[exp] = DataLoader(
                dataset_class(valid_files[idx], valid_labels[idx], np.array(valid_stats)[idx],
                              valid_ct_files[idx_ct], valid_ct_labels[idx_ct], np.array(valid_ct_stats)[idx_ct],
                              self.eval_transform,
                              'valid'), batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
            if eval_bn_batch_size > 0:
                bn_loaders[exp] = DataLoader(
                    dataset_class(valid_files[idx], valid_labels[idx], np.array(valid_stats)[idx],
                                  valid_ct_files[idx_ct], valid_ct_labels[idx_ct], np.array(valid_ct_stats)[idx_ct],
                                  self.eval_transform,
                                  'valid'), batch_size=eval_bn_batch_size, shuffle=False, num_workers=num_workers)

        criterion = lb.get_criterion(self.criterion)
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            train_loss, train_acc, train_loss2, train_acc2 = self._train_epoch(train_loader, criterion)
            valid_loss_dict = dict()
            valid_acc_dict = dict()
            for exp in valid_exps:
                valid_loss, valid_acc, valid_loss2, valid_acc2 = self._eval_epoch(valid_loaders[exp], criterion)
                valid_loss_dict[exp] = valid_loss
                valid_acc_dict[exp] = valid_acc
                valid_loss_dict[exp + '_control'] = valid_loss2
                valid_acc_dict[exp + '_control'] = valid_acc2

            valid_loss = np.sum([valid_loss_dict[exp] * c.TEST_COUNT[exp] / tw for exp in valid_exps])
            valid_acc = np.sum([valid_acc_dict[exp] * c.TEST_COUNT[exp] / tw for exp in valid_exps])
            valid_loss2 = np.sum([valid_loss_dict[exp + '_control'] * c.TEST_COUNT[exp] / tw for exp in valid_exps])
            valid_acc2 = np.sum([valid_acc_dict[exp + '_control'] * c.TEST_COUNT[exp] / tw for exp in valid_exps])
            logging.info('Epoch {} - Train / Valid Loss: {:.6f} / {:.6f}'.format(epoch, train_loss, valid_loss))
            logging.info(
                'Epoch {} - Train / Valid Acc: {:.4f}% / {:.4f}%'.format(epoch, train_acc * 100, valid_acc * 100))
            logging.info('Epoch {} - Train / Valid Loss2: {:.6f} / {:.6f}'.format(epoch, train_loss2, valid_loss2))
            logging.info(
                'Epoch {} - Train / Valid Acc2: {:.4f}% / {:.4f}%'.format(epoch, train_acc2 * 100, valid_acc2 * 100))

            for exp in valid_exps:
                logging.info('Epoch {} - {} Valid Loss / Acc: {:.6f} / {:.4f}%'.format(epoch, exp, valid_loss_dict[exp],
                                                                                       valid_acc_dict[exp] * 100))
                logging.info('Epoch {} - {} Valid Loss2 / Acc2: {:.6f} / {:.4f}%'.format(epoch, exp, valid_loss_dict[
                    exp + '_control'],
                                                                                         valid_acc_dict[
                                                                                             exp + '_control'] * 100))

            self.loss_history['train'].append(train_loss)
            self.acc_history['train'].append(train_acc)
            self.loss_history['train2'].append(train_loss2)
            self.acc_history['train2'].append(train_acc2)
            self.loss_history['valid'].append(valid_loss_dict)
            self.acc_history['valid'].append(valid_acc_dict)

            # save best model
            if self.cell_type and self.output_ckpt_path:
                group_str = str(self.plate_group) + '_' if self.plate_group is not None else ''
                tar_name = os.path.join(self.output_ckpt_path, '%s_%s%d.tar' % (self.cell_type, group_str, epoch))
            else:
                tar_name = os.path.join(self.ckpt_path, '%d.tar' % epoch)
            self._save_ckpt(tar_name, epoch)
            if best_loss > valid_loss:
                best_loss = valid_loss
                best_epoch = epoch
                if self.cell_type and self.output_ckpt_path:
                    group_str = str(self.plate_group) + '_' if self.plate_group is not None else ''
                    tar_name = os.path.join(self.output_ckpt_path, '%s_%sbest.tar' % (self.cell_type, group_str))
                else:
                    tar_name = os.path.join(self.ckpt_path, 'best.tar')
                self._save_ckpt(tar_name, epoch)

            if epoch - best_epoch >= patient:
                logging.info('Loss not improving for %d epochs! Break.', epoch - best_epoch)
                break
        return best_loss

    def predict(self, files, stats, labels=None, files_ct=None, stats_ct=None, labels_ct=None, dataset='test',
                dataset_class=CellDataset, batch_size=32,
                eval_bn_batch_size=0, tta=(True, True, True, True)):
        if eval_bn_batch_size > 0:
            dataloader = DataLoader(dataset_class(files, labels, stats, files_ct, labels_ct, stats_ct,
                                                  self.eval_transform, 'valid'),
                                    batch_size=eval_bn_batch_size, shuffle=False, num_workers=6)
            self.model = self._eval_batchnorm_param(self.model, dataloader)

        dataloader = DataLoader(dataset_class(files, labels, stats, files_ct, labels_ct, stats_ct,
                                              self.eval_transform, dataset, tta=tta),
                                batch_size=batch_size, shuffle=False, num_workers=6)

        if self.freeze_eval:
            self.model.eval()
        prediction = []
        labels = np.array([])
        prediction2 = []
        labels2 = np.array([])
        for images, label, images2, label2 in tqdm(dataloader):
            images = images.to(self.device)
            images2 = images2.to(self.device)
            if self.precision == 8:
                images = images.half()
                images2 = images2.half()
            with torch.set_grad_enabled(False):
                outputs, outputs2 = self._predict_batch(images, images2)
            prediction.append(outputs.cpu().numpy())
            prediction2.append(outputs2.cpu().numpy())
            labels = np.concatenate((labels, label))
            labels2 = np.concatenate((labels2, label2))
        return np.vstack(prediction), labels, np.vstack(prediction2), labels2

    def _eval_batchnorm_param(self, model, dataloader):
        logging.info('Evaluating batchnorm param')
        model.train()
        for images, _, images2, _ in tqdm(dataloader):
            images = images.to(self.device)
            images2 = images2.to(self.device)
            with torch.set_grad_enabled(False):
                if len(images.size()) == 5:
                    _, _, C, H, W = images.size()
                    images = images.view(-1, C, H, W)
                    _, _, C, H, W = images2.size()
                    images2 = images2.view(-1, C, H, W)
                model(images, images2)
        return model


class UDAModel(Model):

    def __init__(self, model_name='resnet', ckpt_path=None, ckpt_epoch=None,
                 ckpt_full_path=None, output_ckpt_path=None, cell_type=None, criterion='cross_entropy',
                 train_transform=list(), progress_func=tqdm, lr=0.0001, load_optimizer=True,
                 freeze_eval=True, precision=16, plate_group=None, train_control=False, optimizer='adam'):
        super(UDAModel, self).__init__(
            model_name=model_name,
            ckpt_path=ckpt_path,
            ckpt_epoch=ckpt_epoch,
            ckpt_full_path=ckpt_full_path,
            output_ckpt_path=output_ckpt_path,
            cell_type=cell_type,
            criterion=criterion,
            train_transform=train_transform,
            progress_func=progress_func,
            lr=lr, load_optimizer=load_optimizer,
            freeze_eval=freeze_eval,
            precision=precision,
            plate_group=plate_group,
            train_control=train_control,
            optimizer=optimizer)
