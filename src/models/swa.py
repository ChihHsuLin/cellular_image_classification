import argparse
import glob
import logging
import numpy as np
import os
import pandas as pd
import torch
import yaml

from functools import partial

import constants as c
import models as m
import util

from dataset import CellDataset, SubimageDataset, Subimage128Dataset, SubimageListDataset, SubimageControlDataset


def get_best_epoch_cell(ckpt_path, cell_type):
    model_ckpt = torch.load(ckpt_path)
    loss_history = model_ckpt['loss']
    best_epoch = -1
    best_loss = np.inf
    for i, loss_dict in enumerate(loss_history['valid']):
        if loss_dict[cell_type] < best_loss:
            best_loss = loss_dict[cell_type]
            best_epoch = i
    assert best_epoch != -1
    logging.info('%s best epoch: %d best loss: %f', cell_type, best_epoch, best_loss)
    return best_epoch


def get_args():
    parser = argparse.ArgumentParser(description='Cellular Classification cell type-specific model')
    parser.add_argument('-y', '--yaml', type=str, default='config/train_ft0.yml', help='Configure file')
    parser.add_argument('-a', '--augment', type=str, default='config/aug.yml', help='Augmentation configure file')
    parser.add_argument('-log', action='store_true', default=False, help='Save log file')
    return parser.parse_args()


def main():
    args = get_args()
    config = yaml.load(open(args.yaml), Loader=yaml.FullLoader)
    if args.log:
        util.add_log_file(config['output_path'])
    config_aug = yaml.load(open(args.augment), Loader=yaml.FullLoader)
    df_train = pd.read_csv(c.TRAIN_CSV)
    logging.info('Load stats')
    stats = util.load_raw_stat()

    aug_list = []
    for aug in config_aug:
        key = list(aug.keys())[0]
        aug_list.append((key, aug[key]))
    # fixme: use aug after find better augmentation
    transform = util.get_transform(c.BASE_AUG, [])

    skip_exp = config.get('skip_exp', [])
    valid_exp = config.get('valid_exp', c.VALIDATION_EXPS)
    train_control = config.get('train_control', False)
    epoch_step = config['epoch_step']
    ckpt_prefix = config['ckpt_prefix']
    logging.info('Validation experiments: %s', str(valid_exp))
    for cell_type in zip(config['cell_type']):
        assert cell_type in {'RPE', 'HUVEC', 'HEPG2', 'U2OS'}
        files, ids, labels = util.df2file(df_train, 'train', return_label=True, cell_type=cell_type)
        train_files, train_labels, train_ids, valid_files, valid_labels, valid_ids = util.train_valid_experiment_split(
            files, labels, ids, skip_exp=skip_exp, valid_exp=valid_exp)
        valid_files, valid_labels, valid_ids = util.rm_control(valid_files, valid_labels, valid_ids)

        train_stats = [stats[id_code] for id_code in train_ids]
        valid_stats = [stats[id_code] for id_code in valid_ids]

        logging.info('Train %d Valid %d in %s', len(train_files), len(valid_files), cell_type)
        lr = config.get('lr', 0.0001)
        freeze_eval = config.get('freeze_eval', True)
        load_optimizer = config.get('load_optimizer', True)
        batch_size = config.get('batch_size', 32)
        eval_batch_size = config.get('eval_batch_size', batch_size)
        criterion = config.get('criterion', 'cross_entropy')
        norm = config.get('normalize', 'self')
        ds_class = partial(globals().get(config['dataset_class']), norm=norm)
        optimizer = config.get('optimizer', 'adam')
        gaussian_sigma = config.get('gaussian_sigma', 0)

        if os.path.isdir(ckpt_prefix):
            # find best loss for cell type
            max_epoch = -1
            for path in glob.iglob(os.path.join(ckpt_prefix, '*.tar')):
                epoch = path.split('/')[-1].split('.')[0]
                if epoch.isdigit():
                    max_epoch = max(max_epoch, int(epoch))
            path = os.path.join(ckpt_prefix, '%d.tar' % max_epoch)
            ckpt_full_path = os.path.join(ckpt_full_path, '%d.tar' % get_best_epoch_cell(path, cell_type))

        model = m.Model(model_name=config['model_name'],
                        ckpt_path=config.get('ckpt_path', None),
                        ckpt_epoch=config['ckpt_epoch'],
                        output_ckpt_path=config['output_path'],
                        cell_type=cell_type,
                        train_transform=transform,
                        freeze_eval=freeze_eval,
                        lr=lr, load_optimizer=load_optimizer,
                        criterion=criterion,
                        train_control=train_control,
                        optimizer=optimizer, gaussian_sigma=gaussian_sigma)
        model.get_swa_from_ckpts(train_files, train_labels, train_stats, valid_files, valid_labels, valid_stats,
                                 ckpt_prefix=ckpt_prefix, cell_type=cell_type, first_epoch=first_epoch,
                                 last_epoch=last_epoch, epoch_step=epoch_step, batch_size=batch_size,
                                 eval_batch_size=eval_batch_size, dataset_class=ds_class)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    main()
