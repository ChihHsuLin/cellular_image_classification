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
    df_train_control = pd.read_csv(c.TRAIN_CONTROL_CSV)
    df_test_control = pd.read_csv(c.TEST_CONTROL_CSV)
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
    logging.info('Validation experiments: %s', str(valid_exp))
    for cell_type in config['cell_type']:
        assert cell_type in {'RPE', 'HUVEC', 'HEPG2', 'U2OS'}
        files, ids, labels = util.df2file(df_train, 'train', return_label=True, cell_type=cell_type)
        if '2in2out' not in config['model_name'] and train_control:
            ct_files, ct_ids, ct_labels = util.df2file(df_train_control, 'train', return_label=True, cell_type=cell_type)
            files = np.concatenate((files, ct_files))
            ids = np.concatenate((ids, ct_ids))
            labels = np.concatenate((labels, ct_labels))
        train_files, train_labels, train_ids, valid_files, valid_labels, valid_ids = util.train_valid_experiment_split(
            files, labels, ids, skip_exp=skip_exp, valid_exp=valid_exp)
        valid_files, valid_labels, valid_ids = util.rm_control(valid_files, valid_labels, valid_ids)

        train_stats = [stats[id_code] for id_code in train_ids]
        valid_stats = [stats[id_code] for id_code in valid_ids]

        test_files, test_ids, test_labels = util.df2file(df_test_control, 'test', return_label=True, cell_type=cell_type)
        test_stats = [stats[id_code] for id_code in test_ids]

        if '2in2out' in config['model_name']:
            files, ids, labels = util.df2file(df_train_control, 'train_control', return_label=True, cell_type=cell_type)
            train_ct_files, train_ct_labels, train_ct_ids, valid_ct_files, valid_ct_labels, valid_ct_ids = util.train_valid_experiment_split(files, labels, ids, valid_exp=valid_exp)
            train_ct_stats = [stats[id_code] for id_code in train_ct_ids]
            valid_ct_stats = [stats[id_code] for id_code in valid_ct_ids]

        logging.info('Train %d Valid %d in %s', len(train_files), len(valid_files), cell_type)
        lr = config.get('lr', 0.0001)
        freeze_eval = config.get('freeze_eval', True)
        load_optimizer = config.get('load_optimizer', True)
        batch_size = config.get('batch_size', 32)
        eval_batch_size = config.get('eval_batch_size', batch_size)
        eval_bn_batch_size = config.get('eval_bn_batch_size', 0)
        criterion = config.get('criterion', 'cross_entropy')
        norm = config.get('normalize', 'self')
        ds_class = partial(globals().get(config['dataset_class']), norm=norm)
        optimizer = config.get('optimizer', 'adam')
        gaussian_sigma = config.get('gaussian_sigma', 0)

        ckpt_full_path = config.get('ckpt_full_path', None)
        if ckpt_full_path is not None and os.path.isdir(ckpt_full_path):
            # find best loss for cell type
            max_epoch = -1
            for path in glob.iglob(os.path.join(ckpt_full_path, '*.tar')):
                epoch = path.split('/')[-1].split('.')[0]
                if epoch.isdigit():
                    max_epoch = max(max_epoch, int(epoch))
            path = os.path.join(ckpt_full_path, '%d.tar' % max_epoch)
            ckpt_full_path = os.path.join(ckpt_full_path, '%d.tar' % get_best_epoch_cell(path, cell_type))

        if '2in2out' in config['model_name']:
            labmda_control = config.get('labmda_control', 0.1)
            model = m.MulInOutModel(model_name=config['model_name'],
                                    ckpt_path=config.get('ckpt_path', None),
                                    ckpt_full_path=ckpt_full_path,
                                    ckpt_epoch=config['ckpt_epoch'],
                                    output_ckpt_path=config['output_path'],
                                    cell_type=cell_type,
                                    train_transform=transform,
                                    freeze_eval=freeze_eval,
                                    lr=lr, load_optimizer=load_optimizer,
                                    labmda_control=labmda_control,
                                    criterion=criterion,
                                    optimizer=optimizer)
            model.train(train_files, train_labels, train_stats, valid_files, valid_labels, valid_stats, 
                        train_ct_files, train_ct_labels, train_ct_stats, valid_ct_files, valid_ct_labels, valid_ct_stats,
                        epochs=config['epoch'], patient=config['patient'], valid_exps=[cell_type],
                        batch_size=batch_size, eval_batch_size=eval_batch_size, eval_bn_batch_size=eval_bn_batch_size,
                        dataset_class=ds_class)
        else:
            model = m.Model(model_name=config['model_name'],
                            ckpt_path=config.get('ckpt_path', None),
                            ckpt_full_path=ckpt_full_path,
                            ckpt_epoch=config['ckpt_epoch'],
                            output_ckpt_path=config['output_path'],
                            cell_type=cell_type,
                            train_transform=transform,
                            freeze_eval=freeze_eval,
                            lr=lr, load_optimizer=load_optimizer,
                            criterion=criterion,
                            train_control=train_control,
                            optimizer=optimizer, gaussian_sigma=gaussian_sigma)
            model.train(train_files, train_labels, train_stats, valid_files, valid_labels, valid_stats,
                        test_files, test_labels, test_stats,
                        epochs=config['epoch'], patient=config['patient'], valid_exps=[cell_type],
                        batch_size=batch_size, eval_batch_size=eval_batch_size, eval_bn_batch_size=eval_bn_batch_size,
                        dataset_class=ds_class, restore_loss=False)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    main()
