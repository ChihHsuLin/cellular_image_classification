import argparse
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd
import yaml

from functools import partial

import constants as c
import models as m
import util

from dataset import CellDataset, SubimageDataset, Subimage128Dataset, SubimageListDataset, SubimageControlDataset


def get_args():
    parser = argparse.ArgumentParser(description='Cellular Classification')
    parser.add_argument('-y', '--yaml', type=str, default='config/train1.yml', help='Configure file')
    parser.add_argument('-a', '--augment', type=str, default='config/aug.yml', help='Augmentation configure file')
    parser.add_argument('-log', action='store_true', default=False, help='Save log file')
    return parser.parse_args()


def main():
    args = get_args()
    config = yaml.load(open(args.yaml), Loader=yaml.FullLoader)
    if args.log:
        util.add_log_file(config['ckpt_path'])

    config_aug = yaml.load(open(args.augment), Loader=yaml.FullLoader)
    df_train = pd.read_csv(c.TRAIN_CSV)
    df_train_control = pd.read_csv(c.TRAIN_CONTROL_CSV)
    logging.info('Load stats')
    stats = util.load_raw_stat()

    skip_exp = config.get('skip_exp', [])
    valid_exp = config.get('valid_exp', c.VALIDATION_EXPS)
    train_control = config.get('train_control', False)
    logging.info('Validation experiments: %s', str(valid_exp))
    files, ids, labels = util.df2file(df_train, 'train', return_label=True)
    if train_control:
        ct_files, ct_ids, ct_labels = util.df2file(df_train_control, 'train', return_label=True)
        files = np.concatenate((files, ct_files))
        ids = np.concatenate((ids, ct_ids))
        labels = np.concatenate((labels, ct_labels))
    train_files, train_labels, train_ids, valid_files, valid_labels, valid_ids = util.train_valid_experiment_split(
        files, labels, ids, skip_exp=skip_exp, valid_exp=valid_exp)
    valid_files, valid_labels, valid_ids = util.rm_control(valid_files, valid_labels, valid_ids)

    train_stats = [stats[id_code] for id_code in train_ids]
    valid_stats = [stats[id_code] for id_code in valid_ids]

    df_test_control = pd.read_csv(c.TEST_CONTROL_CSV)
    test_files, test_ids, test_labels = util.df2file(df_test_control, 'test', return_label=True)
    test_stats = [stats[id_code] for id_code in test_ids]

    if '2in2out' in config['model_name']:
        ct_files, ct_ids, ct_labels = util.df2file(df_train_control, 'train_control', return_label=True)
        train_ct_files, train_ct_labels, train_ct_ids, valid_ct_files, valid_ct_labels, valid_ct_ids = util.train_valid_experiment_split(
            ct_files, ct_labels, ct_ids, valid_exp=valid_exp)
        train_ct_stats = [stats[id_code] for id_code in train_ct_ids]
        valid_ct_stats = [stats[id_code] for id_code in valid_ct_ids]

    logging.info('Train %d Valid %d', len(train_files), len(valid_files))

    aug_list = []
    for aug in config_aug:
        key = list(aug.keys())[0]
        aug_list.append((key, aug[key]))
    # fixme: use aug after find better augmentation
    transform = util.get_transform(c.BASE_AUG, [])
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
    if '2in2out' in config['model_name']:
        labmda_control = config.get('labmda_control', 0.1)
        model = m.MulInOutModel(model_name=config['model_name'],
                                ckpt_path=config['ckpt_path'],
                                ckpt_epoch=config['ckpt_epoch'],
                                train_transform=transform,
                                freeze_eval=freeze_eval,
                                lr=lr, load_optimizer=load_optimizer,
                                labmda_control=labmda_control,
                                criterion=criterion, optimizer=optimizer)
        model.train(train_files, train_labels, train_stats, valid_files, valid_labels, valid_stats,
                    train_ct_files, train_ct_labels, train_ct_stats, valid_ct_files, valid_ct_labels, valid_ct_stats,
                    epochs=config['epoch'],
                    batch_size=batch_size, balance_exp=config['balance_exp'], num_workers=mp.cpu_count(),
                    dataset_class=ds_class, eval_bn_batch_size=eval_bn_batch_size,
                    eval_batch_size=eval_batch_size, patient=config['patient'])
    else:
        model = m.Model(model_name=config['model_name'],
                        ckpt_path=config['ckpt_path'],
                        ckpt_epoch=config['ckpt_epoch'],
                        train_transform=transform,
                        freeze_eval=freeze_eval,
                        lr=lr, load_optimizer=load_optimizer,
                        criterion=criterion,
                        train_control=train_control, optimizer=optimizer,
                        gaussian_sigma=gaussian_sigma)
        model.train(train_files, train_labels, train_stats, valid_files, valid_labels, valid_stats,
                    test_files, test_labels, test_stats, epochs=config['epoch'],
                    batch_size=batch_size, balance_exp=config['balance_exp'], num_workers=mp.cpu_count(),
                    dataset_class=ds_class, eval_batch_size=eval_batch_size,
                    eval_bn_batch_size=eval_bn_batch_size, patient=config['patient'])
    logging.info('Training done')


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    main()
