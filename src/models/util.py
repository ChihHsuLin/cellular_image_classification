import logging
import datetime
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pickle

from albumentations import RandomRotate90, Flip, Transpose, Resize, RandomCrop, RandomScale
from albumentations import Blur, MedianBlur, GaussianBlur, MotionBlur, GridDistortion, ElasticTransform
from albumentations import Rotate, ShiftScaleRotate, Cutout, CoarseDropout
from albumentations import OneOf

import constants as c
import rio


def cache_c6_npy(df, dataset):
    for row in df.itertuples():
        paths = [[
            rio.image_path(
                dataset, row.experiment, row.plate, row.well, 1, channel)
            for channel in rio.DEFAULT_CHANNELS
        ], [
            rio.image_path(
                dataset, row.experiment, row.plate, row.well, 2, channel)
            for channel in rio.DEFAULT_CHANNELS
        ]]

        npy_paths = [rio.image_npy_path(dataset, row.experiment, row.plate, row.well, 1),
                     rio.image_npy_path(dataset, row.experiment, row.plate, row.well, 2)]
        for npy_path, path in zip(npy_paths, paths):
            if not os.path.exists(npy_path):
                img = rio.load_images_as_tensor(path)
                np.save(npy_path, img)


def load_raw_stat():
    df_stat = pd.read_csv(c.STAT_CSV)
    stats = dict()
    for row in df_stat.itertuples():
        if row.id_code not in stats:
            stats[row.id_code] = [np.zeros(12), np.zeros(12)]
        stats[row.id_code][0][row.channel - 1 + (row.site - 1) * 6] = row.mean
        stats[row.id_code][1][row.channel - 1 + (row.site - 1) * 6] = row.std
    return stats


def group_by_labels(labels):
    np.random.seed(c.SEED)
    train_index = np.array([])
    valid_index = np.array([])
    for i in range(labels.max() + 1):
        idx = np.where(labels == i)[0]
        idx = np.random.permutation(idx)
        train_index = np.concatenate((train_index, idx[:len(idx) * 9 // 10]))
        valid_index = np.concatenate((valid_index, idx[len(idx) * 9 // 10:]))
    assert len(set(train_index) | set(valid_index)) == len(labels)
    return train_index.astype(np.int), valid_index.astype(np.int)


def train_valid_experiment_split(files, labels, ids, used_exps=c.EXPS, skip_exp=[], valid_exp=c.VALIDATION_EXPS):
    np.random.seed(c.SEED)
    exps = sorted(set([x.split('_')[0] for x in ids]))
    train_exp = set()
    for exp in exps:
        if exp not in valid_exp:
            train_exp.add(exp)

    train_index = []
    valid_index = []
    for i, file_id in enumerate(ids):
        exp = file_id.split('_')[0]
        exp_prefix = exp.split('-')[0]
        if exp_prefix not in used_exps or exp in skip_exp:
            continue
        if exp in train_exp:
            train_index.append(i)
        else:
            valid_index.append(i)

    train_files = files[train_index]
    valid_files = files[valid_index]

    train_labels = labels[train_index]
    valid_labels = labels[valid_index]

    train_ids = ids[train_index]
    valid_ids = ids[valid_index]

    valid_count = np.bincount(valid_labels)
    logging.info('#valid label max / min: %d / %d', valid_count.max(), valid_count.min())
    return train_files, train_labels, train_ids, valid_files, valid_labels, valid_ids


def df2file(df, dataset, return_label=False, cell_type=None):
    files = []
    ids = []
    labels = []
    if cell_type:
        df = df.loc[df.experiment.str.startswith(cell_type)]
    for i, row in enumerate(df.itertuples()):
        dataset_prefix = dataset.replace('_control', '')
        png_path = [rio.image_npy_path(dataset_prefix, row.experiment, row.plate, row.well, 1),
                    rio.image_npy_path(dataset_prefix, row.experiment, row.plate, row.well, 2)]
        if return_label:
            labels.append(row.sirna)
        ids.append(row.id_code)
        files.append(png_path)

    if return_label:
        if 'control' in dataset:
            labels = [x - c.N_CLASS for x in labels]
        return np.array(files), np.array(ids), np.array(labels)
    return np.array(files), np.array(ids)


def get_exp_index(exp, files):
    idx = []
    for i, f in enumerate(files):
        if isinstance(f, np.ndarray) or isinstance(f, list):
            x = f[0]
        else:
            x = f

        if exp in x:
            idx.append(i)
    return idx


def _transform(name, params=None):
    f = globals().get(name)
    return f(**params)


def get_transform(base_list, aug_list):
    transform = []
    for trans, param in base_list:
        transform.append(_transform(trans, param))

    one_of = []
    for trans, param in aug_list:
        one_of.append(_transform(trans, param))

    if len(one_of) > 0:
        transform.append(OneOf(one_of, p=1))
    return transform


def add_log_file(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(message)s',
                        filename='{}/{}.log'.format(log_dir, start_time),
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='[%(asctime)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def load_data_by_set(dataset, cell_type):
    if dataset in ('train', 'valid'):
        df_train = pd.read_csv(c.TRAIN_CSV)
        stats = load_raw_stat()
        files, ids, labels = df2file(df_train, 'train', return_label=True, cell_type=cell_type)
        train_files, train_labels, train_ids, valid_files, valid_labels, valid_ids = train_valid_experiment_split(
            files, labels, ids)
        train_stats = [stats[id_code] for id_code in train_ids]
        valid_stats = [stats[id_code] for id_code in valid_ids]

        if dataset == 'train':
            files, stats, ids, labels = train_files, train_stats, train_ids, train_labels
        else:
            files, stats, ids, labels = valid_files, valid_stats, valid_ids, valid_labels
    else:
        df_test = pd.read_csv(c.TEST_CSV)
        stats = load_raw_stat()
        files, ids = df2file(df_test, 'test', return_label=False, cell_type=cell_type)
        stats = [stats[id_code] for id_code in ids]
        labels = None
    return files, stats, ids, labels


def get_g2rna():
    g2rna = [set(), set(), set(), set()]
    for line in open('plate_dist.csv'):
        line = line.split(',')
        if line[0].isdigit():
            g2rna[int(line[1])].add(int(line[0]))
    assert len(g2rna[0] | g2rna[1] | g2rna[2] | g2rna[3]) == c.N_CLASS

    masks = []
    for i in range(4):
        idx = np.zeros(c.N_CLASS)
        idx[list(g2rna[i])] = 1
        zidx = np.where(idx == 0)[0]
        masks.append(zidx)
    return g2rna, masks


def print_ckpt_log(ckpt_epoch, loss_history, acc_history, pp_acc_history):
    logging.info('Epoch %d', ckpt_epoch)
    logging.info('Train Loss: {:.6f} Acc: {:.4f}%'.format(loss_history['train'][-1],
                                                          acc_history['train'][-1] * 100))
    valid_loss_dict = loss_history['valid'][-1]
    valid_acc_dict = acc_history['valid'][-1]
    if len(pp_acc_history['valid']) > 0:
        valid_pp_acc_dict = pp_acc_history['valid'][-1]
    else:
        valid_pp_acc_dict = None
    tw = 0
    loss = 0.0
    acc = 0.0
    pp_acc = 0.0
    for exp in c.EXPS:
        if exp not in valid_loss_dict:
            continue
        tw += c.TEST_COUNT[exp]
        loss += valid_loss_dict[exp] * c.TEST_COUNT[exp]
        acc += valid_acc_dict[exp] * c.TEST_COUNT[exp]
        if valid_pp_acc_dict is not None:
            pp_acc += valid_pp_acc_dict[exp] * c.TEST_COUNT[exp]
    logging.info('Valid Loss: {:.6f} Acc: {:.4f}% Plate Acc: {:.4f}%'.format(loss / tw, acc / tw * 100, pp_acc / tw * 100))
    for exp in c.EXPS:
        if exp not in valid_loss_dict:
            continue
        if valid_pp_acc_dict is not None:
            logging.info(
                '{} Valid Loss / Acc: {:.6f} / {:.4f}% / {:.4f}%'.format(exp, valid_loss_dict[exp], valid_acc_dict[exp] * 100, valid_pp_acc_dict[exp] * 100))
        else:
            logging.info('{} Valid Loss / Acc: {:.6f} / {:.4f}%'.format(exp, valid_loss_dict[exp], valid_acc_dict[exp] * 100))

    if 'test' not in loss_history or len(loss_history['test']) == 0:
        return

    loss = 0.0
    acc = 0.0
    pp_acc = 0.0
    test_loss_dict = loss_history['test'][-1]
    test_acc_dict = acc_history['test'][-1]
    for exp in c.EXPS:
        if exp not in test_loss_dict:
            continue
        loss += test_loss_dict[exp] * c.TEST_COUNT[exp]
        acc += test_acc_dict[exp] * c.TEST_COUNT[exp]
    logging.info(
        'Test Loss: {:.6f} Acc: {:.4f}% Plate Acc: {:.4f}%'.format(loss / tw, acc / tw * 100, pp_acc / tw * 100))
    for exp in c.EXPS:
        if exp not in test_loss_dict:
            continue
        logging.info(
            '{} Test Loss / Acc: {:.6f} / {:.4f}%'.format(exp, test_loss_dict[exp], test_acc_dict[exp] * 100))


def rm_control(files, labels, ids):
    idx = [i for i, label in enumerate(labels) if label < c.N_CLASS]
    return files[idx], labels[idx], ids[idx]


def update_control_class(df_control):
    df_pg = pd.read_csv('plate_group.csv', index_col='plate')
    pg_dict = df_pg.to_dict()['group']
    df_control['sirna'] = df_control.apply(
        lambda x: x['sirna'] + pg_dict[x['id_code'].rsplit('_', 1)[0]] * c.N_CLASS_CONTROL, axis=1)
    return df_control


def load_group_dict():
    df = pd.read_csv('plate_group.csv')
    group_dict = dict()
    for row in df.itertuples():
        cell = row.plate.split('-')[0]
        key = (cell, row.group)
        if key not in group_dict:
            group_dict[key] = set()
        group_dict[key].add(row.plate)
    return group_dict


def write_inference_result(dataset, output_path, ids, prediction):
    out_file = 'output/%s_%s.csv' % (output_path, dataset)
    if os.path.isfile(out_file):
        write_header = False
        logging.info('Appending prediction to previous file %s!' % out_file)
    else:
        write_header = True

    with open(out_file, 'a+') as f:
        # only write header first time
        if write_header:
            f.write('id_code,sirna\n')
        for id_code, pred in zip(ids, prediction):
            f.write('%s,%d\n' % (id_code, np.argmax(pred)))

    out_file = 'output/%s_%s_prob.pkl' % (output_path, dataset)
    if os.path.exists(out_file):
        with open(out_file, 'rb') as f:
            ids_pre, prediction_pre = pickle.load(f)
        ids = np.concatenate((ids_pre, ids))
        prediction = np.vstack((prediction_pre, prediction))

    with open(out_file, 'wb') as f:
        pickle.dump((ids, prediction), f)
