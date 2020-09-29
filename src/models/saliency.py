import argparse
import glob
import logging
import numpy as np
import os
import pandas as pd
import torch
import yaml

import constants as c
import models as m
import util


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
    parser.add_argument('-y', '--yaml', type=str, default='config/saliency.yml', help='Configure file')
    return parser.parse_args()


def main():
    args = get_args()
    config = yaml.load(open(args.yaml), Loader=yaml.FullLoader)
    cell_type = config['cell_type']
    output_dir = config['output_dir']
    model_name = config['model_name']
    valid_exp = config.get('valid_exp', c.VALIDATION_EXPS)
    batch_size = config.get('batch_size', 32)
    precision = config['precision']
    dataset = config['dataset']
    method = config.get('method', 'saliency')
    if isinstance(cell_type, str):
        cell_types = [cell_type]
    else:
        cell_types = cell_type
    os.makedirs(output_dir, exist_ok=True)

    df_train = pd.read_csv(c.TRAIN_CSV)

    for qq in range(1, 17):
        valid_exp = ['HUVEC-%02d' % qq]
        logging.info('%s', valid_exp[0])
        for cell_type in cell_types:
            files, ids, labels = util.df2file(df_train, 'train', return_label=True, cell_type=cell_type)
            stats = util.load_raw_stat()

            train_files, train_labels, train_ids, valid_files, valid_labels, valid_ids = util.train_valid_experiment_split(
                files, labels, ids, valid_exp=valid_exp)
            train_stats = [stats[id_code] for id_code in train_ids]
            valid_stats = [stats[id_code] for id_code in valid_ids]

            ckpt_full_path = config.get('ckpt_full_path', None)
            if ckpt_full_path is not None and os.path.isdir(ckpt_full_path):
                # find best loss for cell type
                max_epoch = -1
                for path in glob.iglob(os.path.join(ckpt_full_path, '%s_*.tar' % cell_type)):
                    epoch = path.split('/')[-1].split('.')[0].split('_')[1]
                    if epoch.isdigit():
                        max_epoch = max(max_epoch, int(epoch))
                assert max_epoch != -1
                path = os.path.join(ckpt_full_path, '%s_%d.tar' % (cell_type, max_epoch))
                ckpt_full_path = os.path.join(ckpt_full_path, '%s_%d.tar' % (cell_type, get_best_epoch_cell(path, cell_type)))

            model = m.Model(model_name, ckpt_full_path=ckpt_full_path, freeze_eval=False, precision=precision,
                            training=False, load_optimizer=False)

            if method == 'saliency':
                if dataset == 'train':
                    model.saliency_map(train_files, train_stats, train_labels, output_dir, batch_size)
                elif dataset == 'valid':
                    model.saliency_map(valid_files, valid_stats, valid_labels, output_dir, batch_size)
                else:
                    assert False
            elif method == 'probability':
                if dataset == 'train':
                    model.predict_proba_by_subimage(train_files, train_stats, train_labels, output_dir, batch_size)
                elif dataset == 'valid':
                    model.predict_proba_by_subimage(valid_files, valid_stats, valid_labels, output_dir, batch_size)
                else:
                    assert False
            else:
                assert False

    """ Visualization example
    img = np.load('data/train/RPE-07/Plate1/B03_s1.npy')
    imgg = np.zeros_like(img, dtype=np.float)
    raw_imgg = []
    for idx in range(4):
        raw_imgg.append(np.load('vis/mishefficientnet_b2_v2_ft_swa/RPE-07/Plate1/B03_s1_%d.npy' % idx))
    imgg[:256, :256, :] = raw_imgg[0]
    imgg[:256, 256:, :] = raw_imgg[1]
    imgg[256:, :256, :] = raw_imgg[2]
    imgg[256:, 256:, :] = raw_imgg[3]
    
    fig = plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        im = img[:, :, i]
        im = np.stack((im,)*3, axis=-1) / im.max()
        ig = np.abs(imgg[:, :, i])
        ig = (ig / ig.max())
        im[:, :, 0] = np.maximum(im[:, :, 0], ig)
        plt.imshow(im)
        
    fig = plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        im = np.abs(imgg[:, :, i])
        im = im / im.max()
        im = np.stack((im,)*3, axis=-1) / im.max()
        im[:, :, 1:] = 0.0
        plt.imshow(im)
    """


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    main()
