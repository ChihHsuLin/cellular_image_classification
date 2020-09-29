import argparse
import logging
import numpy as np
import os
import pickle
import yaml
import pandas as pd

from functools import partial

import constants as c
import models as m
import util

from dataset import CellDataset, SubimageDataset, SubimageListDataset, SubimageControlDataset


def get_args():
    parser = argparse.ArgumentParser(description='Cellular Classification')
    parser.add_argument('-y', '--yaml', type=str, default='config/test_ft.yml', help='Configure file')
    parser.add_argument('-log', action='store_true', default=False, help='Save log file')
    return parser.parse_args()


def main():
    args = get_args()
    config = yaml.load(open(args.yaml), Loader=yaml.FullLoader)
    if args.log:
        util.add_log_file('/'.join(config['model'].split('/')[:-1]))
    # get subsample data
    if isinstance(config['cell_type'], str):
        if config['cell_type'] == 'all':
            used_exps = c.EXPS
        else:
            used_exps = [config['cell_type']]
    elif isinstance(config['cell_type'], list):
        used_exps = config['cell_type']
    else:
        raise Exception('cell_type must be str or list, not %s' % type(config['cell_type']))

    acc = 0.0
    sample_total_n = 0
    dataset = config['dataset']
    freeze_eval = config.get('freeze_eval', True)
    batch_size = config.get('batch_size', 8)
    eval_bn_batch_size = config.get('eval_bn_batch_size', 0)
    tta = config.get('tta', [True] * 4)
    norm = config.get('normalize', 'self')
    ds_class = partial(globals().get(config['dataset_class']), norm=norm)
    ckpt_full_paths = config.get('ckpt_full_path', None)
    precision = config.get('precision', 16)
    optimizer = config.get('optimizer', 'adam')
    gaussian_sigma = config.get('gaussian_sigma', 0)
    for i, cell_type in enumerate(used_exps):
        assert cell_type in {'RPE', 'HUVEC', 'HEPG2', 'U2OS'}
        if isinstance(ckpt_full_paths, list):
            ckpt_full_path = ckpt_full_paths[i]
            # Ensure the model name matches the cell type
            assert cell_type in ckpt_full_path
        else:
            ckpt_full_path = ckpt_full_paths
        if '2in2out' in config['model_name']:
            labmda_control = config.get('labmda_control', 0.1)
            model = m.MulInOutModel(model_name=config['model_name'],
                                    ckpt_path=config.get('model', None),
                                    ckpt_full_path=ckpt_full_path,
                                    cell_type=cell_type,
                                    freeze_eval=freeze_eval,
                                    precision=precision,
                                    labmda_control=labmda_control,
                                    optimizer=optimizer)
        else:
            model = m.Model(config['model_name'], ckpt_path=config['model'], cell_type=cell_type,
                            freeze_eval=freeze_eval, precision=precision, load_optimizer=False, training=False,
                            gaussian_sigma=gaussian_sigma)
        if dataset in ('train', 'valid'):
            df_train = pd.read_csv(c.TRAIN_CSV)
            stats = util.load_raw_stat()
            files, ids, labels = util.df2file(df_train, 'train', return_label=True, cell_type=cell_type)
            train_files, train_labels, train_ids, valid_files, valid_labels, valid_ids = util.train_valid_experiment_split(
                files, labels, ids)
            train_stats = [stats[id_code] for id_code in train_ids]
            valid_stats = [stats[id_code] for id_code in valid_ids]

            if dataset == 'train':
                files, stats, ids, labels = train_files, train_stats, train_ids, train_labels
            else:
                files, stats, ids, labels = valid_files, valid_stats, valid_ids, valid_labels
        else:
            df_test = pd.read_csv(c.TEST_CSV)
            stats = util.load_raw_stat()
            files, ids = util.df2file(df_test, 'test', return_label=False, cell_type=cell_type)
            stats = [stats[id_code] for id_code in ids]
            labels = None
            if '2in2out' in config['model_name']:
                df_test_control = pd.read_csv(c.TEST_CONTROL_CSV)
                files_ct, ids_ct, labels_ct = util.df2file(df_test_control, 'test_control', return_label=True,
                                                           cell_type=cell_type)
                stats_ct = util.load_raw_stat()
                stats_ct = [stats_ct[id_code] for id_code in ids]

        util.print_ckpt_log(model.ckpt_epoch, model.loss_history, model.acc_history, model.pp_acc_history)

        logging.info('Predicting %s', config['output'])
        if dataset in ('train', 'valid'):
            prediction = np.zeros((len(labels), c.N_CLASS))
            idx = util.get_exp_index(cell_type, files)
            prediction_i, labels_pred = model.predict(files[idx], np.array(stats)[idx], labels[idx], dataset='valid',
                                                      dataset_class=ds_class, batch_size=batch_size,
                                                      eval_bn_batch_size=eval_bn_batch_size)
            acc_i = np.sum(np.argmax(prediction_i, axis=1) == labels_pred) / len(labels_pred)
            acc += acc_i * c.TEST_COUNT[cell_type]
            sample_total_n += c.TEST_COUNT[cell_type]
            logging.info('{} Accuracy: {:.4f}%'.format(cell_type, acc_i * 100))
            prediction[idx] = prediction_i
        else:
            if '2in2out' in config['model_name']:
                prediction, _, _, _ = model.predict(files, stats, None, files_ct, stats_ct, labels_ct,
                                                    dataset='test', batch_size=batch_size,
                                                    dataset_class=ds_class, tta=tta,
                                                    eval_bn_batch_size=eval_bn_batch_size)
            else:
                prediction, _ = model.predict(files, stats, None, dataset='test', batch_size=batch_size,
                                              dataset_class=ds_class, tta=tta, eval_bn_batch_size=eval_bn_batch_size)

        out_file = 'output/%s_%s_prob.pkl' % (config['output'], dataset)
        if os.path.exists(out_file):
            with open(out_file, 'rb') as f:
                ids_pre, prediction_pre = pickle.load(f)
            with open(out_file, 'wb') as f:
                pickle.dump((np.concatenate((ids_pre, ids)), np.vstack((prediction_pre, prediction))), f)
        else:
            with open(out_file, 'wb') as f:
                pickle.dump((ids, prediction), f)

        prediction = m.get_plate_postprocessing(ids, prediction)
        out_file = 'output/%s_%s.csv' % (config['output'], dataset)
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

        logging.info('Inference %s done' % cell_type)
    if dataset in ('train', 'valid'):
        logging.info('Accuracy mean for {}: {:.4f}%'.format(';'.join(config['cell_type']), acc / sample_total_n * 100))
    logging.info('Inference done')


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    os.makedirs('output', exist_ok=True)
    main()
