import argparse
import logging
import numpy as np
import os
import pandas as pd
import pickle
import yaml

import constants as c
import models as m
import util

from scipy.special import softmax


def load_prob(dataset, ensemble_file, u2os):
    path = 'output/%s_%s_prob.pkl' % (ensemble_file, dataset)
    if os.path.exists(path):
        ids, prediction = pickle.load(open(path, 'rb'))
        ids = np.array(ids)
        if u2os:
            idx = [i for i, id_i in enumerate(ids) if 'U2OS' in id_i]
        else:
            idx = [i for i, id_i in enumerate(ids) if 'U2OS' not in id_i]
        ids = ids[idx]
        prediction = prediction[idx]
        # make sure order are the same between different prediction outputs
        idx = np.argsort(ids)
        return ids[idx], prediction[idx]
    else:
        return None, None


def print_accuracy(ids, y_true, y_pred):
    y_true = np.array(y_true)
    tw = 0.0
    acc = 0.0
    for exp in c.EXPS:
        idx = [i for i, x in enumerate(ids) if x.startswith(exp)]
        acc_i = np.sum(np.argmax(y_pred[idx], axis=1) == y_true[idx]) / len(y_true[idx])
        acc += acc_i * c.TEST_COUNT[exp]
        tw += c.TEST_COUNT[exp]
        logging.info('{} Accuracy: {:.4f}%'.format(exp, acc_i * 100))
    logging.info('Accuracy: {:.4f}%'.format(acc / tw * 100))


def get_args():
    parser = argparse.ArgumentParser(description='Cellular Classification')
    parser.add_argument('-y', '--yaml', type=str, default='config/ensemble2.yml', help='Configure file')
    parser.add_argument('-f', '--function', type=str, default='mean', help='Ensemble function')
    parser.add_argument('-b', '--balance', action='store_true', default=False, help='Balance class')
    return parser.parse_args()


def calc_ensemble(func, probs):
    if func == 'mean':
        return np.mean(probs, axis=0)
    if func == 'median':
        return np.median(probs, axis=0)
    if func.startswith('rank'):
        ranks = []
        for prob in probs:
            ranks.append(np.argsort(np.argsort(prob, axis=1), axis=1))
        if func == 'rank_mean':
            return np.mean(ranks, axis=0)
        if func == 'rank_median':
            return np.median(ranks, axis=0)
    elif func.startswith('softmax'):
        softmaxs = []
        for prob in probs:
            softmaxs.append(softmax(prob, axis=1))
        if func == 'softmax_mean':
            return np.mean(softmaxs, axis=0)
        if func == 'softmax_median':
            return np.median(softmaxs, axis=0)
    raise NotImplementedError


def main(config, function, balance, u2os):
    # todo: add model based ensemble
    logging.info('Only support mean ensemble now!')
    ensemble_files = config['ensemble_files']

    df_train = pd.read_csv(c.TRAIN_CSV)
    _, ids, labels = util.df2file(df_train, 'train', return_label=True)

    id2label = dict(zip(ids, labels))
    # todo: load training set label and probability
    # load validation set label and probability
    valid_ids = None
    y_pred = []
    for ensemble_file in ensemble_files:
        valid_ids, pred_i = load_prob('valid', ensemble_file, u2os)
        if pred_i is not None and pred_i.shape[1] > c.N_CLASS:
            pred_i = pred_i[:, :c.N_CLASS]
        y_pred.append(pred_i)
    if valid_ids is not None:
        y_true = [id2label[x] for x in valid_ids]
        y_pred = calc_ensemble(function, y_pred)
        print_accuracy(valid_ids, y_true, y_pred)
        y_pred = m.get_plate_postprocessing(valid_ids, y_pred)
        print_accuracy(valid_ids, y_true, y_pred)

    # load testing set probability
    test_ids = None
    y_pred = []
    for ensemble_file in ensemble_files:
        test_ids, pred_i = load_prob('test', ensemble_file, u2os)
        if pred_i.shape[1] > c.N_CLASS:
            pred_i = pred_i[:, :c.N_CLASS]
        y_pred.append(pred_i)

    y_pred = calc_ensemble(function, y_pred)
    out_file = 'output/%s_test_prob.pkl' % config['output']
    if os.path.exists(out_file):
        with open(out_file, 'rb') as f:
            ids_pre, prediction_pre = pickle.load(f)
        with open(out_file, 'wb') as f:
            pickle.dump((np.concatenate((ids_pre, test_ids)), np.vstack((prediction_pre, y_pred))), f)
    else:
        with open(out_file, 'wb') as f:
            pickle.dump((test_ids, y_pred), f)

    if balance:
        y_pred = m.balancing_class_prediction(test_ids, y_pred)
    else:
        y_pred = m.get_plate_postprocessing(test_ids, y_pred)
    out_file = 'output/%s_test.csv' % config['output']
    if os.path.isfile(out_file):
        write_header = False
        logging.info('Appending prediction to previous file %s!' % out_file)
    else:
        write_header = True
    with open(out_file, 'a+') as f:
        if write_header:
            f.write('id_code,sirna\n')
        for id_code, pred in zip(test_ids, y_pred):
            if balance:
                f.write('%s,%d\n' % (id_code, pred))
            else:
                f.write('%s,%d\n' % (id_code, np.argmax(pred)))

    logging.info('Done')


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    args = get_args()
    config = yaml.load(open(args.yaml), Loader=yaml.FullLoader)
    config['general']['output'] = config['output']
    config['u2os']['output'] = config['output']
    main(config['general'], args.function, args.balance, False)
    main(config['u2os'], args.function, args.balance, True)
