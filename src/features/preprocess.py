import ujson as json
import logging
import numpy as np
import pandas as pd

import constants as c

from util import cache_c6_npy


def main():
    df_train = pd.read_csv(c.TRAIN_CSV)
    df_test = pd.read_csv(c.TEST_CSV)
    df_train_control = pd.read_csv(c.TRAIN_CONTROL_CSV)
    df_test_control = pd.read_csv(c.TEST_CONTROL_CSV)
    logging.info('Caching train npy...')
    cache_c6_npy(df_train, 'train')
    logging.info('Caching test npy...')
    cache_c6_npy(df_test, 'test')
    logging.info('Caching train control npy...')
    cache_c6_npy(df_train_control, 'train')
    logging.info('Caching test control npy...')
    cache_c6_npy(df_test_control, 'test')
    logging.info('Caching npy done!')

    logging.info('Calculate mean and std')
    df = pd.read_csv(c.STAT_CSV)
    tbl = dict()
    for row in df.itertuples():
        exp = row.id_code.split('_')[0]
        if exp not in tbl:
            tbl[exp] = [np.zeros(6), np.zeros(6), 0]
        i = row.channel - 1
        tbl[exp][0][i] += row.mean
        tbl[exp][1][i] += row.std
        if i == 0:
            # number of images
            tbl[exp][2] += 1

    mean_dict = dict()
    std_dict = dict()
    for exp in sorted(tbl.keys()):
        count = tbl[exp][2]
        mean_dict[exp] = tbl[exp][0] / count
        std_dict[exp] = tbl[exp][1] / count
    json.dump(mean_dict, open('experiment_mean.json', 'w'), indent=2)
    json.dump(std_dict, open('experiment_std.json', 'w'), indent=2)

    tbl = dict()
    for row in df.itertuples():
        exp = row.id_code.rsplit('_', 1)[0]
        if exp not in tbl:
            tbl[exp] = [np.zeros(6), np.zeros(6), 0]
        i = row.channel - 1
        tbl[exp][0][i] += row.mean
        tbl[exp][1][i] += row.std
        if i == 0:
            # number of images
            tbl[exp][2] += 1

    mean_dict = dict()
    std_dict = dict()
    for exp in sorted(tbl.keys()):
        count = tbl[exp][2]
        mean_dict[exp] = tbl[exp][0] / count
        std_dict[exp] = tbl[exp][1] / count
    json.dump(mean_dict, open('plate_mean.json', 'w'), indent=2)
    json.dump(std_dict, open('plate_std.json', 'w'), indent=2)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    main()
