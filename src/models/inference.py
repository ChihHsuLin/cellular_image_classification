#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import argparse
import logging
import os
import yaml

import ensemble

from dataset import CellDataset, SubimageDataset, SubimageListDataset, SubimageControlDataset


def get_args():
    parser = argparse.ArgumentParser(description='Cellular Classification')
    parser.add_argument('-y', '--yaml', type=str, default='config/test_ft.yml', help='Configure file')
    parser.add_argument('-log', action='store_true', default=False, help='Save log file')
    parser.add_argument('-l', '--level', type=int, default=1, help='Fine tune level')
    return parser.parse_args()


def main():
    args = get_args()
    config = yaml.load(open(args.yaml), Loader=yaml.FullLoader)
    if 'batch_size' not in config:
        config['batch_size'] = 32
    output = config['output']
    config_ensemble = {'ensemble_files': [], 'output': '{}_all'.format(output)}
    if args.level == 0:
        ft_py = 'inference.py'
        # fixme: not tested yet
        raise NotImplementedError
    elif args.level == 1:
        ft_py = 'inference_ft.py'
    elif args.level == 2:
        ft_py = 'inference_l2ft.py'
    else:
        raise Exception('Level unknown')

    for i in range(4):
        config['tta'] = [False] * 4
        config['tta'][i] = True
        config['output'] = '{}_tta_{}'.format(output, i)
        config_ensemble['ensemble_files'].append(config['output'])
        tmp_yml = '{}.tmp'.format(args.yaml)
        with open(tmp_yml, 'w') as outfile:
            yaml.dump(config, outfile)
        logging.info("Running tta {}".format(i))
        if args.log:
            subprocess.call(['python', ft_py, '-y', tmp_yml, '-log'])
        else:
            subprocess.call(['python', ft_py, '-y', tmp_yml])
    os.remove(tmp_yml)
    logging.info("Running mean ensemble")
    ensemble.main(config_ensemble, 'mean', False)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    main()
