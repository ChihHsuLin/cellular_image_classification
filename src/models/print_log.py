import argparse
import glob
import numpy as np
import os

import torch

import constants as c


def get_best_epoch_cell(ckpt_path, cell_type):
    model_ckpt = torch.load(ckpt_path)
    loss_history = model_ckpt['loss']
    acc_history = model_ckpt['acc']
    pp_acc_history = model_ckpt['pp_acc']
    best_epoch = -1
    best_loss = np.inf
    best_acc = 0.0
    best_pp_acc = 0.0
    for i, loss_dict in enumerate(loss_history['valid']):
        if loss_dict[cell_type] < best_loss:
            best_loss = loss_dict[cell_type]
            best_acc = acc_history['valid'][i][cell_type]
            best_pp_acc = pp_acc_history['valid'][i][cell_type]
            best_epoch = i
    assert best_epoch != -1
    return best_loss, best_acc, best_pp_acc


def get_args():
    parser = argparse.ArgumentParser(description='Cellular Classification')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='Checkpoint file')
    return parser.parse_args()


def main():
    args = get_args()
    path = os.path.join(args.checkpoint, 'best.tar')
    if os.path.exists(path):
        # base model
        # model_ckpt = torch.load(path)
        # loss_history = model_ckpt['loss']
        # acc_history = model_ckpt['acc']
        # if 'pp_acc' in model_ckpt:
        #     pp_acc_history = model_ckpt['pp_acc']
        # else:
        #     pp_acc_history = dict()
        max_epoch = -1
        for path in glob.iglob(os.path.join(args.checkpoint, '*.tar')):
            epoch = path.split('/')[-1].split('.')[0]
            if epoch.isdigit():
                max_epoch = max(max_epoch, int(epoch))
        path = os.path.join(args.checkpoint, '%d.tar' % max_epoch)

        for exp in c.EXPS:
            best_loss, best_acc, best_pp_acc = get_best_epoch_cell(path, exp)
            out = [exp, '', '', best_loss, best_acc, best_pp_acc]
            # out = [exp, '', '', loss_history['valid'][-1][exp], acc_history['valid'][-1][exp],
            #        pp_acc_history.get('valid', [{exp: ''}])[-1][exp]]
            print('\t'.join([str(x) for x in out]))
        # if isinstance(loss_history['train'][-1], torch.Tensor):
        #     train_loss = loss_history['train'][-1].cpu().numpy()
        # else:
        #     train_loss = loss_history['train'][-1]
        # out = ['overall', train_loss, acc_history['train'][-1]]
        # print('\t'.join([str(x) for x in out]))
    else:
        # finetune model
        for exp in c.EXPS:
            path = os.path.join(args.checkpoint, '%s_best.tar' % exp)
            if os.path.exists(path):
                model_ckpt = torch.load(path)
                loss_history = model_ckpt['loss']
                acc_history = model_ckpt['acc']
                if 'pp_acc' in model_ckpt:
                    pp_acc_history = model_ckpt['pp_acc']
                else:
                    pp_acc_history = dict()
                if isinstance(loss_history['train'][-1], torch.Tensor):
                    train_loss = loss_history['train'][-1].cpu().numpy()
                else:
                    train_loss = loss_history['train'][-1]
                out = [exp, train_loss, acc_history['train'][-1],
                       loss_history['valid'][-1][exp], acc_history['valid'][-1][exp],
                       pp_acc_history.get('valid', [{exp: ''}])[-1][exp]]
                print('\t'.join([str(x) for x in out]))


if __name__ == '__main__':
    main()
