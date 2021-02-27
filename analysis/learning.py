#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 17/02/21 9:04 PM
@description:  
@version: 1.0
"""


import numpy as np
import json
import matplotlib.pyplot as plt
import argparse


save_path = '../results'


class Learning:

    def __init__(self, args):
        self.args = args
        with open('{}/learning history/{}.json'.format(save_path, self.file_name())) as f:
            self.data = json.load(f)

    def file_name(self):
        if self.args.net_name == 'MLP':
            return '{}_{}_{}_{}_{}_{}'.format(self.args.model_name,
                                              self.args.net_name,
                                              self.args.len_seg,
                                              self.args.optimizer,
                                              self.args.learning_rate,
                                              self.args.num_epoch
                                              )
        else:
            return '{}_{}_{}_{}_{}_{}_{}'.format(self.args.model_name,
                                                 self.args.net_name,
                                                 self.args.len_seg,
                                                 self.args.optimizer,
                                                 self.args.learning_rate,
                                                 self.args.num_epoch,
                                                 self.args.num_hidden_map
                                                 )

    def plot_learning(self):
        font = {'style': 'normal',
                'weight': 'bold',
                'color': 'k',
                'size': 12
                }
        fig, ax = plt.subplots()
        loss = self.data['Loss']
        loss_x = self.data['MSE']
        loss_z = self.data['MSE latent']
        best_epoch = self.data['Best epoch']
        best_loss = self.data['Min loss']
        range_ = np.arange(1, len(loss) + 1)
        ax.plot(range_, np.log(loss), c='r', lw=1, label='Overall loss')
        ax.plot(range_, np.log(loss_x), c='b', lw=1, label='Reconstruction loss')
        ax.plot(range_, np.log(loss_z), c='g', lw=1, label='Latent loss')
        ax.scatter(best_epoch, np.log(best_loss),
                   marker='*', s=100, c='r', label='Model save point')
        ax.set_xlim(0)
        ax.set_xlabel('Epoch', fontdict=font)
        ax.set_ylabel('Loss', fontdict=font)
        ax.legend()
        plt.tight_layout()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='AE', type=str)
    parser.add_argument('--net_name', default='MLP', type=str)
    parser.add_argument('--len_seg', default=500, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str)
    # MLP setting
    parser.add_argument('--dim_input', default=384, type=int)
    parser.add_argument('--dim_feature', default=20, type=int)
    # Conv2D setting
    parser.add_argument('--num_feature_map', default=128, type=int)
    parser.add_argument('--num_hidden_map', default=256, type=int)
    parser.add_argument('--num_epoch', default=10000, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    args = parser.parse_args()
    plt.rcParams['font.family'] = 'Arial'
    learning = Learning(args)
    learning.plot_learning()
    plt.show()
