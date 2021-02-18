#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 18/02/21 7:34 PM
@description:  
@version: 1.0
"""


import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import argparse


save_path = '../results'


class Loss:

    def __init__(self, args):
        self.args = args
        self.datasets = ['W-2', 'W-5', 'W-7']
        self.spots = ['Wall-A2', 'Wall-B1',
                      'Floor-A1', 'Floor-A2',
                      'Floor-B1', 'Floor-B2'
                      ]
        self.colors = ['b', 'g', 'r']
        self.labels = ['25%', '50%', '100%']
        self.font = {'family': 'Arial',
                     'style': 'normal',
                     'weight': 'bold',
                     'size': 12,
                     'color': 'k',
                     }

    def load_data(self, dataset):
        with open('{}/damage index/{}_{}.json'.
                  format(save_path, dataset, self.file_name())) as f:
            data = json.load(f)
        return data

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

    def spot_loss(self, dataset):
        data = self.load_data(dataset)
        losses = np.empty((0, 2))
        for _, loss_value in data.items():
            loss = np.array([loss_value['Reconstruction loss'],
                             loss_value['Latent loss']]).reshape(-1, 2)
            losses = np.concatenate((losses, loss), axis=0)
        return losses

    def spot_loss_3d(self, dataset):
        data = self.load_data(dataset)
        losses = np.empty((0, 3))
        for i, spot in enumerate(self.spots):
            loss_l1 = np.array([data['L1-{}'.format(spot)]['Reconstruction loss'],
                                data['L1-{}'.format(spot)]['Latent loss'],
                                i + 1
                                ]
                               ).reshape(-1, 3)
            loss_l2 = np.array([data['L2-{}'.format(spot)]['Reconstruction loss'],
                                data['L2-{}'.format(spot)]['Latent loss'],
                                i + 1
                                ]
                               ).reshape(-1, 3)
            loss = np.vstack((loss_l1, loss_l2))
            losses = np.concatenate((losses, loss), axis=0)
        return losses

    def plot_loss(self):
        fig, ax = plt.subplots()
        for i, dataset in enumerate(self.datasets):
            losses = self.spot_loss(dataset)
            ax.scatter(losses[:, 0], losses[:, 1], alpha=0.8,
                       c=self.colors[i], label=self.labels[i]
                       )
        ax.set_xlabel('Reconstruction loss', fontdict=self.font)
        ax.set_ylabel('Latent loss', fontdict=self.font)
        ax.legend()

    def plot_loss_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, dataset in enumerate(self.datasets):
            losses = self.spot_loss_3d(dataset)
            ax.scatter(losses[:, 0], losses[:, 1], losses[:, 2],
                       alpha=0.8, c=self.colors[i], label=self.labels[i]
                       )
        x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10)
        y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 10)
        x, y = np.meshgrid(x, y)
        for i in range(len(self.spots)):
            z = np.full_like(x, i + 1)
            ax.plot_surface(x, y, z, color='k', alpha=0.1)
        if self.args.net_name == 'Conv2D':
            x_major_locator = MultipleLocator(0.002)
            ax.xaxis.set_major_locator(x_major_locator)
        ax.set_xlabel('Reconstruction loss', fontdict=self.font)
        ax.set_ylabel('Latent loss', fontdict=self.font)
        ax.set_zticks(np.arange(1, len(self.spots) + 1))
        ax.set_zticklabels(self.spots)
        ax.legend(bbox_to_anchor=(0.99, 1.05), ncol=3)


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
    loss = Loss(args)
    # loss.spot_loss_3d('W-2')
    loss.plot_loss_3d()
    plt.show()
