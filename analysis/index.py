#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 18/02/21 9:28 AM
@description:  
@version: 1.0
"""


import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse


info_path = '../data/info'
save_path = '../results'


class Index:

    def __init__(self, args):
        self.args = args
        self.datasets = ['W-2', 'W-5', 'W-7']
        self.spots = np.load('{}/spots.npy'.format(info_path))
        self.locs = {'Wall-A2': [-1000, 0], 'Wall-B1': [4475, 3700],
                     'Floor-A1': [0, 2700], 'Floor-A2': [0, 0],
                     'Floor-B1': [4475, 2700], 'Floor-B2': [4475, 0]
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

    def spot_coords(self):
        coords = np.empty((0, 2))
        for _, loc in self.locs.items():
            coords = np.concatenate((coords, np.array(loc).reshape(1, -1)), axis=0)
        return coords

    def spot_indices(self, dataset):
        data = self.load_data(dataset)
        indices = []
        for spot, _ in self.locs.items():
            index = data['{}-{}'.format(self.args.level, spot)]['Damage index']
            indices.append(index)
        return np.array(indices)

    def damage_surface(self, dataset):
        locs = [['Floor-A2', 'Floor-B2'],
                ['Floor-A1', 'Floor-B1']
                ]
        data = self.load_data(dataset)
        x = np.linspace(250, 4475 + 250, 2)
        y = np.linspace(250, 2700 + 250, 2)
        x, y = np.meshgrid(x, y)
        z = np.empty((0, 2))
        for loc in locs:
            tmp = np.array([data['{}-{}'.format(self.args.level, loc[0])]['Damage index'],
                            data['{}-{}'.format(self.args.level, loc[1])]['Damage index']]).reshape(-1, 2)
            z = np.concatenate((z, tmp), axis=0)
        return x, y, z

    def plot_index(self):
        font = {'family': 'Arial',
                'style': 'normal',
                'weight': 'bold',
                'size': 12,
                'color': 'k',
                }
        colors = ['b', 'g', 'r']
        labels = ['25%', '50%', '100%']
        coords = self.spot_coords()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = coords[:, 0]
        y = coords[:, 1]
        z = np.zeros_like(x)
        dx = dy = 500 * np.ones_like(z)
        tmp = np.zeros_like(z)
        for i, dataset in enumerate(self.datasets):
            indices = self.spot_indices(dataset)
            dz = indices if i == 0 else indices - tmp
            ax.bar3d(x, y, z, dx, dy, dz, color=colors[i], zsort='average')
            z += dz
            tmp = indices
            # Plot damage surface
            a, b, c = self.damage_surface(dataset)
            ax.plot_surface(a, b, c, alpha=0.5, color=colors[i])
        for idx, (spot_name, _) in enumerate(self.locs.items()):
            ax.text(coords[idx, 0] + 600, coords[idx, 1], indices[idx] + 0.05,
                    spot_name, ha='center', va='bottom', fontdict=font, fontsize=8
                    )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('EW', fontdict=font)
        ax.set_ylabel('NS', fontdict=font)
        ax.set_zlabel('Probability of damage', fontdict=font)
        handles = [mpatches.Rectangle((0, 0), 1, 1) for _ in range(len(labels))]
        hmap = dict(zip(handles, [Handler(idx) for idx in range(len(labels))]))
        ax.legend(handles=handles, labels=labels, handler_map=hmap,
                  bbox_to_anchor=(0.99, 1.05), ncol=3
                  )
        plt.tight_layout()


class Handler:

    def __init__(self, index):
        self.index = index

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle((x0, y0), width, height,
                                   facecolor='b',
                                   transform=handlebox.get_transform())
        patch2 = mpatches.Rectangle((x0 + 3 * width / 5, y0), 2 * width / 5, height,
                                    facecolor='g',
                                    transform=handlebox.get_transform()
                                    )
        patch3 = mpatches.Rectangle((x0 + 21 * width / 25, y0), 4 * width / 25, height,
                                    facecolor='r',
                                    transform=handlebox.get_transform()
                                    )
        if self.index == 0:
            handlebox.add_artist(patch)
        elif self.index == 1:
            handlebox.add_artist(patch)
            handlebox.add_artist(patch2)
        else:
            handlebox.add_artist(patch)
            handlebox.add_artist(patch2)
            handlebox.add_artist(patch3)
        return patch


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
    parser.add_argument('--level', default='L1', type=str)
    args = parser.parse_args()
    plt.rcParams['font.family'] = 'Arial'
    d = Index(args)
    d.plot_index()
    plt.show()
