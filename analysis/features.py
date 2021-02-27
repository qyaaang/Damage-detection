#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 17/02/21 2:46 PM
@description:  
@version: 1.0
"""


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse


data_path = '../data/data_processed'
info_path = '../data/info'
save_path = '../results'


class FeatureReader:

    def __init__(self, args, dataset):
        self.args = args
        if dataset == 'W-1':
            self.feat = np.load('{}/features/{}.npy'.format(save_path, self.file_name()))
        else:
            self.feat = np.load('{}/features/test/{}_{}.npy'.
                                format(save_path, dataset, self.file_name())
                                )

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


def make_dataset(args, datasets):
    features = {}
    if args.net_name == 'MLP':
        data = np.empty((0, args.dim_feature))
    else:
        data = np.empty((0, args.num_hidden_map * 8))
    target = np.empty(0)
    for i, dataset in enumerate(datasets):
        f = FeatureReader(args, dataset)
        feat = f.feat
        if args.net_name == 'Conv2D': feat = feat.reshape(len(feat), -1)
        t = np.full(len(feat), i)
        data = np.concatenate((data, feat), axis=0)
        target = np.concatenate((target, t), axis=0)
    features['data'] = data
    features['target'] = target
    return features


def feat_dim_reduction(args, features):
    x = features['data']
    y = features['target']
    if args.decomposition == 'PCA':
        pca = PCA(n_components=args.num_component)
        x_r = pca.fit(x).transform(x)
    else:
        lda = LinearDiscriminantAnalysis(n_components=args.num_component)
        x_r = lda.fit(x, y).transform(x)
    return x_r, y


def show_feature(args, x_r, y):
    target_names = ['Intact', '25%', '50%', '100%']
    colors = ['g', 'b', 'y', 'r']
    if args.num_component == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
            ax.scatter(x_r[y == i, 0], x_r[y == i, 1], x_r[y == i, 2],
                       color=color, alpha=.8, lw=1,
                       label=target_name
                       )
        ax.legend(loc='best', shadow=False, scatterpoints=1)
    else:
        fig, ax = plt.subplots()
        for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
            ax.scatter(x_r[y == i, 0], x_r[y == i, 1],
                       color=color, alpha=.8, lw=1,
                       label=target_name
                       )
        ax.legend(loc='best', shadow=False, scatterpoints=1)
    plt.tight_layout()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decomposition', default='LDA', type=str)
    parser.add_argument('--num_component', default=3, type=int)
    parser.add_argument('--data_source', default='FFT', type=str)
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
    datasets = ['W-1', 'W-2', 'W-5', 'W-7']
    features = make_dataset(args, datasets)
    x_r, y = feat_dim_reduction(args, features)
    show_feature(args, x_r, y)
    plt.show()
