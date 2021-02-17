#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 16/02/21 5:59 PM
@description:  
@version: 1.0
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys


if sys.platform == 'darwin':
    data_path = '/Users/qunyang/Dropbox (Uni of Auckland)/Concrete-shared/' \
                'Qun/ILEE project/Processed data/V2.0'
else:
    data_path = r'H:\EngFiles\ILEE project\Data\Processed data\V2.0'
with open('../data/info/folders.json') as f:
    folders = json.load(f)


class SegmentShow:

    def __init__(self, base_path, folders, white_noise, len_seg):
        self.white_noise = white_noise
        self.data = pd.read_csv('{}/{}/{}-processed.csv'.format(base_path,
                                                                folders[white_noise],
                                                                white_noise),
                                skiprows=[0, 1, 2, 4])
        self.fft_data = np.load('../data/data_processed/FFT/{}/{}_FFT.npy'.
                                format(len_seg, white_noise))
        self.spots = np.load('../data/info/spots.npy')
        self.len_seg = len_seg  # Length of segmentation
        with open('../data/info/segments_{}.json'.format(len_seg)) as f:
            self.num_seg = json.load(f)
        self.dirs = ['NS', 'EW', 'V']

    def show_segments(self, spot_name):
        fig, axs = plt.subplots(nrows=3, ncols=1)
        num_seg = self.num_seg[self.white_noise]
        for i, ax in enumerate(axs):
            signal = self.data['A-{}-{}'.format(spot_name, self.dirs[i])]
            ax.plot(signal, label=self.dirs[i])
            ax.legend(loc='upper right')
            ax.set_xticks([])
            ax.set_yticks([])
            # Show segments
            for seg_idx in range(num_seg):
                ax.axvline(x=seg_idx * self.len_seg, lw=0.3, ls='--', c='k')

    def show_fft_segments(self, spot_name, seg_idx=25):
        fft_data = self.fft_data[self.spots == spot_name].squeeze(0)
        num_channel = fft_data.shape[0]
        colors = ['b', 'r', 'g']
        fig_, ax_ = plt.subplots(figsize=(10, 1.5))
        ax_.set_xticks([])
        ax_.set_yticks([])
        _fig = plt.figure()
        _ax = _fig.add_subplot(111, projection='3d')
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.set_zticks([])
        # _ax.axis('off')
        for i in range(num_channel):
            fig, ax = plt.subplots(figsize=(5, 1.5))
            ax.plot(fft_data[i, seg_idx], c=colors[i], label=self.dirs[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(loc='upper right')
            # MLP
            seg = np.arange(i * 128, (i + 1) * 128)
            ax_.plot(seg, fft_data[i, seg_idx], c=colors[i])
            # Conv2D
            x = - np.ones(128) * (i + 1)
            y = np.arange(0, 128)
            z = fft_data[i, seg_idx]
            _ax.plot(x, y, z, c=colors[i])


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Arial'
    show = SegmentShow(data_path, folders, 'W-1', 500)
    show.show_segments('L1-Wall-B1')
    show.show_fft_segments('L1-Wall-B1')
    plt.show()
