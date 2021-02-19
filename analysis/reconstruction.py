#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 17/02/21 11:22 AM
@description:  
@version: 1.0
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
from data_processing.dataset_reader import DatasetReader
from models.AutoEncoder import AutoEncoder
import argparse


data_path = '../data/data_processed'
info_path = '../data/info'
save_path = '../results'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Reconstruction:

    def __init__(self, args):
        self.args = args
        white_noise = DatasetReader(white_noise='W-1',
                                    data_path=data_path,
                                    data_source=args.data_source,
                                    len_seg=self.args.len_seg
                                    )
        self.dataset, _ = white_noise(args.net_name)
        self.spots = np.load('{}/spots.npy'.format(info_path))
        self.AE = AutoEncoder(args)

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

    def load_model(self):
        path = '{}/models/{}/{}.model'.format(save_path,
                                              self.args.model_name,
                                              self.file_name()
                                              )
        self.AE.load_state_dict(torch.load(path, map_location=torch.device(device)))  # Load AutoEncoder

    def show_reconstruct(self, seg_idx=25):
        self.load_model()
        fig, axs = plt.subplots(nrows=int(len(self.spots) / 2),
                                ncols=2,
                                figsize=(15, 15)
                                )
        with torch.no_grad():
            num_seg = int(self.dataset.shape[0] / len(self.spots))
            spots_l1, spots_l2 = np.hsplit(self.spots, 2)
            for i, (spot_l1, spot_l2) in enumerate(zip(spots_l1, spots_l2)):
                # L1 sensors
                x = self.dataset[i * num_seg + seg_idx]
                x = x.to(device)
                axs[i][0].set_title('{}-{}'.format(spot_l1, seg_idx))
                axs[i][0].plot(x.view(-1).detach().cpu().numpy(), c='b', label='Original')
                if self.args.net_name == 'Conv2D': x = x.unsqueeze(0).unsqueeze(2)
                x_hat, _, _ = self.AE(x)
                axs[i][0].plot(x_hat.view(-1).detach().cpu().numpy(),
                               ls='--',
                               c='r',
                               label='Reconstructed')
                axs[i][0].axvline(x=127, ls='--', c='k')
                axs[i][0].axvline(x=255, ls='--', c='k')
                axs[i][0].set_xticks(np.linspace(self.args.dim_input / 6,
                                                 self.args.dim_input - self.args.dim_input / 6,
                                                 3
                                                 )
                                     )
                axs[i][0].set_xticklabels(['NS', 'EW', 'V'])
                axs[i][0].legend(loc='upper center')
                # L2 sensors
                x = self.dataset[(i + 5) * num_seg + seg_idx]
                x = x.to(device)
                axs[i][1].plot(x.view(-1).detach().cpu().numpy(), c='b', label='Original')
                axs[i][1].set_title('{}-{}'.format(spot_l2, seg_idx))
                if self.args.net_name == 'Conv2D': x = x.unsqueeze(0).unsqueeze(2)
                x_hat, _, _ = self.AE(x)
                axs[i][1].plot(x_hat.view(-1).detach().cpu().numpy(),
                               ls='--',
                               c='r',
                               label='Reconstructed')
                axs[i][1].axvline(x=127, ls='--', c='k')
                axs[i][1].axvline(x=255, ls='--', c='k')
                axs[i][1].set_xticks(np.linspace(self.args.dim_input / 6,
                                                 self.args.dim_input - self.args.dim_input / 6,
                                                 3
                                                 )
                                     )
                axs[i][1].set_xticklabels(['NS', 'EW', 'V'])
                axs[i][1].legend(loc='upper center')
            plt.subplots_adjust(hspace=0.5)
            plt.show()

    def show_latent_reconstruction(self, seg_idx=25):
        self.load_model()
        fig, axs = plt.subplots(nrows=int(len(self.spots) / 2),
                                ncols=2,
                                figsize=(15, 15)
                                )
        with torch.no_grad():
            num_seg = int(self.dataset.shape[0] / len(self.spots))
            spots_l1, spots_l2 = np.hsplit(self.spots, 2)
            for i, (spot_l1, spot_l2) in enumerate(zip(spots_l1, spots_l2)):
                # L1 sensors
                x = self.dataset[i * num_seg + seg_idx]
                x = x.to(device)
                axs[i][0].set_title('{}-{}'.format(spot_l1, seg_idx))
                if self.args.net_name == 'Conv2D': x = x.unsqueeze(0).unsqueeze(2)
                _, z, z_hat = self.AE(x)
                axs[i][0].plot(z.view(-1).detach().cpu().numpy(),
                               ls='--',
                               c='b',
                               label='Original')
                axs[i][0].plot(z_hat.view(-1).detach().cpu().numpy(),
                               ls='--',
                               c='r',
                               label='Reconstructed')
                axs[i][0].set_xticks(np.linspace(0, z.view(-1).size(0) - 1, 3))
                axs[i][0].set_xticklabels([1, '...', z.view(-1).size(0)])
                axs[i][0].legend(loc='upper center')
                # L2 sensors
                x = self.dataset[(i + 5) * num_seg + seg_idx]
                x = x.to(device)
                axs[i][1].set_title('{}-{}'.format(spot_l2, seg_idx))
                if self.args.net_name == 'Conv2D': x = x.unsqueeze(0).unsqueeze(2)
                _, z, z_hat = self.AE(x)
                axs[i][1].plot(z.view(-1).detach().cpu().numpy(),
                               ls='--',
                               c='b',
                               label='Original')
                axs[i][1].plot(z_hat.view(-1).detach().cpu().numpy(),
                               ls='--',
                               c='r',
                               label='Reconstructed')
                axs[i][1].set_xticks(np.linspace(0, z.view(-1).size(0) - 1, 3))
                axs[i][1].set_xticklabels([1, '...', z.view(-1).size(0)])
                axs[i][1].legend(loc='upper center')
            plt.subplots_adjust(hspace=0.5)
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    show = Reconstruction(args)
    show.show_reconstruct()
    show.show_latent_reconstruction()
