#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 7/02/21 5:22 PM
@description:  
@version: 1.0
"""

import torch
import numpy as np
import data_processing as dp
import argparse
import json
from models.AutoEncoder import AutoEncoder

data_path = './data/data_processed'
info_path = './data/info'
save_path = './results'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DamageDetection:

    def __init__(self, args):
        self.args = args
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        print('{} detection...'.format(args.dataset))
        white_noise = dp.DatasetReader(white_noise=self.args.dataset,
                                       data_path=data_path,
                                       data_source=args.data_source,
                                       len_seg=self.args.len_seg
                                       )
        _, self.testset = white_noise(args.net_name)
        self.spots = np.load('{}/spots.npy'.format(info_path))
        self.AE = AutoEncoder(args)
        self.latent = np.load('{}/features/{}.npy'.format(save_path, self.file_name()))

    def __call__(self, *args, **kwargs):
        self.test()

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

    def damage_index(self, err):
        return 1 - np.exp(- self.args.alpha * err)

    def test(self):
        path = '{}/models/{}/{}.model'.format(save_path,
                                              self.args.model_name,
                                              self.file_name()
                                              )
        self.AE.load_state_dict(torch.load(path, map_location=torch.device(device)))  # Load AutoEncoder
        self.AE.eval()
        damage_indices = {}
        size_0 = self.testset.size(0)
        size_1 = self.testset.size(1)
        if self.args.net_name == 'MLP':
            if self.args.model_name == 'VAE':
                feats = torch.zeros(size_0 * size_1, int(self.args.dim_feature / 2))
            else:
                feats = torch.zeros(size_0 * size_1, int(self.args.dim_feature))
        else:
            feats = torch.zeros(size_0 * size_1, self.args.num_hidden_map * 8)
        idx = 0
        with torch.no_grad():
            for i, spot in enumerate(self.spots):
                damage_indices[spot] = {}
                x = self.testset[i]
                x_size = x.size(0)
                # z_w1 = self.latent[idx: idx + x_size]
                if self.args.net_name == 'Conv2D': x = x.unsqueeze(2)
                x_hat, z, z_hat = self.AE(x)
                if self.args.net_name == 'Conv2D':  # Flatten
                    z = z.reshape(x_size, -1)
                    z_hat = z_hat.reshape(x_size, -1)
                feats[idx: idx + x_size] = z  # Latent
                loss_x = ((x - x_hat) ** 2).mean()  # Reconstruction loss
                loss_z = ((z - z_hat) ** 2).mean()  # Latent loss
                loss = self.args.beta * loss_x.item() + (1 - self.args.beta) * loss_z.item()  # Overall loss
                damage_index = self.damage_index(loss)
                damage_indices[spot]['Reconstruction loss'] = loss_x.item()
                damage_indices[spot]['Latent loss'] = loss_z.item()
                damage_indices[spot]['Damage index'] = damage_index
                print('\033[1;32m[{}]\033[0m\t'
                      '\033[1;31mReconstruction loss: {:5f}\033[0m\t'
                      '\033[1;33mLatent loss: {:5f}\033[0m\t'
                      '\033[1;34mLoss: {:5f}\033[0m\t'
                      '\033[1;35mDamage index: {:5f}\033[0m'.
                      format(spot, loss_x.item(), loss_z.item(),
                             loss, damage_index)
                      )
                idx += x_size
        damage_indices = json.dumps(damage_indices, indent=2)
        with open('{}/damage index/{}_{}.json'.format(save_path,
                                                      self.args.dataset,
                                                      self.file_name()
                                                      ), 'w') as f:
            f.write(damage_indices)
        np.save('{}/features/test/{}_{}.npy'.
                format(save_path, self.args.dataset, self.file_name()), feats
                )


def main():
    # Hyper-parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='W-2', type=str)
    parser.add_argument('--data_source', default='FFT', type=str)
    parser.add_argument('--model_name', default='AE', type=str)
    parser.add_argument('--net_name', default='MLP', type=str)
    parser.add_argument('--len_seg', default=400, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--initializer', default='xavier_normal_', type=str)
    # MLP setting
    parser.add_argument('--dim_input', default=384, type=int)
    parser.add_argument('--dim_feature', default=20, type=int)
    # Conv2D setting
    parser.add_argument('--num_feature_map', default=128, type=int)
    parser.add_argument('--num_hidden_map', default=256, type=int)
    parser.add_argument('--seed', default=23, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=0.5, type=float)
    args = parser.parse_args()
    detector = DamageDetection(args)
    detector()


if __name__ == '__main__':
    main()
