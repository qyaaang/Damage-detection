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
        self.feat = np.load('{}/features/{}.npy'.format(save_path, self.file_name()))

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

    def test(self):
        path = '{}/models/{}.model'.format(save_path, self.file_name())
        self.AE.load_state_dict(torch.load(path))  # Load AutoEncoder
        self.AE.eval()
        damage_indices = {}
        with torch.no_grad():
            for i, spot in enumerate(self.spots):
                damage_indices[spot] = {}
                data_origin = torch.from_numpy(self.testset[i])
                data_origin = torch.tensor(data_origin, dtype=torch.float32)
                feature_origin = self.feat[i: i + data_origin.size(0)]
                if self.args.net_name == 'Conv2D': data_origin = data_origin.unsqueeze(2)
                data_reconstruct, feature_reconstruct = self.AE(data_origin)
                c_res = ((data_reconstruct - data_origin) ** 2).mean()
                f_res = ((feature_reconstruct - feature_origin) ** 2).mean()
                damage_index = 0
                # damage_indices[spot]['Generate residual'] = res.item()
                # damage_indices[spot]['Discriminate loss'] = np.abs(dis.item())
                print('[{}]\tLoss: {:5f}\tLoss: {:5f}'.
                      format(spot, c_res.item(), f_res.item())
                      )
                i += data_origin.size(0)
        # damage_indices = json.dumps(damage_indices, indent=2)
        # with open('{}/damage index/{}_{}.json'.format(save_path,
        #                                               self.args.dataset,
        #                                               self.file_name()
        #                                  ), 'w') as f:
        #     f.write(damage_indices)


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
    args = parser.parse_args()
    detector = DamageDetection(args)
    detector()


if __name__ == '__main__':
    main()
