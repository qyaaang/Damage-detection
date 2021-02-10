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
from models.Generator import Generator
from models.Discriminator import Discriminator


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
                                       len_seg=self.args.len_seg
                                       )
        self.testset = torch.tensor(torch.from_numpy(white_noise.dataset_), dtype=torch.float32)
        self.spots = np.load('{}/spots.npy'.format(info_path))
        self.Generator = Generator(args)  # Generator
        self.Discriminator = Discriminator(args)  # Discriminator

    def __call__(self, *args, **kwargs):
        self.test()

    def file_name(self):
        return '{}_{}_{}_{}_{}_{}'.format(self.args.model_name,
                                          self.args.net_name,
                                          self.args.len_seg,
                                          self.args.optimizer,
                                          self.args.learning_rate,
                                          self.args.num_epoch
                                          )

    def test(self):
        path_gen = '{}/models/{}_Gen.model'.format(save_path, self.file_name())
        path_dis = '{}/models/{}_Dis.model'.format(save_path, self.file_name())
        self.Generator.load_state_dict(torch.load(path_gen))  # Load Generator
        self.Discriminator.load_state_dict(torch.load(path_dis))  # Load Discriminator
        self.Generator.eval()
        self.Discriminator.eval()
        damage_indices = {}
        beta = 0.5
        with torch.no_grad():
            for i, spot in enumerate(self.spots):
                damage_indices[spot] = {}
                z = torch.randn(self.testset.shape[1], 50)
                data_gen = self.Generator(z)
                data_real = self.testset[i]
                res = ((data_gen - data_real) ** 2).mean()
                dis = self.Discriminator(data_gen).mean() - 1
                loss = beta * res.item() + (1 - beta) * np.abs(dis.item())
                damage_indices[spot]['Generate residual'] = res.item()
                damage_indices[spot]['Discriminate loss'] = np.abs(dis.item())
                damage_indices[spot]['Loss'] = loss
                print('[{}]\tGenerate residual: {:5f}\tDiscriminate loss: {:5f}\tLoss: {:5f}'.
                      format(spot, res.item(), np.abs(dis.item()), loss)
                      )
        damage_indices = json.dumps(damage_indices, indent=2)
        with open('{}/damage index/{}_{}.json'.format(save_path,
                                                      self.args.dataset,
                                                      self.file_name()
                                         ), 'w') as f:
            f.write(damage_indices)


def main():
    # Hyper-parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='W-2', type=str)
    parser.add_argument('--model_name', default='GAN', type=str)
    parser.add_argument('--net_name', default='MLP', type=str)
    parser.add_argument('--len_seg', default=400, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--dim_noise', default=50, type=int)
    parser.add_argument('--dim_input', default=384, type=int)
    parser.add_argument('--dim_hidden', default=1000, type=int)
    parser.add_argument('--dim_output', default=384, type=int)
    parser.add_argument('--seed', default=1993, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--learning_rate', default=1e-2, type=float)
    args = parser.parse_args()
    detector = DamageDetection(args)
    detector()


if __name__ == '__main__':
    main()
