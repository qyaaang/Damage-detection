#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 10/02/21 12:03 PM
@description:  
@version: 1.0
"""


import torch
from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.args = args
        dim_feature = args.dim_feature / 2 if args.model_name == 'VAE' else args.dim_feature
        self.encoder = nn.Sequential(
            nn.Linear(args.dim_input, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, args.dim_feature),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(dim_feature), 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, args.dim_input),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        if self.args.model_name == 'VAE':
            z_ = self.encoder(x)
            miu, sigma = z_.chunk(2, dim=1)
            z = miu + sigma * torch.randn_like(sigma)
            kld = 0.5 * torch.sum(
                torch.pow(miu, 2) +
                torch.pow(sigma, 2) -
                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
            ) / (batch_size * x.size(1))
            output = self.decoder(z)
            return output, z, kld
        else:
            z = self.encoder(x)
            output = self.decoder(z)
            return output, z


class AutoEncoderConv(nn.Module):

    def __init__(self, args):
        super(AutoEncoderConv, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            # [b, 3, 1, 128]
            nn.Conv2d(3, args.num_feature_map, kernel_size=(1, 4),
                      stride=(1, 2), padding=(0, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # [b, num_feature_map, 1, 64]
            nn.Conv2d(args.num_feature_map, args.num_feature_map * 2, kernel_size=(1, 4),
                      stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(args.num_feature_map * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # [b, num_feature_map * 2, 1, 32]
            nn.Conv2d(args.num_feature_map * 2, args.num_feature_map * 4, kernel_size=(1, 4),
                      stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(args.num_feature_map * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # [b, num_feature_map * 4, 1, 16]
            nn.Conv2d(args.num_feature_map * 4, args.num_hidden_map, kernel_size=(1, 4),
                      stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(args.num_hidden_map),
            nn.LeakyReLU(0.2, inplace=True),
            # [b, num_hidden_map, 1, 8]
        )
        self.decoder = nn.Sequential(
            # [b, num_hidden, 1, 8]
            nn.ConvTranspose2d(args.num_hidden_map, args.num_feature_map * 4, kernel_size=(1, 4),
                               stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(args.num_feature_map * 4),
            nn.ReLU(True),
            # [b, num_feature_map * 4, 1, 16]
            nn.ConvTranspose2d(args.num_feature_map * 4, args.num_feature_map * 2, kernel_size=(1, 4),
                               stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(args.num_feature_map * 2),
            nn.ReLU(True),
            # [b, num_feature_map * 2, 1, 32]
            nn.ConvTranspose2d(args.num_feature_map * 2, args.num_feature_map, kernel_size=(1, 4),
                               stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(args.num_feature_map),
            nn.ReLU(True),
            # [b, num_feature_map, 1, 64]
            nn.ConvTranspose2d(args.num_feature_map, 3, kernel_size=(1, 4),
                               stride=(1, 2), padding=(0, 1), bias=False),
            nn.Sigmoid()
            # [b, 3, 1, 128]
        )

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output, z
