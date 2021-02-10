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

from torch import nn


ngf = 128
ndf = 128
nc = 3


class AutoEncoder(nn.Module):

    def __init__(self, args):
        super(AutoEncoder, self).__init__()
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
            nn.Linear(args.dim_feature, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, args.dim_input),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output, z


class AutoEncoderConv(nn.Module):

    def __init__(self):
        super(AutoEncoderConv, self).__init__()
        self.encoder = nn.Sequential(
            # input is (nc) x 1 x 128
            nn.Conv2d(nc, ndf, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 1 x 64
            nn.Conv2d(ndf, ndf * 2, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 1 x 32
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 1 x 16
        )
        self.decoder = nn.Sequential(
            # state size. (ngf*4) x 1 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 1 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 1 x 64
            nn.ConvTranspose2d(ngf, nc, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), bias=False),
            nn.Sigmoid()
            # state size. (nc) x 1 x 128
        )

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output, z
