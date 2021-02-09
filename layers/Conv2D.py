#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 9/02/21 3:26 PM
@description:  
@version: 1.0
"""


from torch import nn


nz = 100
ngf = 128
ndf = 128
nc = 3


class Conv2D_G(nn.Module):

    def __init__(self):
        super(Conv2D_G, self).__init__()
        self.conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=(1, 8), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 1 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
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
            nn.Tanh()
            # state size. (nc) x 1 x 128
        )

    def forward(self, x):
        output = self.conv(x)
        return output


class Conv2D_D(nn.Module):

    def __init__(self):
        super(Conv2D_D, self).__init__()
        self.conv = nn.Sequential(
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
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 1 x 8
            nn.Conv2d(ndf * 8, 1, kernel_size=(1, 8), stride=(1, 1), padding=(0, 0), bias=False),
            nn.Sigmoid()
            # state size. 1 x 1 x 1
        )

    def forward(self, x):
        output = self.conv(x)
        return output.view(-1)
