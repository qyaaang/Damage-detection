#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 7/02/21 2:57 PM
@description:  
@version: 1.0
"""


from torch import nn


class Gen_MLP(nn.Module):

    def __init__(self, dim_noise, dim_hidden, dim_output):
        super(Gen_MLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(dim_noise, dim_hidden),
                                 nn.ReLU(True),
                                 nn.Linear(dim_hidden, dim_hidden),
                                 nn.ReLU(True),
                                 nn.Linear(dim_hidden, dim_hidden),
                                 nn.ReLU(True),
                                 nn.Linear(dim_hidden, dim_output)
                                 )

    def forward(self, z):
        output = self.net(z)
        return output


class Dis_MLP(nn.Module):

    def __init__(self, dim_input, dim_hidden):
        super(Dis_MLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(dim_input, dim_hidden),
                                 nn.ReLU(True),
                                 nn.Linear(dim_hidden, dim_hidden),
                                 nn.ReLU(True),
                                 nn.Linear(dim_hidden, dim_hidden),
                                 nn.ReLU(True),
                                 nn.Linear(dim_hidden, 1),
                                 nn.Sigmoid()
                                 )

    def forward(self, x):
        output = self.net(x)
        return output
