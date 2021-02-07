#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 7/02/21 3:23 PM
@description:  
@version: 1.0
"""


from torch import nn
from layers.MLP import Dis_MLP


net_classes = {'MLP': Dis_MLP}


class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        net = net_classes[args.net_name]
        self.net = net(dim_input=args.dim_input,
                       dim_hidden=args.dim_hidden,
                       )

    def forward(self, x):
        output = self.net(x)
        return output
