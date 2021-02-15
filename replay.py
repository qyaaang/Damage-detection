#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 15/02/21 2:49 PM
@description:  
@version: 1.0
"""


import visdom
import argparse


class Replay:

    def __init__(self, args):
        self.args = args
        self.vis = visdom.Visdom()

    def __call__(self, *args, **kwargs):
        self.replay_log()

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

    def replay_log(self):
        self.vis.replay_log(log_filename='./results/visualization/{}.log'.
                            format(self.file_name())
                            )


def main():
    # Hyper-parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='AE', type=str)
    parser.add_argument('--net_name', default='MLP', type=str)
    parser.add_argument('--len_seg', default=400, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str)
    # Conv2D setting
    parser.add_argument('--num_hidden_map', default=256, type=int)
    parser.add_argument('--num_epoch', default=1000, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    args = parser.parse_args()
    replay = Replay(args)
    replay()


if __name__ == '__main__':
    main()
