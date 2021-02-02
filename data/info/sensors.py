#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 31/01/21 10:51 AM
@description:  
@version: 1.0
"""


import numpy as np

levels = ['L1', 'L2']
components = {'Wall': ['A2', 'B1'],
              'Floor': ['A1', 'A2', 'B1', 'B2']
              }
dirs = ['NS', 'EW', 'V']
spots = []
sensors = []
for level in levels:
    for component in components.keys():
        locations = components[component]
        for location in locations:
            spot = '{}-{}-{}'.format(level, component, location)
            spots.append(spot)
            for d in dirs:
                sensor_name = 'A-{}-{}-{}-{}'.format(level, component, location, d)
                sensors.append(sensor_name)
spots = np.array(spots)
sensors = np.array(sensors)
np.save('./sensors.npy', sensors)
np.save('./spots.npy', spots)
