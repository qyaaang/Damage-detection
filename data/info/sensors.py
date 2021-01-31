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


import json

sensors = {'L1': {'Wall': ['A-L1-Wall-A2-NS',
                           'A-L1-Wall-A2-EW',
                           'A-L1-Wall-A2-V',
                           'A-L1-Wall-B1-NS',
                           'A-L1-Wall-B1-EW',
                           'A-L1-Wall-B1-V'
                           ],
                  'Floor': ['A-L1-Floor-A1-NS',
                            'A-L1-Floor-A1-EW',
                            'A-L1-Floor-A1-V',
                            'A-L1-Floor-A2-NS',
                            'A-L1-Floor-A2-EW',
                            'A-L1-Floor-A2-V',
                            'A-L1-Floor-B1-NS',
                            'A-L1-Floor-B1-EW',
                            'A-L1-Floor-B1-V',
                            'A-L1-Floor-B2-NS',
                            'A-L1-Floor-B2-EW',
                            'A-L1-Floor-B2-V'
                            ]
                  },
           'L2': {'Wall': ['A-L2-Wall-A2-NS',
                           'A-L2-Wall-A2-EW',
                           'A-L2-Wall-A2-V',
                           'A-L2-Wall-B1-NS',
                           'A-L2-Wall-B1-EW',
                           'A-L2-Wall-B1-V'
                           ],
                  'Floor': ['A-L2-Floor-A1-NS',
                            'A-L2-Floor-A1-EW',
                            'A-L2-Floor-A1-V',
                            'A-L2-Floor-A2-NS',
                            'A-L2-Floor-A2-EW',
                            'A-L2-Floor-A2-V',
                            'A-L2-Floor-B1-NS',
                            'A-L2-Floor-B1-EW',
                            'A-L2-Floor-B1-V',
                            'A-L2-Floor-B2-NS',
                            'A-L2-Floor-B2-EW',
                            'A-L2-Floor-B2-V'
                            ]
                  }
           }

sensors = json.dumps(sensors, indent=2)
with open('./sensors.json', 'w') as f:
    f.write(sensors)
