# -*- coding: utf-8 -*-
# time: 2023/12
# Author: whb
'''
This module is for universal arg parse
'''

import argparse


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='/home/whb/29/mytest/csv', help='input file/dir')
    parser.add_argument('-o', '--output', default='/home/whb/29/result', help='output dir')
    # parser.add_argument('-m', '--modelType', default='homegrown', help='choose from homegrown/baseline/resnet')
    parser.add_argument('-sp', '--splitType', default='random', help='choose from random/order')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-n', '--normalize', action='store_true', help='')
    parser.add_argument('-ds', '--dataSource', default='neu',help='choose from neu/simu')
    parser.add_argument('-cf', '--channel_first', action='store_true', help='if set channel first otherwise channel last')
    parser.add_argument('-l', '--location',default='before_fft' , help='where the data collected')
    opts = parser.parse_args()
    return opts
