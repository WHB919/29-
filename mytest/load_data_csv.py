# time: 2023/12 
# Author: whb
# 代码功能：导入数据集

import os
import numpy as np
import tensorflow as tf
import utils
import pandas as pd

def normalizeData(v):
    # 归一化
    # keepdims makes the result shape (1, 1, 3) instead of (3,). This doesn't matter here, but
    # would matter if you wanted to normalize over a different axis.
    v_min = v.min(axis=(0, 1), keepdims=True)
    v_max = v.max(axis=(0, 1), keepdims=True)
    v = (v - v_min) / (v_max - v_min)
    return v

def add_noise(signal, SNR):
    noise = np.random.randn(signal.shape[0], signal.shape[1])  # 产生N(0,1)噪声数据
    noise = noise - np.mean(noise)  # 均值为0
    signal_power = np.linalg.norm(signal) ** 2 / signal.size  # 此处是信号的std**2
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    signal_noise = noise + signal
    return signal_noise

def read_csv(filepath):
    """读取csv文件, 并返回两个数组, 一个包含前5列数据, 一个包含最后一列数据"""
    df = pd.read_csv(filepath, delimiter=',')
    data_array = df.iloc[:, :5].values
    labels_array = df.iloc[:, 5].values
    return data_array, labels_array

def loadData(args):

    dev_dir_list = []
    dev_dir_names = os.listdir(args.root_dir) #返回目录列表 '/信号类型01.csv'
    dev_dir_names = sorted(dev_dir_names)
    # print(dev_dir_names)
    for n in dev_dir_names: #根据目录列表补充路径
        tmp = os.path.join(args.root_dir, n) 
        dev_dir_list.append(tmp) #device地址列表 '/home/whb/29/mytest/csv/信号类型01.csv'
    # print(dev_dir_list)
    n_devices = len(dev_dir_list) #设备个数

    x_train, y_train, x_test, y_test = [], [], [], []
    split_ratio = {'train': 0.8, 'val': 0.2}
    X_data_pd = []
    Y_data_pd = []
    for i, d in enumerate(dev_dir_list):
        pre_X_data, pre_Y_data = read_csv(d)

        if i == 0:
            X_data_pd = pre_X_data
            Y_data_pd = pre_Y_data
        else:
            X_data_pd = np.concatenate((X_data_pd, pre_X_data), axis=0)
            Y_data_pd = np.concatenate((Y_data_pd, pre_Y_data), axis=0)

        # split one class data
        x_train_pd, x_test_pd, y_train_pd, y_test_pd = [], [], [], []
        x_train_pd, y_train_pd, x_test_pd, y_test_pd = utils.splitData(split_ratio, X_data_pd, Y_data_pd)
        if i == 0:
            x_train, x_test = x_train_pd, x_test_pd
            y_train, y_test = y_train_pd, y_test_pd
        else:
            x_train = np.concatenate((x_train, x_train_pd), axis=0)
            x_test = np.concatenate((x_test, x_test_pd), axis=0)
            y_train = np.concatenate((y_train, y_train_pd), axis=0)
            y_test = np.concatenate((y_test, y_test_pd), axis=0)
        del pre_X_data
        del pre_Y_data

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    y_train = tf.keras.utils.to_categorical((y_train-1), n_devices) #one-hot编码
    y_test = tf.keras.utils.to_categorical((y_test-1), n_devices)

    return x_train, y_train, x_test, y_test, n_devices

class loadDataOpts():
    def __init__(self, root_dir):
        self.root_dir = root_dir

if __name__ == '__main__':
    opts = loadDataOpts(root_dir='/home/whb/29/mytest/csv')
    x_train, y_train, x_test, y_test, NUM_CLASS = loadData(opts)
    print('train data shape: ', x_train.shape, 'train label shape: ', y_train.shape)
    print('test data shape: ', x_test.shape, 'test label shape: ', y_test.shape)
    print('NUM_CLASS: ', NUM_CLASS)
    print('all test done!')
