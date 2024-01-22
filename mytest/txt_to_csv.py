# time: 2023/12 
# Author: whb
# 代码功能：将数据集由txt格式转化为csv格式

import os
import numpy as np
import pandas as pd

def txt_to_csv(txt_path):
    csv_path = os.path.join(os.path.dirname(__file__)+'/csv', os.path.splitext(os.path.basename(txt_path))[0]+'.csv')
    data_txt = np.loadtxt(txt_path)
    data_txtDF = pd.DataFrame(data_txt)
    data_txtDF.to_csv(csv_path, header=False, index=False)
    return csv_path

if __name__ == '__main__':
    root_dir='/home/whb/29/训练验证数据集/训练数据集'
    print('Start')
    dev_dir_list = []
    dev_dir_names = os.listdir(root_dir) #返回目录列表 '/home/whb/29/训练验证数据集/训练数据集/信号类型1.txt'
    dev_dir_names = sorted(dev_dir_names)
    for n in dev_dir_names: #根据目录列表补充路径
        tmp = os.path.join(root_dir, n) 
        dev_dir_list.append(tmp) #device地址列表
    # print(dev_dir_list)
    for i, d in enumerate(dev_dir_list):
        d = txt_to_csv(d)
    print('Done')