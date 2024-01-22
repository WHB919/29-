import os
import sys
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf

def read_txt(filepath):
    """读取txt文件, 并返回一个数组, 包含前5列数据
    """
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()  # 去除行末的换行符
            parts = line.split()  # 假设每行的数据是用空格分隔的
            data.append(parts[:5])
    # print(data)
    # 将数组转换为numpy数组
    data_array = np.array(data)
    data_array = data_array.astype(float)
    # print(data_array)

    return data_array

def normalizeData(v):
    # 归一化
    v_min = v.min(axis=(0, 1), keepdims=True)
    v_max = v.max(axis=(0, 1), keepdims=True)
    v = (v - v_min) / (v_max - v_min)
    return v

def zip_with_cmd(to_zipped_dir, zipped_filepath):
    """使用命令行压缩
    """
    dirpath = os.path.dirname(to_zipped_dir)
    os.system("cd %s && zip -r %s submit/"%(dirpath, zipped_filepath))

def zip_with_python(to_zipped_dir, zipped_filepath):
    """使用python包压缩
    """
    zipfile_object = zipfile.ZipFile(zipped_filepath, "w")
    for topdir, _, filenames in os.walk(to_zipped_dir):
        for filename in filenames:
            filepath = os.path.join(topdir, filename)
            arcfilepath = os.path.join("submit", filename)
            zipfile_object.write(filepath, arcname=arcfilepath)

def model_test(test_data, model):
    predictions = model.predict(test_data)
    # print(predictions)
    class_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'UN']
    # 输出分类结果
    label = []
    for i, prediction in enumerate(predictions):
        max_prob = np.max(prediction)
        max_index = np.argmax(prediction)
        if max_prob < 0.5:  # 设置一个阈值，用于判断是否明确指向某个类型
            predicted_label = class_labels[-1]  # -1表示未知类型
        else:
            predicted_label = class_labels[max_index]
        label.append(predicted_label)
    # print(label)
    return label

def main(to_pred_path, result_save_path):
    """主函数, 入参不可修改
        to_pred_path: 待预测文件夹路径
        result_save_path: 结果文件(zip)保存路径
    """
    #! 获取run.py运行文件夹$dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    #! 在$dir内创建一个submit文件夹
    submit_dir = os.path.join(current_dir,"submit")
    os.makedirs(submit_dir)

    #! 获取待预测文件路径, 不可修改
    path1 = os.path.join(to_pred_path, "Scene1.txt") #! 场景1
    path2 = os.path.join(to_pred_path, "Scene2.txt") #! 场景2
    path3 = os.path.join(to_pred_path, "Scene3.txt") #! 场景3
    # print(path1)

    #! 读取待预测文件, 读取文件函数可自行修改
    # print(path1)
    df1 = read_txt(path1)
    df1 = normalizeData(df1)
    # print(path2)
    df2 = read_txt(path2)
    df2 = normalizeData(df2)
    # print(path3)
    df3 = read_txt(path3)
    df3 = normalizeData(df3)
    # print(df1)

    """模型推理(略, 自行完善补充)
    """
    model = tf.keras.models.load_model(os.path.join(current_dir, "mymodel.h5"))
    result1 = model_test(df1, model)
    result2 = model_test(df2, model)
    result3 = model_test(df3, model)
    # print(result1)

    #! 推理结果整理, 请根据实际情况修改
    result1 = pd.DataFrame(result1)
    result2 = pd.DataFrame(result2)
    result3 = pd.DataFrame(result3)

    # print(result1)

    #! 推理结果文件写到新创建的submit文件夹中
    target_path1 = os.path.join(submit_dir, "Scene1.txt") #! 场景1
    target_path2 = os.path.join(submit_dir, "Scene2.txt") #! 场景2
    target_path3 = os.path.join(submit_dir, "Scene3.txt") #! 场景3

    result1.to_csv(target_path1, index=None, header=None) #!!! 注意, 写入文件时, 不要带表头
    result2.to_csv(target_path2, index=None, header=None) #!!! 注意, 写入文件时, 不要带表头
    result3.to_csv(target_path3, index=None, header=None) #!!! 注意, 写入文件时, 不要带表头
    
    #! 推理结果打包成zip， 二选一均可, 或其他压缩函数均可
    zip_with_cmd(submit_dir, result_save_path)
    # zip_with_python(submit_dir, result_save_path)

if __name__ == "__main__":
    #! 以下内容请勿修改, 因修改造成的评估失败由选手自行承担
    to_pred_path  = sys.argv[1] # 数据路径
    result_save_path = sys.argv[2] # 输出路径
    main(to_pred_path,result_save_path) # 运行主函数
