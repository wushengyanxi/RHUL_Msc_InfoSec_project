import sys
import random
from random import shuffle
import numpy as np
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\Machine_Learning_Aglorithm')
from softmax import Softmax_preprocess_training
from softmax import softmax_train

def average_dicts(dict_list):
    # 确保列表不为空
    if not dict_list:
        return {}
    
    # 初始化结果字典
    result = {}
    
    # 遍历列表中的第一个字典的键，设置初始化的平均值
    for key in dict_list[0]:
        result[key] = 0.0

    # 累加每个键对应的值
    for d in dict_list:
        for key, value in d.items():
            result[key] += value

    # 计算每个键的平均值
    for key in result:
        result[key] /= len(dict_list)

    return result

def top_bottom_keys(data_dict):
    # 按值排序字典，获取值最大的15个键
    top_keys = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)[:26]
    # 按值排序字典，获取值最小的15个键
    bottom_keys = sorted(data_dict.items(), key=lambda item: item[1])[:26]
    
    # 将元组列表转换成列表列表
    top_keys = [[key, value] for key, value in top_keys]
    bottom_keys = [[key, value] for key, value in bottom_keys]
    
    return top_keys, bottom_keys

def find_heaviest(weights, features):

    top_indices = np.argsort(weights)[::-1][:25]
    
    heaviest_weight = [[features[index], weights[index]] for index in top_indices]
    heaviest_index = list(top_indices)
    
    return heaviest_weight, heaviest_index

for m in range(0,20):
    average_weights = []

    for x in range(0,10):
        #print("round ", x, " start")
        Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(700,700,1500,1500,1500,1500)

        X_train, y_train, scale_factors, heaviest_features = Softmax_preprocess_training(Training_Data_Set, Features_name)

        weight = softmax_train(X_train, y_train, learning_rate=0.1, epochs=10000)
        
        if x == 0:
            for i in range(0,len(weight)):
                difference = weight[i][0] - weight[i][1]
                if difference < 0:
                    difference = -difference
                average_weights.append(difference)
        else:
            for i in range(0,len(weight)):
                difference = weight[i][0] - weight[i][1]
                if difference < 0:
                    difference = -difference
                average_weights[i] += difference

    average_weights = [i / 10 for i in average_weights]

    heaviest_weight, heaviest_index = find_heaviest(average_weights, heaviest_features)

    #print(heaviest_weight)
    #print()
    print(heaviest_index)
    print()


