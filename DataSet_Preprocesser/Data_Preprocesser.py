import csv
import torch
import time
import math

# "ALLFLOWMETER_HIKARI2021.csv"
# "ALLFLOWMETER_HIKARI2021_simple_version.csv"

'''
第一个模块，只负责读文件，我们把它称为 Read_HIKARI

    我需要做以下几件事情，并且按顺序做

    1，我要读取整个文件，并且将每一个样本逐行取出来
        每一个样本都是一个列表，这个列表包含了他们的所有待处理特征，以及一位标签
        有一个列表存储所有的样本
        
    2，在我将每一行的内容取出来之后，我要把它乱序一遍

    3，将这个列表返回出来


第二个模块，负责区分训练样本和测试样本，我们把它称作 K_set_process

    这个模块将负责把 Read_HIKARI 给到我们的列表拆成 K 个 set，一般来说，将 K 设置成 5 或者 10 就好
    
    K_set_process 将会把整个sample list拆成 k 份
        如果总数除不尽的话，应该前 k-1 份大小相同，最后一份略微不同 


'''

def Read_HIKARI2021_File(DataBase_Name):
    """
    this function is used to load HIKARI2021 during program running
    then return sample list, each element in sample list is [feature_list(_list_),label(_float_)]
    you could get all the samples in HIKARI2021 by combine feature and label
    feature"unnamed", "uid", "originh","responh" and "traffic_category" will be drop
    since those feature is no necessary when we just want Determine whether traffic is malignant
    

    Args:
        DataBase_Name (_String_): this is the name of file which need to be load

    Returns:
        Sample(_List_): list of all samples, or, all the row of database
    """

    Sample = []
    
    with open(DataBase_Name, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            DataBase_CurrentLine = []
            Current_Sample = []
            for i in row:
                DataBase_CurrentLine.append(row[i])
            # drop no necessary feature
            del DataBase_CurrentLine[86]
            del DataBase_CurrentLine[0]
            del DataBase_CurrentLine[0]
            del DataBase_CurrentLine[0]
            del DataBase_CurrentLine[0]
            del DataBase_CurrentLine[1]
            for i in range(len(DataBase_CurrentLine)):
                DataBase_CurrentLine[i] = float(DataBase_CurrentLine[i])
                # transfer element to float so make easier to transfer to tensor
            #Feature.append(DataBase_CurrentLine[:-1])
            #Label.append([DataBase_CurrentLine[-1]])
            Sample.append([DataBase_CurrentLine[:-1],DataBase_CurrentLine[-1]])
            

        
    return Sample

def K_fold(k,Sample):
    """
    this function is used to process sample list in multiple folds

    Args:
        k (_int_): _how many folds we want to divide the samples into_
        Sample (_list_): _the list of samples_

    Returns:
        _list_: _a list which contain servel lists/folds of sample_
    """
    fold_size = len(Sample)//k
    Sample_folds = []
    for x in range(0,k-1):
        current_fold = []
        for i in range(0,fold_size+1):
            current_fold.append(Sample.pop())
        Sample_folds.append(current_fold)
        current_fold = []
    for y in range(0,len(Sample)):
        current_fold.append(Sample.pop())
    Sample_folds.append(current_fold)
    
    return Sample_folds
           
def Standard_Scalar(X, y):
    """
    do standardardization on training set and testing set
    which means, for each sample in training set
    find the mean and sd of a feature, value(v) of this feature of all sample: (v-mean)/sd
    
    for each sample in test set
    do (v-mean)/sd again, use the mean and sd get from training set for each feature 

    Args:
        X (_List_): _in Machine Learning, we say X is the training set_
        y (_List_): _in Machine Learning, we say Y is the testing set_
            the format of X,y should be:
            [[[f1,f2...,fn],[label]],[[f1,f2...,fn],[label]],.....，[[f1,f2...,fn],[label]]]
            each  _[[f1,f2...,fn],[label]]_ is a sample
    
    Returns:
        _List_: _a list of train sample and a list of test sample after standard scalar(with same format as input)_
    """
    
    trainset_mean = []
    trainset_sd = []
    
    for i in range(0,len(X[0][0])):
    # loop <number of features> times
        mean = 0
        sum = 0
        for j in range(0,len(X)):
        # loop <number of samples> times
            sum += X[j][0][i]
            # same feature in different sample
        mean = float(sum/len(X))
        
        sum_for_sd = 0
        for j in range(0,len(X)):
            sum_for_sd += (X[j][0][i]-mean) ** 2
        sd = math.sqrt((sum_for_sd/len(X)))
        
        for n in range(0,len(X)):
            X[n][0][i] = (X[n][0][i] - mean)/sd
        
        trainset_mean.append(mean)
        trainset_sd.append(sd)
        
    for i in range(0,len(y[0][0])):
        for j in range(0,len(y)):
            y[j][0][i] = (y[j][0][i] - trainset_mean[i])/trainset_sd[i]
            # index_j_sample's index_i_feature
          
    return X,y

def Normalization(X, y):
    """
    this function is used to normalization the sample
    for a specific feature, find up bound and low bound in all training sample
    for value(v) of this feature(for both training and testing set): v-low/up-low 

    Args:
        X (_List_): _in Machine Learning, we say X is the training set_
        y (_List_): _in Machine Learning, we say Y is the testing set_
            the format of X,y should be:
            [[[f1,f2...,fn],[label]],[[f1,f2...,fn],[label]],.....，[[f1,f2...,fn],[label]]]
            each  _[[f1,f2...,fn],[label]]_ is a sample

    Returns:
        _List_: _a list of train sample and a list of test sample after standard scalar(with same format as input)_
    """
    Up_Low_Bound = []
    for i in range(0,len(X[0][0])):
    # loop <number of feature> times
        Up_bound = 0
        for j in range(0,len(X)):
            if X[j][0][i] > Up_bound:
            # same feature in different sample
                Up_bound = X[j][0][i]
        Lower_bound = Up_bound
        for j in range(0,len(X)):
            if X[j][0][i] < Lower_bound:
                Lower_bound = X[j][0][i]
        Up_Low_Bound.append([Up_bound,Lower_bound])
                
    for i in range(0,len(X[0][0])):
        for j in range(0,len(X)):
            if Up_Low_Bound[i][0] != Up_Low_Bound[i][1]:
            # maybe all the sample get same value in same feature, then up-low = 0
            # to prevent "n/0" happen, we set this if selection
                X[j][0][i] = (X[j][0][i] - Up_Low_Bound[i][1])/(Up_Low_Bound[i][0]-Up_Low_Bound[i][1])
            else:
                X[j][0][i] = 1
                
    for i in range(0,len(y[0][0])):
        for j in range(0,len(y)):
            if Up_Low_Bound[i][0] != Up_Low_Bound[i][1]:
                y[j][0][i] = (y[j][0][i] - Up_Low_Bound[i][1])/(Up_Low_Bound[i][0]-Up_Low_Bound[i][1])
            else:
                y[j][0][i] = 1
    
    return X,y
    
    
