import csv
import torch
import time

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

#Start_time = time.time()

def K_fold(k,Sample):
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
            
    










def Read_HIKARI2021_File_testcase():
    '''
    test case of Read_HIKARI2021
    '''
    Sample = Read_HIKARI2021_File("ALLFLOWMETER_HIKARI2021_simple_version.csv")
    print(Sample[0])

def K_fold_testcase():
    a = [[1],[2],[3],[4],[5],[6],[7],[8],[9]]
    b = K_fold(5,a)
    c = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]]
    d = K_fold(4,c)
    print(b == [[[9],[8]],[[7],[6]],[[5],[4]],[[3],[2]],[[1]]])
    print(d == [[[15],[14],[13],[12]],[[11],[10],[9],[8]],[[7],[6],[5],[4]],[[3],[2],[1]]])

K_fold_testcase()