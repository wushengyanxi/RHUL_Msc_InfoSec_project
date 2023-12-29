import csv
import torch
import time

# "ALLFLOWMETER_HIKARI2021.csv"
# "ALLFLOWMETER_HIKARI2021_simple_version.csv"

def Read_DataBase_File(DataBase_Name):
    
    DataBase_Content = []
    
    with open(DataBase_Name, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            DataBase_CurrentLine = []
            for i in row:
                DataBase_CurrentLine.append(row[i])
            del DataBase_CurrentLine[86]
            del DataBase_CurrentLine[0]
            del DataBase_CurrentLine[0]
            del DataBase_CurrentLine[0]
            del DataBase_CurrentLine[0]
            del DataBase_CurrentLine[1]
            # 我们在准备数据集的时候，删除掉了原本数据集中的两个IP地址
            # 在模型训练的过程中，我们只关注恶意与否
            # 靠IP地址封掉用户，是携带模型的IDS该负责的事
            for i in range(len(DataBase_CurrentLine)):
                DataBase_CurrentLine[i] = float(DataBase_CurrentLine[i])
            DataBase_Content.append(DataBase_CurrentLine)
    
    DataBase_Content_tensor = torch.as_tensor(DataBase_Content)
        
    return DataBase_Content_tensor

Start_time = time.time()

a = Read_DataBase_File("ALLFLOWMETER_HIKARI2021.csv")


print(type(a))

Endtime = time.time()

timecost = Endtime - Start_time

time_in_minute = timecost/60

time_in_hour = timecost

print("共耗时 ",time_in_minute," 分钟，",time_in_hour," 秒")

print(a.shape)
