import csv
import torch
import time

# "ALLFLOWMETER_HIKARI2021.csv"
# "ALLFLOWMETER_HIKARI2021_simple_version.csv"

def Read_HIKARI2021_File(DataBase_Name):
    """
    this function is used to load HIKARI2021 during program running
    then return feature list and label list of samples in torch.tensor
    you could get all the samples in HIKARI2021 by combine feature and label
    feature"unnamed", "uid", "originh","responh" and "traffic_category" will be drop
    since those feature is no necessary when we just want Determine whether traffic is malignant
    

    Args:
        DataBase_Name (_String_): this is the name of file which need to be load

    Returns:
        FeaFeature_tensor(_torch.tensor_): a tensor which contain feature data in each line
        Label_tensor(_torch.tensor_): a tensor which contain label in each line
    """
    Feature = []
    Label = []
    
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
            Feature.append(DataBase_CurrentLine[:-1])
            Label.append([DataBase_CurrentLine[-1]])
            
    
    Feature_tensor = torch.as_tensor(Feature)
    Label_tensor = torch.as_tensor(Label)
        
    return Feature_tensor,Label_tensor

#Start_time = time.time()

Feature,Label = Read_HIKARI2021_File("ALLFLOWMETER_HIKARI2021_simple_version.csv")

print(Feature[0])
print(Label[0])









#print(type(a))
#Endtime = time.time()
#timecost = Endtime - Start_time
#time_in_minute = timecost/60
#time_in_hour = timecost
#print("共耗时 ",time_in_minute," 分钟，",time_in_hour," 秒")
#print(a.shape)
