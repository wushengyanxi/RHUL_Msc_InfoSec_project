import torch
import time
import random
import csv

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

def data_iter(Batch_size,Features,Labels):
    """
    there are a few reasons we need this function
    --------
    first, this function help us return small batch of sample times by times
    so we don't calculate weight of the whole sample set, just find mean of the result of those small batch
    --------
    second, this function help us shuffle samples each time
    but it won't shuffle the order of sample in original features and labels list 

    Args:
        Batch_size (_integer_): _the size of batch we want_
        Features (_torch.tensor_): _a tensor(shape:[n,1]) which contain value of each feature of all samples_
        Labels (_type_): _a tensor(shape:[n]) which contain value of label of all samples_

    param: 
        Number_sample(): the number of sample we load in
        indices(_list_): a list contain index in integer
        Batch_indices(_torch.tensor_): the index of samples in this batch
    
    
    Yields:
        _type_: _description_
    """
    Num_sample = len(Features)
    indices = list(range(Num_sample))
    random.shuffle(indices)
    for i in range(0,Num_sample,Batch_size):
        Batch_indices = torch.tensor(indices[i: min(i + Batch_size, Num_sample)])
        yield Features[Batch_indices], Labels[Batch_indices]

def linreg(Batch,Features_Weight,Bias):
    """
    caculate the y = wx + b equation

    Args:
        Batch (_torch.tensor_): _current batch_
        Features_Weight (_tensor_): _the weight of features_
        Bia (_torch.tensor_): _bias of equation_

    Returns:
        _torch_tensor_: _result of caculator_
    """
    #print("Batch:",Batch)
    #print("Feature weight:",Features_Weight)
    #print("Bias:",Bias)
    return torch.matmul(Batch,Features_Weight)+Bias

def squared_loss(y_hat, y):
    #print("y_hat:",y_hat)
    #print("y.reshape",y.reshape(y_hat.shape))
    return(y_hat - y.reshape(y_hat.shape)) ** 2/2

def sgd(params, lr, batch_size):
    count = 0
    with torch.no_grad():
        for param in params:
            if count<2:
                #print("param:",param)
                #print("param:",param.grad)
                #print("batch size:",batch_size)
                
                # sgd的运行过程中会导致梯度疯狂变小，然后inf，然后nan
                
                count += 1
            param -= lr*param.grad/batch_size
            param.grad.zero_()

def Linear_Regression_model(Features,Labels,Learning_rate,Num_epochs,Batch_size):
    Features_Weight = torch.normal(0,0.1,size=(Features.shape[1],1),requires_grad=True)
    #print("Feature weight:",Features_Weight)
    # shape of batch will be [Batch_size,2], so w must be [2,1] to follow the rule of metirx multiplication
    Bias = torch.ones(1,requires_grad=True) 
    #print("Bias:",Bias)
    net = linreg
    loss = squared_loss
    
    
    for epoch in range(Num_epochs):
        for X,y in data_iter(Batch_size, Features, Labels):
            l = loss(net(X,Features_Weight,Bias),y)
            l.sum().backward()
            sgd([Features_Weight,Bias],Learning_rate,Batch_size)
        
        with torch.no_grad():
            train_l = loss(net(Features,Features_Weight,Bias),Labels)
            mean_lose = float(train_l.mean())
            print("epoch",epoch,",",format(mean_lose,".12f"))


Feature,Label = Read_HIKARI2021_File("ALLFLOWMETER_HIKARI2021.csv")

print(Feature.shape)

t1 = time.time()
Linear_Regression_model(Feature,Label,0.00000000001,3,5000) 
t2 = time.time()
cost = t2-t1
print("time cost:",cost)