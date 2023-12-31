import random
import torch
from torch import nn

torch.cuda.is_available()

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device()
    return torch.device('cpu')

def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

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
    return torch.matmul(Batch,Features_Weight)+Bias

def squared_loss(y_hat, y):
    return(y_hat - y.reshape(y_hat.shape)) ** 2/2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()

def Linear_Regression_model(Features,Labels,Learning_rate,Num_epochs,Batch_size):
    Features_Weight = torch.normal(0,0.1,size=(Features.shape[1],1),requires_grad=True)
    # shape of batch will be [Batch_size,2], so w must be [2,1] to follow the rule of metirx multiplication
    Bias = torch.zeros(1,requires_grad=True) 
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

# test case
'''
def synthetic_data(w,b, num_examples):
    X = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(X,w)+b
    return X,y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_w = true_w
true_b = 4.2

features, labels = synthetic_data(true_w, true_b, 1000)
features = features
labels = labels


Linear_Regression_model(features,labels,0.03,3,10) 
''' 