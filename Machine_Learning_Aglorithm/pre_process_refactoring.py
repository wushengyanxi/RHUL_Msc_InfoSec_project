import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import random
import time
import numpy as np
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
    
def Softmax_preprocess_training(training_set, features_list):
    
    heaviest_features = ['bwd_pkts_payload.min', 'responp', 'flow_pkts_payload.avg', 'bwd_iat.tot', 'flow_pkts_per_sec', 'payload_bytes_per_second', 'idle.avg', 
     'fwd_pkts_payload.max', 'fwd_pkts_tot', 'flow_iat.tot', 'fwd_iat.tot', 'fwd_pkts_payload.avg', 'fwd_iat.min', 'idle.tot', 'fwd_header_size_tot', 'bwd_data_pkts_tot', 
     'flow_RST_flag_count', 'bwd_header_size_max', 'bwd_iat.std', 'fwd_pkts_payload.min', 'flow_pkts_payload.max', 'flow_FIN_flag_count', 'Label']
    
    indices = {}
    for feature in heaviest_features:
        if feature in features_list:
            index = features_list.index(feature)
            indices[feature] = index
    
    new_training_set = []
    for samples in training_set:
        new_sample = []
        for feature in heaviest_features:
            if feature in indices:
                new_sample.append(samples[indices[feature]])
        new_training_set.append(new_sample)
    # with this heaviest_features, all the avlue in sample should be int or float
    
    for i in new_training_set:
        for j in i:
            if not isinstance(j, (int, float)):
                j = 0
    # change all nun-numeric value to 0
    
    X_train = [] # 2D array for X_train [[sample1],[sample2],..,[samplen]]
    y_train = [] # 1D list for label [0,0,...,1,1]
    
    for i in new_training_set:
        X_train.append(i[:-1])
        y_train.append(i[-1])

    scaler = StandardScaler()
    scaler.fit(X_train)
    
    means = scaler.mean_
    means = means.tolist()
    stds = scaler.scale_
    stds = stds.tolist()
    scale_factors = [means,stds]
    
    for sample in X_train:
        for i in range(0,len(sample)):
            sample[i] = (sample[i] - scale_factors[0][i]) / scale_factors[1][i]
    
    return X_train, y_train, scale_factors, heaviest_features

def each_test_sample_preprocess(test_sample, scale_factors, features_list, heaviest_features):
    indices = {}
    for feature in heaviest_features:
        if feature in features_list:
            index = features_list.index(feature)
            indices[feature] = index
    
    testing_sample = []
    for feature in heaviest_features:
        if feature in indices:
            testing_sample.append(test_sample[indices[feature]])
    
    
    
    for i in range(0,len(scale_factors[0])):
        testing_sample[i] = (testing_sample[i] - scale_factors[0][i]) / scale_factors[1][i]
    
    return testing_sample
    
    



Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(3000,3000,1500,1500,1500,1500)

X_train, y_train, scale_factors, heaviest_features = Softmax_preprocess_training(Training_Data_Set, Features_name)

print(X_train[0])

each_test_sample_preprocess(Training_Data_Set[0], scale_factors, Features_name, heaviest_features)
    
    



    
