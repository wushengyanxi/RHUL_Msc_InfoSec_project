import sys
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from random import shuffle
import numpy as np
from Training_Set_Creator import Training_set_create
from Linear_Regression_1_1 import train_linear_regression
from Linear_Regression_1_1 import LR_preprocess_data
from Linear_Regression_1_1 import linear_regression_predict
from Linear_Regression_1_1 import LR_each_test_sample_preprocess

avg_correct_rate0 = 0
avg_correct_rate1 = 0

'''
for i in range(0,20):
    print("round: ", i)
    Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(700,700,1500,1500,1500,1500)
    X_train, y_train, scale_factors, heaviest_features = LR_preprocess_data(Training_Data_Set, Features_name)
    weights = train_linear_regression(X_train, y_train, 10000, 0.01)

    all_test_sample_benign = Testing_Data_Set[0][:3000] + Testing_Data_Set[1][:3000]
    all_test_sample_malicious = Testing_Data_Set[2][:1500] + Testing_Data_Set[3][:1500] + Testing_Data_Set[4][:1500] + Testing_Data_Set[5][:1500]
    
    for i in range(0,len(all_test_sample_malicious)):
        all_test_sample_malicious[i][22] = all_test_sample_malicious[i][22] * 0.25
        all_test_sample_malicious[i][15] = all_test_sample_malicious[i][15] * 0.8
        all_test_sample_malicious[i][33] = all_test_sample_malicious[i][33] * 0.1
        all_test_sample_malicious[i][75] = all_test_sample_malicious[i][75] * 0.5
        all_test_sample_malicious[i][34] = all_test_sample_malicious[i][34] * 0.5

    count = 0
    for samples in all_test_sample_benign:
        testing_sample = LR_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
        prediction = linear_regression_predict(testing_sample[:-1], weights)
        if prediction == 0: #samples[-1]:
            count += 1

    correct_rate0 = count / len(all_test_sample_benign)
    print("The correct rate for benign sample is: ", correct_rate0)
    avg_correct_rate0 += correct_rate0

    count = 0
    for samples in all_test_sample_malicious:
        testing_sample = LR_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
        prediction = linear_regression_predict(testing_sample[:-1], weights)
        if prediction == 1: #samples[-1]:
            count += 1

    correct_rate1 = count / len(all_test_sample_malicious)
    print("The correct rate for malicious sample is: ", correct_rate1)
    avg_correct_rate1 += correct_rate1

print("LR total correct rate for benign sample is: ", avg_correct_rate0 / 20)
print("LR total correct rate for malicious sample is: ", avg_correct_rate1 / 20)
'''

Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(700,700,1500,1500,1500,1500)
X_train, y_train, scale_factors, heaviest_features = LR_preprocess_data(Training_Data_Set, Features_name)
weights = train_linear_regression(X_train, y_train, 10000, 0.01)

a = heaviest_features.index("fwd_header_size_min")
b = heaviest_features.index("flow_FIN_flag_count")
c = heaviest_features.index("down_up_ratio")
d = heaviest_features.index("fwd_pkts_payload.max")
e = heaviest_features.index("active.tot")
f = heaviest_features.index("fwd_pkts_payload.tot")

print(weights[a])
print(weights[b])
print(weights[c])
print(weights[d])
print(weights[e])
print(weights[f])