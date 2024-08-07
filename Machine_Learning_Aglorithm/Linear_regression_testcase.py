import sys
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from random import shuffle
import numpy as np
from Training_Set_Creator import Training_set_create
from Linear_Regression_1_1 import train_linear_regression
from Linear_Regression_1_1 import LR_preprocess_data
from Linear_Regression_1_1 import linear_regression_predict
from Linear_Regression_1_1 import LR_each_test_sample_preprocess

for i in range(0,20):
    print("round: ", i)
    Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(400,400,1500,1500,1500,1500)
    X_train, y_train, scale_factors, heaviest_features = LR_preprocess_data(Training_Data_Set, Features_name)
    weights = train_linear_regression(X_train, y_train, 10000, 0.001)

    all_test_sample_benign = Testing_Data_Set[0][:3000] + Testing_Data_Set[1][:3000]
    all_test_sample_malicious = Testing_Data_Set[2][:1500] + Testing_Data_Set[3][:1500] + Testing_Data_Set[4][:1500] + Testing_Data_Set[5][:1500]

    count = 0
    for samples in all_test_sample_benign:
        testing_sample = LR_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
        prediction = linear_regression_predict(testing_sample[:-1], weights)
        if prediction == 0: #samples[-1]:
            count += 1

    correct_rate = count / len(all_test_sample_benign)
    print("The correct rate for benign sample is: ", correct_rate)

    count = 0
    for samples in all_test_sample_malicious:
        testing_sample = LR_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
        prediction = linear_regression_predict(testing_sample[:-1], weights)
        if prediction == 1: #samples[-1]:
            count += 1

    correct_rate = count / len(all_test_sample_malicious)
    print("The correct rate for malicious sample is: ", correct_rate)