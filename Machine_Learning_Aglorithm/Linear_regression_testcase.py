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
True_positive = 0
True_negative = 0
False_positive = 0
False_negative = 0

for i in range(0,20):
    print("round: ", i)
    Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(700,700,1500,1500,1500,1500)
    X_train, y_train, scale_factors, heaviest_features = LR_preprocess_data(Training_Data_Set, Features_name)
    weights = train_linear_regression(X_train, y_train, 10000, 0.01)

    all_test_sample_benign = Testing_Data_Set[0][:3000] + Testing_Data_Set[1][:3000]
    all_test_sample_malicious = Testing_Data_Set[2][:1500] + Testing_Data_Set[3][:1500] + Testing_Data_Set[4][:1500] + Testing_Data_Set[5][:1500]

    count = 0
    for samples in all_test_sample_benign:
        testing_sample = LR_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
        prediction = linear_regression_predict(testing_sample[:-1], weights)
        if prediction == 0: #samples[-1]:
            count += 1
    True_positive += count
    False_positive += len(all_test_sample_benign)-count

    correct_rate0 = count / len(all_test_sample_benign)
    print("The correct rate for benign sample is: ", correct_rate0)
    avg_correct_rate0 += correct_rate0

    count = 0
    for samples in all_test_sample_malicious:
        testing_sample = LR_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
        prediction = linear_regression_predict(testing_sample[:-1], weights)
        if prediction == 1: #samples[-1]:
            count += 1
    True_negative += count
    False_negative += len(all_test_sample_malicious)-count

    correct_rate1 = count / len(all_test_sample_malicious)
    print("The correct rate for malicious sample is: ", correct_rate1)
    avg_correct_rate1 += correct_rate1

print("LR total correct rate for benign sample is: ", avg_correct_rate0 / 20)
print("LR total correct rate for malicious sample is: ", avg_correct_rate1 / 20)

True_positive = True_positive/20
True_negative = True_negative/20
False_positive = False_positive/20
False_negative = False_negative/20

Accuracy = (True_positive+True_negative)/(True_positive+True_negative+False_positive+False_negative)
print("Accuracy: ", Accuracy)
Precision = True_positive/(True_positive+False_positive)
print("Precision: ", Precision)
Recall = True_positive/(True_positive+False_negative)
print("Recall: ", Recall)
F1_score = 2*(Precision*Recall)/(Precision+Recall)
print("F1 score: ", F1_score)
