import sys
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from random import shuffle
import numpy as np
from Training_Set_Creator import Training_set_create
from Linear_Regression_1_1 import train_linear_regression
from Linear_Regression_1_1 import LR_preprocess_data
from Linear_Regression_1_1 import linear_regression_predict
from Linear_Regression_1_1 import LR_each_test_sample_preprocess

Accuracy_list = []
Precision_list = []
Recall_list = []
F1_score_list = []



for m in range(0,1):
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
        weights = train_linear_regression(X_train, y_train, 10000, 0.05)

        all_test_sample_benign = Testing_Data_Set[0][:3000] + Testing_Data_Set[1][:3000]
        all_test_sample_malicious = Testing_Data_Set[2][:1500] + Testing_Data_Set[3][:1500] + Testing_Data_Set[4][:1500] + Testing_Data_Set[5][:1500]

        count = 0
        for samples in all_test_sample_benign:
            testing_sample = LR_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
            prediction = linear_regression_predict(testing_sample[:-1], weights)
            if prediction == 0: #samples[-1]:
                count += 1
        True_negative += count
        False_negative += len(all_test_sample_benign)-count

        correct_rate0 = count / len(all_test_sample_benign)
        print("The correct rate for benign sample is: ", correct_rate0)
        avg_correct_rate0 += correct_rate0

        count = 0
        for samples in all_test_sample_malicious:
            testing_sample = LR_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
            prediction = linear_regression_predict(testing_sample[:-1], weights)
            if prediction == 1: #samples[-1]:
                count += 1
        True_positive += count
        False_positive += len(all_test_sample_malicious)-count

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
    Accuracy_list.append(Accuracy)
    Precision = True_positive/(True_positive+False_positive)
    Precision_list.append(Precision)
    Recall = True_positive/(True_positive+False_negative)
    Recall_list.append(Recall)
    F1_score = 2*(Precision*Recall)/(Precision+Recall)
    F1_score_list.append(F1_score)
    
print("Accuracy:", Accuracy_list)
print("Precision:", Precision_list)
print("Recall:", Recall_list)
print("F1_score:", F1_score_list)


average = sum(Accuracy_list) / len(Accuracy_list)
print("Accuracy:", average)
average = sum(Precision_list) / len(Precision_list)
print("Precision:", average)
average = sum(Recall_list) / len(Recall_list)
print("Recall:", average)
average = sum(F1_score_list) / len(F1_score_list)
print("F1_score:", average)
'''
Accuracy: [0.8569833333333334, 0.8560958333333333, 0.857375, 0.8551666666666666, 0.8575833333333334, 0.8558583333333333, 0.8561000000000001, 0.8564083333333333, 0.8563166666666666, 0.8551791666666666, 0.8547333333333332, 0.8559000000000001, 0.8558416666666667, 0.8571416666666667, 0.8574666666666666, 0.8569500000000001, 0.8576333333333332, 0.8571875, 0.8575625, 0.8574958333333333]
Precision: [0.9977833333333334, 0.9972583333333334, 0.9962166666666668, 0.9941666666666666, 0.9959333333333334, 0.9964, 0.9960166666666668, 0.9969250000000001, 0.9967916666666666, 0.9972333333333333, 0.9959833333333332, 0.99545, 0.9957083333333333, 0.99725, 0.996175, 0.9971166666666667, 0.9958416666666667, 0.9974833333333333, 0.9962416666666666, 0.9962833333333333]
Recall: [0.7785450478568455, 0.777695462018859, 0.7797054564902623, 0.7779081898800209, 0.7800840709119866, 0.7777792233136017, 0.7782393540825628, 0.7782194062085295, 0.7781659445463653, 0.7765959517953444, 0.7765951916829109, 0.7782425142678445, 0.7780592310898105, 0.7789697056487834, 0.7798356057146585, 0.7787945847435563, 0.7801854148984788, 0.7789238047517099, 0.7799234096409908, 0.7798237546393232]
F1_score: [0.8746347592332866, 0.8738968668645642, 0.874763099933412, 0.872841673983026, 0.8748920220787397, 0.8736199905015892, 0.8737627019518971, 0.8740994578480513, 0.8740144822698146, 0.8731926754787608, 0.8727126688572472, 0.8735465754025712, 0.8735305298865365, 0.8746975799083421, 0.8748289362947784, 0.8745358865662914, 0.8749203792510158, 0.8747583777601572, 0.8749098919435164, 0.8748632500210385]
Accuracy: 0.8565489583333333
Precision: 0.9964129166666666
Recall: 0.7786145662091222
F1_score: 0.8741510903017318

'''