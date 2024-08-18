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



for m in range(0,20):
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
[0.8575458333333332, 0.8539291666666667, 0.8564666666666667, 0.8572041666666668, 0.8563375000000001, 0.8547166666666667, 0.857375, 0.85655, 0.8570833333333333, 0.8563583333333332]
[0.7183833333333334, 0.7118249999999999, 0.7163083333333334, 0.7177916666666667, 0.7159333333333334, 0.7130166666666667, 0.71795, 0.7156166666666667, 0.7182, 0.7157166666666667]
[0.995438851745361, 0.9944583503114267, 0.9953104374609203, 0.9953085820593707, 0.9954694506564082, 0.9949995348404502, 0.9955626430007627, 0.9964955440029706, 0.9944154705312226, 0.9958258933747652]
[0.8345167738781517, 0.8297336020787295, 0.8330700419650905, 0.8340716855249078, 0.8328720377309105, 0.8307312905355548, 0.834267786073265, 0.833016451963371, 0.8340333288172335, 0.8328500916381408]
APRF

Accuracy: 0.8563566666666667
Precision: 0.7160741666666667
Recall: 0.9953284757983658
F1_score: 0.8329163090205356

----------------------

Accuracy: [0.8562749999999999, 0.856125, 0.8556499999999999, 0.8576666666666667, 0.8566958333333333, 0.8576708333333333, 0.8569500000000001, 0.8558833333333333, 0.8546958333333333, 0.8563416666666667, 0.8574624999999999, 0.857575, 0.8566666666666667, 0.8561833333333334, 0.8560791666666667, 0.8563583333333334, 0.8559541666666667, 0.8570916666666667, 0.85875, 0.8572666666666667]
Precision: [0.7157333333333332, 0.7149083333333333, 0.7138333333333333, 0.7188833333333333, 0.7168500000000001, 0.718175, 0.7172, 0.7170833333333333, 0.7135583333333334, 0.7162, 0.7179666666666666, 0.7182749999999999, 0.7164750000000001, 0.71595, 0.7163083333333334, 0.7168666666666667, 0.7154166666666667, 0.717, 0.7215, 0.7174166666666667]
Recall: [0.9955720412657934, 0.9962953500255494, 0.9964636359406264, 0.9950860517694827, 0.9951988153221422, 0.9960703181886479, 0.9954198473282442, 0.9926402731635291, 0.9941946195734207, 0.9951138179376142, 0.9957813709966367, 0.9956681452730801, 0.995634249716284, 0.9950199203187251, 0.9942397779191487, 0.9942442384595827, 0.995120028746624, 0.9960869665887148, 0.994486560992419, 0.9959970382710908]
F1_score: [0.8327725796286419, 0.8324664738874764, 0.8317958478180651, 0.8347299363303853, 0.8333971138894676, 0.8345979343505019, 0.8337111304853241, 0.8326559838984363, 0.8308178938227405, 0.8329279047905177, 0.8343558282208589, 0.8345242244684559, 0.8332961803502719, 0.8327259333927809, 0.8326947761013297, 0.833073474012454, 0.8324000950206282, 0.8338098053087054, 0.8362793393219357, 0.8340599507837779]
Accuracy: 0.8566670833333333
Precision: 0.71678
Recall: 0.9952166533898679
F1_score: 0.8333546202941378

'''