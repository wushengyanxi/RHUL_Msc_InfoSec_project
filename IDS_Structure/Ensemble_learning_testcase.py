import sys
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from random import shuffle
import numpy as np
from Training_Set_Creator import Training_set_create
from Ensemble_learning_model import Ensemble_Learning_Training
from Ensemble_learning_model import Ensemble_Learning_Decision

Accuracy_list = []
Precision_list = []
Recall_list = []
F1_score_list = []


for m in range(0,10):
    
    True_positive = 0
    True_negative = 0
    False_positive = 0
    False_negative = 0

    for i in range(0,20):
        print("round: ", i)

        Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(700,700,1500,1500,1500,1500)
        Ensemble_param = Ensemble_Learning_Training(Features_name, Training_Data_Set)

        all_test_sample_benign = Testing_Data_Set[0][:3000] + Testing_Data_Set[1][:3000]
        all_test_sample_malicious = Testing_Data_Set[2][:1500] + Testing_Data_Set[3][:1500] + Testing_Data_Set[4][:1500] + Testing_Data_Set[5][:1500]

        count = 0

        for sample in all_test_sample_benign:
            predict_decition, reliable = Ensemble_Learning_Decision(Ensemble_param, sample, Features_name)
            if predict_decition == sample[-1]:
                count += 1
        True_positive += count
        False_positive += len(all_test_sample_benign)-count

        for sample in all_test_sample_malicious:
            predict_decition, reliable = Ensemble_Learning_Decision(Ensemble_param, sample, Features_name)
            if predict_decition == sample[-1]:
                count += 1
        True_negative += count
        False_negative += len(all_test_sample_malicious)-count  

    
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
# let's try to test the effectiveness of Adversarial Machine Learning Attacks
count = 0
for i in range(0,len(all_test_sample_malicious)):
    all_test_sample_malicious[i][22] = all_test_sample_malicious[i][22] * 1.5
    all_test_sample_malicious[i][15] = all_test_sample_malicious[i][15] * 1.8
    all_test_sample_malicious[i][33] = all_test_sample_malicious[i][33] * 0.1
    all_test_sample_malicious[i][75] = all_test_sample_malicious[i][75] * 0.5
    all_test_sample_malicious[i][34] = all_test_sample_malicious[i][34] * 1.5

for i in all_test_sample_malicious:
    predict_decition, reliable = Ensemble_Learning_Decision(Ensemble_param, i, Features_name)
    if predict_decition == i[-1]:
        count += 1

correct_rate = count / len(all_test_sample_malicious)
print(correct_rate)

'''
