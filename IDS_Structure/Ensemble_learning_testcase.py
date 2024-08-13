import sys
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from random import shuffle
import numpy as np
from Training_Set_Creator import Training_set_create
from Ensemble_learning_model import Ensemble_Learning_Training
from Ensemble_learning_model import Ensemble_Learning_Decision

Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(700,700,1500,1500,1500,1500)

Ensemble_param = Ensemble_Learning_Training(Features_name, Training_Data_Set)

all_test_sample_malicious = Testing_Data_Set[2][:1500] + Testing_Data_Set[3][:1500] + Testing_Data_Set[4][:1500] + Testing_Data_Set[5][:1500]

count = 0

for i in all_test_sample_malicious:
    predict_decition, reliable = Ensemble_Learning_Decision(Ensemble_param, i, Features_name)
    if predict_decition == i[-1]:
        count += 1

correct_rate = count / len(all_test_sample_malicious)
print(correct_rate)

print()
print("atfer causative integrity attack")

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