import sys
import random
import time
import numpy as np

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
from softmax import softmax, cross_entropy_loss, compute_gradient
from softmax import Softmax_preprocess_training
from softmax import softmax_train
from softmax import softmax_predict
from softmax import softmax_each_test_sample_preprocess
#from softmax import softmax_predict



'''
for i in range(0,10):

    test_sample = each_test_sample_preprocess(Training_Data_Set[i], scale_factors, Features_name, heaviest_features)

    predict = softmax_predict(test_sample[:-1], weight)

    print(predict)
'''

print("start predicting test sample") 

Accuracy_list = []
Precision_list = []
Recall_list = []
F1_score_list = []


for m in range(0,10):
    avg_correct_rate0 = 0
    avg_correct_rate1 = 0
    True_positive = 0
    True_negative = 0
    False_positive = 0
    False_negative = 0

    for i in range(0,20):
        
        print("round: ", i)
        
        Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(700,700,1500,1500,1500,1500)

        X_train, y_train, scale_factors, heaviest_features = Softmax_preprocess_training(Training_Data_Set, Features_name)

        weight = softmax_train(X_train, y_train, learning_rate=0.08, epochs=10000)

        all_test_sample = []
        for i in range(0,len(Testing_Data_Set)):
            all_test_sample += Testing_Data_Set[i]

        random.shuffle(all_test_sample)

        all_test_sample_benign = Testing_Data_Set[0][:3000] + Testing_Data_Set[1][:3000]
        all_test_sample_malicious = Testing_Data_Set[2][:1500] + Testing_Data_Set[3][:1500] + Testing_Data_Set[4][:1500] + Testing_Data_Set[5][:1500]
        count = 0

        for samples in all_test_sample_benign:
            test_sample = softmax_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
            predict = softmax_predict(test_sample[:-1], weight)
            if predict == samples[-1]:
                count += 1
        True_positive += count
        False_positive += len(all_test_sample_benign)-count

        correct_rate0 = count / len(all_test_sample_benign)
        print("The correct rate for benign sample is: ", correct_rate0)
        avg_correct_rate0 += correct_rate0

        count = 0
        for samples in all_test_sample_malicious:
            test_sample = softmax_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
            predict = softmax_predict(test_sample[:-1], weight)
            if predict == samples[-1]:
                count += 1
        True_negative += count
        False_negative += len(all_test_sample_malicious)-count

        correct_rate1 = count / len(all_test_sample_malicious)
        print("The correct rate for malicious sample is: ", correct_rate1)
        avg_correct_rate1 += correct_rate1

    print("softmax total correct rate for benign sample is: ", avg_correct_rate0 / 20)
    print("softmax total correct rate for malicious sample is: ", avg_correct_rate1 / 20)

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
Accuracy: [0.8921958333333332, 0.8923958333333334, 0.8916291666666667, 0.8903958333333334, 0.8906999999999999, 0.890875, 0.8919458333333333, 0.8920583333333333, 0.8908999999999999, 0.8913666666666666]
Precision: [0.7855083333333334, 0.7858583333333333, 0.7843249999999999, 0.7815333333333333, 0.7824333333333334, 0.782825, 0.7848916666666668, 0.7851499999999999, 0.783125, 0.7839166666666667]
Recall: [0.9985804332856614, 0.9986445129247812, 0.9986418665846128, 0.999051910560012, 0.9986810755616066, 0.9986286516137263, 0.9987275599902447, 0.9986856331220454, 0.9983109184779145, 0.9984927610070903]
F1_score: [0.8793208796846942, 0.8795649882712854, 0.878602734227317, 0.8770064477517031, 0.8774297248803828, 0.8776556981893604, 0.8789913535255499, 0.879137079993655, 0.8777214267701532, 0.8782887980131833]
Accuracy: 0.8914462499999999
Precision: 0.7839566666666666
Recall: 0.9986445323127695
F1_score: 0.8783719131307285


'''