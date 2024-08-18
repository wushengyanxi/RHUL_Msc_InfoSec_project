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
        True_negative += count
        False_negative += len(all_test_sample_benign)-count

        correct_rate0 = count / len(all_test_sample_benign)
        print("The correct rate for benign sample is: ", correct_rate0)
        avg_correct_rate0 += correct_rate0

        count = 0
        for samples in all_test_sample_malicious:
            test_sample = softmax_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
            predict = softmax_predict(test_sample[:-1], weight)
            if predict == samples[-1]:
                count += 1
        True_positive += count
        False_positive += len(all_test_sample_malicious)-count

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

-----------------

Accuracy: [0.8904375, 0.8923625000000001, 0.8908458333333333, 0.8921791666666666, 0.8920708333333334, 0.8908291666666666, 0.8905625, 0.8914541666666665, 0.8914083333333334, 0.8916833333333334, 0.8916166666666666, 0.8870041666666666, 0.890625, 0.8921291666666666, 0.8909333333333332, 0.891575, 0.8924583333333334, 0.8914208333333332, 0.8916999999999999, 0.8890833333333333]
Precision: [0.7819083333333333, 0.785725, 0.7826666666666666, 0.7854416666666666, 0.785275, 0.7827333333333333, 0.7820083333333334, 0.7839833333333333, 0.7838416666666667, 0.7843500000000001, 0.7844, 0.7750416666666666, 0.78245, 0.7855666666666666, 0.7832333333333332, 0.7842916666666667, 0.786, 0.7837916666666667, 0.7845666666666666, 0.779025]
Recall: [0.9986801911594095, 0.9987289078140392, 0.9987558088837372, 0.9986226333132026, 0.9985588487744916, 0.998628491234039, 0.9988717042587841, 0.9986306750029191, 0.9986940456977832, 0.9987478777589135, 0.9985148724911953, 0.9986685135672025, 0.9984687041408793, 0.9983373047392109, 0.9982581357804402, 0.9985464499427067, 0.9986236103758602, 0.9987894105278807, 0.9985151875106058, 0.9988994080311159]
F1_score: [0.8770991759872495, 0.8795141949656495, 0.8776052738545205, 0.8792954665248642, 0.8791662973657571, 0.8775980229565022, 0.8772359509598171, 0.8783839930534484, 0.8783195756919285, 0.8786594473487678, 0.8786006309854948, 0.8727578133430928, 0.8773582261093825, 0.8792631456220122, 0.8777690612275392, 0.8785448910628605, 0.8796456050361389, 0.8783250610499186, 0.8787052938101994, 0.8753663629638646]
Accuracy: 0.8911189583333334
Precision: 0.783315
Recall: 0.9986270390502208
F1_score: 0.8779606744959505


'''