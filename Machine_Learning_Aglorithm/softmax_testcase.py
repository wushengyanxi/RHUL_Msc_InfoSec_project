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
Accuracy: [0.8908416666666666, 0.8917541666666666, 0.890125, 0.89165, 0.8903666666666666, 0.8920666666666666, 0.8911499999999999, 0.8904583333333334, 0.8911666666666667, 0.8925666666666666, 0.8916833333333334, 0.8935333333333333, 0.8910916666666666, 0.8917875000000001, 0.8913416666666667, 0.8913166666666666, 0.8918166666666667, 0.8921541666666667, 0.8909249999999999, 0.89175]
Precision: [0.9986416666666668, 0.9988416666666667, 0.99875, 0.9988666666666667, 0.998625, 0.9986916666666666, 0.9986499999999999, 0.9987583333333334, 0.99875, 0.99875, 0.99845, 0.998775, 0.9985916666666667, 0.9986583333333333, 0.99845, 0.998625, 0.9987666666666667, 0.9987333333333333, 0.9985916666666667, 0.9989083333333333]
Recall: [0.8215216079850828, 0.8226504965648829, 0.820497021975765, 0.8224960887107842, 0.8208888767108273, 0.823154062779037, 0.8219341563786008, 0.8209422434105978, 0.8219037169112604, 0.8238019301091529, 0.8227627314178981, 0.8251042971815666, 0.821886145404664, 0.8227931534030443, 0.8222996993946715, 0.8221729763848676, 0.8227750775736606, 0.8232506062000701, 0.8216607240811848, 0.8226094236813571]
F1_score: [0.9014638623096829, 0.9022246811265294, 0.9008907430375466, 0.9021420077371186, 0.901076012662511, 0.9024662073120223, 0.9017155756207674, 0.9011624409756684, 0.9017380182078097, 0.9028792695605009, 0.902132337439388, 0.9036711428119037, 0.9016629044394282, 0.9022356568253598, 0.9018539287784242, 0.9018490784710672, 0.9022689973952451, 0.9025412400829884, 0.9015272344267229, 0.9022271731685471]
Accuracy: 0.8914772916666667
Precision: 0.9986937499999999
Recall: 0.8223552518129488
F1_score: 0.9019864256194616

'''