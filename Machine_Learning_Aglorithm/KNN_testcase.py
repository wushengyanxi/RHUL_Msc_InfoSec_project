import sys
import random
import time

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
from KNN import KNN_preprocess_training
from KNN import KNN_train
from KNN import KNN_predict
from KNN import KNN_each_test_sample_preprocess

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

    for x in range(0,2):
        
        print("round: ", x)

        Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(700,700,1500,1500,1500,1500)

        X_train, y_train, scale_factors, heaviest_features = KNN_preprocess_training(Training_Data_Set, Features_name)

        Kdtree = KNN_train(X_train, y_train)

        all_test_sample_benign = Testing_Data_Set[0][:3000] + Testing_Data_Set[1][:3000]
        all_test_sample_malicious = Testing_Data_Set[2][:1500] + Testing_Data_Set[3][:1500] + Testing_Data_Set[4][:1500] + Testing_Data_Set[5][:1500]

        count = 0
        for samples in all_test_sample_benign:
            testing_sample = KNN_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
            prediction = KNN_predict(testing_sample[:-1], Kdtree, k=3)
            if prediction == 0: #samples[-1]:
                count += 1
        True_negative += count
        False_negative += len(all_test_sample_benign)-count

        correct_rate0 = count / len(all_test_sample_benign)
        print("The correct rate for benign sample is: ", correct_rate0)
        avg_correct_rate0 += correct_rate0

        count = 0
        for samples in all_test_sample_malicious:
            testing_sample = KNN_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
            prediction = KNN_predict(testing_sample[:-1], Kdtree, k=3)
            if prediction == 1: #samples[-1]:
                count += 1
        True_positive += count
        False_positive += len(all_test_sample_malicious)-count

        correct_rate1 = count / len(all_test_sample_malicious)
        print("The correct rate for malicious sample is: ", correct_rate1)
        avg_correct_rate1 += correct_rate1
        
    print("knn total correct rate for benign sample is: ", avg_correct_rate0 / 20)
    print("knn total correct rate for malicious sample is: ", avg_correct_rate1 / 20)

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
Accuracy: [0.9299458333333334, 0.9298208333333333, 0.9281125, 0.9285125, 0.9297916666666667, 0.929, 0.9291833333333334, 0.9284083333333333, 0.9283708333333334, 0.9289458333333334, 0.9291, 0.9290583333333334, 0.9292875, 0.9286291666666666, 0.929025, 0.9288958333333334, 0.9293666666666667, 0.9285583333333334, 0.9297083333333334, 0.9289499999999999]
Precision: [0.9943500000000001, 0.9941166666666666, 0.9939083333333333, 0.9936833333333334, 0.9944583333333333, 0.994125, 0.9937666666666667, 0.9937916666666666, 0.994375, 0.9942083333333334, 0.9939583333333334, 0.9934583333333333, 0.9937833333333334, 0.9930666666666667, 0.993675, 0.9944750000000001, 0.9933500000000001, 0.9934083333333333, 0.9937333333333332, 0.9939250000000001]
Recall: [0.8808847088005788, 0.8808470734174597, 0.8783277242232548, 0.8790999771455533, 0.8805711334120425, 0.8795620437956204, 0.8800885608856089, 0.8788653715768062, 0.8784166783224505, 0.8794218025548602, 0.8798297508224776, 0.8801012875029529, 0.8802397419526273, 0.8796958623998818, 0.8799034800318781, 0.8791651625546084, 0.8806554567215343, 0.8793558761913193, 0.8809302188141779, 0.8796185671932387]
F1_score: [0.9341846181549145, 0.9340604703422842, 0.9325504024770221, 0.9328863523456722, 0.9340560425798373, 0.9333411571411807, 0.9334794520547945, 0.9328019648639769, 0.9328059224283833, 0.9332989130222208, 0.9334183733360464, 0.9333505574345484, 0.9335718395641128, 0.932949719139608, 0.9333348987930306, 0.9332718122773609, 0.9336142483434892, 0.9329091734360082, 0.933937955718459, 0.9332848188924624]
Accuracy: 0.9290335416666666
Precision: 0.9938808333333334
Recall: 0.8797790239159466
F1_score: 0.9333554346172706

'''