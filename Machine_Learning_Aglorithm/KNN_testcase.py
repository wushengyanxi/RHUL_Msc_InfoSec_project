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


for m in range(0,10):
    avg_correct_rate0 = 0
    avg_correct_rate1 = 0
    True_positive = 0
    True_negative = 0
    False_positive = 0
    False_negative = 0

    for x in range(0,20):
        
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
        True_positive += count
        False_positive += len(all_test_sample_benign)-count

        correct_rate0 = count / len(all_test_sample_benign)
        print("The correct rate for benign sample is: ", correct_rate0)
        avg_correct_rate0 += correct_rate0

        count = 0
        for samples in all_test_sample_malicious:
            testing_sample = KNN_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
            prediction = KNN_predict(testing_sample[:-1], Kdtree, k=3)
            if prediction == 1: #samples[-1]:
                count += 1
        True_negative += count
        False_negative += len(all_test_sample_malicious)-count

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
Accuracy: [0.929, 0.9293666666666667, 0.9290875000000001, 0.92905, 0.9295416666666667, 0.9284375, 0.9297041666666666, 0.9288708333333332, 0.9294541666666666, 0.9289499999999999]
Precision: [0.8639666666666667, 0.8649583333333334, 0.8644750000000001, 0.8640749999999999, 0.8657833333333333, 0.86375, 0.86505, 0.8634, 0.8640083333333334, 0.86315]
Recall: [0.993141236876389, 0.9928545464980582, 0.9927650656024806, 0.9931325785874374, 0.9923207702152859, 0.9921033740129217, 0.9935204770153996, 0.9934891213669969, 0.9941319168112912, 0.9939543988945186]
F1_score: [0.9240614638668045, 0.924504101682536, 0.92418917293635, 0.9241196759444934, 0.924743433408397, 0.923488138990979, 0.9248450884922244, 0.9238876974590592, 0.9245139170638405, 0.9239456219225005]
Accuracy: 0.92914625
Precision: 0.8642616666666667
Recall: 0.9931413485880778
F1_score: 0.9242298311767184

'''