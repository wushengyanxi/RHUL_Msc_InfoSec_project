import sys
import random
import time

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
from KNN import KNN_preprocess_training
from KNN import KNN_train
from KNN import KNN_predict
from KNN import each_test_sample_preprocess

for x in range(0,20):
    
    print("round: ", x)

    Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(3000,3000,1500,1500,1500,1500)

    X_train, y_train, scale_factors, heaviest_features = KNN_preprocess_training(Training_Data_Set, Features_name)

    Kdtree = KNN_train(X_train, y_train)

    all_test_sample_benign = Testing_Data_Set[0][:3000] + Testing_Data_Set[1][:3000]
    all_test_sample_malicious = Testing_Data_Set[2][:1500] + Testing_Data_Set[3][:1500] + Testing_Data_Set[4][:1500] + Testing_Data_Set[5][:1500]

    count = 0
    for samples in all_test_sample_benign:
        testing_sample = each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
        prediction = KNN_predict(testing_sample[:-1], Kdtree, k=3)
        if prediction == 0: #samples[-1]:
            count += 1

    correct_rate = count / len(all_test_sample_benign)
    print("The correct rate for benign sample is: ", correct_rate)

    count = 0
    for samples in all_test_sample_malicious:
        testing_sample = each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
        prediction = KNN_predict(testing_sample[:-1], Kdtree, k=3)
        if prediction == 1: #samples[-1]:
            count += 1

    correct_rate = count / len(all_test_sample_malicious)
    print("The correct rate for malicious sample is: ", correct_rate)
