import sys
import random
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
from SVM import svm_train
from SVM import Softmax_preprocess_training
from SVM import svm_predict
from SVM import svm_each_test_sample_preprocess


for i in range(0,20):
    print("round: ", i)
    
    Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(3000,3000,1500,1500,1500,1500)

    X, y, scale_factors, heaviest_features = Softmax_preprocess_training(Training_Data_Set, Features_name)

    model = svm_train(X, y)
    
    all_test_sample = []
    for i in range(0,len(Testing_Data_Set)):
        random.shuffle(Testing_Data_Set[i])
        all_test_sample += Testing_Data_Set[i]


    all_test_sample_benign = Testing_Data_Set[0][:3000] + Testing_Data_Set[1][:3000]
    all_test_sample_malicious = Testing_Data_Set[2][:1500] + Testing_Data_Set[3][:1500] + Testing_Data_Set[4][:1500] + Testing_Data_Set[5][:1500]

    count = 0

    for samples in all_test_sample_benign:
        test_sample = svm_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
        predict = svm_predict(model, test_sample[:-1])
        if predict == samples[-1]:
            count += 1

    correct_rate = count / len(all_test_sample_benign)
    print("The correct rate for benign sample is: ", correct_rate)

    count = 0
    for samples in all_test_sample_malicious:
        test_sample = svm_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
        predict = svm_predict(model, test_sample[:-1])
        if predict == samples[-1]:
            count += 1

    correct_rate = count / len(all_test_sample_malicious)
    print("The correct rate for malicious sample is: ", correct_rate)

