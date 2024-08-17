import sys
import random
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
from SVM import svm_train
from SVM import svm_preprocess_training
from SVM import svm_predict
from SVM import svm_each_test_sample_preprocess

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

        X, y, scale_factors, heaviest_features = svm_preprocess_training(Training_Data_Set, Features_name)

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
        True_positive += count
        False_positive += len(all_test_sample_benign)-count

        correct_rate0 = count / len(all_test_sample_benign)
        print("The correct rate for benign sample is: ", correct_rate0)
        avg_correct_rate0 += correct_rate0

        count = 0
        for samples in all_test_sample_malicious:
            test_sample = svm_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
            predict = svm_predict(model, test_sample[:-1])
            if predict == samples[-1]:
                count += 1
        True_negative += count
        False_negative += len(all_test_sample_malicious)-count

        correct_rate1 = count / len(all_test_sample_malicious)
        print("The correct rate for malicious sample is: ", correct_rate1)
        avg_correct_rate1 += correct_rate1

    print("svm total correct rate for benign sample is: ", avg_correct_rate0 / 20)
    print("svm total correct rate for malicious sample is: ", avg_correct_rate1 / 20)

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
Accuracy: [0.9098083333333332, 0.910225, 0.9096458333333334, 0.9088291666666666, 0.9111625000000001, 0.9109958333333334, 0.9095500000000001, 0.9094375, 0.9100208333333333, 0.9112916666666667]
Precision: [0.8246083333333333, 0.8251916666666667, 0.8257, 0.8231333333333334, 0.8291999999999999, 0.8265333333333333, 0.8250833333333333, 0.8225083333333334, 0.82435, 0.8287166666666667]
Recall: [0.9939830440372871, 0.9942866896939514, 0.9922986790582156, 0.9933925356772901, 0.9917770534940047, 0.9945351903658916, 0.9928004171346062, 0.9956020456539939, 0.9948008326712858, 0.9926533708650256]
F1_score: [0.9014083224019822, 0.9018816714634412, 0.9013659126780166, 0.9002839135408075, 0.9032310771618806, 0.9027847939962409, 0.9012051263380179, 0.900815471599958, 0.9015899634978285, 0.9033072639906985]
Accuracy: 0.9100966666666667
Precision: 0.8255025
Recall: 0.9936129858651552
F1_score: 0.9017873516668871

'''