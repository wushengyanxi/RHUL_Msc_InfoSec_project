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


for m in range(0,20):
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
Accuracy: [0.929, 0.9293666666666667, 0.9290875000000001, 0.92905, 0.9295416666666667, 0.9284375, 0.9297041666666666, 0.9288708333333332, 0.9294541666666666, 0.9289499999999999]
Precision: [0.8639666666666667, 0.8649583333333334, 0.8644750000000001, 0.8640749999999999, 0.8657833333333333, 0.86375, 0.86505, 0.8634, 0.8640083333333334, 0.86315]
Recall: [0.993141236876389, 0.9928545464980582, 0.9927650656024806, 0.9931325785874374, 0.9923207702152859, 0.9921033740129217, 0.9935204770153996, 0.9934891213669969, 0.9941319168112912, 0.9939543988945186]
F1_score: [0.9240614638668045, 0.924504101682536, 0.92418917293635, 0.9241196759444934, 0.924743433408397, 0.923488138990979, 0.9248450884922244, 0.9238876974590592, 0.9245139170638405, 0.9239456219225005]
Accuracy: 0.92914625
Precision: 0.8642616666666667
Recall: 0.9931413485880778
F1_score: 0.9242298311767184

--------------------

Accuracy: [0.9285749999999999, 0.9287916666666667, 0.9288375, 0.9295625, 0.9302041666666667, 0.9283333333333333, 0.9298625, 0.9295749999999999, 0.9289, 0.9293208333333334, 0.9300958333333333, 0.92875, 0.9291541666666665, 0.9283791666666666, 0.9295499999999999, 0.9293166666666666, 0.9286958333333334, 0.9294333333333334, 0.929, 0.9295874999999999]
Precision: [0.8626499999999999, 0.8636583333333333, 0.863875, 0.8657916666666666, 0.866375, 0.8629666666666667, 0.865625, 0.8656666666666667, 0.863425, 0.8641000000000001, 0.8656916666666666, 0.86325, 0.8646333333333334, 0.8629166666666667, 0.8648416666666667, 0.8648166666666666, 0.8634, 0.86525, 0.8636833333333334, 0.8648833333333333]
Recall: [0.9936646892818062, 0.99301510041392, 0.9928741775134328, 0.9923587563876021, 0.9931601723330881, 0.9927525116956821, 0.9932302573075931, 0.9925283292885669, 0.9935274149933836, 0.9937228669726968, 0.9936868082989775, 0.9933831990794016, 0.9927378845141847, 0.9929139218901323, 0.9934048052072365, 0.992900880214313, 0.9930891698377249, 0.9926765841906, 0.9934626739772248, 0.9934431564740455]
F1_score: [0.9235339774643363, 0.9238304927618911, 0.9238934267342219, 0.9247646810120386, 0.9254454092691415, 0.9233210884838975, 0.9250477551729204, 0.9247669832904541, 0.9239185682566009, 0.9243894502712318, 0.925283798648811, 0.9237560192616372, 0.9242678662895576, 0.9233622692151644, 0.92467590323874, 0.9244432567254587, 0.92371472257265, 0.9245934923150902, 0.9240384443929317, 0.9247163330675232]
Accuracy: 0.9291962500000001
Precision: 0.8643750000000001
Recall: 0.9931266679935806
F1_score: 0.9242881969222149

'''