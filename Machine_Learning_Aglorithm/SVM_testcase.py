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
        True_negative += count
        False_negative += len(all_test_sample_benign)-count

        correct_rate0 = count / len(all_test_sample_benign)
        print("The correct rate for benign sample is: ", correct_rate0)
        avg_correct_rate0 += correct_rate0

        count = 0
        for samples in all_test_sample_malicious:
            test_sample = svm_each_test_sample_preprocess(samples, scale_factors, Features_name, heaviest_features)
            predict = svm_predict(model, test_sample[:-1])
            if predict == samples[-1]:
                count += 1
        True_positive += count
        False_positive += len(all_test_sample_malicious)-count

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
Accuracy: [0.9102208333333333, 0.90985, 0.9094875, 0.910525, 0.9105, 0.9101208333333333, 0.9110833333333334, 0.9116458333333334, 0.911275, 0.9106625, 0.9110916666666667, 0.9110041666666666, 0.9090833333333334, 0.9097291666666667, 0.9106083333333332, 0.9097875000000001, 0.910075, 0.9090541666666666, 0.9118416666666667, 0.9094083333333333]
Precision: [0.9956666666666667, 0.9947916666666666, 0.9936083333333333, 0.9951416666666667, 0.9949250000000001, 0.9946499999999999, 0.9953666666666666, 0.9966499999999999, 0.9958333333333333, 0.9941666666666666, 0.99485, 0.9969166666666667, 0.9946833333333334, 0.995175, 0.993425, 0.9937833333333334, 0.9949666666666667, 0.9960916666666667, 0.9964916666666667, 0.994225]
Recall: [0.8503490929277544, 0.850334078896756, 0.8505160889941435, 0.8511061379251361, 0.8511998973349874, 0.8508129762558184, 0.8517842370995807, 0.8518315396613985, 0.8517826849330692, 0.8518933740833041, 0.8521077500678077, 0.8507385203990926, 0.8492856329690346, 0.8499291850228102, 0.8522620034888043, 0.8508479655248682, 0.8505563708379044, 0.8484054823300612, 0.8522121497192053, 0.8500313488557668]       
F1_score: [0.9172882110653456, 0.9169079750831457, 0.916510690305893, 0.917505416660264, 0.917467782465362, 0.9171260954254342, 0.9179949890096378, 0.9185679010449269, 0.9181925053977427, 0.9175476174910879, 0.9179629529953635, 0.9180451156669314, 0.9162521493490543, 0.9168352481891082, 0.9174452431159477, 0.9167778166428991, 0.9171115395546405, 0.9163360662963943, 0.9187218611226357, 0.916491265805282]        
Accuracy: 0.9103527083333333
Precision: 0.9950704166666666
Recall: 0.8508993258663653
F1_score: 0.9173544221343548

'''