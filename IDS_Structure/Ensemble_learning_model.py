import sys
import random
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\Machine_Learning_Aglorithm')
from KNN import KNN_preprocess_training
from KNN import KNN_train
from KNN import KNN_predict
from KNN import KNN_each_test_sample_preprocess
from softmax import Softmax_preprocess_training
from softmax import softmax_train
from softmax import softmax_predict
from softmax import softmax_each_test_sample_preprocess
from Linear_Regression_1_1 import train_linear_regression
from Linear_Regression_1_1 import LR_preprocess_data
from Linear_Regression_1_1 import linear_regression_predict
from Linear_Regression_1_1 import LR_each_test_sample_preprocess
from SVM import svm_train
from SVM import svm_preprocess_training
from SVM import svm_predict
from SVM import svm_each_test_sample_preprocess

def split_list_into_five_parts_manual(lst):
    avg = len(lst) // 5
    remainder = len(lst) % 5
    out = []
    last = 0

    for i in range(5):
        part_size = avg + (1 if remainder > 0 else 0)
        out.append(lst[last:last + part_size])
        last += part_size
        remainder -= 1

    return out

def create_K_fold_training_set(Data):
    k_fold = []
    random.shuffle(Data)
    result = split_list_into_five_parts_manual(Data)
    train1 = result[1] + result[2] + result[3] + result[4]
    test1 = result[0]
    k_fold.append([train1, test1])
    train2 = result[0] + result[2] + result[3] + result[4]
    test2 = result[1]
    k_fold.append([train2, test2])
    train3 = result[0] + result[1] + result[3] + result[4]
    test3 = result[2]
    k_fold.append([train3, test3])
    train4 = result[0] + result[1] + result[2] + result[4]
    test4 = result[3]
    k_fold.append([train4, test4])
    train5 = result[0] + result[1] + result[2] + result[3]
    test5 = result[4]
    k_fold.append([train5, test5])
    
    return k_fold
        

def Ensemble_Learning_Training(feature_list, Data):
    
    Ensemble_parameters = []
    
    k_fold = create_K_fold_training_set(Data)
    
    LR_correct_rate = 0
    LR_weights = None
    LR_scale_factors = None
    for i in range(0,5):
        training_set = k_fold[i][0]
        testing_set = k_fold[i][1]
        X_train, y_train, current_LR_scale_factors, LR_heaviest_features = LR_preprocess_data(training_set, feature_list)
        current_LR_weights = train_linear_regression(X_train, y_train, 10000, 0.001)
        count = 0
        for sample in testing_set:
            LR_sample = LR_each_test_sample_preprocess(sample, current_LR_scale_factors, feature_list, LR_heaviest_features)
            if linear_regression_predict(LR_sample[:-1], current_LR_weights) == sample[-1]:
                count += 1
        current_correct_rate = count / len(testing_set)
        if current_correct_rate > LR_correct_rate:
            LR_correct_rate = current_correct_rate
            LR_weights = current_LR_weights
            LR_scale_factors = current_LR_scale_factors       
    Ensemble_parameters.append([LR_weights, LR_scale_factors, LR_heaviest_features])
    
    
    svm_correct_rate = 0
    svm_model = None
    svm_scale_factors = None
    for i in range(0,5):
        training_set = k_fold[i][0]
        testing_set = k_fold[i][1]
        X_train, y_train, current_svm_scale_factors, svm_heaviest_features = svm_preprocess_training(training_set, feature_list)
        current_svm_model = svm_train(X_train, y_train)
        count = 0
        for sample in testing_set:
            svm_sample = svm_each_test_sample_preprocess(sample, current_svm_scale_factors, feature_list, svm_heaviest_features)
            if svm_predict(current_svm_model, svm_sample[:-1]) == sample[-1]:
                count += 1
        current_correct_rate = count / len(testing_set)
        if current_correct_rate > svm_correct_rate:
            svm_correct_rate = current_correct_rate
            svm_model = current_svm_model
            svm_scale_factors = current_svm_scale_factors
    Ensemble_parameters.append([svm_model, svm_scale_factors, svm_heaviest_features])
    
    
    knn_correct_rate = 0
    Kdtree = None
    knn_scale_factors = None
    for i in range(0,5):
        training_set = k_fold[i][0]
        testing_set = k_fold[i][1]
        X_train, y_train, current_knn_scale_factors, knn_heaviest_features = KNN_preprocess_training(training_set, feature_list)
        current_Kdtree = KNN_train(X_train, y_train)
        count = 0
        for sample in testing_set:
            knn_sample = KNN_each_test_sample_preprocess(sample, current_knn_scale_factors, feature_list, knn_heaviest_features)
            if KNN_predict(knn_sample[:-1], current_Kdtree) == sample[-1]:
                count += 1
        current_correct_rate = count / len(testing_set)
        if current_correct_rate > knn_correct_rate:
            knn_correct_rate = current_correct_rate
            Kdtree = current_Kdtree
            knn_scale_factors = current_knn_scale_factors
    Ensemble_parameters.append([Kdtree, knn_scale_factors, knn_heaviest_features])
    
    
    softmax_correct_rate = 0
    softmax_weights = None
    softmax_scale_factors = None
    for i in range(0,5):
        training_set = k_fold[i][0]
        testing_set = k_fold[i][1]
        X_train, y_train, current_softmax_scale_factors, softmax_heaviest_features = Softmax_preprocess_training(training_set, feature_list)
        current_softmax_weights = softmax_train(X_train, y_train, learning_rate=0.08, epochs=10000)
        count = 0
        for sample in testing_set:
            softmax_sample = softmax_each_test_sample_preprocess(sample, current_softmax_scale_factors, feature_list, softmax_heaviest_features)
            if softmax_predict(softmax_sample[:-1], current_softmax_weights) == sample[-1]:
                count += 1
        current_correct_rate = count / len(testing_set)
        if current_correct_rate > softmax_correct_rate:
            softmax_correct_rate = current_correct_rate
            softmax_weights = current_softmax_weights
            softmax_scale_factors = current_softmax_scale_factors
    Ensemble_parameters.append([softmax_weights, softmax_scale_factors, softmax_heaviest_features])
    print("model training finished with LR_correct_rate:",LR_correct_rate,"svm_correct_rate:",svm_correct_rate,"knn_correct_rate:",knn_correct_rate,"softmax_correct_rate:",softmax_correct_rate)
    
    return Ensemble_parameters

def Ensemble_Learning_Decision(param, new_sample, features_list):
    #print("length of new sample which given to ensemble model:",len(new_sample))
    LR_sample = LR_each_test_sample_preprocess(new_sample, param[0][1], features_list, param[0][2])
    svm_sample = svm_each_test_sample_preprocess(new_sample, param[1][1], features_list, param[1][2])
    knn_sample = KNN_each_test_sample_preprocess(new_sample, param[2][1], features_list, param[2][2])
    softmax_sample = softmax_each_test_sample_preprocess(new_sample, param[3][1], features_list, param[3][2])
    
    benign_count = 0
    malicious_count = 0
    '''
    # a backup of runable code
    
    if linear_regression_predict(LR_sample[:-1], param[0][0]) == 0:
        benign_count += 1
    else:
        malicious_count += 1
    
    if svm_predict(param[1][0], svm_sample[:-1]) == 0:
        benign_count += 1
    else:
        malicious_count += 1
    
    if KNN_predict(knn_sample[:-1], param[2][0]) == 0:
        benign_count += 1
    else:
        malicious_count += 1
    
    if softmax_predict(softmax_sample[:-1], param[3][0]) == 0:
        benign_count += 1
    else:
        malicious_count += 1
    
    predict_decision = 0
    
    if benign_count != malicious_count:
        if benign_count > malicious_count:
            predict_decision = 0
        else:
            predict_decision = 1
    else:
        predict_decision = random.randint(0,1)
    
    '''
    
    linear_regression_prediction = linear_regression_predict(LR_sample[:-1], param[0][0])
    if linear_regression_prediction == 0:
        benign_count += 1
    else:
        malicious_count += 1
    
    svm_prediction = svm_predict(param[1][0], svm_sample[:-1])
    if svm_prediction == 0:
        benign_count += 1
    else:
        malicious_count += 1
    
    knn_prediction = KNN_predict(knn_sample[:-1], param[2][0])
    if knn_prediction == 0:
        benign_count += 1
    else:
        malicious_count += 1
    
    softmax_prediction = softmax_predict(softmax_sample[:-1], param[3][0])
    if softmax_prediction == 0:
        benign_count += 1
    else:
        malicious_count += 1
    
    predict_decision = 0
    reliable = 0
    
    if linear_regression_prediction == svm_prediction == knn_prediction == softmax_prediction:
        reliable = 1
    
    if benign_count != malicious_count:
        if benign_count > malicious_count:
            predict_decision = 0
        else:
            predict_decision = 1
    else:
        predict_decision = random.randint(0,1)
        
        # 其实需要研究一下，如果是平局的话，应该怎么办
    
    return predict_decision, reliable