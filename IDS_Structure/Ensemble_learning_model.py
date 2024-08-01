import sys
import random
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\Machine_Learning_Aglorithm')
from ..Machine_Learning_Aglorithm.KNN import KNN_preprocess_training
from ..Machine_Learning_Aglorithm.KNN import KNN_train
from ..Machine_Learning_Aglorithm.KNN import KNN_predict
from ..Machine_Learning_Aglorithm.KNN import KNN_each_test_sample_preprocess
from ..Machine_Learning_Aglorithm.softmax import Softmax_preprocess_training
from ..Machine_Learning_Aglorithm.softmax import softmax_train
from ..Machine_Learning_Aglorithm.softmax import softmax_predict
from ..Machine_Learning_Aglorithm.softmax import softmax_each_test_sample_preprocess
from ..Machine_Learning_Aglorithm.Linear_Regression_1_1 import train_linear_regression
from ..Machine_Learning_Aglorithm.Linear_Regression_1_1 import LR_preprocess_data
from ..Machine_Learning_Aglorithm.Linear_Regression_1_1 import linear_regression_predict
from ..Machine_Learning_Aglorithm.Linear_Regression_1_1 import LR_each_test_sample_preprocess
from ..Machine_Learning_Aglorithm.SVM import svm_train
from ..Machine_Learning_Aglorithm.SVM import svm_preprocess_training
from ..Machine_Learning_Aglorithm.SVM import svm_predict
from ..Machine_Learning_Aglorithm.SVM import svm_each_test_sample_preprocess


def Ensemble_Learning_Training(feature_list, Data):
    
    Ensemble_parameters = []
    
    X_train, y_train, LR_scale_factors, LR_heaviest_features = LR_preprocess_data(Data, feature_list)
    LR_weights = train_linear_regression(X_train, y_train, 10000, 0.001)
    Ensemble_parameters.append([LR_weights, LR_scale_factors, LR_heaviest_features])
    
    X_train, y_train, svm_scale_factors, svm_heaviest_features = svm_preprocess_training(Data, feature_list)
    svm_model = svm_train(X_train, y_train)
    Ensemble_parameters.append([svm_model, svm_scale_factors, svm_heaviest_features])
    
    X_train, y_train, knn_scale_factors, knn_heaviest_features = KNN_preprocess_training(Data, feature_list)
    Kdtree = KNN_train(X_train, y_train)
    Ensemble_parameters.append([Kdtree, knn_scale_factors, knn_heaviest_features])
    
    X_train, y_train, softmax_scale_factors, softmax_heaviest_features = Softmax_preprocess_training(Data, feature_list)
    softmax_weights = softmax_train(X_train, y_train, learning_rate=0.08, epochs=10000)
    Ensemble_parameters.append([softmax_weights, softmax_scale_factors, softmax_heaviest_features])
    
    return Ensemble_parameters

def Ensemble_Learning_Decision(param, new_sample, features_list):
    
    LR_sample = LR_each_test_sample_preprocess(new_sample[:-1], param[0][1], features_list, param[0][2])
    svm_sample = svm_each_test_sample_preprocess(new_sample[:-1], param[1][1], features_list, param[1][2])
    knn_sample = KNN_each_test_sample_preprocess(new_sample[:-1], param[2][1], features_list, param[2][2])
    softmax_sample = softmax_each_test_sample_preprocess(new_sample[:-1], param[3][1], features_list, param[3][2])
    
    benign_count = 0
    malicious_count = 0
    
    if linear_regression_predict(LR_sample, param[0][0]) == 0:
        benign_count += 1
    else:
        malicious_count += 1
    
    if svm_predict(param[1][0], svm_sample) == 0:
        benign_count += 1
    else:
        malicious_count += 1
    
    if KNN_predict(knn_sample, param[2][0]) == 0:
        benign_count += 1
    else:
        malicious_count += 1
    
    if softmax_predict(softmax_sample, param[3][0]) == 0:
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
        
        # 其实需要研究一下，如果是平局的话，应该怎么办