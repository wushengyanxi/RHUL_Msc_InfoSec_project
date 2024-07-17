import sys
from random import shuffle
import time

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
from Linear_Regression_1_1 import train_linear_regression
from Linear_Regression_1_1 import LR_preprocess_data
from Linear_Regression_1_1 import linear_regression_predict
from Linear_Regression_1_1 import one_step_LR



Sample_Amount_Increment = [300,600,900,1200,1500,1800,2100,2400,2700,3000]
Epochs_Increment = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
Learning_Rate_Increment = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]

def get_correct_rate(features, weights, test_sample, scale_factors):
    count = 0
    for sample in test_sample:
        predict_label = linear_regression_predict(features, weights, sample, scale_factors)
        if predict_label == sample[-1]:
            count += 1
        if count % 100 == 0:
            print("currently counting the correct rate",time.time())
    
    correct_rate =  count / len(test_sample)
    return correct_rate

Sample_Epochs_LearningRate_CorrectRate = []

for sample in Sample_Amount_Increment:
    Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(sample,sample,sample,sample,sample,sample)
    Test_sets = []
    for i in Testing_Data_Set:
        Test_sets += i
    for epoch in Epochs_Increment:
        for learning_rate in Learning_Rate_Increment:
            print("code is running with sample amount: {}, epoch: {}, learning rate: {}".format(sample, epoch, learning_rate))
            scale_factors, weight_result = one_step_LR(Training_Data_Set, Features_name, epochs=epoch, learning_rate=learning_rate)
            correct_rate = get_correct_rate(Features_name, weight_result, Test_sets, scale_factors)
            Sample_Epochs_LearningRate_CorrectRate.append([sample, epoch, learning_rate, correct_rate])


print(Sample_Epochs_LearningRate_CorrectRate[0])
print(Sample_Epochs_LearningRate_CorrectRate[1])
print(Sample_Epochs_LearningRate_CorrectRate[2])