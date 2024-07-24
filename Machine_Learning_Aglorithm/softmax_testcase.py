import sys
import random
import time

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
from softmax import softmax, cross_entropy_loss, train_softmax_classifier
from softmax import Softmax_preprocess_data
from softmax import train_softmax_classifier
from softmax import classify_softmax

Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create()

X_train, y_train, numeric_features, scale_factors = Softmax_preprocess_data(Training_Data_Set, Features_name)

weight, bias = train_softmax_classifier(X_train, y_train, learning_rate=0.01, epochs=1000)

print(weight)
print(bias)
print(len(weight))

print("start predicting test sample") 

all_test_sample = []
for i in range(0,len(Testing_Data_Set)):
    all_test_sample += Testing_Data_Set[i]

random.seed(65)
random.shuffle(all_test_sample)

all_test_sample = all_test_sample[:30000]
count = 0

for samples in all_test_sample:
    test_sample = samples[:-1]
    prediction = classify_softmax(test_sample, weight, bias, Features_name, numeric_features, scale_factors)
    if prediction == samples[-1]:
        count += 1

correct_rate = count / len(all_test_sample)
print("The correct rate is: ", correct_rate)