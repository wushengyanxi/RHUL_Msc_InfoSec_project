import sys
import random
import time

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
from KNN import KNN_preprocess_data
from KNN import KNN_train
from KNN import KNN_predict

Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create()

X_train, y_train, numberic_features, scale_factors = KNN_preprocess_data(Training_Data_Set, Features_name)


print(Training_Data_Set[0])
print()
print(len(Training_Data_Set[0]))
print()
print(X_train[0])
print()
print(len(X_train[0]))
print(len(Features_name))
print(len(numberic_features))
print(len(scale_factors))


Kdtree = KNN_train(X_train, y_train)

all_test_sample = []
for i in range(0,len(Testing_Data_Set)):
    all_test_sample += Testing_Data_Set[i]

random.seed(65)
random.shuffle(all_test_sample)

all_test_sample = all_test_sample[:30000]


#test_sample = test_sample[:-1]
'''
print(test_sample)
print()
print(Features_name)
print()
print(scale_factors)
'''

start = time.time()

count = 0
for i in range(0,len(all_test_sample)):    
    
    test_sample = all_test_sample[i][:-1]
    predict = KNN_predict(test_sample,Features_name,numberic_features, Kdtree, scale_factors, k=3)
    if predict == all_test_sample[i][-1]:
        count += 1
end = time.time()
print("模型运行时间是：", end-start)
print("模型最终正确率是：", count/len(all_test_sample))





