import sys
from random import shuffle
import numpy as np
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\Machine_Learning_Aglorithm')
from Linear_Regression_1_1 import train_linear_regression
from Linear_Regression_1_1 import LR_preprocess_data


Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(500,500,1500,1500,1500,1500)
X_train, y_train, scale_factors, heaviest_features = LR_preprocess_data(Training_Data_Set, Features_name)

weights = train_linear_regression(X_train, y_train, 10000, 0.004)
print(weights)

def find_top_n_indices(arr, n):
    top_n_indices = np.argsort(arr)[-n:]
    top_n_indices = top_n_indices[np.argsort(arr[top_n_indices])[::-1]]
    return top_n_indices.tolist()

def find_bottom_n_indices(arr, n):
    bottom_n_indices = np.argsort(arr)[:n]
    bottom_n_indices = bottom_n_indices[np.argsort(arr[bottom_n_indices])]
    return bottom_n_indices.tolist()

top15 = find_top_n_indices(weights, 15)
bot15 = find_bottom_n_indices(weights, 15)
print(top15)
print()
print(bot15)