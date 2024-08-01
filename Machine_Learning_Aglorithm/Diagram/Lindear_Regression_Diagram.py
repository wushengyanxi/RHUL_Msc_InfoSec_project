import multiprocessing
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\Machine_Learning_Aglorithm')
from Training_Set_Creator import Training_set_create
from Linear_Regression_1_1 import train_linear_regression
from Linear_Regression_1_1 import LR_preprocess_data
from Linear_Regression_1_1 import linear_regression_predict
from Linear_Regression_1_1 import LR_each_test_sample_preprocess
from Linear_Regression_1_1 import one_step_LR



def get_correct_rate(features, weights, test_sample, scale_factors, heaviest_features,share_data, sample, epoch, learning_rate):
    count = 0
    for i in test_sample:
        testing_sample = LR_each_test_sample_preprocess(i, scale_factors, features, heaviest_features)
        predict_label = linear_regression_predict(testing_sample[:-1], weights)
        if predict_label == testing_sample[-1]:
            count += 1
        
    correct_rate =  count / len(test_sample) 
    share_data['sample_epoch_learningrate_correctrate'].append([sample, epoch, learning_rate, correct_rate])

def split_list_into_sublists(lst):
    # 确保列表长度大于100
    
    # 每个子列表的长度
    sublist_length = 10
    # 子列表的数量
    number_of_sublists = len(lst) // sublist_length
    # 分割子列表
    sublists = [lst[i * sublist_length:(i + 1) * sublist_length] for i in range(number_of_sublists)]
    
    # 检查是否有剩余元素需要放入最后一个子列表
    remainder = len(lst) % sublist_length
    if remainder != 0:
        sublists.append(lst[-remainder:])
    
    return sublists


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    shared_data = manager.dict({
        'sample_amount_increment': manager.list(),
        'epochs_increment': manager.list(),
        'learning_rate_increment': manager.list(),
        'sample_epoch_learningrate_correctrate': manager.list()
    })
    
    Sample_Amount_Increment = [500,1000,2000]
    Epochs_Increment = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
    Learning_Rate_Increment = [0.01,0.02,0.03,0.04,0.05]
    
    shared_data['sample_amount_increment'][:] = Sample_Amount_Increment
    shared_data['epochs_increment'][:] = Epochs_Increment
    shared_data['learning_rate_increment'][:] = Learning_Rate_Increment
    
    processes = []
    
    for sample in Sample_Amount_Increment:
        
        Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(sample,sample,sample,sample,sample,sample)
        Test_sets = []
        for i in Testing_Data_Set:
            Test_sets += i
        
        for epoch in Epochs_Increment:
            
            for learning_rate in Learning_Rate_Increment:
                print("code is running with sample amount: {}, epoch: {}, learning rate: {}".format(sample, epoch, learning_rate))
                #scale_factors, weight_result, heaviest_features = one_step_LR(Training_Data_Set, Features_name, epochs=epoch, learning_rate=learning_rate)
                X_train, y_train, scale_factors, heaviest_features = LR_preprocess_data(Training_Data_Set, Features_name)
                weight_result = train_linear_regression(X_train, y_train, epoch, learning_rate)
                proc = multiprocessing.Process(target=get_correct_rate, args = (Features_name, weight_result, Test_sets, scale_factors, heaviest_features, shared_data, sample, epoch, learning_rate))
                processes.append(proc)
    
    
    sub_lists = split_list_into_sublists(processes)
    
    for i in sub_lists:
        for x in i:
            x.start()
        for x in i:
            x.join()
    
        
    Result = list(shared_data['sample_epoch_learningrate_correctrate'])
    
    print(Result)


# Result = []
# parameter_name = ["sample amount", "epoch", "learning rate", "correct rate"]

# print(len(Result))

def correct_rate_line_chart_drawing(result, parameter_name):
    # Get the desktop path
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    # Extract parameter information
    sample_amount_index = 0
    correct_rate_index = len(parameter_name) - 1
    parameter_indices = list(range(1, correct_rate_index))
    
    # Extract unique sample amounts
    sample_amounts = sorted(set(row[sample_amount_index] for row in result))
    num_columns = len(sample_amounts)
    num_rows = len(parameter_indices)
    
    # Prepare the figure
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(5*num_columns, 5*num_rows))
    
    for col, sample_amount in enumerate(sample_amounts):
        filtered_results = [row for row in result if row[sample_amount_index] == sample_amount]
        
        for row, param_index in enumerate(parameter_indices):
            ax = axes[row, col] if num_rows > 1 else axes[col]
            other_params_indices = parameter_indices.copy()
            other_params_indices.remove(param_index)
            
            # Determine fixed values for other parameters
            fixed_values = {}
            for other_param_index in other_params_indices:
                values = sorted(set(item[other_param_index] for item in filtered_results))
                median_value = values[len(values) // 2 - 1] if len(values) % 2 == 0 else values[len(values) // 2]
                fixed_values[other_param_index] = median_value
            
            # Filter results based on fixed values
            filtered_final_results = [
                row for row in filtered_results
                if all(row[idx] == fixed_values[idx] for idx in other_params_indices)
            ]
            
            # Extract x and y values for the line plot
            x_values = [row[param_index] for row in filtered_final_results]
            y_values = [row[correct_rate_index] for row in filtered_final_results]
            
            ax.plot(x_values, y_values, marker='o')
            ax.set_xlabel(parameter_name[param_index])
            ax.set_ylabel(parameter_name[correct_rate_index])
            ax.set_title(f'{parameter_name[sample_amount_index]}: {sample_amount}')
            ax.grid(True)
    
    plt.tight_layout()
    # Save the figure
    save_path = os.path.join(desktop_path, 'correct_rate_line_chart.png')
    plt.savefig(save_path)
    plt.show()
    

# correct_rate_line_chart_drawing(Result, parameter_name)