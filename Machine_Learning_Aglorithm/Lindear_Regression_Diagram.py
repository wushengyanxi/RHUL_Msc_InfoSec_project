import multiprocessing
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
from Linear_Regression_1_1 import train_linear_regression
from Linear_Regression_1_1 import LR_preprocess_data
from Linear_Regression_1_1 import linear_regression_predict
from Linear_Regression_1_1 import one_step_LR


'''
def get_correct_rate(features, weights, test_sample, scale_factors, share_data, sample, epoch, learning_rate):
    count = 0
    for i in test_sample:
        predict_label = linear_regression_predict(features, weights, i, scale_factors)
        if predict_label == i[-1]:
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
                scale_factors, weight_result = one_step_LR(Training_Data_Set, Features_name, epochs=epoch, learning_rate=learning_rate)
                proc = multiprocessing.Process(target=get_correct_rate, args = (Features_name, weight_result, Test_sets, scale_factors, shared_data, sample, epoch, learning_rate))
                processes.append(proc)
    
    
    sub_lists = split_list_into_sublists(processes)
    
    for i in sub_lists:
        for x in i:
            x.start()
        for x in i:
            x.join()
    
        
    Result = list(shared_data['sample_epoch_learningrate_correctrate'])
    
    print(Result)
'''

Result = [[5000, 5000, 0.01, 0.9443489617021801], [5000, 5000, 0.02, 0.9539978521516471], [5000, 5000, 0.03, 0.9212154439742176], [5000, 5000, 0.04, 0.9002913763683293], [5000, 5000, 0.05, 0.9465851059242422], [5000, 10000, 0.01, 0.9567966258657615], [5000, 10000, 0.02, 0.925231879326241], [5000, 10000, 0.03, 0.9472319245008717], [5000, 10000, 0.04, 0.9069115131653248], [5000, 10000, 0.05, 0.9268006710486059], [5000, 15000, 0.01, 0.9464557422089163], [5000, 15000, 0.02, 0.9457370549015501], [5000, 15000, 0.03, 0.8889114762042633], [5000, 15000, 0.04, 0.9136035186930569], [5000, 15000, 0.05, 0.9221579510430207], [5000, 20000, 0.01, 0.9451477313095099], [5000, 20000, 0.02, 0.9059299916426933], [5000, 20000, 0.03, 0.8841927806833279], [5000, 20000, 0.04, 0.9170326838453468], [5000, 20000, 0.05, 0.8940654908942318], [5000, 25000, 0.01, 0.9163406906379684], [5000, 25000, 0.02, 0.915658964392124], [5000, 25000, 0.03, 0.9198335109517679], [5000, 25000, 0.04, 0.9149772381462795], [5000, 25000, 0.05, 0.8975706315618718], [5000, 30000, 0.01, 0.9265090893410459], [5000, 30000, 0.02, 0.9148129667617387], [5000, 30000, 0.03, 0.8889176363811835], [5000, 30000, 0.04, 0.9180573266064201], [5000, 30000, 0.05, 0.9103940665175904], [5000, 35000, 0.01, 0.917303731629839], [5000, 35000, 0.02, 0.9037554491898341], [5000, 35000, 0.03, 0.9103591588483755], [5000, 35000, 0.04, 0.9153550623307235], [5000, 35000, 0.05, 0.9037944636436626], [5000, 40000, 0.01, 0.9194680071211645], [5000, 40000, 0.02, 0.9261086778412276], [5000, 40000, 0.03, 0.9174926437220611], [5000, 40000, 0.04, 0.8992605734303356], [5000, 40000, 0.05, 0.8768108353405243], [5000, 45000, 0.01, 0.9115172721093883], [5000, 45000, 0.02, 0.8978786404078859], [5000, 45000, 0.03, 0.9138396588083343], [5000, 45000, 0.04, 0.9135439703161609], [5000, 45000, 0.05, 0.9069299936960856], [5000, 50000, 0.01, 0.9233735592886227], [5000, 50000, 0.02, 0.9115562865632167], [5000, 50000, 0.03, 0.914168201577416], [5000, 50000, 0.04, 0.8980654991078011], [5000, 50000, 0.05, 0.9045028839894949], [8000, 5000, 0.01, 0.9650020972691348], [8000, 5000, 0.02, 0.9605448484447093], [8000, 5000, 0.03, 0.9374219043203744], [8000, 5000, 0.04, 0.9465174294103362], [8000, 10000, 0.01, 0.9484667858798596], [8000, 10000, 0.02, 0.9496964478883811], [8000, 10000, 0.03, 0.9321522396626708], [8000, 5000, 0.05, 0.9324966333311256], [8000, 10000, 0.04, 0.9261319734198733], [8000, 10000, 0.05, 0.9283374174890169], [8000, 15000, 0.01, 0.9446850784820187], [8000, 15000, 0.02, 0.9186105040068878], [8000, 15000, 0.03, 0.9259752301476919], [8000, 15000, 0.05, 0.9256175905689118], [8000, 15000, 0.04, 0.9253416341038038], [8000, 20000, 0.01, 0.9374682650065126], [8000, 20000, 0.02, 0.9152813652118241], [8000, 20000, 0.03, 0.9229882773693622], [8000, 20000, 0.04, 0.9253416341038038], [8000, 20000, 0.05, 0.9004216614786851], [8000, 25000, 0.01, 0.9519968209815219], [8000, 25000, 0.02, 0.9265470119433958], [8000, 25000, 0.03, 0.922372342539241], [8000, 25000, 0.04, 0.9063138839216726], [8000, 25000, 0.05, 0.9200852153564254], [8000, 30000, 0.01, 0.9455129478773429], [8000, 30000, 0.02, 0.9073271960615493], [8000, 30000, 0.03, 0.9232598185310286], [8000, 30000, 0.04, 0.9018036514559463], [8000, 30000, 0.05, 0.8969556482769279], [8000, 35000, 0.01, 0.9321853544384838], [8000, 35000, 0.02, 0.9300527628761287], [8000, 35000, 0.03, 0.907234474689273], [8000, 35000, 0.04, 0.9140406649446984], [8000, 35000, 0.05, 0.9035521116188711], [8000, 40000, 0.01, 0.9277258979623375], [8000, 40000, 0.02, 0.9235379826478575], [8000, 40000, 0.03, 0.9019979248073824], [8000, 40000, 0.04, 0.9221979380532928], [8000, 40000, 0.05, 0.9135880963419211], [8000, 45000, 0.02, 0.916376360465373], [8000, 45000, 0.01, 0.9286244122127293], [8000, 45000, 0.03, 0.9174426562465505], [8000, 45000, 0.04, 0.9088438527937832], [8000, 45000, 0.05, 0.9108660617700951], [8000, 50000, 0.01, 0.9317769388701238], [8000, 50000, 0.02, 0.9108219087356778], [8000, 50000, 0.03, 0.9216327792127514], [8000, 50000, 0.04, 0.9188467227410204], [8000, 50000, 0.05, 0.9180365145594631], [11000, 5000, 0.01, 0.9684096483082125], [11000, 5000, 0.02, 0.9555039645688977], [11000, 5000, 0.03, 0.9514417696502131], [11000, 5000, 0.04, 0.8362978307974379], [11000, 5000, 0.05, 0.9428911588923018], [11000, 10000, 0.01, 0.9640879110412649], [11000, 10000, 0.02, 0.93068790627902], [11000, 10000, 0.03, 0.9350572659951901], [11000, 10000, 0.04, 0.9306688572993309], [11000, 10000, 0.05, 0.9230206919541872], [11000, 15000, 0.01, 0.9447246231873705], [11000, 15000, 0.02, 0.9365954711050789], [11000, 15000, 0.03, 0.9380027144796057], [11000, 15000, 0.04, 0.9255470628854442], [11000, 15000, 0.05, 0.9102483510726956], [11000, 20000, 0.01, 0.9370431221277711], [11000, 20000, 0.02, 0.9287044312689001], [11000, 20000, 0.03, 0.9406981451056028], [11000, 20000, 0.04, 0.9387646736671666], [11000, 20000, 0.05, 0.9252518037002643], [11000, 25000, 0.01, 0.9359525680405744], [11000, 25000, 0.02, 0.9348810629330666], [11000, 25000, 0.03, 0.9178560373360002], [11000, 25000, 0.04, 0.9332380884348882], [11000, 25000, 0.05, 0.9024192204205063], [11000, 30000, 0.02, 0.9301211991332714], [11000, 30000, 0.03, 0.9082243969807368], [11000, 30000, 0.04, 0.9096816439269472], [11000, 30000, 0.05, 0.9051503678834203], [11000, 30000, 0.01, 0.9327428149629735], [11000, 35000, 0.01, 0.9321213420006191], [11000, 35000, 0.02, 0.9242469700216682], [11000, 35000, 0.03, 0.9226492368502512], [11000, 35000, 0.04, 0.9323832654713432], [11000, 35000, 0.05, 0.9182465414196253], [11000, 40000, 0.01, 0.9310403124032669], [11000, 40000, 0.02, 0.9319975236326404], [11000, 40000, 0.04, 0.9240874348167726], [11000, 40000, 0.03, 0.9296997404576517], [11000, 40000, 0.05, 0.931116508322023], [11000, 45000, 0.01, 0.9293973379050885], [11000, 45000, 0.02, 0.9261042455413482], [11000, 45000, 0.03, 0.9304402695430626], [11000, 45000, 0.04, 0.9262661618687049], [11000, 45000, 0.05, 0.9078553229992619], [11000, 50000, 0.01, 0.9271614639140892], [11000, 50000, 0.02, 0.9254946781912994], [11000, 50000, 0.03, 0.8971164606995737], [11000, 50000, 0.04, 0.9281877276948354], [11000, 50000, 0.05, 0.9203252613281901]]
parameter_name = ["sample amount", "epoch", "learning rate", "correct rate"]

print(len(Result))

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
    

correct_rate_line_chart_drawing(Result, parameter_name)