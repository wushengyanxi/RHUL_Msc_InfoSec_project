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

Result = [[5000, 5000, 0.01, 0.9525767601077042], [5000, 5000, 0.02, 0.9112218429256982], [5000, 5000, 0.03, 0.9248879030130986], [5000, 5000, 0.04, 0.9358120224137048], [5000, 5000, 0.05, 0.9127607452765565], [5000, 10000, 0.01, 0.9377968459143187], [5000, 10000, 0.02, 0.9325027182214767], [5000, 10000, 0.03, 0.9294913273080214], [5000, 10000, 0.04, 0.9147892121237422], [5000, 10000, 0.05, 0.9289865825172344], [5000, 15000, 0.01, 0.9187588591249699], [5000, 15000, 0.02, 0.932254140899698], [5000, 15000, 0.03, 0.908582369226507], [5000, 15000, 0.04, 0.908206656938628], [5000, 15000, 0.05, 0.9044609192806817], [5000, 20000, 0.01, 0.9351497820678977], [5000, 20000, 0.02, 0.9327626807641001], [5000, 20000, 0.03, 0.9230036489632808], [5000, 20000, 0.04, 0.9119884477959161], [5000, 20000, 0.05, 0.8692236607659597], [5000, 25000, 0.01, 0.9144761185505096], [5000, 25000, 0.02, 0.8923887901115561], [5000, 25000, 0.03, 0.9202275526139518], [5000, 25000, 0.04, 0.9138897796770012], [5000, 25000, 0.05, 0.9186715724318263], [5000, 30000, 0.01, 0.9364894430539716], [5000, 30000, 0.02, 0.8969903168696715], [5000, 30000, 0.03, 0.908457131797214], [5000, 30000, 0.04, 0.9228784115339878], [5000, 30000, 0.05, 0.916398323336477], [5000, 35000, 0.01, 0.9335786215913123], [5000, 35000, 0.02, 0.9227228135157751], [5000, 35000, 0.03, 0.890580437534037], [5000, 35000, 0.04, 0.9097854075624432], [5000, 35000, 0.05, 0.8891402070971671], [5000, 40000, 0.01, 0.9496602460346224], [5000, 40000, 0.02, 0.9243831582223117], [5000, 40000, 0.03, 0.9095159573357824], [5000, 40000, 0.04, 0.90521234385644], [5000, 40000, 0.05, 0.8860624023954504], [5000, 45000, 0.01, 0.935037827396257], [5000, 45000, 0.02, 0.9025728701572489], [5000, 45000, 0.03, 0.8720452979986679], [5000, 45000, 0.04, 0.9128480319697001], [5000, 45000, 0.05, 0.8859940910703815], [5000, 50000, 0.01, 0.9228385632610309], [5000, 50000, 0.02, 0.8972996153692891], [5000, 50000, 0.03, 0.877775859157228], [5000, 50000, 0.04, 0.8960870893493157], [5000, 50000, 0.05, 0.8964305435114678], [8000, 5000, 0.01, 0.9566764028860475], [8000, 5000, 0.02, 0.8916494187283595], [8000, 5000, 0.03, 0.9542545989128963], [8000, 5000, 0.04, 0.9460065380969882], [8000, 5000, 0.05, 0.9416484515542488], [8000, 10000, 0.01, 0.9558736483741803], [8000, 10000, 0.02, 0.9306033232102443], [8000, 10000, 0.03, 0.9393620519565933], [8000, 10000, 0.04, 0.9356287598893552], [8000, 10000, 0.05, 0.9207768342457009], [8000, 15000, 0.01, 0.9524150337543764], [8000, 15000, 0.02, 0.9541327349749502], [8000, 15000, 0.03, 0.9419985685823162], [8000, 15000, 0.04, 0.931609184285355], [8000, 15000, 0.05, 0.9344526761707642], [8000, 20000, 0.01, 0.9188676325512118], [8000, 20000, 0.02, 0.9370891927964872], [8000, 20000, 0.03, 0.9351335667446854], [8000, 20000, 0.04, 0.9190610673733486], [8000, 20000, 0.05, 0.9183763081029847], [8000, 25000, 0.01, 0.9484902412132232], [8000, 25000, 0.02, 0.9383252413099407], [8000, 25000, 0.03, 0.91194460026694], [8000, 25000, 0.04, 0.9433796932123721], [8000, 25000, 0.05, 0.9263438884267946], [8000, 30000, 0.01, 0.9454223649341355], [8000, 30000, 0.02, 0.9310888446138074], [8000, 30000, 0.03, 0.9415091784823104], [8000, 30000, 0.04, 0.9296129369209045], [8000, 30000, 0.05, 0.9348550206008086], [8000, 35000, 0.01, 0.9448343230748399], [8000, 35000, 0.02, 0.9285413080062673], [8000, 35000, 0.03, 0.9059732673075808], [8000, 35000, 0.04, 0.939408476313906], [8000, 35000, 0.05, 0.9029595527786912], [8000, 40000, 0.01, 0.939002263187419], [8000, 40000, 0.02, 0.9147203899646015], [8000, 40000, 0.03, 0.9259376753003076], [8000, 40000, 0.04, 0.8832156604832002], [8000, 40000, 0.05, 0.8905661837243941], [8000, 45000, 0.01, 0.9317794069288353], [8000, 45000, 0.02, 0.9255740178346906], [8000, 45000, 0.03, 0.9159545041298335], [8000, 45000, 0.04, 0.8868444977464843], [8000, 45000, 0.05, 0.8927075072054471], [8000, 50000, 0.01, 0.9421726599222392], [8000, 50000, 0.02, 0.9051028106079656], [8000, 50000, 0.03, 0.9110644718262182], [8000, 50000, 0.04, 0.9124958895100296], [8000, 50000, 0.05, 0.8870514730061706], [11000, 5000, 0.01, 0.959271216804142], [11000, 5000, 0.02, 0.9357599858259346], [11000, 5000, 0.03, 0.9575919837785696], [11000, 5000, 0.04, 0.9424040789810422], [11000, 5000, 0.05, 0.94317971533752], [11000, 10000, 0.01, 0.9303777782152489], [11000, 10000, 0.02, 0.9323562415103254], [11000, 10000, 0.03, 0.9616453727582338], [11000, 10000, 0.04, 0.9120991397129752], [11000, 10000, 0.05, 0.9367502017835699], [11000, 15000, 0.01, 0.9545957438431404], [11000, 15000, 0.02, 0.936454908754454], [11000, 15000, 0.03, 0.9430911274287852], [11000, 15000, 0.04, 0.9312754690237612], [11000, 15000, 0.05, 0.8985963738016025], [11000, 20000, 0.01, 0.9643915979290115], [11000, 20000, 0.02, 0.932460578380613], [11000, 20000, 0.03, 0.9313601196921077], [11000, 20000, 0.04, 0.9399688958009331], [11000, 20000, 0.05, 0.9316101344567592], [11000, 25000, 0.01, 0.941955233576786], [11000, 25000, 0.02, 0.9377640411835345], [11000, 25000, 0.03, 0.8896332460578381], [11000, 25000, 0.04, 0.9328011496741934], [11000, 25000, 0.05, 0.9235427289013131], [11000, 30000, 0.01, 0.9347756757288816], [11000, 30000, 0.02, 0.9402326909069433], [11000, 30000, 0.03, 0.9272673583085616], [11000, 30000, 0.04, 0.9305254247298069], [11000, 30000, 0.05, 0.9234994192570427], [11000, 35000, 0.01, 0.9401736323011202], [11000, 35000, 0.02, 0.9069472606649999], [11000, 35000, 0.03, 0.8933145658208161], [11000, 35000, 0.04, 0.9087367364214423], [11000, 35000, 0.05, 0.909606866547237], [11000, 40000, 0.01, 0.9351221528830442], [11000, 40000, 0.02, 0.9233478355020965], [11000, 40000, 0.03, 0.9317695926924818], [11000, 40000, 0.04, 0.9341732779494852], [11000, 40000, 0.05, 0.9204086855522964], [11000, 45000, 0.01, 0.91653640963049], [11000, 45000, 0.02, 0.9340689410791976], [11000, 45000, 0.03, 0.9365966494084297], [11000, 45000, 0.04, 0.9199047187826053], [11000, 45000, 0.05, 0.9255093804752249], [11000, 50000, 0.01, 0.9379628718231392], [11000, 50000, 0.02, 0.9180798078626691], [11000, 50000, 0.03, 0.9253499222395023], [11000, 50000, 0.04, 0.9198338484556174], [11000, 50000, 0.05, 0.9284504990452193]]

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