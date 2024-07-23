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

Result = [[2000, 5000, 0.01, 0.9359259900087984], [2000, 5000, 0.02, 0.7678904722812262], [2000, 5000, 0.03, 0.9153195969650897], [2000, 5000, 0.04, 0.9040620087689912],
          [2000, 5000, 0.05, 0.897967523072902], [2000, 10000, 0.01, 0.909368684172744], [2000, 10000, 0.02, 0.9219791708848877], [2000, 10000, 0.03, 0.9082992501076796],
          [2000, 10000, 0.04, 0.860303564657505], [2000, 10000, 0.05, 0.8940781699240536], [2000, 15000, 0.01, 0.9104454809508208], [2000, 15000, 0.02, 0.8549895265407398],
          [2000, 15000, 0.03, 0.9268404021513847], [2000, 15000, 0.04, 0.8589451441067004], [2000, 15000, 0.05, 0.8526978821156019], [2000, 20000, 0.01, 0.9149091257146433],
          [2000, 20000, 0.02, 0.8604673850220329], [2000, 20000, 0.03, 0.8580468931191766], [2000, 20000, 0.04, 0.8451603046690644], [2000, 20000, 0.05, 0.8217726467848873],
          [2000, 25000, 0.01, 0.8781470996432765], [2000, 25000, 0.02, 0.8630406532198984], [2000, 25000, 0.03, 0.8280751291235795], [2000, 25000, 0.04, 0.8280051833499608],
          [2000, 25000, 0.05, 0.7829012034354419], [2000, 30000, 0.01, 0.8987976689650602], [2000, 30000, 0.02, 0.8354525675620953], [2000, 30000, 0.03, 0.8216732501592187],
          [2000, 30000, 0.04, 0.8151995847429861], [2000, 30000, 0.05, 0.807833190373989], [2000, 35000, 0.01, 0.8563866013348599], [2000, 35000, 0.02, 0.845243135190455],
          [2000, 35000, 0.03, 0.8255552405950545], [2000, 35000, 0.04, 0.8071907936636492], [2000, 35000, 0.05, 0.8097585398267554], [2000, 40000, 0.01, 0.8570971031405652],
          [2000, 40000, 0.02, 0.8331517197456919], [2000, 40000, 0.03, 0.8229948571449608], [2000, 40000, 0.04, 0.7961614495709379], [2000, 40000, 0.05, 0.7840000883525562],
          [2000, 45000, 0.01, 0.8966091025220974], [2000, 45000, 0.02, 0.8243403929479934], [2000, 45000, 0.03, 0.8024050302055301], [2000, 45000, 0.04, 0.8011680944194317],
          [2000, 45000, 0.05, 0.7910885403053317], [2000, 50000, 0.01, 0.904133795220863], [2000, 50000, 0.02, 0.8349887166423083], [2000, 50000, 0.03, 0.7960823004060537],
          [2000, 50000, 0.04, 0.7926402320727142], [2000, 50000, 0.05, 0.7903044113695015], [3000, 5000, 0.01, 0.9413376315427023], [3000, 5000, 0.02, 0.9320500746354774],
          [3000, 5000, 0.03, 0.9186826931309303], [3000, 5000, 0.04, 0.9103853126314496], [3000, 5000, 0.05, 0.9001634163319547], [3000, 10000, 0.01, 0.9388882477972298],
          [3000, 10000, 0.02, 0.9331723986465107], [3000, 10000, 0.03, 0.9009674693547847], [3000, 10000, 0.04, 0.908285840849616], [3000, 10000, 0.05, 0.9114908855378407],
          [3000, 15000, 0.01, 0.9120455332248855], [3000, 15000, 0.02, 0.9200748960500895], [3000, 15000, 0.03, 0.9064599704436065], [3000, 15000, 0.04, 0.8952441752686691],
          [3000, 15000, 0.05, 0.9016654320482134], [3000, 20000, 0.01, 0.9224684427800878], [3000, 20000, 0.02, 0.8819456594165405], [3000, 20000, 0.03, 0.9000684934056484],
          [3000, 20000, 0.04, 0.8990615658932619], [3000, 20000, 0.05, 0.8895934693026701], [3000, 25000, 0.01, 0.9082951470188617], [3000, 25000, 0.02, 0.8892379736374838],
          [3000, 25000, 0.03, 0.8904831390825606], [3000, 25000, 0.04, 0.8571763593521418], [3000, 25000, 0.05, 0.8907679078614795], [3000, 30000, 0.01, 0.9047439128346964],
          [3000, 30000, 0.02, 0.899256995447422], [3000, 30000, 0.03, 0.8996627444265353], [3000, 30000, 0.04, 0.8782287754198013], [3000, 30000, 0.05, 0.8428597485845316],
          [3000, 35000, 0.01, 0.9077460830333645], [3000, 35000, 0.02, 0.9052780869494005], [3000, 35000, 0.03, 0.8797922118530817], [3000, 35000, 0.04, 0.8856904619210166],
          [3000, 35000, 0.05, 0.9051087146691285], [3000, 40000, 0.01, 0.8579804123749716], [3000, 40000, 0.02, 0.88907232382491], [3000, 40000, 0.03, 0.9022163572675598],
          [3000, 40000, 0.04, 0.8724328932135691], [3000, 40000, 0.05, 0.8672419120083086], [3000, 45000, 0.01, 0.8698253045909194], [3000, 45000, 0.03, 0.8856067063978053],
          [3000, 45000, 0.02, 0.902674220794449], [3000, 45000, 0.04, 0.8975669951123999], [3000, 45000, 0.05, 0.8746663738325411], [3000, 50000, 0.01, 0.9097171296796072],
          [3000, 50000, 0.02, 0.8800397559550177], [3000, 50000, 0.03, 0.8906674012336258], [3000, 50000, 0.04, 0.8909130841017127], [3000, 50000, 0.05, 0.9028007846961909],
          [5000, 5000, 0.01, 0.9456241852451333], [5000, 5000, 0.02, 0.950106167184378], [5000, 5000, 0.03, 0.9501441179205273], [5000, 5000, 0.04, 0.9324097389179107],
          [5000, 5000, 0.05, 0.9265311698883679], [5000, 10000, 0.01, 0.9539543718299276], [5000, 10000, 0.02, 0.9107721267023277], [5000, 10000, 0.03, 0.9467095762990062],
          [5000, 10000, 0.04, 0.916071947005592], [5000, 10000, 0.05, 0.8979049296108721], [5000, 15000, 0.01, 0.9364609800018596], [5000, 15000, 0.02, 0.8815272894255967],
          [5000, 15000, 0.03, 0.9257247167451931], [5000, 15000, 0.04, 0.9091250647534436], [5000, 15000, 0.05, 0.8938214304011962], [5000, 20000, 0.01, 0.9491118578972636],
          [5000, 20000, 0.02, 0.9221421672526893], [5000, 20000, 0.03, 0.9193338127776334], [5000, 20000, 0.04, 0.9213869476033162], [5000, 20000, 0.05, 0.8890567154776384],
          [5000, 25000, 0.01, 0.9093736420752222], [5000, 25000, 0.02, 0.9358442805394318], [5000, 25000, 0.03, 0.9173527843506344], [5000, 25000, 0.04, 0.8746771815506291],
          [5000, 25000, 0.05, 0.9187626541985848], [5000, 30000, 0.01, 0.9328537625308587], [5000, 30000, 0.02, 0.9262332566095951], [5000, 30000, 0.03, 0.9200776472061617],
          [5000, 30000, 0.04, 0.8874172436759842], [5000, 30000, 0.05, 0.8871648712805906], [5000, 35000, 0.01, 0.9311023360575637], [5000, 35000, 0.02, 0.9225368549086431],
          [5000, 35000, 0.03, 0.8939599505881415], [5000, 35000, 0.04, 0.884728434019799], [5000, 35000, 0.05, 0.8895405873635434], [5000, 40000, 0.01, 0.9487740963455339],
          [5000, 40000, 0.02, 0.9217531722071579], [5000, 40000, 0.03, 0.915946709576299], [5000, 40000, 0.04, 0.8801288047984911], [5000, 40000, 0.05, 0.8937436313920899],
          [5000, 45000, 0.01, 0.8993470575845495], [5000, 45000, 0.02, 0.8912293951221919], [5000, 45000, 0.03, 0.8869087038115822], [5000, 45000, 0.04, 0.9214609515388075],
          [5000, 45000, 0.05, 0.8854324201753704], [5000, 50000, 0.01, 0.9207626579936584], [5000, 50000, 0.02, 0.8814741583949874], [5000, 50000, 0.03, 0.9206336254907505], 
          [5000, 50000, 0.04, 0.8889295805115379], [5000, 50000, 0.05, 0.9184021222051655]]

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