import time
import sys
import csv
import socket
import multiprocessing
import pickle

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project')
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\IDS_Structure')
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
from Get_Training_Result import Ensemble_Learning_Training


def initialize_log_file(features_list, training_data_set):
    with open('IDS_traffic_log.txt', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(features_list)
        writer.writerows(training_data_set)


def read_log_file():
    def convert_to_original_type(value):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    with open('IDS_traffic_log.txt', 'r', newline='') as file:
        reader = csv.reader(file)
        new_features_list = next(reader)
        new_training_data_set = [list(map(convert_to_original_type, row)) for row in reader]
    return new_features_list, new_training_data_set


def Ensemble_Learning_Decision(a, b):
    return 1

def decision_process(proc_id, shared_data):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_address = ("127.0.0.1", 8081 + proc_id)
    server_socket.bind(server_address)
    server_socket.listen(5)

    print(f"Decision process {proc_id} is listening on port {8081 + proc_id}")

    while True:
        conn, client_address = server_socket.accept()
        try:
            data = conn.recv(4096)
            print(f"Connection from {client_address} in process {proc_id}")
            user_traffic = pickle.loads(data)
            scale_weight = list(shared_data["scale_weight"])
            sample_decision = Ensemble_Learning_Decision(user_traffic[1], scale_weight)
            shared_data['sample_buffer'].append(sample_decision)
        finally:
            conn.close()

'''

def start_decision_processes(num_processes, shared_data):
    processes = []
    for i in range(num_processes):
        proc = multiprocessing.Process(target=decision_process, args=(i, shared_data))
        proc.start()
        processes.append(proc)
    return processes
'''


def load_adjustment_process():
    '''
    global buffer_limit
    while True:
        # 检查系统负载并调整 buffer_limit
        # 示例代码，假设负载通过一些API获取
        system_load = get_system_load()  # 伪代码，需替换为实际获取负载的方法
        if system_load > 0.8:
            buffer_limit = 10000
        elif system_load < 0.2:
            buffer_limit = 20000
        else:
            buffer_limit = 15000
        time.sleep(100)
    '''

def parameter_update_process():
    while True:
        if len(shared_data['sample_buffer']) >= buffer_limit:
            shared_data['sample_to_be_written'].extend(shared_data['sample_buffer'])
            shared_data['sample_buffer'].clear()

            with open('IDS_traffic_log.txt', 'a') as file:
                for sample in shared_data['sample_to_be_written']:
                    file.write(','.join(map(str, sample)) + '\n')

            shared_data['sample_to_be_written'].clear()
            new_features_list, new_training_data_set = read_log_file()
            shared_data['Features_List'][:] = new_features_list
            shared_data['Training_Data_Set'][:] = new_training_data_set

            shared_data['scale_weight'][:] = Ensemble_Lrarning_Training(shared_data['Features_List'], shared_data['Training_Data_Set'])
        time.sleep(10)


def testcase(param):
    List = list(param['Features_List'])
    print("test case", List)

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    shared_data = manager.dict({
        'sample_buffer': manager.list(),
        'sample_to_be_written': manager.list(),
        'scale_weight': manager.list(),
        'Features_List': manager.list(),
        'Training_Data_Set': manager.list(),
        'Testing_Data_Set': manager.list()
    })

    Features_List, Training_Data_Set, Testing_Data_Set = Training_set_create()
    print("Training data set has sample amount: ", len(Training_Data_Set))
    print("Testing data set sample amount: ", len(Testing_Data_Set))
    scale_weight = []

    shared_data['Features_List'][:] = Features_List
    shared_data['Training_Data_Set'][:] = Training_Data_Set
    shared_data['Testing_Data_Set'][:] = Testing_Data_Set
    shared_data['scale_weight'][:] = scale_weight

    Testing_Data_Sender = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
    Testing_Data_Sender.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    Testing_Data_Sender.bind(("127.0.0.1", 8081))
    Testing_Data_Sender.listen(2)

    print("Data pre-load successful, currently sending testing set to client script")

    Client_Main_Socket, Client_Main_Socket_address = Testing_Data_Sender.accept()
    Shake_Hand_message = Client_Main_Socket.recv(1024)
    print(Shake_Hand_message)

    normal_Features_List = list(shared_data['Features_List'])
    normal_Testing_Data_Set = list(shared_data['Testing_Data_Set'])
    Testing_Data_Packet = pickle.dumps([normal_Features_List, normal_Testing_Data_Set])

    chunk_size = 40960

    for i in range(0, len(Testing_Data_Packet), chunk_size):
        Client_Main_Socket.send(Testing_Data_Packet[i:i + chunk_size])
        Transfer_progress = i / len(Testing_Data_Packet) * 100
        print("current transfer progress is: ", Transfer_progress)
    print("Testing data set transmission done")
    Client_Main_Socket.close()
    Testing_Data_Sender.close()

    num_processes = 10  # 可以调整的参数

    processes = []

    for i in range(0,num_processes):
        proc = multiprocessing.Process(target=decision_process, args=(i, shared_data))
        processes.append(proc)



    # 启动负载调整进程和参数更新进程
    parameter_update = multiprocessing.Process(target=parameter_update_process, args=(shared_data,))
    processes.append(parameter_update)

    for i in range(0,len(processes)):
        processes[i].start()

    for proc in processes:
        proc.join()




    # 把start_decision_processes删掉，直接for循环启动decision_process，避免manager对象被嵌套传输






