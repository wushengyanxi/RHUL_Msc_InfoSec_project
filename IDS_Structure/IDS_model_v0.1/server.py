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

def decision_process(proc_id):
    global sample_buffer, scale_weight

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
            sample_decision = Ensemble_Learning_Decision(user_traffic[1], scale_weight)
            sample_buffer.append(sample_decision)
        finally:
            conn.close()


def start_decision_processes(num_processes):
    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=decision_process, args=(i,))
        p.start()
        processes.append(p)
    return processes


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
    global sample_buffer, sample_to_be_written, scale_weight
    while True:
        if len(sample_buffer) >= buffer_limit:
            sample_to_be_written.extend(sample_buffer)
            sample_buffer = []

            with open('IDS_traffic_log.txt', 'a') as file:
                for sample in sample_to_be_written:
                    file.write(','.join(map(str, sample)) + '\n')

            sample_to_be_written = []
            new_features_list, new_training_data_set = read_log_file()
            Features_List = new_features_list
            Training_Data_Set = new_training_data_set

            scale_weight = Ensemble_Lrarning_Training(Features_List, Training_Data_Set)
        time.sleep(10)


if __name__ == '__main__':
    global Features_List, Training_Data_Set, Testing_Data_Set, scale_weight

    scale_weight = []

    Features_List, Training_Data_Set, Testing_Data_Set = Training_set_create()
    print("Training data set has sample amount: ", len(Training_Data_Set))
    print("Testing data set sample amount: ", len(Testing_Data_Set))

    # initialize_log_file(Features_List, Training_Data_Set)
    # scale_weight = Ensemble_Learning_Training(Features_List, Training_Data_Set)

    Testing_Data_Sender = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
    Testing_Data_Sender.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    Testing_Data_Sender.bind(("127.0.0.1", 8081))
    Testing_Data_Sender.listen(2)

    print("Data pre-load successful, currently sending testing set to client script")

    Client_Main_Socket, Client_Main_Socket_address = Testing_Data_Sender.accept()
    Shake_Hand_message = Client_Main_Socket.recv(1024)
    print(Shake_Hand_message)

    Testing_Data_Packet = pickle.dumps([Features_List, Testing_Data_Set])

    chunk_size = 40960

    for i in range(0, len(Testing_Data_Packet), chunk_size):
        Client_Main_Socket.send(Testing_Data_Packet[i:i + chunk_size])
        Transfer_progress = i/len(Testing_Data_Packet)*100
        print("current transfer progress is: ",Transfer_progress)
    print("Testing data set transmission done")
    Client_Main_Socket.close()
    Testing_Data_Sender.close()

    # 上方代码仅用于在客户端与服务器初始化时传递testing set用

    num_processes = 10  # 可以调整的参数
    processes = start_decision_processes(num_processes)

    # 启动负载调整进程和参数更新进程
    # load_adjustment = multiprocessing.Process(target=load_adjustment_process)
    parameter_update = multiprocessing.Process(target=parameter_update_process)
    # load_adjustment.start()
    parameter_update.start()

    # processes.append(load_adjustment)
    processes.append(parameter_update)






