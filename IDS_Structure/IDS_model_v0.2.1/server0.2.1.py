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
from Get_Training_Result import Ensemble_Learning_Decision


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


def decision_process(proc_id, share_data, lock):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_address = ("127.0.0.1", 8081 + proc_id)
    server_socket.bind(server_address)
    server_socket.listen(5)

    print(f"Decision process {proc_id} is listening on port {8081 + proc_id}")

    while True:
        conn, client_address = server_socket.accept()

        with lock:
            final_flag = share_data['final_analysis'].value
        if final_flag > 3:
            sys.exit()

        try:
            data = conn.recv(4096)
            print(f"Connection from {client_address} in process {proc_id}")
            user_traffic = pickle.loads(data)

            with lock:
                print(f"Decision process {proc_id} try to access the value of scale weight")
                scale_weight = list(share_data["scale_weight"])
                features_list = list(share_data["Features_List"])
                print(f"Decision process {proc_id} access the value of scale weight successful")
                print(scale_weight)

            sample_decision = Ensemble_Learning_Decision(scale_weight, user_traffic[1], features_list)

            with lock:
                share_data['decision_record'].append([sample_decision, user_traffic[1][-1]])
                user_traffic[1][-1] = sample_decision
                share_data['sample_buffer'].append(user_traffic[1])
                print("代码确实成功执行过")
        except TypeError as e:
            continue
        finally:
            conn.close()


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


def parameter_update_process(share_data, lock):
    while True:
        final_flag = share_data['final_analysis'].value
        if final_flag > 3:
            print("周期性模拟结束")
            all_decision = list(share_data['decision_record'])
            count = 0
            for each_decision in all_decision:
                if each_decision[0] == each_decision[1]:
                    count = count + 1
            decision_times = len(all_decision)
            correct_rate = count/decision_times
            print("模型的最终正确率是", correct_rate)
            sys.exit()

        
        if len(share_data['sample_buffer']) >= share_data['buffer_limit'].value:
            print("len(share_data['sample_buffer'])的值是",len(share_data['sample_buffer']))
            print("share_data['buffer_limit'].value的值是",share_data['buffer_limit'].value)
            share_data['sample_to_be_written'].extend(share_data['sample_buffer'])
            share_data['sample_buffer'][:] = []

            with open('IDS_traffic_log.txt', 'a') as file:
                for sample in share_data['sample_to_be_written']:
                    file.write(','.join(map(str, sample)) + '\n')

            share_data['sample_to_be_written'][:] = []
            new_features_list, new_training_data_set = read_log_file()
            share_data['Features_List'][:] = new_features_list
            share_data['Training_Data_Set'][:] = new_training_data_set

            share_data['scale_weight'][:] = Ensemble_Learning_Training(list(share_data['Features_List']),
                                                                       list(share_data['Training_Data_Set']))
            print("成功在 parameter_update_process 中更新了一次权重参数")
            with lock:
                share_data['final_analysis'].value += 1

        time.sleep(10)


def testcase(param):
    List = list(param['Features_List'])
    print("test case", List)


if __name__ == '__main__':
    Lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    shared_data = manager.dict({
        'sample_buffer': manager.list(),
        'decision_record': manager.list(),
        'sample_to_be_written': manager.list(),
        'scale_weight': manager.list(),
        'Features_List': manager.list(),
        'Training_Data_Set': manager.list(),
        'Testing_Data_Set': manager.list(),
        'buffer_limit': manager.Value('i', 1000),
        'final_analysis': manager.Value('i', 0)
    })

    Features_List, Training_Data_Set, Testing_Data_Set = Training_set_create()
    print("Training data set has sample amount: ", len(Training_Data_Set))
    print("Testing data set sample amount: ", len(Testing_Data_Set))
    Scale_Weight = Ensemble_Learning_Training(Features_List, Training_Data_Set)
    Decision_Record = []

    shared_data['Features_List'][:] = Features_List
    shared_data['Training_Data_Set'][:] = Training_Data_Set
    shared_data['Testing_Data_Set'][:] = Testing_Data_Set
    shared_data['scale_weight'][:] = Scale_Weight
    shared_data['decision_record'][:] = Decision_Record
    '''
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
    '''

    num_processes = 10  # 可以调整的参数

    processes = []

    for i in range(0, num_processes):
        proc = multiprocessing.Process(target=decision_process, args=(i, shared_data, Lock))
        processes.append(proc)



    # 启动负载调整进程和参数更新进程
    parameter_update = multiprocessing.Process(target=parameter_update_process, args=(shared_data, Lock))
    processes.append(parameter_update)

    for i in range(0, len(processes)):
        processes[i].start()

    for proc in processes:
        proc.join()




    # 把start_decision_processes删掉，直接for循环启动decision_process，避免manager对象被嵌套传输





