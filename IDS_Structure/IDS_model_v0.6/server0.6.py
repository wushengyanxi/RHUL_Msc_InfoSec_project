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
from Ensemble_learning_model import Ensemble_Learning_Training
from Ensemble_learning_model import Ensemble_Learning_Decision
from normal_distribution_calculation import get_normal_distribution


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

def specific_client_traffic_log_update(specific_user_traffic_log, client_traffic, normal_distribution):

    if specific_user_traffic_log[1] >= 15:
        specific_user_traffic_log[1] = 0
        specific_user_traffic_log[2] = [0 for i in range(7, 86)]
        specific_user_traffic_log[3] = 0
        # 在这里，如果特定用户的流量日志中的特征超过了15次，那么就将这个特定用户的流量日志清零
        # 我们的系统找出攻击者的方式，是检查用户流量超过标准差或被判为恶性的次数
        # 这是为了防止某个长期连接的良性用户的凭据被披露后，攻击者依靠该用户长期积攒的“流量信用”展开攻击
        # 也为了避免极大值导致的潜在风险（鲁棒性方面）
        
    
    specific_user_traffic_log[1] += 1

    for specific_feature in range(7, 86):
        current_value = client_traffic[specific_feature]
        mean = normal_distribution[specific_feature-7][0]
        std = normal_distribution[specific_feature-7][1]
        upper_bound = mean + 2*std
        lower_bound = mean - 2*std
        if current_value > upper_bound or current_value < lower_bound:

            specific_user_traffic_log[2][specific_feature-7] += 1
            print(specific_user_traffic_log)

def find_client(client_ID, user_traffic_log):
    for i in range(0,len(user_traffic_log)):
        if user_traffic_log[i][0][0] == client_ID:
            return i
    return -1

def decision_process(proc_id, share_data, lock):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_address = ("127.0.0.1", 8081 + proc_id)
    server_socket.bind(server_address)
    server_socket.listen(10)
    count = 0

    print(f"Decision process {proc_id} is listening on port {8081 + proc_id}")

    while True:
        conn, client_address = server_socket.accept()

        with lock:
            final_flag = share_data['final_analysis'].value
        if final_flag > 3:
            sys.exit()

        try:
            data = conn.recv(4096)
            #print(f"Connection from {client_address} in process {proc_id}")
            user_traffic = pickle.loads(data)
            client_ID = user_traffic[0]
            client_traffic = user_traffic[1]

            with lock:
                attacker_list = list(share_data['attacker_list'])
                print(attacker_list)
            if len(attacker_list) != 0:    
                for i in attacker_list:
                    if i[0] == client_ID: 
                        #try:
                            #conn.send("You are an attacker, please stop the connection".encode("utf-8"))
                        #except OSError as e:
                            #pass
                        #finally:
                            #conn.close()
                        #continue
                        conn.close()
                        break
                    else:
                        continue
            
            while True:
                try:
                    with lock:
                        client_list = list(share_data['client_list'])
                        if client_ID not in client_list:
                            share_data['client_list'].append(client_ID)
                            zero_list = [0 for i in range(7, 86)]
                            share_data['user_traffic_log'].append([[client_ID,time.time()],0,zero_list,0])
                            # The four elements in specific client log, which is one of the element in user traffic log, are 
                            # [client_ID, time when first time connect to server]
                            # total connection times
                            # In index[7-86], the number of times each feature exceeds three standard deviations from the mean
                            # Number of times diagnosed as malicious
                    break
                except OSError as e:
                    time.sleep(0.1)
            
            while True:
                try:
                    with lock:
                        user_traffic_log = list(share_data['user_traffic_log'])
                    break
                except OSError as e:
                    time.sleep(0.1)    

            while True:
                try:
                    with lock:
                        normal_distribution_param = list(share_data['normal_distribution_param'])
                    break
                except OSError as e:
                    time.sleep(0.1)
            
            
            client_index = find_client(client_ID, user_traffic_log)
            if client_index != -1:
                specific_client_log = user_traffic_log[client_index] # 获取用户的流量日志模板
                specific_client_traffic_log_update(specific_client_log, client_traffic, normal_distribution_param)
            
            while True:    
                try:
                    with lock:
                        user_traffic_log = list(share_data['user_traffic_log'])
                        if client_ID == user_traffic_log[client_index][0][0]:
                            user_traffic_log[client_index] = specific_client_log
                            share_data['user_traffic_log'][:] = user_traffic_log
                            print("更新了特定用户的流量日志")
                    break
                except OSError as e:
                    time.sleep(0.1)

            with lock:
                scale_weight = list(share_data["scale_weight"])
                features_list = list(share_data["Features_List"])

            sample_decision, reliable = Ensemble_Learning_Decision(scale_weight, client_traffic, features_list)
            count += 1
            specific_client_log[3] += sample_decision
            most_unusual_count = max(specific_client_log[2])
            traffic_amount_of_this_client = specific_client_log[1]
            if traffic_amount_of_this_client > 10:
                unusual_percentage = most_unusual_count/traffic_amount_of_this_client
                malicious_percentage = specific_client_log[3]/traffic_amount_of_this_client
                if unusual_percentage > 0.5 or malicious_percentage > 0.7:
                    with lock:
                        time_now = time.time()
                        start_time = share_data['start_time'].value
                        time_interval = time_now - start_time
                        # share_data['attacker_list'].append([client_ID,time_interval])
                        attacker_list = list(share_data['attacker_list'])
                        in_list = False
                        for i in attacker_list:
                            if i[0] == client_ID:
                                in_list = True
                        if in_list == False:
                            share_data['attacker_list'].append([client_ID,time_interval])                            
                        # record clieng_ID and the time when the client is diagnosed as an attacker into attack list
            
            while True:
                try:
                    with lock:
                        user_traffic_log = list(share_data['user_traffic_log'])
                        user_traffic_log[client_index] = specific_client_log
                        share_data['user_traffic_log'][:] = user_traffic_log
                    break
                except OSError as e:
                    time.sleep(0.1)
                

            #conn.send(str(sample_decision).encode("utf-8"))
            #message = "decision_process" + str(proc_id) + " has received the data and made a decision: " + str(count)
            #try:
                #conn.send(message.encode("utf-8"))
            #except OSError as e:
                #print("Error sending message to client")

            # send the decision back to client to test the correctness of the decision

            with lock:
                share_data['decision_record'].append([sample_decision, client_traffic[-1]])
                client_traffic[-1] = sample_decision
                if reliable == 1:
                    share_data['sample_buffer'].append(client_traffic)
                print("END decision process")
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
            print("periodic simulation has beed ended")
            all_decision = list(share_data['decision_record'])
            count = 0
            for each_decision in all_decision:
                if each_decision[0] == each_decision[1]:
                    count = count + 1
            decision_times = len(all_decision)
            correct_rate = count/decision_times
            print("system's final correct rate is", correct_rate)
            attacker_list = list(share_data['attacker_list'])
            print(attacker_list)
            sys.exit()

        '''
        if len(share_data['sample_buffer']) >= share_data['buffer_limit'].value:
            #print("len(share_data['sample_buffer'])的值是",len(share_data['sample_buffer']))
            #print("share_data['buffer_limit'].value的值是",share_data['buffer_limit'].value)
            share_data['sample_to_be_written'].extend(share_data['sample_buffer'])
            share_data['sample_buffer'][:] = []

            with open('IDS_traffic_log.txt', 'a') as file:
                for sample in share_data['sample_to_be_written']:
                    file.write(','.join(map(str, sample)) + '\n')

            share_data['sample_to_be_written'][:] = []
            new_features_list, new_training_data_set = read_log_file()
            # share_data['Features_List'][:] = new_features_list
            share_data['Training_Data_Set'][:] = new_training_data_set

            share_data['scale_weight'][:] = Ensemble_Learning_Training(list(share_data['Features_List']),
                                                                       list(share_data['Training_Data_Set']))
            print("成功在 parameter_update_process 中更新了一次权重参数")
            with lock:
                share_data['final_analysis'].value += 1
        '''
        if len(share_data['sample_buffer']) >= share_data['buffer_limit'].value:
            #share_data['sample_to_be_written'].extend(share_data['sample_buffer'])
            sample_buffer = list(share_data['sample_buffer'])
            share_data['sample_buffer'][:] = []
            original_training_set = list(share_data['Training_Data_Set'])
            new_training_set = original_training_set + sample_buffer
            share_data['Training_Data_Set'][:] = new_training_set

            share_data['scale_weight'][:] = Ensemble_Learning_Training(list(share_data['Features_List']),
                                                                       list(share_data['Training_Data_Set']))
            print("成功在 parameter_update_process 中更新了一次权重参数")
            with lock:
                share_data['final_analysis'].value += 1

        time.sleep(10)


def send_test_set(share_data):
    Testing_Data_Sender = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
    Testing_Data_Sender.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    Testing_Data_Sender.bind(("127.0.0.1", 8081))
    Testing_Data_Sender.listen(2)

    print("Data pre-load successful, currently sending testing set to client script")

    Client_Main_Socket, Client_Main_Socket_address = Testing_Data_Sender.accept()
    Shake_Hand_message = Client_Main_Socket.recv(1024)
    print(Client_Main_Socket_address, Shake_Hand_message)

    normal_Features_List = list(share_data['Features_List'])
    normal_Testing_Data_Set = list(share_data['Testing_Data_Set'])
    Testing_Data_Packet = pickle.dumps([normal_Features_List, normal_Testing_Data_Set])

    chunk_size = 40960

    for i in range(0, len(Testing_Data_Packet), chunk_size):
        Client_Main_Socket.send(Testing_Data_Packet[i:i + chunk_size])
        Transfer_progress = i / len(Testing_Data_Packet) * 100
        print("current transfer progress is: ", Transfer_progress)
    print("Testing data set transmission done")
    Client_Main_Socket.close()
    Testing_Data_Sender.close()


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
        'final_analysis': manager.Value('i', 0),
        'normal_distribution_param': manager.list(),
        'user_traffic_log': manager.list(),
        'client_list': manager.list(),
        'attacker_list': manager.list(),
        'start_time': manager.Value('d', time.time())
    })

    Features_List, Training_Data_Set, Testing_Data_Set = Training_set_create()
    print("Training data set has sample amount: ", len(Training_Data_Set))
    print("Testing data set sample amount: ", len(Testing_Data_Set))
    Scale_Weight = Ensemble_Learning_Training(Features_List, Training_Data_Set)
    Features_List, Benign_Training_Data_Set, No_Need_Testing_Data_Set = Training_set_create(1500,1500,0,0,0,0) # new
    Decision_Record = []
    normal_distribution_param = get_normal_distribution(Benign_Training_Data_Set) # new
    User_Traffic_Log = []
    Client_List = []
    attacker_list = []

    shared_data['Features_List'][:] = Features_List
    shared_data['Training_Data_Set'][:] = Training_Data_Set
    shared_data['Testing_Data_Set'][:] = Testing_Data_Set
    shared_data['scale_weight'][:] = Scale_Weight
    shared_data['decision_record'][:] = Decision_Record
    shared_data['normal_distribution_param'][:] = normal_distribution_param
    shared_data['user_traffic_log'][:] = User_Traffic_Log
    shared_data['client_list'][:] = Client_List
    shared_data['attacker_list'][:] = attacker_list
    
    
    
    
    
    # send_test_set(shared_data)

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





