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
    """
    Updates the traffic log for a specific user based on their traffic patterns and predefined thresholds.

    This function performs two main tasks:
    
    1. **Resetting the Traffic Log**: 
       If the counter in `specific_user_traffic_log[1]` exceeds or equals 15, the function resets the traffic log for the specific user. 
       This involves:
       - Resetting the counter to 0.
       - Clearing the list of feature anomalies by setting `specific_user_traffic_log[2]` to a list of zeros.
       - Resetting the anomaly count in `specific_user_traffic_log[3]` to 0.

       This mechanism prevents an attacker from exploiting the accumulated "traffic credit" of a benign user 
       whose credentials may have been compromised. It also helps mitigate risks due to extreme values and 
       enhances the robustness of the system.

    2. **Updating Feature Anomalies**:
       The function increments the counter in `specific_user_traffic_log[1]` and checks each feature in 
       the `client_traffic` data against its expected normal distribution (defined by `mean` and `std`). 
       If a feature value exceeds two standard deviations from the mean (either above or below), 
       it increments the corresponding counter in `specific_user_traffic_log[2]`.

    Args:
        specific_user_traffic_log (list): A list tracking the traffic log of a specific user, containing:
            - `specific_user_traffic_log[1]` (int): A counter for the number of traffic events.
            - `specific_user_traffic_log[2]` (list): A list tracking the number of anomalies detected in specific features.
            - `specific_user_traffic_log[3]` (int): A counter for the total number of anomalies.
        client_traffic (list): A list representing the current traffic data for the specific user.
        normal_distribution (list): A list of tuples where each tuple contains the mean and standard deviation 
                                    for the corresponding feature in `client_traffic`.

    Note:
        The features considered start from index 7 up to 85 in `client_traffic`, corresponding to the features 
        in the range 7 to 85 in the `specific_user_traffic_log[2]`.
    """
    if specific_user_traffic_log[1] >= 15:
        specific_user_traffic_log[1] = 0
        specific_user_traffic_log[2] = [0 for i in range(7, 86)]
        specific_user_traffic_log[3] = 0

    
    specific_user_traffic_log[1] += 1

    for specific_feature in range(7, 86):
        current_value = client_traffic[specific_feature]
        mean = normal_distribution[specific_feature-7][0]
        std = normal_distribution[specific_feature-7][1]
        upper_bound = mean + 2*std
        lower_bound = mean - 2*std
        if current_value > upper_bound or current_value < lower_bound:

            specific_user_traffic_log[2][specific_feature-7] += 1
            #print(specific_user_traffic_log)

def find_client(client_ID, user_traffic_log):
    """
    Finds the index of a client in the user traffic log.

    Args:
        client_ID (str): The ID of the client to search for.
        user_traffic_log (list): A list of user traffic logs.

    Returns:
        int: The index of the client if found, otherwise -1.
    """
    for i in range(0,len(user_traffic_log)):
        if user_traffic_log[i][0][0] == client_ID:
            return i
    return -1

def decision_process(proc_id, share_data, lock):
    """
    Handles client connections and updates traffic logs for anomaly detection and decision making.

    This function creates a server socket to listen for incoming client connections on a specified port. 
    It processes client traffic data, updates user traffic logs, and makes decisions based on the traffic data 
    using an ensemble learning model. The function also identifies potential attackers by analyzing the 
    frequency of anomalous behavior in the client's traffic.

    Args:
        proc_id (int): The process ID used to differentiate between different decision processes.
        share_data (multiprocessing.Manager.dict): A shared dictionary containing the following keys:
            - 'final_analysis': A `Value` object indicating whether the process should exit.
            - 'attacker_list': A list of clients identified as attackers.
            - 'client_list': A list of connected client IDs.
            - 'user_traffic_log': A list of logs for each user's traffic.
            - 'normal_distribution_param': A list of tuples containing mean and standard deviation for each feature.
            - 'scale_weight': A list of weights for the ensemble learning decision.
            - 'Features_List': A list of features used for the decision-making process.
            - 'start_time': A `Value` object storing the time when the system started.
            - 'decision_record': A list of decisions made by the ensemble model.
            - 'sample_buffer': A list of traffic samples classified as reliable by the ensemble model.
        lock (threading.Lock): A lock to ensure thread-safe access to the shared data.

    Workflow:
        - The server listens for client connections on a port determined by `proc_id`.
        - When a client connects, it checks whether the client has already been identified as an attacker.
        - If not, it updates the client's traffic log and checks for anomalies.
        - The ensemble learning model is used to make a decision based on the client's traffic.
        - If the client exhibits a high level of anomalous or malicious behavior, they are added to the attacker list.
        - Traffic logs and decisions are updated in the shared data structure.

    Notes:
        - The function exits if the `final_analysis` flag exceeds a threshold value.
        - It handles potential issues with shared data access using a lock and retries if an `OSError` occurs.

    """
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
                specific_client_log = user_traffic_log[client_index] # Get the user's traffic log template
                specific_client_traffic_log_update(specific_client_log, client_traffic, normal_distribution_param)
            
            while True:    
                try:
                    with lock:
                        user_traffic_log = list(share_data['user_traffic_log'])
                        if client_ID == user_traffic_log[client_index][0][0]:
                            user_traffic_log[client_index] = specific_client_log
                            share_data['user_traffic_log'][:] = user_traffic_log
                            # print("Updated traffic logs for specific users")
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


def parameter_update_process(share_data, lock):
    """
    Periodically updates the parameters of the system based on incoming data samples.

    This function continuously monitors the `final_analysis` flag and the `sample_buffer` in the 
    shared data structure. If the `final_analysis` flag exceeds a threshold, it calculates and prints 
    the system's final accuracy and exits the process. If the number of samples in `sample_buffer` 
    exceeds the buffer limit, the samples are added to the training set, and the model's weights are 
    updated using ensemble learning.

    Args:
        share_data (multiprocessing.Manager.dict): A shared dictionary containing the following keys:
            - 'final_analysis': A `Value` object indicating the completion of the simulation.
            - 'decision_record': A list of decisions made by the system.
            - 'attacker_list': A list of identified attackers.
            - 'sample_buffer': A list of new samples awaiting addition to the training set.
            - 'buffer_limit': A `Value` object specifying the limit of samples in the buffer.
            - 'Training_Data_Set': The current training data set used for model training.
            - 'Features_List': A list of features used for training the model.
            - 'scale_weight': A list of weights used in the ensemble learning model.
        lock (threading.Lock): A lock to ensure thread-safe access to the shared data.

    Workflow:
        - The process checks if the `final_analysis` flag exceeds 3. If so, it calculates the system's 
          correct rate, prints the final results, and exits.
        - If the `sample_buffer` exceeds the `buffer_limit`, the samples are added to the training set, 
          and the model's weights are updated. The `final_analysis` flag is then incremented.
        - The function waits for 10 seconds between each check.

    Notes:
        - The function assumes the `Ensemble_Learning_Training` function is defined elsewhere and is responsible 
          for training the model and returning updated weights.
        - The function uses a lock to ensure safe access to shared data during updates.

    """
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

        if len(share_data['sample_buffer']) >= share_data['buffer_limit'].value:
            #share_data['sample_to_be_written'].extend(share_data['sample_buffer'])
            sample_buffer = list(share_data['sample_buffer'])
            share_data['sample_buffer'][:] = []
            original_training_set = list(share_data['Training_Data_Set'])
            new_training_set = original_training_set + sample_buffer
            share_data['Training_Data_Set'][:] = new_training_set

            share_data['scale_weight'][:] = Ensemble_Learning_Training(list(share_data['Features_List']),
                                                                       list(share_data['Training_Data_Set']))
            print("Successfully updated the weight parameter once in parameter_update_process")
            with lock:
                share_data['final_analysis'].value += 1

        time.sleep(10)


def send_test_set(share_data):
    """
    Sends a test set to a client over a socket connection.

    This function sets up a server socket that listens for incoming connections on "127.0.0.1" at port 8081.
    Once a connection is established, it receives a handshake message from the client, prepares the test set
    data, and sends it to the client in chunks. The function tracks and prints the progress of the data transfer.
    After the entire test set is sent, the connection is closed.

    Args:
        share_data (dict): A dictionary containing the data to be sent. It should include the following keys:
            - 'Features_List' (list): A list of feature names or indices to be sent.
            - 'Testing_Data_Set' (list): A list of test data samples to be sent.

    Note:
        The function assumes the client will connect to "127.0.0.1" at port 8081 and that the data will be 
        transmitted in chunks of 40960 bytes. The data is serialized using `pickle` before sending.
    """
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
    Features_List, Benign_Training_Data_Set, No_Need_Testing_Data_Set = Training_set_create(1500,1500,0,0,0,0)
    Decision_Record = []
    normal_distribution_param = get_normal_distribution(Benign_Training_Data_Set)
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

    num_processes = 10  # Adjustable parameters

    processes = []

    for i in range(0, num_processes):
        proc = multiprocessing.Process(target=decision_process, args=(i, shared_data, Lock))
        processes.append(proc)



    # Start the load adjustment process and parameter update process
    parameter_update = multiprocessing.Process(target=parameter_update_process, args=(shared_data, Lock))
    processes.append(parameter_update)

    for i in range(0, len(processes)):
        processes[i].start()

    for proc in processes:
        proc.join()






