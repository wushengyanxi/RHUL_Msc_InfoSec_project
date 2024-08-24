import socket
import pickle
import threading
import random
import time
import sys

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create


def user_Strategy():
    """
    Generates two random integers within specified ranges.

    The function returns a tuple containing two random integers:
    - The first integer is between 5 and 10, inclusive.
    - The second integer is between 20 and 30, inclusive.

    Returns:
        tuple: A tuple containing two elements (time_low, time_high), where:
               - time_low is an integer between 5 and 10.
               - time_high is an integer between 20 and 30.
    """
    time_low = random.randint(5, 10)
    time_high = random.randint(20, 30)
    return time_low, time_high


def connect_to_server(client_socket, server_address):
    """
    Attempts to establish a connection to a server using a socket.

    This function attempts to connect a client socket to a specified server address.
    If the connection is successful, it returns True. If a socket error occurs during
    the attempt, the function catches the exception and returns False.

    Args:
        client_socket (socket.socket): The client socket object used for the connection.
        server_address (tuple): The server address (host, port) to connect to.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    try:
        client_socket.connect(server_address)
        return True
    except socket.error as e:
        return False


def standard_User_Script(time_low, time_high):
    """
    Simulates a benign client sending data to a server in a loop with randomized timing.

    This function continuously sends data samples labeled as "benign" from a global 
    dictionary `Sample_dict` to a server. The server's port number is randomly selected 
    within a specified range, and a connection is attempted. If the connection is successful,
    the client sends the data and then waits for a random amount of time before reconnecting
    and sending another sample.

    Args:
        time_low (int): The lower bound of the sleep time range (in seconds) between sending samples.
        time_high (int): The upper bound of the sleep time range (in seconds) between sending samples.

    Global Variables:
        Sample_dict (dict): A global dictionary containing data samples categorized under "Benign".

    Note:
        This function assumes that `Sample_dict` contains a list of benign samples under the key "Benign".
        The function creates a new client socket for each connection attempt.
    """
    global Sample_dict
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    client_fileno = client_socket.fileno()
    random_int = random.randint(1, 9999)
    client_ID = client_fileno + random_int
    client_ID = "benign" + str(client_ID)


    while True:
        
        sample_index = random.randint(0, len(Sample_dict["Benign"]))
        test_sample = Sample_dict["Benign"][sample_index]

        random_port = random.randint(1, 10)
        port_num = 8081 + random_port
        server_address = ("127.0.0.1", port_num)

        if connect_to_server(client_socket, server_address):
            message = [client_ID, test_sample]
            transfer_data_package = pickle.dumps(message)
            client_socket.send(transfer_data_package)

            client_socket.close()
            sleep_time = random.randint(time_low, time_high)
            time.sleep(sleep_time)
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def BruteForce_User_Script():
    """
    Simulates a brute force attack client sending data to a server in a loop with varying strategies.

    This function continuously sends data samples labeled as "Bruteforce" from a global 
    dictionary `Sample_dict` to a server. The server's port number is randomly selected 
    within a specified range, and a connection is attempted. If the connection is successful,
    the client sends the data and then waits for a time period determined by a randomly chosen
    attack strategy before reconnecting and sending another sample.

    Global Variables:
        Sample_dict (dict): A global dictionary containing data samples categorized under "Bruteforce".

    Note:
        This function assumes that `Sample_dict` contains a list of brute force attack samples under 
        the key "Bruteforce". The function uses a predefined attack strategy that varies the delay 
        between successive attacks.
    """
    global Sample_dict
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    client_fileno = client_socket.fileno()
    random_int = random.randint(1, 9999)
    client_ID = client_fileno + random_int
    client_ID = "Bruteforce" + str(client_ID)
    sleep_time = random.randint(0,2)
    attack_strategy = [["crazy",1],["normal",3],["latency",5]]

    while True:
        sample_index = random.randint(0, len(Sample_dict["Bruteforce"]))
        test_sample = Sample_dict["Bruteforce"][sample_index]

        random_port = random.randint(1, 10)
        port_num = 8081 + random_port
        server_address = ("127.0.0.1", port_num)

        if connect_to_server(client_socket, server_address):

            message = [client_ID, test_sample]
            transfer_data_package = pickle.dumps(message)
            client_socket.send(transfer_data_package)
            client_socket.close()
            time.sleep(attack_strategy[sleep_time][1])
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def Scanner_User_Script():
    """
    Simulates a network scanner client sending data to a server in a loop with varying strategies.

    This function continuously sends data samples labeled as "Bruteforce_XML" from a global 
    dictionary `Sample_dict` to a server. The server's port number is randomly selected 
    within a specified range, and a connection is attempted. If the connection is successful,
    the client sends the data and then waits for a time period determined by a randomly chosen
    attack strategy before reconnecting and sending another sample.

    Global Variables:
        Sample_dict (dict): A global dictionary containing data samples categorized under "Bruteforce_XML".

    Note:
        This function assumes that `Sample_dict` contains a list of XML-formatted brute force attack 
        samples under the key "Bruteforce_XML". The function uses a predefined attack strategy that varies 
        the delay between successive attacks.
    """
    global Sample_dict
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    client_fileno = client_socket.fileno()
    random_int = random.randint(1, 9999)
    client_ID = client_fileno + random_int
    client_ID = "Scanner" + str(client_ID)
    sleep_time = random.randint(0,2)
    attack_strategy = [["crazy",1],["normal",3],["latency",5]]
    message_amount = 0


    while True:
        sample_index = random.randint(0, len(Sample_dict["Bruteforce_XML"]))
        test_sample = Sample_dict["Bruteforce"][sample_index]

        random_port = random.randint(1, 10)
        port_num = 8081 + random_port
        server_address = ("127.0.0.1", port_num)

        if connect_to_server(client_socket, server_address):

            message = [client_ID, test_sample]
            transfer_data_package = pickle.dumps(message)
            client_socket.send(transfer_data_package)
            message_amount += 1
            client_socket.close()
            time.sleep(attack_strategy[sleep_time][1])
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

def Causative_Attacker_User_Script():
    """
    Simulates a causative attacker client sending modified data to a server in a loop with varying strategies.

    This function continuously sends modified data samples labeled as "Bruteforce_XML" from a global 
    dictionary `Sample_dict` to a server. The server's port number is randomly selected 
    within a specified range, and a connection is attempted. If the connection is successful,
    the client modifies specific fields of the sample before sending it. The function then waits 
    for a time period determined by a randomly chosen attack strategy before reconnecting and sending 
    another modified sample.

    Global Variables:
        Sample_dict (dict): A global dictionary containing data samples categorized under "Bruteforce_XML".

    Note:
        This function assumes that `Sample_dict` contains a list of XML-formatted brute force attack 
        samples under the key "Bruteforce_XML". The function modifies specific fields in the sample 
        before sending to simulate a causative attack.
    """
    global Sample_dict
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    client_fileno = client_socket.fileno()
    random_int = random.randint(1, 9999)
    client_ID = client_fileno + random_int
    client_ID = "Causative_Attacker" + str(client_ID)
    sleep_time = random.randint(0,2)
    attack_strategy = [["crazy",1],["normal",3],["latency",5]]
    message_amount = 0


    while True:
        sample_index = random.randint(0, len(Sample_dict["Bruteforce_XML"]))
        test_sample = Sample_dict["Bruteforce"][sample_index]
        test_sample[22] = test_sample[22] * 1.5
        test_sample[15] = test_sample[15] * 1.8
        test_sample[33] = test_sample[33] * 0.1
        test_sample[75] = test_sample[75] * 0.5
        test_sample[34] = test_sample[34] * 1.5

        random_port = random.randint(1, 10)
        port_num = 8081 + random_port
        server_address = ("127.0.0.1", port_num)

        if connect_to_server(client_socket, server_address):

            message = [client_ID, test_sample]
            transfer_data_package = pickle.dumps(message)
            client_socket.send(transfer_data_package)
            message_amount += 1
            client_socket.close()
            time.sleep(attack_strategy[sleep_time][1])
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

def multi_client_script_thread():
    """
    Launches multiple client script threads based on specified types and amounts.

    This function initializes and starts multiple threads to simulate different types of 
    client behaviors (e.g., standard users, brute force attackers, scanners, and causative attackers).
    The number of threads for each type is determined by the global variable `Amount_and_types`. 
    The threads run scripts that simulate specific behaviors, with each script running in its own thread.

    Global Variables:
        Client_Script_Threads (list): A global list to store the created thread objects.
        Amount_and_types (list): A global list containing tuples that specify the number 
        of threads to be created for each type of client behavior.
        Example: [(num_standard_users, type), (num_bruteforce, type), ...]

    Note:
        The function assumes the following global variables are defined and populated before calling:
        - `Client_Script_Threads`: A list that will store the created thread objects.
        - `Amount_and_types`: A list where each element is a tuple, with the first value indicating 
          the number of threads to create and the second indicating the type (e.g., standard user, 
          brute force attacker).

    Execution:
        The function first prints a message indicating the start of client script execution. It then 
        iterates over the specified amounts and types of client scripts and appends the corresponding 
        threads to `Client_Script_Threads`. Finally, it starts each thread in the list.
    """
    global Client_Script_Threads, Amount_and_types
    print("client scripts start to run")
    
    for i in range(0, Amount_and_types[0][0]):
        time_low, time_high = user_Strategy()
        Client_Script_Threads.append(
            threading.Thread(target=standard_User_Script, args=(time_low, time_high)))
    
    for i in range(0, Amount_and_types[1][0]):
        Client_Script_Threads.append(
            threading.Thread(target=BruteForce_User_Script))
    
    for i in range(0, Amount_and_types[2][0]):
        Client_Script_Threads.append(
            threading.Thread(target=Scanner_User_Script))
    
    for i in range(0, Amount_and_types[3][0]):
        Client_Script_Threads.append(
            threading.Thread(target=Causative_Attacker_User_Script))
    
    for thread in Client_Script_Threads:
        thread.start()

def receive_test_set():
    """
    Receives a test set from a server and returns the features list and testing data set.

    This function connects to a server at "127.0.0.1" on port 8081, sends a handshake message, 
    and then continuously receives data packets from the server. The received packets are accumulated 
    and deserialized to extract the features list and testing data set. Once all data is received, 
    the connection is closed and the extracted data is returned.

    Returns:
        tuple: A tuple containing two elements:
            - Features_List (list): A list of feature names or indices received from the server.
            - Testing_Data_Set (list): A list of test data samples received from the server.

    Note:
        The function uses a buffer size of 40960 bytes for receiving data packets. The server is expected 
        to send the data in chunks, and the function accumulates these chunks until the entire data set is received.
    """
    Client_Main_Socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
    Client_Main_Socket.connect(("127.0.0.1", 8081))

    Shake_Hand_Message = "Client main procedure starting..."

    Client_Main_Socket.send(Shake_Hand_Message.encode("utf-8"))
    Testing_Data_Packet = b''

    print("Start receving test set from server......")
    while True:
        Packet = Client_Main_Socket.recv(40960)
        if not Packet:
            break
        Testing_Data_Packet += Packet

    Recv_Data = pickle.loads(Testing_Data_Packet)
    Features_List = Recv_Data[0]
    Testing_Data_Set = Recv_Data[1]
    Client_Main_Socket.close()
    
    return Features_List, Testing_Data_Set

# Features_List, Testing_Data_Set = receive_test_set()

Features_List, Training_Data_Set, Testing_Data_Set = Training_set_create()

'''
print(Features_List)
Amount_Of_Testing_Set = 0
for i in range(0,6):
    Amount_Of_Testing_Set += len(Testing_Data_Set[i])
print(Amount_Of_Testing_Set)
'''

# 上方代码仅用于在客户端与服务器初始化时传递testing set用

Background_TestingSet = Testing_Data_Set[0]
Benign_TestingSet = Testing_Data_Set[1]
Bruteforce_XML_TestingSet = Testing_Data_Set[2]
Bruteforce_TestingSet = Testing_Data_Set[3]
Probing_TestingSet = Testing_Data_Set[4]
XMRIGCC_CryptoMiner_TestingSet = Testing_Data_Set[5]

Sample_dict = {
    "Background": Background_TestingSet,
    "Benign": Benign_TestingSet,
    "Bruteforce_XML": Bruteforce_XML_TestingSet,
    "Bruteforce": Bruteforce_TestingSet,
    "Probing": Probing_TestingSet,
    "XMRIGCC": XMRIGCC_CryptoMiner_TestingSet
}

Amount_and_types = [[5, "Benign"],[3,"BruteForce"],[5,"scanner"],[3,"Causative_Attacker"]]
# Amount_and_types is a summary of all types of users. Each element is a list. The first element is the number of users of this type, and the second element is the type of user of this type.
Client_Script_Threads = []


multi_client_script_thread()
