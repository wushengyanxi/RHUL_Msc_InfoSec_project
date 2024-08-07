import socket
import pickle
import threading
import random
import time
import sys

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create


def user_Strategy():
    time_low = random.randint(1, 2)
    time_high = random.randint(3, 10)
    return time_low, time_high


def connect_to_server(client_socket, server_address):
    try:
        client_socket.connect(server_address)
        return True
    except socket.error as e:
        return False


def standard_User_Script(time_low, time_high):
    global Sample_dict
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    client_fileno = client_socket.fileno()
    random_int = random.randint(1, 9999)
    client_ID = client_fileno + random_int
    client_ID = str(client_ID)
    message_amount = 0
    count = 0

    while True:
        if message_amount%20 == 0 and message_amount != 0:
            print(f"Client {client_ID} has sent {message_amount} messages and got {count} correct responses.")
        
        sample_index = random.randint(0, len(Sample_dict["Benign"]))
        test_sample = Sample_dict["Benign"][sample_index]

        random_port = random.randint(1, 10)
        port_num = 8081 + random_port
        server_address = ("127.0.0.1", port_num)

        if connect_to_server(client_socket, server_address):
            try:
                message = [client_ID, test_sample]
                transfer_data_package = pickle.dumps(message)
                client_socket.send(transfer_data_package)
                message_amount += 1
                '''
                这里还缺一段代码
                当服务器判定客户端为攻击者时，记录，并要求客户端关闭此线程（在这个项目中更省算力）
                另一个方法是，在服务端维护一个攻击者列表（在真实场景中貌似更合理）
                '''
                response = client_socket.recv(4096).decode("utf-8")
                if int(response) == test_sample[-1]:
                    count += 1
            except socket.error as e:
                print(f"Communication error: {e}")
            finally:
                client_socket.close()
                sleep_time = random.randint(time_low, time_high)
                time.sleep(sleep_time)
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def BruteForce_User_Script():
    global Sample_dict
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    client_fileno = client_socket.fileno()
    random_int = random.randint(1, 9999)
    client_ID = client_fileno + random_int
    client_ID = str(client_ID)
    sleep_time = random.randint(0,2)
    attack_strategy = [["crazy",0.5],["normal",3],["latency",15]]
    message_amount = 0
    count = 0

    while True:
        if message_amount%20 == 0 and message_amount != 0:
            print(f"Client {client_ID} has sent {message_amount} messages and got {count} correct responses.")
        sample_index = random.randint(0, len(Sample_dict["Bruteforce"]))
        test_sample = Sample_dict["Bruteforce"][sample_index]

        random_port = random.randint(1, 10)
        port_num = 8081 + random_port
        server_address = ("127.0.0.1", port_num)

        if connect_to_server(client_socket, server_address):
            try:
                message = [client_ID, test_sample]
                transfer_data_package = pickle.dumps(message)
                client_socket.send(transfer_data_package)
                message_amount += 1
                '''
                这里还缺一段代码
                当服务器判定客户端为攻击者时，记录，并要求客户端关闭此线程（在这个项目中更省算力）
                另一个方法是，在服务端维护一个攻击者列表（在真实场景中貌似更合理）
                '''
                response = client_socket.recv(4096).decode("utf-8")
                if int(response) == test_sample[-1]:
                    count += 1
            except socket.error as e:
                print(f"Communication error: {e}")
            finally:
                client_socket.close()
                time.sleep(attack_strategy[sleep_time][1])
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def Scanner_User_Script():
    global Sample_dict
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    client_fileno = client_socket.fileno()
    random_int = random.randint(1, 9999)
    client_ID = client_fileno + random_int
    client_ID = str(client_ID)
    sleep_time = random.randint(0,2)
    attack_strategy = [["crazy",0.5],["normal",3],["latency",15]]
    message_amount = 0
    count = 0

    while True:
        if message_amount%20 == 0 and message_amount != 0:
            print(f"Client {client_ID} has sent {message_amount} messages and got {count} correct responses.")
        sample_index = random.randint(0, len(Sample_dict["Bruteforce_XML"]))
        test_sample = Sample_dict["Bruteforce"][sample_index]

        random_port = random.randint(1, 10)
        port_num = 8081 + random_port
        server_address = ("127.0.0.1", port_num)

        if connect_to_server(client_socket, server_address):
            try:
                message = [client_ID, test_sample]
                transfer_data_package = pickle.dumps(message)
                client_socket.send(transfer_data_package)
                message_amount += 1
                '''
                这里还缺一段代码
                当服务器判定客户端为攻击者时，记录，并要求客户端关闭此线程（在这个项目中更省算力）
                另一个方法是，在服务端维护一个攻击者列表（在真实场景中貌似更合理）
                '''
                response = client_socket.recv(4096).decode("utf-8")
                if int(response) == test_sample[-1]:
                    count += 1
            except socket.error as e:
                print(f"Communication error: {e}")
            finally:
                client_socket.close()
                time.sleep(attack_strategy[sleep_time][1])
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

def multi_client_script_thread():
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
    
    for thread in Client_Script_Threads:
        thread.start()

def receive_test_set():
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

Amount_and_types = [[50, "Benign"],[3,"BruteForce"],[5,"scanner"],[]]
# Amount_and_types 是所有类型的用户的汇总，每个元素是一个列表，第一个元素是该类型用户的数量，第二个元素是该类型用户的类型
# 理想情况下，Amount_and_type中应该含有 [[50,良性用户],[3，爆破用户],[5，扫描用户],[2，数据污染用户],[？未知策略攻击者？]]
# 最好是在multi_client_script_thread()函数中，使用若干个for循环来依次启动各类用户，对于每一类而言，使用一个for循环来依次启动每个用户，从而保证每个用户的行为尽可能独立和可定制化
Client_Script_Threads = []


multi_client_script_thread()
