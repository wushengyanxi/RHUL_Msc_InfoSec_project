import pandas as pd
from random import shuffle

def Testing_set_create():
    F_path_Tues_Mor = (r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\Model_Generalization_CICIDS2017\Tuesday-WorkingHours.pcap_ISCX.csv")
    F_path_Wedn = (r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\Model_Generalization_CICIDS2017\\Wednesday-workingHours.pcap_ISCX.csv")
    F_path_Thur_Mor = (r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\Model_Generalization_CICIDS2017\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    F_path_Thur_Aft = (r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\Model_Generalization_CICIDS2017\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
    F_path_Fri_Mor = (r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\Model_Generalization_CICIDS2017\Friday-WorkingHours-Morning.pcap_ISCX.csv")
    F_path_Fri_Aft_scan = (r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\Model_Generalization_CICIDS2017\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    F_path_Fri_Aft_DDos = (r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\Model_Generalization_CICIDS2017\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    
    df = pd.read_csv(F_path_Thur_Mor)
    features_name = df.columns.tolist()
    
    Testing_set_come_from = [F_path_Thur_Mor]
    Testing_set = []
    Step2_Testing_set = []
    
    for i in Testing_set_come_from:
        df = pd.read_csv(i)
        Testing_Data = df.to_numpy().tolist()
        for i in Testing_Data:
            i[0] = 0
            i[1] = 0
        shuffle(Testing_Data)
        
        Testing_set.append(Testing_Data)
    
    if len(Testing_set) == 1:
        Step2_Testing_set = Testing_set[0]
    else:
        for i in Testing_set:
            Step2_Testing_set += i
    shuffle(Step2_Testing_set)
    
    Final_Testing_set = []
    count = 0
    
    while len(Final_Testing_set) < 1500:
        if Step2_Testing_set[count][-1] == "BENIGN":
            if count == 750:
                print("adding benign test sample")
            Step2_Testing_set[count][-1] = 0
            Final_Testing_set.append(Step2_Testing_set[count])
        count += 1
    
    count = 0
    
    while len(Final_Testing_set) < 3000:
        if Step2_Testing_set[count][-1] != "BENIGN":
            if count == 750:
                print("adding malicious test sample")
            Step2_Testing_set[count][-1] = 1
            Final_Testing_set.append(Step2_Testing_set[count])
        count += 1
    

    return Final_Testing_set, features_name

if __name__ == "__main__":
    Testing_Set, features_name = Testing_set_create()
    