import pandas as pd
from random import shuffle
import random


def Training_set_create(Background=2000, Benign=3000, Bruteforce_XML=2000, Bruteforce=2000, Probing=2000,
                        XMRIGCC_CryptoMiner=2000):
    """
    Creates training and testing datasets by sampling from preprocessed CSV files.

    Args:
        Background (int): Number of samples to use from the Background set.
        Benign (int): Number of samples to use from the Benign set.
        Bruteforce_XML (int): Number of samples to use from the Bruteforce-XML set.
        Bruteforce (int): Number of samples to use from the Bruteforce set.
        Probing (int): Number of samples to use from the Probing set.
        XMRIGCC_CryptoMiner (int): Number of samples to use from the XMRIGCC CryptoMiner set.

    Returns:
        tuple: A tuple containing:
            - features_name (list): The list of feature names.
            - training_Data_Set (list): The combined training dataset.
            - testing_Data_Set (list): The testing datasets split by category.

    
    
    read csv file and return DataFrame

    The number of samples which have value "Bruteforce-XML" on feature "traffic_category" is 5145
    The number of samples which have value "Bruteforce" on feature "traffic_category" is 5884
    The number of samples which have value "Background" on feature "traffic_category" is 170151
    The number of samples which have value "Benign" on feature "traffic_category" is 347431
    The number of samples which have value "Probing" on feature "traffic_category" is 23388
    The number of samples which have value "XMRIGCC CryptoMiner" on feature "traffic_category" is 3279
    """
    file_path_Background = (r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser\Sub_database"
                            r"\Background_set.csv")
    file_path_Benign = (r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser\Sub_database\Benign_set"
                        r".csv")
    file_path_Bruteforce_XML = (r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser\Sub_database"
                                r"\Bruteforce-XML_set.csv")
    file_path_Bruteforce = (r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser\Sub_database"
                            r"\Bruteforce_set.csv")
    file_path_Probing = (r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser\Sub_database"
                         r"\Probing_set.csv")
    file_path_XMRIGCC_CryptoMiner = (r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser"
                                     r"\Sub_database\XMRIGCC CryptoMiner_set.csv")

    if Background > 170151:
        Background = 170151
        print("Value of parameter 'Background' has already set as 170151, which is it's upper bound value.")

    if Benign > 347431:
        Benign = 347431
        print("Value of parameter 'Benign' has already set as 347431, which is it's upper bound value.")

    if Bruteforce_XML > 5145:
        Bruteforce_XML = 5145
        print("Value of parameter 'Bruteforce_XML' has already set as 347431, which is it's upper bound value.")

    if Bruteforce > 5884:
        Bruteforce = 5884
        print("Value of parameter 'Bruteforce_XML' has already set as 5884, which is it's upper bound value.")

    if Probing > 23388:
        Probing = 23388
        print("Value of parameter 'Probing' has already set as 23388, which is it's upper bound value.")

    if XMRIGCC_CryptoMiner > 3279:
        XMRIGCC_CryptoMiner = 3279
        print("Value of parameter 'XMRIGCC_CryptoMiner' has already set as 23388, which is it's upper bound value.")


    df = pd.read_csv(file_path_Background)
    features_name = df.columns.tolist()

    df_Background = pd.read_csv(file_path_Background)
    Background_Data = df_Background.to_numpy().tolist()
    for i in Background_Data:
        i[0] = 0
        i[1] = 0
    shuffle(Background_Data)
    Background_Data_trimmed_list = Background_Data[:Background]
    Background_TestingSet = Background_Data[Background:]

    df_Benign = pd.read_csv(file_path_Benign)
    Benign_Data = df_Benign.to_numpy().tolist()
    for i in Benign_Data:
        i[0] = 0
        i[1] = 0
    shuffle(Benign_Data)
    Benign_Data_trimmed_list = Benign_Data[:Benign]
    Benign_TestingSet = Benign_Data[Benign:]

    df_Bruteforce_XML = pd.read_csv(file_path_Bruteforce_XML)
    Bruteforce_XML_Data = df_Bruteforce_XML.to_numpy().tolist()
    for i in Bruteforce_XML_Data:
        i[0] = 0
        i[1] = 0
    shuffle(Bruteforce_XML_Data)
    Bruteforce_XML_Data_trimmed_list = Bruteforce_XML_Data[:Bruteforce_XML]
    Bruteforce_XML_TestingSet = Bruteforce_XML_Data[Bruteforce_XML:]

    df_Bruteforce = pd.read_csv(file_path_Bruteforce)
    Bruteforce_Data = df_Bruteforce.to_numpy().tolist()
    for i in Bruteforce_Data:
        i[0] = 0
        i[1] = 0
    shuffle(Bruteforce_Data)
    Bruteforce_Data_trimmed_list = Bruteforce_Data[:Bruteforce]
    Bruteforce_TestingSet = Bruteforce_Data[Bruteforce:]

    df_Probing = pd.read_csv(file_path_Probing)
    Probing_Data = df_Probing.to_numpy().tolist()
    for i in Probing_Data:
        i[0] = 0
        i[1] = 0
    shuffle(Probing_Data)
    Probing_Data_trimmed_list = Probing_Data[:Probing]
    Probing_TestingSet = Probing_Data[Probing:]

    df_XMRIGCC_CryptoMiner = pd.read_csv(file_path_XMRIGCC_CryptoMiner)
    XMRIGCC_CryptoMiner_Data = df_XMRIGCC_CryptoMiner.to_numpy().tolist()
    for i in XMRIGCC_CryptoMiner_Data:
        i[0] = 0
        i[1] = 0
    shuffle(XMRIGCC_CryptoMiner_Data)
    XMRIGCC_CryptoMiner_Data_trimmed_list = XMRIGCC_CryptoMiner_Data[:XMRIGCC_CryptoMiner]
    XMRIGCC_CryptoMiner_TestingSet = XMRIGCC_CryptoMiner_Data[XMRIGCC_CryptoMiner:]

    # print(column_names)
    # print(Bruteforce_XML_Data[150])
    # print(type(Bruteforce_XML_Data[150]))

    training_Data_Set = (Background_Data_trimmed_list + Benign_Data_trimmed_list + Bruteforce_XML_Data_trimmed_list +
                         Bruteforce_Data_trimmed_list + Probing_Data_trimmed_list +
                         XMRIGCC_CryptoMiner_Data_trimmed_list)

    testing_Data_Set = [Background_TestingSet, Benign_TestingSet, Bruteforce_XML_TestingSet, Bruteforce_TestingSet,
                        Probing_TestingSet, XMRIGCC_CryptoMiner_TestingSet]

    return features_name, training_Data_Set, testing_Data_Set
    # return sample with label, 88 length for each sample

if __name__ == "__main__":

    Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create()
    '''
    print("the features list is: ", Features_name)
    print()
    print("length of features list is: ", len(Features_name))
    print()
    print("one of the sample in training set is: ")
    print(Training_Data_Set[0])
    print()
    print("length of one of the sample in training set is: ", len(Training_Data_Set[0]))
    '''
    print(Features_name.index("fwd_header_size_min"))
    print(Features_name.index("flow_FIN_flag_count"))
    print(Features_name.index("down_up_ratio"))
    print(Features_name.index("fwd_pkts_payload.max"))
    print(Features_name.index("active.tot"))
    print(Features_name.index("fwd_pkts_payload.tot"))

 
    print("the features list is: ", Features_name)
    print("length of feature, ", len(Features_name))  


 
