import pandas as pd
from random import shuffle


def Training_set_create(Background=2000, Benign=10000, Bruteforce_XML=2000, Bruteforce=2000, Probing=2000,
                        XMRIGCC_CryptoMiner=2000):
    """
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
    shuffle(Background_Data)
    Background_Data_trimmed_list = Background_Data[:Background]
    Background_TestingSet = Background_Data[Background:]

    df_Benign = pd.read_csv(file_path_Benign)
    Benign_Data = df_Benign.to_numpy().tolist()
    shuffle(Benign_Data)
    Benign_Data_trimmed_list = Benign_Data[:Benign]
    Benign_TestingSet = Benign_Data[Benign:]

    df_Bruteforce_XML = pd.read_csv(file_path_Bruteforce_XML)
    Bruteforce_XML_Data = df_Bruteforce_XML.to_numpy().tolist()
    shuffle(Bruteforce_XML_Data)
    Bruteforce_XML_Data_trimmed_list = Bruteforce_XML_Data[:Bruteforce_XML]
    Bruteforce_XML_TestingSet = Bruteforce_XML_Data[Bruteforce_XML:]

    df_Bruteforce = pd.read_csv(file_path_Bruteforce)
    Bruteforce_Data = df_Bruteforce.to_numpy().tolist()
    shuffle(Bruteforce_Data)
    Bruteforce_Data_trimmed_list = Bruteforce_Data[:Bruteforce]
    Bruteforce_TestingSet = Bruteforce_Data[Bruteforce:]

    df_Probing = pd.read_csv(file_path_Probing)
    Probing_Data = df_Probing.to_numpy().tolist()
    shuffle(Probing_Data)
    Probing_Data_trimmed_list = Probing_Data[:Probing]
    Probing_TestingSet = Probing_Data[Probing:]

    df_XMRIGCC_CryptoMiner = pd.read_csv(file_path_XMRIGCC_CryptoMiner)
    XMRIGCC_CryptoMiner_Data = df_XMRIGCC_CryptoMiner.to_numpy().tolist()
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
    print("the features list is: ", Features_name)
    print("one of the sample in training set is: ", Training_Data_Set[0])
