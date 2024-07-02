import time
import sys
import csv

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



