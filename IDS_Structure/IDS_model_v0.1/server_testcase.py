import time
import sys
import socket
import multiprocessing
import csv

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project')
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\IDS_Structure')
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
from Get_Training_Result import Ensemble_Learning_Training
from server import initialize_log_file
from server import read_log_file

'''


This is a testcase for "initialize_log_file" and "read_log_file"

We need make sure the feature list and the sample will keep exactly same when they were written in file and been read

In testcase, when we get Features_List and Training_Data_Set by Training_set_create()

we first store them in variables, for training set data, we choose the first and the 1314th to store

and after new Features_List and Training_Data_Set been read from file, we check if the corresponding content is exactly the same
--------------

Features_List, Training_Data_Set, Testing_Data_Set = Training_set_create()

test_feature = Features_List
testcase_sample = Training_Data_Set[0]
testcase_sample2 = Training_Data_Set[1314]

initialize_log_file(Features_List, Training_Data_Set)

Features_List, Training_Data_Set = read_log_file()

if Features_List == test_feature and testcase_sample == Training_Data_Set[0] and testcase_sample2 == Training_Data_Set[1314]:
    print("pass test")
else:
    print("not pass")

'''