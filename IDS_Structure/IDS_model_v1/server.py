import time
import sys
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project')
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\IDS_Structure')
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
from Get_Training_Result import Ensemble_Lrarning_Training

Features_List, Training_Data_Set, Testing_Data_Set = Training_set_create()
Training_Result = Ensemble_Lrarning_Training(Features_List, Training_Data_Set)


