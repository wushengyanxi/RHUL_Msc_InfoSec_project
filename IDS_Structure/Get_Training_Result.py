import sys

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\Machine_Learning_Aglorithm')
from Linear_Regression_1_1 import one_step_LR


def Ensemble_Learning_Training(feature_list, Data):
    LR_scale_factors, LR_weight = one_step_LR(Data, feature_list)
    Training_Result = [[LR_scale_factors, LR_weight]]

    return Training_Result
