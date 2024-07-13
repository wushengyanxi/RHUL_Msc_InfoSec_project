import sys

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\Machine_Learning_Aglorithm')
from Linear_Regression_1_1 import one_step_LR


def Ensemble_Learning_Training(feature_list, Data):
    LR_scale_factors, LR_weight = one_step_LR(Data, feature_list)
    Training_Result = [[LR_scale_factors, LR_weight]]

    return Training_Result


def Ensemble_Learning_Decision(param, new_sample, features_list):
    LR_scale_factor = param[0][0]
    LR_weights = param[0][1]
    predict_score = 0
    predict_label = 0

    for n in range(0, len(features_list)):
        index = next((i for i, sublist in enumerate(LR_scale_factor) if sublist[0] == features_list[n]), -1)
        if index != -1:
            new_sample[n] = new_sample[n] * LR_scale_factor[index][1]
            predict_score += new_sample[n] * LR_weights[features_list[n]]

    if predict_score >= 0.5:
        predict_label = 1

    return predict_label



