import sys
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create

features_name, training_Data_Set, testing_Data_Set = Training_set_create()


def get_normal_distribution(data_set):
    # get the mean and standard deviation of each feature
    normal_distribution = []
    for i in range(7,86):
        feature = [row[i] for row in data_set]
        mean = sum(feature) / len(feature)
        if mean == 0:
            mean = 0.0000001
            std_dev = 0.0000001
        else:
            std_dev = (sum([(x - mean) ** 2 for x in feature]) / len(feature)) ** 0.5
        normal_distribution.append([mean, std_dev])
    return normal_distribution