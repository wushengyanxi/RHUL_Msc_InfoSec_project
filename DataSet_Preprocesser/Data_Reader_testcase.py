import sys

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project')
from Data_Reader import Database_Reader

feature_list, data = Database_Reader("ALLFLOWMETER_HIKARI2021.csv", full_read=True, traffic_category=True)
print("Features:", feature_list)
print("Data Sample:", data[0])

print("--------------------------")

feature_list, data = Database_Reader("ALLFLOWMETER_HIKARI2021.csv", full_read=True, traffic_category=False)
print("Features:", feature_list)
print("Data Sample:", data[0])

print("--------------------------")

feature_list, data = Database_Reader("ALLFLOWMETER_HIKARI2021.csv", full_read=False,
                                         feature_list=['uid', 'originh', 'originp'], traffic_category=True)
print("Features:", feature_list)
print("Data Sample:", data[0])

print("--------------------------")

feature_list, data = Database_Reader("ALLFLOWMETER_HIKARI2021.csv", full_read=False,
                                         feature_list=['uid', 'originh', 'originp'], traffic_category=False)
print("Features:", feature_list)
print("Data Sample:", data[0])