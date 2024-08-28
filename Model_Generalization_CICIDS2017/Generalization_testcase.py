'''
B = ['Flow ID', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Protocol', ' Timestamp', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min', ' Label']



match_feature = {
 'bwd_pkts_payload.max': 'Bwd Packet Length Max',
 'fwd_iat.std': ' Fwd IAT Std',
 'bwd_subflow_bytes': ' Subflow Bwd Bytes',
 'fwd_iat.min': ' Fwd IAT Min',
 'fwd_pkts_payload.tot': ' Total Fwd Packets',
 'flow_pkts_payload.tot': None,
 'active.tot': None,
 'responp': ' Destination Port',
 'active.avg': 'Active Mean',
 'fwd_pkts_per_sec': 'Fwd Packets/s',
 'flow_iat.min': ' Flow IAT Min',
 'bwd_header_size_tot': ' Bwd Header Length',
 'bwd_data_pkts_tot': ' Total Length of Bwd Packets',
 'bwd_header_size_max': None,
 'flow_pkts_payload.avg': None,
 'flow_pkts_payload.min': None,
 'bwd_pkts_payload.avg': ' Bwd Packet Length Mean',
 'fwd_PSH_flag_count': 'Fwd PSH Flags', # ?
 'down_up_ratio': ' Down/Up Ratio',
 'fwd_pkts_payload.max': ' Fwd Packet Length Max',
 'fwd_iat.tot': 'Fwd IAT Total',
 'flow_RST_flag_count': None,
 'bwd_header_size_min': None,
 'flow_pkts_per_sec': ' Flow Packets/s',
 'bwd_iat.tot': 'Bwd IAT Total',
 'bwd_pkts_per_sec': ' Bwd Packets/s',
 'fwd_iat.max': ' Fwd IAT Max',
 'flow_iat.tot': None,
 'fwd_header_size_min': ' Bwd IAT Min',
 'flow_CWR_flag_count': None,
 'bwd_iat.max': ' Bwd IAT Max',
 'bwd_URG_flag_count': ' Bwd URG Flags',
 'flow_ACK_flag_count': ' ACK Flag Count',
 'flow_iat.max': ' Flow IAT Max',
 'fwd_pkts_tot': None,
 'bwd_pkts_tot': None,
 'fwd_header_size_tot': None,
 'bwd_iat.avg': ' Bwd IAT Mean',
 'flow_pkts_payload.std': None,
 'Label': ' Label',
 'flow_duration': ' Flow Duration',
 'bwd_iat.std': ' Bwd IAT Std',
 'fwd_pkts_payload.avg': ' Fwd Packet Length Mean',
 'fwd_pkts_payload.min': ' Fwd Packet Length Min',
 'payload_bytes_per_second': None,
 'flow_FIN_flag_count': None,
 'fwd_bulk_rate': ' Fwd Avg Bulk Rate',
 'flow_iat.std': ' Flow IAT Std',
 'bwd_pkts_payload.min': ' Bwd Packet Length Min',
 'bwd_pkts_payload.std': ' Bwd Packet Length Std'
}
'''
import sys
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\IDS_Structure')
from Training_Set_Creator import Training_set_create
from CICIDS2017_Training_Set_Creator import Testing_set_create
from generalization_test_model import Ensemble_Learning_Training
from generalization_test_model import Ensemble_Learning_Decision

match_feature = {
    'Bwd Packet Length Max': 'bwd_pkts_payload.max',
    ' Fwd IAT Std': 'fwd_iat.std',
    ' Subflow Bwd Bytes': 'bwd_subflow_bytes',
    ' Fwd IAT Min': 'fwd_iat.min',
    ' Total Fwd Packets': 'fwd_pkts_payload.tot',
    ' Destination Port': 'responp',
    'Active Mean': 'active.avg',
    'Fwd Packets/s': 'fwd_pkts_per_sec',
    ' Flow IAT Min': 'flow_iat.min',
    ' Bwd Header Length': 'bwd_header_size_tot',
    ' Total Length of Bwd Packets': 'bwd_data_pkts_tot',
    ' Bwd Packet Length Mean': 'bwd_pkts_payload.avg',
    'Fwd PSH Flags': 'fwd_PSH_flag_count',
    ' Down/Up Ratio': 'down_up_ratio',
    ' Fwd Packet Length Max': 'fwd_pkts_payload.max',
    'Fwd IAT Total': 'fwd_iat.tot',
    ' Flow Packets/s': 'flow_pkts_per_sec',
    'Bwd IAT Total': 'bwd_iat.tot',
    ' Bwd Packets/s': 'bwd_pkts_per_sec',
    ' Fwd IAT Max': 'fwd_iat.max',
    ' Bwd IAT Min': 'fwd_header_size_min',
    ' Bwd IAT Max': 'bwd_iat.max',
    ' Bwd URG Flags': 'bwd_URG_flag_count',
    ' ACK Flag Count': 'flow_ACK_flag_count',
    ' Flow IAT Max': 'flow_iat.max',
    ' Bwd IAT Mean': 'bwd_iat.avg',
    ' Label': 'Label',
    ' Flow Duration': 'flow_duration',
    ' Bwd IAT Std': 'bwd_iat.std',
    ' Fwd Packet Length Mean': 'fwd_pkts_payload.avg',
    ' Fwd Packet Length Min': 'fwd_pkts_payload.min',
    ' Fwd Avg Bulk Rate': 'fwd_bulk_rate',
    ' Flow IAT Std': 'flow_iat.std',
    ' Bwd Packet Length Min': 'bwd_pkts_payload.min',
    ' Bwd Packet Length Std': 'bwd_pkts_payload.std'
}


Testing_Set, CICIDS_features_name = Testing_set_create()
Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(700,700,2500,2500,1500,1500)


def update_list(A, match_feature):
    for i in range(len(A)):
        if A[i] in match_feature:
            A[i] = match_feature[A[i]]
    return A

CICIDS_features_name = update_list(CICIDS_features_name, match_feature)

print(type(Testing_Set))
print(type(Training_Data_Set))

Ensemble_param = Ensemble_Learning_Training(Features_name, Training_Data_Set)

print(Testing_Set[1])

benign_count = 0
malicious_count = 0
for i in Testing_Set:
    predict_decition, reliable = Ensemble_Learning_Decision(Ensemble_param, i, CICIDS_features_name)
    if predict_decition == i[-1] == 0:
        benign_count += 1
    if predict_decition == i[-1] == 1:
        malicious_count += 1

print(benign_count / 1500)
print(malicious_count / 1500)
