import sys
import random
from random import shuffle

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create

Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create(700,700,1500,1500,1500,1500)

'''
fwd_header_size_min, flow_FIN_flag_count, down_up_ratio, fwd_pkts_payload.max, active.tot, fwd_pkts_payload.tot

with index 17,22,15,33,75,34 in Features_name

For the feature sets adopted by each model, the above four features appear in each feature set, 
and account for a large proportion, and their values ​are more determined by the client rather than the server.

Therefore, I decided to use them as the attack surface for causal attacks.

In order to understand the attack strategy, first, 
we need to calculate the average value of each attribute for samples of different categories.

'''
print("for Background samples:")
count17 = 0
count22 = 0
count15 = 0
count33 = 0
count75 = 0
count34 = 0
for i in Testing_Data_Set[0]:
    count17 += i[17]
    count22 += i[22]
    count15 += i[15]
    count33 += i[33]
    count75 += i[75]
    count34 += i[34]
    
print("mean value of fwd_header_size_min =",count17/len(Testing_Data_Set[0]))
print("mean value of flow_FIN_flag_count =",count22/len(Testing_Data_Set[0]))
print("mean value of down_up_ratio =",count15/len(Testing_Data_Set[0]))
print("mean value of fwd_pkts_payload.max =",count33/len(Testing_Data_Set[0]))
print("mean value of active.tot =",count75/len(Testing_Data_Set[0]))
print("mean value of fwd_pkts_payload.tot =",count34/len(Testing_Data_Set[0]))

# ------------------------------------------------

print("for Benign samples:")
count17 = 0
count22 = 0
count15 = 0
count33 = 0
count75 = 0
count34 = 0
for i in Testing_Data_Set[1]:
    count17 += i[17]
    count22 += i[22]
    count15 += i[15]
    count33 += i[33]
    count75 += i[75]
    count34 += i[34]
    
print("mean value of fwd_header_size_min =",count17/len(Testing_Data_Set[1]))
print("mean value of flow_FIN_flag_count =",count22/len(Testing_Data_Set[1]))
print("mean value of down_up_ratio =",count15/len(Testing_Data_Set[1]))
print("mean value of fwd_pkts_payload.max =",count33/len(Testing_Data_Set[1]))
print("mean value of active.tot =",count75/len(Testing_Data_Set[1]))
print("mean value of fwd_pkts_payload.tot =",count34/len(Testing_Data_Set[1]))

# ------------------------------------------------

print("for Bruteforce_XML samples:")
count17 = 0
count22 = 0
count15 = 0
count33 = 0
count75 = 0
count34 = 0
for i in Testing_Data_Set[2]:
    count17 += i[17]
    count22 += i[22]
    count15 += i[15]
    count33 += i[33]
    count75 += i[75]
    count34 += i[34]   
print("mean value of fwd_header_size_min =",count17/len(Testing_Data_Set[2]))
print("mean value of flow_FIN_flag_count =",count22/len(Testing_Data_Set[2]))
print("mean value of down_up_ratio =",count15/len(Testing_Data_Set[2]))
print("mean value of fwd_pkts_payload.max =",count33/len(Testing_Data_Set[2]))
print("mean value of active.tot =",count75/len(Testing_Data_Set[2]))
print("mean value of fwd_pkts_payload.tot =",count34/len(Testing_Data_Set[2]))

# ------------------------------------------------

print("for Bruteforce samples:")
count17 = 0
count22 = 0
count15 = 0
count33 = 0
count75 = 0
count34 = 0
for i in Testing_Data_Set[3]:
    count17 += i[17]
    count22 += i[22]
    count15 += i[15]
    count33 += i[33]
    count75 += i[75]
    count34 += i[34]    
print("mean value of fwd_header_size_min =",count17/len(Testing_Data_Set[3]))
print("mean value of flow_FIN_flag_count =",count22/len(Testing_Data_Set[3]))
print("mean value of down_up_ratio =",count15/len(Testing_Data_Set[3]))
print("mean value of fwd_pkts_payload.max =",count33/len(Testing_Data_Set[3]))
print("mean value of active.tot =",count75/len(Testing_Data_Set[3]))
print("mean value of fwd_pkts_payload.tot =",count34/len(Testing_Data_Set[3]))

# ------------------------------------------------

print("for probing samples:")
count17 = 0
count22 = 0
count15 = 0
count33 = 0
count75 = 0
count34 = 0
for i in Testing_Data_Set[4]:
    count17 += i[17]
    count22 += i[22]
    count15 += i[15]
    count33 += i[33]
    count75 += i[75]
    count34 += i[34]   
print("mean value of fwd_header_size_min =",count17/len(Testing_Data_Set[4]))
print("mean value of flow_FIN_flag_count =",count22/len(Testing_Data_Set[4]))
print("mean value of down_up_ratio =",count15/len(Testing_Data_Set[4]))
print("mean value of fwd_pkts_payload.max =",count33/len(Testing_Data_Set[4]))
print("mean value of active.tot =",count75/len(Testing_Data_Set[4]))
print("mean value of fwd_pkts_payload.tot =",count34/len(Testing_Data_Set[4]))

# ------------------------------------------------

print("for XMRIGCC_CryptoMiner samples:")
count17 = 0
count22 = 0
count15 = 0
count33 = 0
count75 = 0
count34 = 0
for i in Testing_Data_Set[5]:
    count17 += i[17]
    count22 += i[22]
    count15 += i[15]
    count33 += i[33]
    count75 += i[75]
    count34 += i[34]   
print("mean value of fwd_header_size_min =",count17/len(Testing_Data_Set[5]))
print("mean value of flow_FIN_flag_count =",count22/len(Testing_Data_Set[5]))
print("mean value of down_up_ratio =",count15/len(Testing_Data_Set[5]))
print("mean value of fwd_pkts_payload.max =",count33/len(Testing_Data_Set[5]))
print("mean value of active.tot =",count75/len(Testing_Data_Set[5]))
print("mean value of fwd_pkts_payload.tot =",count34/len(Testing_Data_Set[5]))

# ------------------------------------------------