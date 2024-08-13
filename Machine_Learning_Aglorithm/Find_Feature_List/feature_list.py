from collections import Counter
from collections import defaultdict
import random

model = ['originp','responp', 'flow_duration', 'fwd_pkts_tot',
         'bwd_pkts_tot', 'fwd_data_pkts_tot', 'bwd_data_pkts_tot',
         'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec',
         'down_up_ratio', 'fwd_header_size_tot', 'fwd_header_size_min', 
         'fwd_header_size_max', 'bwd_header_size_tot', 'bwd_header_size_min', 'bwd_header_size_max',
         'flow_FIN_flag_count', 'flow_SYN_flag_count', 'flow_RST_flag_count', 'fwd_PSH_flag_count', 
         'bwd_PSH_flag_count', 'flow_ACK_flag_count', 'fwd_URG_flag_count',
         'bwd_URG_flag_count', 'flow_CWR_flag_count', 'flow_ECE_flag_count', 
         'fwd_pkts_payload.min', 'fwd_pkts_payload.max', 'fwd_pkts_payload.tot', 'fwd_pkts_payload.avg',
         'fwd_pkts_payload.std', 'bwd_pkts_payload.min', 'bwd_pkts_payload.max', 'bwd_pkts_payload.tot',
         'bwd_pkts_payload.avg', 'bwd_pkts_payload.std', 'flow_pkts_payload.min', 'flow_pkts_payload.max',
         'flow_pkts_payload.tot', 'flow_pkts_payload.avg', 'flow_pkts_payload.std', 'fwd_iat.min',
         'fwd_iat.max', 'fwd_iat.tot', 'fwd_iat.avg', 'fwd_iat.std', 'bwd_iat.min', 'bwd_iat.max',
         'bwd_iat.tot', 'bwd_iat.avg', 'bwd_iat.std', 'flow_iat.min', 'flow_iat.max', 'flow_iat.tot',
         'flow_iat.avg', 'flow_iat.std', 'payload_bytes_per_second', 'fwd_subflow_pkts',
         'bwd_subflow_pkts', 'fwd_subflow_bytes', 'bwd_subflow_bytes', 'fwd_bulk_bytes',
         'bwd_bulk_bytes', 'fwd_bulk_packets', 'bwd_bulk_packets', 'fwd_bulk_rate', 'bwd_bulk_rate',
         'active.min', 'active.max', 'active.tot', 'active.avg', 'active.std',
         'idle.min', 'idle.max', 'idle.tot', 'idle.avg', 'idle.std', 'fwd_init_window_size',
         'bwd_init_window_size', 'fwd_last_window_size', 'Label']

heaviest_features = ['originp','responp', 'flow_duration', 'fwd_pkts_tot',
         'bwd_pkts_tot', 'fwd_data_pkts_tot', 'bwd_data_pkts_tot',
         'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec',
         'down_up_ratio', 'fwd_header_size_tot', 'fwd_header_size_min', 
         'fwd_header_size_max', 'bwd_header_size_tot', 'bwd_header_size_min', 'bwd_header_size_max',
         'flow_FIN_flag_count', 'flow_SYN_flag_count', 'flow_RST_flag_count', 'fwd_PSH_flag_count', 
         'bwd_PSH_flag_count', 'flow_ACK_flag_count', 'fwd_URG_flag_count',
         'bwd_URG_flag_count', 'flow_CWR_flag_count', 'flow_ECE_flag_count', 
         'fwd_pkts_payload.min', 'fwd_pkts_payload.max', 'fwd_pkts_payload.tot', 'fwd_pkts_payload.avg',
         'fwd_pkts_payload.std', 'bwd_pkts_payload.min', 'bwd_pkts_payload.max', 'bwd_pkts_payload.tot',
         'bwd_pkts_payload.avg', 'bwd_pkts_payload.std', 'flow_pkts_payload.min', 'flow_pkts_payload.max',
         'flow_pkts_payload.tot', 'flow_pkts_payload.avg', 'flow_pkts_payload.std', 'fwd_iat.min',
         'fwd_iat.max', 'fwd_iat.tot', 'fwd_iat.avg', 'fwd_iat.std', 'bwd_iat.min', 'bwd_iat.max',
         'bwd_iat.tot', 'bwd_iat.avg', 'bwd_iat.std', 'flow_iat.min', 'flow_iat.max', 'flow_iat.tot',
         'flow_iat.avg', 'flow_iat.std', 'payload_bytes_per_second', 'fwd_subflow_pkts',
         'bwd_subflow_pkts', 'bwd_subflow_bytes', 'fwd_bulk_bytes',
         'bwd_bulk_bytes', 'fwd_bulk_packets', 'bwd_bulk_packets', 'fwd_bulk_rate', 'bwd_bulk_rate',
         'active.min', 'active.max', 'active.tot', 'active.avg', 'active.std',
         'idle.min', 'idle.max', 'idle.tot', 'idle.avg', 'idle.std',
         'bwd_init_window_size', 'fwd_last_window_size', 'Label']


G1_25000 = [1, 32, 15, 57, 12, 10, 17, 47, 51, 18, 28, 3, 9, 39, 74, 11, 53, 2, 40, 30, 68, 69, 60, 54, 49]
G2 = [1, 32, 12, 15, 57, 17, 10, 28, 60, 47, 39, 51, 7, 31, 18, 26, 9, 6, 40, 49, 48, 35, 69, 14, 8]
G3 = [1, 32, 12, 15, 57, 17, 10, 51, 54, 28, 39, 60, 47, 7, 34, 58, 20, 30, 8, 4, 40, 65, 24, 69, 3]
G4 = [32, 15, 12, 17, 57, 47, 24, 51, 10, 30, 40, 69, 28, 58, 68, 18, 70, 39, 36, 9, 6, 3, 27, 2]
G6 = [1, 32, 57, 15, 12, 17, 10, 28, 51, 73, 9, 47, 18, 52, 29, 69, 30, 49, 34, 3, 35, 31, 26, 24, 8]
G7 = [1, 32, 12, 15, 17, 28, 10, 57, 18, 30, 40, 68, 67, 7, 39, 58, 49, 20, 69, 47, 5, 48, 53, 51, 52]
G8 = [1, 32, 15, 17, 12, 57, 51, 30, 69, 47, 49, 10, 28, 40, 54, 11, 39, 72, 74, 7, 18, 37, 78, 60, 50]
G9 = [1, 32, 15, 12, 57, 28, 47, 17, 10, 30, 2, 40, 54, 34, 49, 60, 18, 45, 61, 36, 67, 78, 29, 58, 51]
G10 = [1, 32, 15, 57, 12, 17, 28, 10, 40, 51, 18, 49, 7, 30, 8, 47, 26, 39, 69, 29, 75, 37, 53, 9, 43]
G11 = [1, 32, 57, 15, 12, 17, 42, 28, 47, 10, 51, 40, 58, 30, 29, 18, 69, 60, 45, 35, 73, 7, 56, 24, 44]
G12 = [1, 32, 28, 10, 12, 57, 15, 47, 5, 17, 69, 49, 30, 31, 39, 18, 11, 70, 52, 36, 20, 60, 53, 40, 67]
G13 = [1, 32, 15, 12, 57, 10, 17, 30, 49, 28, 47, 39, 40, 4, 59, 48, 51, 18, 45, 42, 41, 8, 78, 29, 67]
G14 = [1, 32, 12, 15, 10, 17, 57, 51, 47, 18, 24, 60, 28, 49, 39, 3, 2, 16, 68, 29, 30, 40, 69, 41, 26]
G15 = [1, 32, 15, 12, 57, 10, 28, 47, 3, 51, 17, 60, 69, 30, 18, 39, 49, 14, 40, 53, 48, 54, 59, 7, 74]
G16 = [1, 32, 15, 12, 57, 51, 17, 47, 10, 28, 31, 69, 30, 40, 41, 58, 35, 18, 3, 49, 7, 26, 34, 45, 4]
G17 = [1, 32, 15, 10, 12, 28, 3, 57, 47, 18, 51, 17, 30, 29, 8, 69, 59, 34, 20, 39, 35, 60, 45, 7, 49]
G18 = [1, 32, 57, 15, 12, 30, 51, 47, 39, 10, 17, 28, 18, 69, 49, 3, 60, 74, 20, 7, 5, 40, 22, 42, 16]
G19 = [1, 32, 15, 12, 57, 51, 10, 69, 28, 17, 18, 39, 30, 47, 29, 22, 49, 36, 43, 68, 4, 50, 24, 35, 60]
G20 = [1, 32, 15, 10, 57, 47, 12, 51, 28, 30, 18, 3, 69, 17, 49, 20, 34, 39, 40, 60, 58, 44, 74, 31, 24]
G21 = [1, 32, 57, 15, 12, 28, 10, 49, 51, 17, 30, 47, 39, 40, 26, 69, 18, 31, 36, 29, 22, 20, 43, 5, 60]
G22 = [1, 32, 12, 15, 57, 10, 17, 28, 51, 5, 47, 30, 58, 39, 41, 11, 18, 52, 69, 59, 48, 40, 74, 60, 14]
G23 = [1, 32, 15, 12, 51, 10, 47, 28, 57, 17, 18, 69, 30, 34, 31, 3, 29, 49, 11, 68, 39, 20, 36, 60, 16]
G24 = [1, 32, 15, 57, 12, 10, 47, 30, 17, 51, 3, 60, 28, 34, 49, 69, 39, 20, 40, 2, 11, 18, 41, 45, 43]
G25_10000 = [1, 32, 78, 57, 17, 12, 47, 9, 22, 28, 4, 5, 24, 15, 8, 10, 49, 30, 7, 3, 26, 37, 14, 58, 2]
G26 = [1, 32, 15, 78, 12, 17, 28, 9, 57, 49, 40, 26, 51, 60, 10, 54, 8, 42, 24, 33, 47, 7, 58, 45, 53]
G27 = [1, 32, 12, 17, 15, 57, 7, 78, 2, 47, 8, 39, 10, 40, 9, 28, 30, 46, 51, 45, 55, 61, 22, 52, 37]
G28 = [1, 32, 15, 12, 57, 17, 40, 9, 8, 24, 28, 7, 10, 30, 69, 34, 26, 74, 47, 39, 67, 18, 43, 75, 23]
G29 = [1, 32, 12, 17, 57, 78, 28, 40, 8, 15, 7, 24, 20, 47, 59, 10, 30, 39, 68, 22, 74, 60, 3, 70, 58]
G30 = [1, 32, 17, 12, 7, 15, 8, 28, 78, 57, 74, 40, 3, 11, 47, 73, 52, 16, 20, 30, 54, 29, 58, 59, 51]
G31 = [1, 32, 12, 17, 7, 28, 15, 57, 8, 6, 30, 74, 10, 9, 23, 78, 3, 54, 40, 24, 55, 34, 47, 59, 44]
G32 = [1, 32, 78, 17, 12, 8, 47, 28, 74, 57, 21, 30, 49, 40, 7, 14, 10, 15, 59, 34, 19, 3, 67, 54, 45]
G33 = [1, 32, 17, 12, 15, 57, 78, 7, 8, 41, 69, 47, 5, 74, 28, 73, 10, 40, 26, 14, 22, 30, 9, 6, 49]
G34 = [1, 32, 9, 17, 12, 78, 57, 15, 22, 28, 30, 26, 7, 4, 8, 3, 46, 43, 36, 47, 40, 55, 14, 10, 69]
G35 = [1, 32, 78, 17, 12, 15, 57, 9, 40, 61, 2, 8, 51, 26, 4, 28, 58, 22, 60, 10, 30, 41, 24, 53, 19]
G36 = [1, 32, 78, 9, 17, 12, 15, 57, 49, 69, 7, 8, 53, 10, 73, 30, 40, 11, 27, 41, 19, 47, 48, 60, 2]
G37 = [1, 32, 12, 7, 17, 15, 57, 10, 8, 28, 40, 30, 47, 78, 44, 24, 22, 36, 39, 26, 54, 45, 55, 74, 9]
G38 = [1, 32, 17, 12, 8, 7, 57, 51, 10, 15, 6, 61, 28, 30, 58, 47, 55, 4, 42, 5, 22, 9, 40, 20, 27]
G39 = [1, 32, 8, 17, 54, 57, 78, 15, 51, 12, 30, 7, 28, 10, 40, 19, 52, 56, 44, 39, 53, 47, 3, 72, 21]
G40 = [1, 32, 7, 12, 17, 15, 78, 57, 47, 40, 10, 49, 24, 9, 28, 4, 74, 52, 30, 75, 6, 25, 3, 14, 29]


top_25_level = [G1_25000,G2,G3,G4,G6,G7,G8,G9,G10,G11,G12,G13,G14,G15,G16,G17,G18,G19,G20,G21,G22,G23,G24,G25_10000,G26,G27,G28,G29,G30,G31,G32,G33,G34,G35,G36,G37,G38,G39,G40]

'''
def unique_elements(lst):
    seen = set()
    unique_lst = []
    for item in lst:
        if item not in seen:
            unique_lst.append(item)
            seen.add(item)
    return unique_lst


top25_level = unique_elements(top_25_level)

# this is add those lists one by one, rather than append them together


print(top25_level)

'''

top25_level = [1, 32, 15, 57, 12, 10, 17, 47, 51, 18, 28, 3, 9, 39, 74, 11, 53, 2, 40, 30, 68, 69, 60, 54, 49, 7, 31, 26, 6, 48, 35, 14, 8, 34, 58, 20, 4, 65, 24, 70, 36, 27, 73, 52, 29, 67, 5, 72, 37, 78, 50, 45, 61, 75, 43, 42, 56, 44, 59, 41, 16, 22, 33, 46, 55, 23, 21, 19, 25]
# all the number which at least exist 1 time in those list above
'''
def frequency_sorted_elements(lst):
    # Count the frequency of each element in the list
    counter = Counter(lst)
    
    # Initialize an empty list to store the sorted items
    sorted_items = []
    
    # Iterate through the counter and store elements with their frequencies
    for item in counter.items():
        # Extract the element and its frequency
        element = item[0]
        frequency = item[1]
        # Append the element and its frequency to the sorted_items list
        sorted_items.append((element, frequency))
    
    # Sort the list: first by frequency in descending order, 
    # and then by the element value in ascending order
    sorted_items.sort(key=lambda x: (-x[1], x[0]))
    
    # Extract the elements from the sorted list
    sorted_elements = []
    for item in sorted_items:
        sorted_elements.append(item[0])
    
    return sorted_elements

top25_level_with_highest_frequency = frequency_sorted_elements(top_25_level)
'''
top25_level_with_highest_frequency = [12, 15, 17, 32, 57, 1, 10, 28, 47, 30, 40, 51, 7, 49, 69, 39, 18, 8, 60, 3, 9, 78, 24, 58, 74, 26, 20, 29, 22, 34, 54, 45, 2, 4, 53, 5, 11, 14, 36, 41, 52, 59, 31, 68, 6, 35, 43, 48, 67, 42, 44, 55, 73, 16, 19, 37, 61, 27, 70, 75, 21, 23, 46, 50, 56, 72, 25, 33, 65]
# sort by the frequency which each number show in those list above, from left to right (12 has highest frequency)


def most_frequent_values(nested_list):
    if not nested_list:
        return []

    # Initialize a list of dictionaries to count occurrences of each index
    count_dicts = [defaultdict(int) for _ in range(len(nested_list[0]))]

    # Count occurrences of each value at each index
    for sublist in nested_list:
        for index, value in enumerate(sublist):
            count_dicts[index][value] += 1

    # Find the most frequent value at each index
    result = []
    for count_dict in count_dicts:
        most_frequent_value = max(count_dict, key=count_dict.get)
        result.append(most_frequent_value)

    return result

heaviest_features_seq = most_frequent_values(top_25_level)
heaviest_features_seq = [1, 32, 15, 12, 12, 17, 10, 28, 51, 28, 40, 60, 28, 40, 49, 69, 49, 39, 40, 29, 48, 37, 24, 60, 49]

unique_heaviest_features_seq = []
[unique_heaviest_features_seq.append(x) for x in heaviest_features_seq if x not in unique_heaviest_features_seq]
unique_heaviest_features_seq = [1, 32, 15, 12, 17, 10, 28, 51, 40, 60, 49, 69, 39, 29, 48, 37, 24] # len:17


'''
So far, we know the following information

First, after 40 tests, each test is trained for 10 cycles, each cycle is tens of thousands of rounds, 
and the average is taken to see which features have the largest proportion. This can be referred to unique_heaviest_features_seq

On the other hand, we also know which features appear most frequently in the so-called "top 25 weighted" list,
which can be found in top25_level_with_highest_frequency

Tip: For numbers with the same frequency, the list is sorted by value.

'''

def difference(A, B):
    # transfer list to set
    set_A = set(A)
    result = [x for x in B if x not in set_A]
    return result

different = difference(unique_heaviest_features_seq, top25_level_with_highest_frequency) # len 52
random.shuffle(different)
different = [42, 56, 70, 36, 43, 2, 8, 7, 33, 6, 19, 27, 41, 11, 22, 46, 20, 54, 30, 44, 65, 35, 4, 14, 31, 34, 72, 67, 75, 45, 47, 59, 53, 3, 57, 52, 16, 25, 9, 50, 18, 68, 74, 23, 61, 55, 73, 26, 58, 78, 21, 5]
'''
I retain 17 features in unique_heaviest_features_seq as core features to ensure accuracy, 
and randomly select 8 features from difference to make up 25 features, 
to ensure that each model does not completely rely on the same features for prediction.

'''





F1 = [1, 32, 15, 12, 17, 10, 28, 51, 40, 60, 49, 69, 39, 29, 48, 37, 24, 42, 56, 70, 36, 43, 2, 8, 7]
F2 = [1, 32, 15, 12, 17, 10, 28, 51, 40, 60, 49, 69, 39, 29, 48, 37, 24, 33, 6, 19, 27, 41, 11, 22, 46]
F3 = [1, 32, 15, 12, 17, 10, 28, 51, 40, 60, 49, 69, 39, 29, 48, 37, 24, 20, 54, 30, 44, 65, 35, 4, 14]
F4 = [1, 32, 15, 12, 17, 10, 28, 51, 40, 60, 49, 69, 39, 29, 48, 37, 24, 31, 34, 72, 67, 75, 45, 47, 59]
F5 = [1, 32, 15, 12, 17, 10, 28, 51, 40, 60, 49, 69, 39, 29, 48, 37, 24, 53, 3, 57, 52, 16, 25, 9, 50]
F6 = [1, 32, 15, 12, 17, 10, 28, 51, 40, 60, 49, 69, 39, 29, 48, 37, 24, 18, 68, 74, 23, 61, 55, 73, 26]

F7 = []

for i in F6:
    F7.append(heaviest_features[i])
F7 = F7+['Label']
print(F7)

F1 = ['responp', 'bwd_pkts_payload.min', 'bwd_header_size_min', 'fwd_header_size_min',
      'flow_FIN_flag_count', 'down_up_ratio', 'fwd_pkts_payload.max', 'bwd_iat.std',
      'flow_pkts_payload.avg', 'bwd_subflow_bytes', 'bwd_iat.tot', 'active.tot',
      'flow_pkts_payload.tot', 'fwd_pkts_payload.tot', 'bwd_iat.max', 'flow_pkts_payload.min',
      'bwd_URG_flag_count', 'fwd_iat.min', 'flow_iat.std', 'active.avg', 'bwd_pkts_payload.std',
      'fwd_iat.max', 'flow_duration', 'bwd_pkts_per_sec', 'fwd_pkts_per_sec','Label'] # given to SVM

F2 = ['responp', 'bwd_pkts_payload.min', 'bwd_header_size_min', 'fwd_header_size_min', 'flow_FIN_flag_count',
      'down_up_ratio', 'fwd_pkts_payload.max', 'bwd_iat.std', 'flow_pkts_payload.avg', 'bwd_subflow_bytes',
      'bwd_iat.tot', 'active.tot', 'flow_pkts_payload.tot', 'fwd_pkts_payload.tot', 'bwd_iat.max',
      'flow_pkts_payload.min', 'bwd_URG_flag_count', 'bwd_pkts_payload.max', 'bwd_data_pkts_tot',
      'flow_RST_flag_count', 'fwd_pkts_payload.min', 'flow_pkts_payload.std', 'fwd_header_size_tot',
      'flow_ACK_flag_count', 'fwd_iat.std', 'Label'] # given to softmax

F3 = ['responp', 'bwd_pkts_payload.min', 'bwd_header_size_min', 'fwd_header_size_min', 'flow_FIN_flag_count',
      'down_up_ratio', 'fwd_pkts_payload.max', 'bwd_iat.std', 'flow_pkts_payload.avg', 'bwd_subflow_bytes',
      'bwd_iat.tot', 'active.tot', 'flow_pkts_payload.tot', 'fwd_pkts_payload.tot', 'bwd_iat.max',
      'flow_pkts_payload.min', 'bwd_URG_flag_count', 'fwd_PSH_flag_count', 'flow_iat.tot', 'fwd_pkts_payload.avg',
      'fwd_iat.tot', 'fwd_bulk_rate', 'bwd_pkts_payload.avg', 'bwd_pkts_tot', 'bwd_header_size_tot', 'Label'] # given to knn

F4 = ['responp', 'bwd_pkts_payload.min', 'bwd_header_size_min', 'fwd_header_size_min', 'flow_FIN_flag_count',
      'down_up_ratio', 'fwd_pkts_payload.max', 'bwd_iat.std', 'flow_pkts_payload.avg', 'bwd_subflow_bytes',
      'bwd_iat.tot', 'active.tot', 'flow_pkts_payload.tot', 'fwd_pkts_payload.tot', 'bwd_iat.max', 'flow_pkts_payload.min',
      'bwd_URG_flag_count', 'fwd_pkts_payload.std', 'bwd_pkts_payload.tot', 'idle.min', 'active.min',
      'idle.avg', 'fwd_iat.avg', 'bwd_iat.min', 'bwd_subflow_pkts', 'Label']

F5 = ['responp', 'bwd_pkts_payload.min', 'bwd_header_size_min', 'fwd_header_size_min', 'flow_FIN_flag_count', 
      'down_up_ratio', 'fwd_pkts_payload.max', 'bwd_iat.std', 'flow_pkts_payload.avg', 'bwd_subflow_bytes', 
      'bwd_iat.tot', 'active.tot', 'flow_pkts_payload.tot', 'fwd_pkts_payload.tot', 'bwd_iat.max', 'flow_pkts_payload.min', 
      'bwd_URG_flag_count', 'flow_iat.max', 'fwd_pkts_tot', 'payload_bytes_per_second', 'flow_iat.min', 'bwd_header_size_max', 
      'flow_CWR_flag_count', 'flow_pkts_per_sec', 'bwd_iat.avg', 'Label'] # given to LR

F6 = ['responp', 'bwd_pkts_payload.min', 'bwd_header_size_min', 'fwd_header_size_min', 'flow_FIN_flag_count', 'down_up_ratio',
      'fwd_pkts_payload.max', 'bwd_iat.std', 'flow_pkts_payload.avg', 'bwd_subflow_bytes', 'bwd_iat.tot', 'active.tot',
      'flow_pkts_payload.tot', 'fwd_pkts_payload.tot', 'bwd_iat.max', 'flow_pkts_payload.min', 'bwd_URG_flag_count',
      'flow_SYN_flag_count', 'active.max', 'idle.tot', 'fwd_URG_flag_count', 'fwd_bulk_bytes', 'flow_iat.avg', 'idle.max',
      'flow_ECE_flag_count', 'Label']