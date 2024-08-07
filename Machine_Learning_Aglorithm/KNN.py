import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from collections import Counter

def KNN_preprocess_training(training_set, features_list):
    
    heaviest_features = ['bwd_pkts_payload.min', 'responp', 'flow_pkts_payload.avg', 'bwd_iat.tot', 'flow_pkts_per_sec', 'payload_bytes_per_second', 'idle.avg', 
     'fwd_pkts_payload.max', 'fwd_pkts_tot', 'flow_iat.tot', 'fwd_iat.tot', 'fwd_pkts_payload.avg', 'fwd_iat.min', 'idle.tot', 'fwd_header_size_tot', 'bwd_data_pkts_tot', 
     'flow_RST_flag_count', 'bwd_header_size_max', 'bwd_iat.std', 'fwd_pkts_payload.min', 'flow_pkts_payload.max', 'flow_FIN_flag_count', 'Label']
    
    indices = {}
    for feature in heaviest_features:
        if feature in features_list:
            index = features_list.index(feature)
            indices[feature] = index
    
    new_training_set = []
    for samples in training_set:
        new_sample = []
        for feature in heaviest_features:
            if feature in indices:
                new_sample.append(samples[indices[feature]])
        new_training_set.append(new_sample)
    # with this heaviest_features, all the avlue in sample should be int or float
    
    for i in new_training_set:
        for j in i:
            if not isinstance(j, (int, float)):
                j = 0
    # change all nun-numeric value to 0
    
    X_train = [] # 2D array for X_train [[sample1],[sample2],..,[samplen]]
    y_train = [] # 1D list for label [0,0,...,1,1]
    
    for i in new_training_set:
        X_train.append(i[:-1])
        y_train.append(i[-1])

    scaler = StandardScaler()
    scaler.fit(X_train)
    
    means = scaler.mean_
    means = means.tolist()
    stds = scaler.scale_
    stds = stds.tolist()
    scale_factors = [means,stds]
    
    for sample in X_train:
        for i in range(0,len(sample)):
            sample[i] = (sample[i] - scale_factors[0][i]) / scale_factors[1][i]
    
    return X_train, y_train, scale_factors, heaviest_features


def KNN_train(X_train, y_train):
    """
    Train the K-Nearest Neighbors model by building a KDTree from the training data and associating labels.

    Args:
        X_train (numpy.ndarray): Training data features.
        y_train (numpy.ndarray): Training data labels.

    Returns:
        list: The trained KDTree model and the training data labels.
    """
    print("KNN training started")
    kdtree = KDTree(X_train)
    return [kdtree, y_train]


def KNN_each_test_sample_preprocess(test_sample, scale_factors, features_list, heaviest_features):
    indices = {}
    for feature in heaviest_features:
        if feature in features_list:
            index = features_list.index(feature)
            indices[feature] = index
    
    testing_sample = []
    for feature in heaviest_features:
        if feature in indices:
            testing_sample.append(test_sample[indices[feature]])
    
    
    
    for i in range(0,len(scale_factors[0])):
        testing_sample[i] = (testing_sample[i] - scale_factors[0][i]) / scale_factors[1][i]
    
    return testing_sample
    

def KNN_predict(test_sample, kdtree_list, k=3):


    # Step 4: Retrieve KDTree and training labels
    kdtree = kdtree_list[0]
    y_train = kdtree_list[1]

    # Step 5: Query KDTree to find k nearest neighbors
    distances, indices = kdtree.query(test_sample, k=k)

    # Step 6: Determine the most common label among the nearest neighbors
    nearest_labels = [y_train[idx].item() if isinstance(y_train[idx], np.ndarray) else y_train[idx] for idx in indices]
    most_common_label = Counter(nearest_labels).most_common(1)[0][0]

    return int(most_common_label)



if __name__ == '__main__':

    # Example usage:
    data = [
        [1.0, 2.0, 0], 
        [1.5, 1.8, 0], 
        [5.0, 8.0, 1], 
        [8.0, 8.0, 1], 
        [1.2, 0.5, 0], 
        [9.0, 11.0, 1]
    ]
    features_list = ['feature1', 'feature2']
    X, y, numeric_features, scale_factors = KNN_preprocess_data(data, features_list)
    kdtree_with_labels = KNN_train(X, y)
    test_sample = np.array([[2.0, 3.0]])  # Example test sample
    predicted_label = KNN_predict(kdtree_with_labels, test_sample, k=3, scale_factors=scale_factors)
    print("Predicted label:", predicted_label)
