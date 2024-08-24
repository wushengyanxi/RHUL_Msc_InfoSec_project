import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from collections import Counter
import numbers

def KNN_preprocess_training(training_set, features_list):
    """
    Preprocess the training data for a K-Nearest Neighbors (KNN) model by selecting, reordering, 
    and scaling features.

    This function selects the most relevant features from the training data, replaces non-numeric 
    values with zeros, and standardizes the selected features by removing the mean and scaling to 
    unit variance.

    Parameters:
    - training_set (list of lists): The complete training dataset, where each inner list 
                                    represents a sample with features and a label.
    - features_list (list of str): A list of feature names corresponding to the columns 
                                   in each sample.

    Returns:
    - X_train (list of lists): The processed feature data for training, with each inner list 
                               representing a sample and containing only the selected features.
    - y_train (list): The list of labels corresponding to each sample in `X_train`.
    - scale_factors (list of lists): A list containing two lists: the means and standard deviations 
                                     used for scaling each feature.
    - heaviest_features (list of str): The list of selected features that were used to create `X_train`.

    Notes:
    - The function first selects a predefined set of "heaviest" features from the input feature list.
    - Non-numeric values in the selected features are replaced with zeros to ensure compatibility 
      with numerical operations.
    - The features in `X_train` are standardized using `StandardScaler` from scikit-learn, with 
      the means and standard deviations saved in `scale_factors` for future use.
    """
    heaviest_features = ['responp', 'bwd_pkts_payload.min', 'bwd_header_size_min', 'fwd_header_size_min', 'flow_FIN_flag_count',
      'down_up_ratio', 'fwd_pkts_payload.max', 'bwd_iat.std', 'flow_pkts_payload.avg', 'bwd_subflow_bytes',
      'bwd_iat.tot', 'active.tot', 'flow_pkts_payload.tot', 'fwd_pkts_payload.tot', 'bwd_iat.max',
      'flow_pkts_payload.min', 'bwd_URG_flag_count', 'fwd_PSH_flag_count', 'flow_iat.tot', 'fwd_pkts_payload.avg',
      'fwd_iat.tot', 'fwd_bulk_rate', 'bwd_pkts_payload.avg', 'bwd_pkts_tot', 'bwd_header_size_tot', 'Label']
    
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
        for x in range(0,len(i)):
            if not isinstance(i[x], numbers.Number): #(int, float)):
                i[x] = 0
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
    """
    Prepares a test sample for KNN by selecting and scaling important features.

    This function selects specific features from the test sample, ensures they are numeric, 
    and then scales them using provided scale factors.

    Args:
        test_sample (list): The feature values of the test sample.
        scale_factors (list of lists): Contains the means and standard deviations for scaling.
        features_list (list): The list of all features.
        heaviest_features (list): The most important features to select and process.

    Returns:
        list: The processed and scaled values of the important features.
    """
    indices = {}
    for feature in heaviest_features:
        if feature in features_list:
            index = features_list.index(feature)
            indices[feature] = index
    
    testing_sample = []
    for feature in heaviest_features:
        if feature in indices:
            testing_sample.append(test_sample[indices[feature]])
    
    for i in range(0,len(testing_sample)):
        if not isinstance(testing_sample[i],numbers.Number):# (int, float)):
            testing_sample[i] = 0
    
    for i in range(0,len(scale_factors[0])):
        testing_sample[i] = (testing_sample[i] - scale_factors[0][i]) / scale_factors[1][i]
    
    return testing_sample
    

def KNN_predict(test_sample, kdtree_list, k=3):
    """
    Predicts the label for a test sample using the k-nearest neighbors (KNN) algorithm.

    This function retrieves a KDTree and associated training labels from `kdtree_list`, 
    queries the KDTree to find the `k` nearest neighbors to the `test_sample`, and then 
    determines the most common label among those neighbors as the predicted label.

    Args:
        test_sample (array-like): The feature vector of the sample to be classified.
        kdtree_list (list): A list containing the KDTree object and the corresponding 
                            training labels. The KDTree should be at index 0, and the 
                            training labels should be at index 1.
        k (int, optional): The number of nearest neighbors to consider. Default is 3.

    Returns:
        int: The predicted label for the test sample, based on the majority label 
             of the `k` nearest neighbors.
    """
    kdtree = kdtree_list[0]
    y_train = kdtree_list[1]

    distances, indices = kdtree.query(test_sample, k=k)

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
