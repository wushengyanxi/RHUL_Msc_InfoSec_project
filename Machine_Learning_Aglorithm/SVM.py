import numpy as np
import numbers
from sklearn.preprocessing import StandardScaler
from sklearn import svm



def svm_preprocess_training(training_set, features_list):
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
    heaviest_features = ['responp', 'bwd_pkts_payload.min',
    'bwd_header_size_min', 'fwd_header_size_min',
    'flow_FIN_flag_count', 'down_up_ratio',
    'fwd_pkts_payload.max', 'bwd_iat.std',
    'flow_pkts_payload.avg', 'bwd_subflow_bytes',
    'bwd_iat.tot', 'active.tot',
    'flow_pkts_payload.tot', 'fwd_pkts_payload.tot',
    'bwd_iat.max', 'flow_pkts_payload.min',
    'bwd_URG_flag_count', 'fwd_iat.min', 'flow_iat.std',
    'active.avg', 'bwd_pkts_payload.std',
    'fwd_iat.max', 'flow_duration', 
    'bwd_pkts_per_sec', 'fwd_pkts_per_sec','Label']
    
    #heaviest_features = ['responp', 'bwd_pkts_payload.min', 'fwd_header_size_min',
    #  'down_up_ratio', 'fwd_pkts_payload.max', 'bwd_iat.std',
    #   'bwd_subflow_bytes', 'bwd_iat.tot',
    #  'fwd_pkts_payload.tot', 'bwd_iat.max',
    #  'bwd_URG_flag_count', 'fwd_iat.min', 'flow_iat.std', 'active.avg', 'bwd_pkts_payload.std',
    #  'fwd_iat.max', 'flow_duration', 'bwd_pkts_per_sec', 'fwd_pkts_per_sec','Label']
    
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
            if not isinstance(i[x], numbers.Number):#(int, float)):
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
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    return X_train, y_train, scale_factors, heaviest_features

def svm_train(X, y, kernel='linear', C=1.0, max_iter=60000):
    """
    Train an SVM (Support Vector Machine) model.

    This function trains a Support Vector Machine (SVM) model using the specified kernel and 
    regularization parameters. It leverages the `svm.SVC` class from the scikit-learn library.

    Parameters:
    - X (array-like): Feature data with shape (n_samples, n_features).
    - y (array-like): Target labels with shape (n_samples,).
    - kernel (str): The kernel type to be used in the algorithm, default is 'linear'.
                    Other options include 'poly', 'rbf', 'sigmoid', etc.
    - C (float): Regularization parameter, default is 1.0. The strength of the regularization 
                 is inversely proportional to C.
    - max_iter (int): The maximum number of iterations for training, default is 60,000. 
                      If -1, no limit on iterations.

    Returns:
    - svm.SVC: The trained SVM model.
    """
    #print("SVM training start")
    model = svm.SVC(kernel=kernel, C=C, max_iter=max_iter)
    model.fit(X, y)
    return model

def svm_predict(model, X):
    """
    Make predictions using a trained SVM model.

    This function uses a trained SVM model to predict the labels for a given set of input features.

    Parameters:
    - model (svm.SVC): The trained SVM model.
    - X (array-like): Feature data with shape (n_samples, n_features).
                      If a single sample is provided, ensure it is reshaped to have 2 dimensions.

    Returns:
    - numpy.ndarray: Predicted labels with shape (n_samples,).
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
        
    return model.predict(X)

def svm_each_test_sample_preprocess(test_sample, scale_factors, features_list, heaviest_features):
    """
    Preprocesses a test sample by selecting and scaling important features for an SVM model.

    Args:
        test_sample (list): The feature values of the test sample.
        scale_factors (list of lists): Contains the means and standard deviations for scaling.
        features_list (list): The list of all available features.
        heaviest_features (list): The most important features to select and process.

    Returns:
        numpy.ndarray: The processed and scaled test sample ready for SVM classification.
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
    
    testing_sample = np.array(testing_sample)
    
    return testing_sample




