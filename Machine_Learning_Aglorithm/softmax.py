import numpy as np
from sklearn.preprocessing import StandardScaler
import numbers

def Softmax_preprocess_training(training_set, features_list):
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
      'flow_pkts_payload.min', 'bwd_URG_flag_count', 'bwd_pkts_payload.max', 'bwd_data_pkts_tot',
      'flow_RST_flag_count', 'fwd_pkts_payload.min', 'flow_pkts_payload.std', 'fwd_header_size_tot',
      'flow_ACK_flag_count', 'fwd_iat.std', 'Label']
    

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
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    return X_train, y_train, scale_factors, heaviest_features

# Softmax function
def softmax(z):
    """
    Compute the softmax of each row of the input array.

    The softmax function transforms the input values into probabilities that sum to one. 
    This is typically used in classification tasks where the output represents the probability 
    distribution over different classes.

    Parameters:
    - z (array-like): The input array of logits (raw predictions). It can be either a 1D array 
                      representing a single sample or a 2D array where each row corresponds to 
                      a sample.

    Returns:
    - numpy.ndarray: The array of softmax probabilities corresponding to the input logits.
                     If the input is a 1D array, the output will be a 2D array with one row.
    """
    if z.ndim == 1:
        z = np.expand_dims(z, axis=0)
    
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)

# Cross Entropy Loss Function
def cross_entropy_loss(y_true, y_pred):
    """
    Compute the cross-entropy loss between the true labels and the predicted probabilities.

    Cross-entropy loss is commonly used in classification tasks to measure the performance 
    of a model's output. It quantifies the difference between the true labels and the predicted 
    probability distributions.

    Parameters:
    - y_true (array-like): True class labels as a 1D array of integers, where each integer 
                           corresponds to the class index.
    - y_pred (array-like): Predicted probabilities as a 2D array, where each row contains 
                           the predicted probability distribution for a sample.

    Returns:
    - float: The average cross-entropy loss over all samples.
    """
    n_samples = y_true.shape[0]
    logp = - np.log(y_pred[range(n_samples), y_true])
    loss = np.sum(logp) / n_samples
    return loss

# Computing Gradients
def compute_gradient(X, y_true, y_pred):
    """
    Compute the gradient of the loss with respect to the model's weights.

    The gradient indicates the direction and magnitude of change required to reduce the loss. 
    It is used in the optimization process to update the model's weights.

    Parameters:
    - X (array-like): Input features as a 2D array, where each row corresponds to a sample.
    - y_true (array-like): True labels in one-hot encoded format, as a 2D array.
    - y_pred (array-like): Predicted probabilities as a 2D array, where each row contains 
                           the predicted probability distribution for a sample.

    Returns:
    - numpy.ndarray: The gradient of the loss with respect to the weights, 
                     with the same shape as the weights matrix.
    """
    n_samples = X.shape[0]
    grad = X.T.dot(y_pred - y_true) / n_samples
    return grad

# Training Function
def softmax_train(X, y, learning_rate=0.4, epochs=30000):
    """
    Train a softmax regression model using gradient descent.

    This function trains a softmax regression model to classify the input data into multiple 
    classes. The training process minimizes the cross-entropy loss using gradient descent.

    Parameters:
    - X (array-like): Input features as a 2D array, where each row corresponds to a sample.
    - y (array-like): True class labels as a 1D array of integers, where each integer corresponds 
                      to the class index.
    - learning_rate (float): The learning rate for the gradient descent optimizer. Default is 0.01.
    - epochs (int): The number of training iterations. Default is 30,000.

    Returns:
    - numpy.ndarray: The trained weights of the softmax regression model.
    """
    #print("softmax training start")
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    y = y.astype(int).ravel()    
    # Initialize weights
    W = np.random.randn(n_features, n_classes)
    
    # Convert labels to one-hot encoding
    y_one_hot = np.zeros((n_samples, n_classes))
    y_one_hot[np.arange(n_samples), y] = 1
    
    for epoch in range(epochs):
        # Calculate predicted values
        z = X.dot(W)
        y_pred = softmax(z)
        
        # Calculating Losses
        loss = cross_entropy_loss(y, y_pred)
        
        # Computing Gradients
        dW = compute_gradient(X, y_one_hot, y_pred)
        
        # Update weights
        W -= learning_rate * dW
        
        #if epoch % 1000 == 0:
            #print(f'Epoch {epoch}, Loss: {loss}')
    
    return W

def softmax_each_test_sample_preprocess(test_sample, scale_factors, features_list, heaviest_features):
    """
    Preprocesses a test sample by selecting and scaling important features for a softmax model.

    Args:
        test_sample (list): The feature values of the test sample.
        scale_factors (list of lists): Contains the means and standard deviations for scaling.
        features_list (list): The list of all features.
        heaviest_features (list): The most important features to select and process.

    Returns:
        numpy.ndarray: The processed and scaled test sample.
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

def softmax_predict(X, W):
    """
    Predict class labels for given input data using a trained softmax regression model.

    This function computes the class scores by multiplying the input features by the model weights, 
    then applies the softmax function to convert these scores into probabilities. The class with 
    the highest probability is selected as the predicted label.

    Parameters:
    - X (array-like): Input features data as a 2D array (n_samples, n_features).
    - W (array-like): Trained weights of the softmax regression model as a 2D array 
                      (n_features, n_classes).

    Returns:
    - numpy.ndarray: Predicted class labels as a 1D array (n_samples,).
                     Each entry is an integer corresponding to the predicted class for the sample.
    """
    z = X.dot(W)
    y_pred = softmax(z)
    return np.argmax(y_pred, axis=1)

