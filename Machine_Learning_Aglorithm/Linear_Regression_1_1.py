import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numbers


def LR_preprocess_data(training_set, features_list):
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
      'bwd_iat.tot', 'active.tot', 'flow_pkts_payload.tot', 'fwd_pkts_payload.tot', 'bwd_iat.max', 'flow_pkts_payload.min', 
      'bwd_URG_flag_count', 'flow_iat.max', 'fwd_pkts_tot', 'payload_bytes_per_second', 'flow_iat.min', 'bwd_header_size_max', 
      'flow_CWR_flag_count', 'flow_pkts_per_sec', 'bwd_iat.avg', 'Label']
    
    #heaviest_features = ['responp', 'bwd_pkts_payload.min','fwd_header_size_min',
    #  'down_up_ratio', 'fwd_pkts_payload.max', 'bwd_iat.std','bwd_subflow_bytes', 
    #  'bwd_iat.tot', 'fwd_pkts_payload.tot', 'bwd_iat.max',
    #  'bwd_URG_flag_count', 'flow_iat.max','flow_iat.min', 
    #  'flow_pkts_per_sec', 'bwd_iat.avg', 'Label']

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



class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def train_linear_regression(X, y, epochs=10000, learning_rate=0.05):
    """
    Train a linear regression model using gradient descent.

    This function trains a linear regression model using the mean squared error loss and 
    stochastic gradient descent optimizer from the PyTorch library.

    Parameters:
    - X (array-like): Input features data as a 2D array-like structure.
    - y (array-like): Target variable data as a 1D array-like structure.
    - epochs (int): Number of training epochs. Default is 10,000.
    - learning_rate (float): Learning rate for the optimizer. Default is 0.05.

    Steps:
    - The function first converts input arrays X and y into PyTorch tensors.
    - It initializes a linear regression model, a loss function (MSE), and an optimizer (SGD).
    - The training loop runs for the specified number of epochs. In each epoch:
        * The model performs a forward pass (calculates predictions).
        * The loss is computed using the predictions and actual targets.
        * The optimizer updates the model parameters based on the loss gradient.
    - Every 1000 epochs, it prints the current epoch number and the loss.

    Returns:
    - numpy.ndarray: The trained weights of the model after all epochs are completed.

    Notes:
    - The training duration is printed at the end of the training process.
    """
    #print("linear regression training start")
    # Convert to tensor
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    y = y.unsqueeze(1)

    # Initialize model, loss function, and optimizer
    input_dim = X.shape[1]
    model = LinearRegressionModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss for every 100 epochs
        #if (epoch + 1) % 1000 == 0:
            #print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Extract weights
    weights = model.linear.weight.data.numpy().flatten()

    return weights
# return a dictionary of weight, key for features name and value for numberic weight

def LR_each_test_sample_preprocess(test_sample, scale_factors, features_list, heaviest_features):
    """
    Preprocesses a test sample by selecting and scaling important features.

    Args:
        test_sample (list): The feature values of the test sample.
        scale_factors (list of lists): Contains the means and standard deviations for scaling.
        features_list (list): The list of all features.
        heaviest_features (list): The most important features to select and process.

    Returns:
        numpy.ndarray: The processed and scaled test sample.
    """
    indices = {}
    count = 0
    for feature in heaviest_features:
        if feature in features_list:
            count += 1
            index = features_list.index(feature)
            indices[feature] = index
    #print("count:",count)
    #print("feature list?:",features_list)
    #print(indices)
    testing_sample = []
    for feature in heaviest_features:
        if feature in indices:
            testing_sample.append(test_sample[indices[feature]])
    #print("length of testing sample:",len(testing_sample))
    
    for i in range(0,len(testing_sample)):
        if not isinstance(testing_sample[i],numbers.Number):# (int, float)):
            testing_sample[i] = 0
    
    for i in range(0,len(scale_factors[0])):
        testing_sample[i] = (testing_sample[i] - scale_factors[0][i]) / scale_factors[1][i]
    
    testing_sample = np.array(testing_sample)
    
    return testing_sample
# return a preprocessed sample for testing



def linear_regression_predict(test_sample, weights):
    """
    Predict the label for a test sample using a trained linear regression model.

    This function calculates a prediction score by multiplying each element of the test sample
    by its corresponding weight and summing the results. The prediction score is then thresholded
    to classify the sample as either 1 or 0.

    Parameters:
    - test_sample (array-like): A single test sample containing features, should have the same
                                number of elements as there are weights.
    - weights (array-like): The weights of the trained linear regression model, as a 1D array-like
                            structure.

    Returns:
    - int: The predicted label for the test sample, either 1 (positive class) or 0 (negative class).

    Notes:
    - The threshold for classification is set at -0.1. If the prediction score is greater than or equal to -0.1,
      the function returns 1, otherwise it returns 0.
    """
    predict_score = 0
    for i in range(0,len(weights)):
        predict_score += weights[i] * test_sample[i]

    predict_label = 1 if predict_score >= -0.1 else 0

    return predict_label




def one_step_LR(Data, feature_list, epochs=10000, learning_rate=0.001):
    X_train, y_train, scale_factors, heaviest_features = LR_preprocess_data(Data, feature_list)
    weight_result = train_linear_regression(X_train, y_train, epochs, learning_rate)

    return scale_factors, weight_result, heaviest_features
    # scale_factors: 2D-array
    # weight_result: dictionary
