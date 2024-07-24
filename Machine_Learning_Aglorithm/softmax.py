import numpy as np
from sklearn.preprocessing import StandardScaler

def Softmax_preprocess_data(Data, features_list):
    Data = np.array(Data, dtype=object)

    # Determine numeric features and filter data
    numeric_features = []
    numeric_data = []

    for feature in range(Data.shape[1] - 1):  # Exclude the last column (labels)
        if all(isinstance(x, (int, float)) for x in Data[:, feature]):
            numeric_features.append(feature)
            numeric_data.append(Data[:, feature].astype(float))
        else:
            numeric_data.append(np.zeros(Data.shape[0]))  # Set non-numeric features to 0

    # Include labels in numeric data
    labels = Data[:, -1].astype(float)
    numeric_data.append(labels)

    # Convert to numpy array
    numeric_data = np.array(numeric_data).T

    # Replace NaN and Inf values with column mean
    for i in range(numeric_data.shape[1]):
        col = numeric_data[:, i]
        valid_mask = np.isfinite(col)
        if np.sum(valid_mask) == 0:
            raise ValueError(f"All values are invalid in column {i}")
        mean_value = col[valid_mask].mean()
        col[~valid_mask] = mean_value

    # Standardize features
    scaler = StandardScaler()
    X = numeric_data[:, :-1]
    y = numeric_data[:, -1].reshape(-1, 1)  # Training sample labels
    X = scaler.fit_transform(X)  # The scaled training samples

    # Get scale factors for each feature
    scale_factors = [[features_list[i], 1 / scaler.scale_[i]] for i in numeric_features]
    return X, y, numeric_features, scale_factors

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    # Ensure that softmax calculations are stable and avoid overflow
    # Calculate the exponential of the above results element by element, that is, calculate each element of e
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    # Sum each row of the above exponential matrix while keeping the dimension unchanged
    # Divide each element of exp_z by the sum of the row in which the element is located to obtain the softmax probability
    
def cross_entropy_loss(y_true, y_pred):
    """
    Computes the cross-entropy loss between the true labels and the predicted probabilities.

    Cross-entropy loss, also known as log loss, measures the performance of a classification model
    whose output is a probability value between 0 and 1. The loss increases as the predicted probability
    diverges from the actual label.

    Args:
        y_true (numpy.ndarray): One-hot encoded true labels of shape (num_samples, num_classes).
        y_pred (numpy.ndarray): Predicted probabilities of shape (num_samples, num_classes).

    Returns:
        float: The cross-entropy loss value.

    Note:
        A small value (1e-8) is added to `y_pred` to avoid taking the logarithm of zero,
        which would result in an undefined value.
    """
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

def train_softmax_classifier(X, y, learning_rate=0.01, epochs=1000):
    """
    Train a softmax classifier from scratch.

    Args:
        X (numpy.ndarray): Standardized features of the dataset.
        y (numpy.ndarray): Labels of the dataset.
        learning_rate (float): Learning rate for gradient descent.
        epochs (int): Number of iterations for training.

    Returns:
        weight (numpy.ndarray): Trained weights of the softmax classifier.
        bias (numpy.ndarray): Trained bias of the softmax classifier.
    """
    num_samples, num_features = X.shape
    num_classes = len(np.unique(y))
    weights = np.random.randn(num_features, num_classes)
    bias = np.zeros((1, num_classes))

    # One-hot encode labels
    y_one_hot = np.eye(num_classes)[y.astype(int).flatten()]

    # Gradient descent
    for epoch in range(epochs):
        logits = np.dot(X, weights) + bias # wx+b = log-odd
        probs = softmax(logits)
        loss = cross_entropy_loss(y_one_hot, probs)

        gradient_w = np.dot(X.T, (probs - y_one_hot)) / num_samples
        gradient_b = np.mean(probs - y_one_hot, axis=0, keepdims=True)

        weights -= learning_rate * gradient_w
        bias -= learning_rate * gradient_b

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return weights, bias

def classify_softmax(test_sample, weights, bias, features_list, numeric_features, scale_factors):
    """
    Classify samples using the trained softmax classifier.

    Args:
        test_sample (numpy.ndarray): Standardized features of the test dataset.
        weights (numpy.ndarray): Trained weights of the softmax classifier.
        bias (numpy.ndarray): Trained bias of the softmax classifier.
        features_list (list): List of feature names.
        numeric_features (list): List of indices of numeric features.
        scale_factors (list): List of scaling factors for numeric features.
        

    Returns:
        numpy.ndarray: Predicted class labels for the test dataset.
    """
    # Step 2: Set non-numeric features to 0
    for i, feature in enumerate(features_list[:-1]):  # Exclude the last column (label)
        if feature not in numeric_features:
            test_sample[i] = 0

    # Step 3: Scale numeric features
    for feature, scale in scale_factors:
        if feature in features_list:
            index = features_list.index(feature)
            test_sample[index] *= scale

    logits = np.dot(test_sample, weights) + bias
    probs = softmax(logits)
    return np.argmax(probs, axis=1)

