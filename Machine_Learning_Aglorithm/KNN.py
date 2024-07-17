import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from collections import Counter

def KNN_preprocess_data(Data, features_list):
    """
    Preprocesses the input data for machine learning tasks by converting all
    features to numeric format, handling missing values, and standardizing the features.

    The function first checks each feature to determine if it is numeric. Non-numeric
    features are set to zero. Then, it handles any missing or infinite values in the
    numeric data by replacing them with the mean of their respective columns. Finally,
    it standardizes the numeric features using a StandardScaler.

    Args:
        Data (list or numpy.ndarray): The dataset to preprocess, where the last column
            is assumed to be the labels.

    Returns:
        tuple:
            - numpy.ndarray: The standardized features of the dataset.
            - numpy.ndarray: The labels of the dataset.
            - list: The indices of the columns that were considered numeric.
            - list: The scale factors for each feature.
    """
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
    y = numeric_data[:, -1].reshape(-1, 1) # Training sample labels
    X = scaler.fit_transform(X) # The scaled training samples

    # Get scale factors for each feature
    scale_factors = [[features_list[i], 1 / scaler.scale_[i]] for i in numeric_features] 
    # The scaling ratio of each feature of the training sample

    return X, y, numeric_features, scale_factors


def KNN_train(X_train, y_train):
    """
    Train the K-Nearest Neighbors model by building a KDTree from the training data and associating labels.

    Args:
        X_train (numpy.ndarray): Training data features.
        y_train (numpy.ndarray): Training data labels.

    Returns:
        list: The trained KDTree model and the training data labels.
    """
    kdtree = KDTree(X_train)
    return [kdtree, y_train]


def KNN_predict(test_sample, feature_list, numeric_features, kdtree_list, scale_factor, k=3):
    """
    Predicts the label of a test sample using the K-Nearest Neighbors algorithm.

    Args:
        test_sample (list): The test sample to predict, with numeric and non-numeric values.
        feature_list (list): The list of features corresponding to the test sample.
        numeric_features (list): The list of features that are numeric.
        kdtree_list (list): A list containing the KDTree and the training labels.
        scale_factor (list): A list of lists, where each sublist contains a feature and its scaling factor.
        k (int): The number of nearest neighbors to consider.

    Returns:
        int: The predicted label for the test sample.
    """
    # Step 2: Set non-numeric features to 0
    for i, feature in enumerate(feature_list[:-1]):  # Exclude the last column (label)
        if feature not in numeric_features:
            test_sample[i] = 0

    # Step 3: Scale numeric features
    for feature, scale in scale_factor:
        if feature in feature_list:
            index = feature_list.index(feature)
            test_sample[index] *= scale

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
