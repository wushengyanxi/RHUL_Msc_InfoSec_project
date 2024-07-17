import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import time


def preprocess_data(Data, features_list):
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

    Raises:
        ValueError: If all values in a column are NaN or infinite.


    Note:
        Non-numeric features are transformed to zero and do not contribute to the model input.
    """
    Data = np.array(Data, dtype=object)

    numeric_features = []
    numeric_data = []

    for feature in range(Data.shape[1] - 1):  # Exclude the last column (labels)
        if all(isinstance(x, (int, float)) for x in Data[:, feature]):
            numeric_features.append(feature)
            numeric_data.append(Data[:, feature].astype(float))
        else:
            numeric_data.append(np.zeros(Data.shape[0]))  # Set non-numeric features to 0

    labels = Data[:, -1].astype(float)
    numeric_data.append(labels)

    numeric_data = np.array(numeric_data).T

    for i in range(numeric_data.shape[1]):
        col = numeric_data[:, i]
        valid_mask = np.isfinite(col)
        if np.sum(valid_mask) == 0:
            raise ValueError(f"All values are invalid in column {i}")
        mean_value = col[valid_mask].mean()
        col[~valid_mask] = mean_value

    scaler = StandardScaler()
    X = numeric_data[:, :-1]
    y = numeric_data[:, -1].reshape(-1, 1)
    X = scaler.fit_transform(X)
    scale_factors = [[features_list[i], 1 / scaler.scale_[i]] for i in numeric_features]

    return X, y, numeric_features, scale_factors


class SVM(nn.Module):
    def __init__(self, input_dim):
        """
        Initializes the SVM model.

        This model uses a single linear layer to perform the operations of a
        Support Vector Machine (SVM). The input dimension is specified to
        match the number of features in the training data.

        Args:
            input_dim (int): The number of input features.
        """
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
               Defines the forward pass of the SVM model.

               The forward pass computes the linear transformation of the input
               features and returns the result. This is equivalent to calculating
               the dot product of the input features and the weight vector, plus
               the bias term.

               Args:
                   x (torch.Tensor): The input tensor of shape (batch_size, input_dim),
                                     where batch_size is the number of samples and
                                     input_dim is the number of features.

               Returns:
                   torch.Tensor: The output tensor of shape (batch_size, 1), containing
                                 the linear transformation results for each sample.
               """
        return self.linear(x)


def train_svm(feature_list, Data, epochs=10000, learning_rate=0.001, C=1.0):
    """
    Trains a Support Vector Machine (SVM) model using the provided data.

    This function preprocesses the input data, initializes an SVM model, and
    trains it for a specified number of epochs. It uses stochastic gradient
    descent (SGD) for optimization and prints the loss value every 100 epochs.
    The function returns the weights of the trained model.

    Args:
        feature_list (list): A list of feature names corresponding to the columns
                             in the data.
        Data (list or numpy.ndarray): The dataset to train on, where the last
                                      column is assumed to be the labels.
        epochs (int, optional): The number of epochs to train the model. Default is 10000.
        learning_rate (float, optional): The learning rate for the SGD optimizer. Default is 0.001.
        C (float, optional): The regularization parameter for the SVM. Default is 1.0.

    Returns:
        list: A list of strings representing each feature and its corresponding weight
              in the format ["feature1: weight1", "feature2: weight2", ...].

    Raises:
        ValueError: If all values in a column are NaN or infinite.
    """
    X, y, numeric_features, scale_factors = preprocess_data(Data, feature_list)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    input_dim = X.shape[1]
    model = SVM(input_dim)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()

        outputs = model(X)
        hinge_loss = torch.mean(torch.clamp(1 - y * outputs, min=0))
        loss = hinge_loss + 0.5 * C * torch.sum(model.linear.weight ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    end_time = time.time()
    print("Training time: {:.2f} seconds".format(end_time - start_time))

    weights = model.linear.weight.data.numpy().flatten()
    result = {feature_list[i]: 0.0 for i in range(len(feature_list))}
    for i, feature in enumerate(numeric_features):
        result[feature_list[feature]] = weights[i]

    return result
