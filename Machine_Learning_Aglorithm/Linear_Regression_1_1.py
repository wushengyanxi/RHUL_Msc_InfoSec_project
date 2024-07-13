import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import time


def LR_preprocess_data(Data, features_list):
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
    y = numeric_data[:, -1].reshape(-1, 1)
    X = scaler.fit_transform(X)

    # Get scale factors for each feature
    scale_factors = [[features_list[i], 1 / scaler.scale_[i]] for i in numeric_features]

    return X, y, numeric_features, scale_factors


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def train_linear_regression(X, y, numeric_features, feature_list, epochs=10000, learning_rate=0.001):
    # Convert to tensor
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Initialize model, loss function, and optimizer
    input_dim = X.shape[1]
    model = LinearRegressionModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    start_time = time.time()
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
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    end_time = time.time()
    print("Training time: {:.2f} seconds".format(end_time - start_time))

    # Extract weights
    weights = model.linear.weight.data.numpy().flatten()

    # Create result dictionary
    result = {feature_list[i]: 0.0 for i in range(len(feature_list))}
    for i, feature in enumerate(numeric_features):
        result[feature_list[feature]] = weights[i]
    '''
    # Create result list
    result = []
    for i in range(len(weights)):
        result.append([feature_list[i], float(weights[i])])
    '''
    return result


def linear_regression_predict(features, weights, sample, scale_factors):
    # sample = preprocess_data(sample) 这些数据需要被初始化
    predict_score = 0
    predict_label = 0
    for n in range(0, len(features)):
        index = next((i for i, sublist in enumerate(scale_factors) if sublist[0] == features[n]), -1)
        if index != -1:
            sample[n] = sample[n] * scale_factors[index][1]
            predict_score += sample[n] * weights[features[n]]

    if predict_score >= 0.5:
        predict_label = 1

    return predict_label


def one_step_LR(Data, feature_list, epochs=10000, learning_rate=0.001):
    X, y, numeric_features, scale_factors = LR_preprocess_data(Data, feature_list)
    weight_result = train_linear_regression(X, y, numeric_features, feature_list, epochs, learning_rate)

    return scale_factors, weight_result


'''
预处理数据函数 (preprocess_data)：

将数据转换为 numpy 数组。
过滤数值特征并处理非数值特征。
用列均值替换 NaN 和 Inf 值。
标准化特征。
线性回归模型类 (LinearRegressionModel)：

定义线性回归模型结构。
训练线性回归模型的函数 (train_linear_regression)：

将数据转换为 tensor。
初始化模型、损失函数和优化器。
训练模型并监控训练时间和损失值。
示例数据和执行流程：

生成示例数据。
进行数据预处理。
打印预处理后的数据形状。
训练模型并打印权重。
通过这个完整的流程，可以确保数据处理和模型训练的正确性。如果训练时间仍然非常短，可以进一步检查数据集和计算资源的使用情况

'''
