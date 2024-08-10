import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import time
import numbers


def LR_preprocess_data(training_set, features_list):
    #heaviest_features = ['bwd_pkts_payload.max', 'flow_ACK_flag_count', 'fwd_iat.max', 'active.tot', 'active.avg', 'bwd_header_size_max', 'flow_iat.tot', 'idle.max',
    #                    'active.min', 'bwd_pkts_payload.tot', 'flow_SYN_flag_count', 'bwd_iat.std', 'flow_FIN_flag_count', 'originp', 'Unnamed: 0.1', 'fwd_PSH_flag_count',
    #                   'bwd_header_size_min', 'flow_duration', 'bwd_pkts_payload.avg', 'fwd_pkts_payload.std', 'fwd_header_size_max', 'flow_pkts_payload.avg', 'fwd_bulk_rate',
    #                  'fwd_header_size_tot', 'fwd_pkts_payload.avg', 'bwd_bulk_bytes', 'bwd_pkts_payload.min', 'fwd_iat.min', 'flow_pkts_payload.std', 'flow_iat.max', 'Label']
    heaviest_features = ['flow_ACK_flag_count', 'fwd_iat.max', 'active.tot', 'bwd_header_size_max', 'flow_iat.tot','active.min','flow_SYN_flag_count', 'bwd_iat.std',
                     'flow_FIN_flag_count', 'originp', 'Unnamed: 0.1', 'fwd_PSH_flag_count','bwd_header_size_min', 'flow_duration', 'fwd_pkts_payload.std', 'fwd_header_size_max',
                        'flow_pkts_payload.avg', 'fwd_bulk_rate','fwd_header_size_tot', 'fwd_pkts_payload.avg', 'bwd_bulk_bytes', 'bwd_pkts_payload.min', 'fwd_iat.min', 'flow_pkts_payload.std',
                        'flow_iat.max', 'Label']
    '''
    heaviest_features = ['originp', 'responp', 'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot',
                     'fwd_data_pkts_tot', 'bwd_data_pkts_tot', 'fwd_pkts_per_sec', 'bwd_pkts_per_sec',
                     'flow_pkts_per_sec', 'down_up_ratio', 'fwd_header_size_tot', 'fwd_header_size_min', 
                     'fwd_header_size_max', 'bwd_header_size_tot', 'bwd_header_size_min', 'bwd_header_size_max',  
                    'flow_CWR_flag_count', 'fwd_pkts_payload.min','fwd_pkts_payload.max', 'fwd_pkts_payload.tot',
                    'fwd_pkts_payload.avg', 'fwd_pkts_payload.std', 'bwd_pkts_payload.min', 'bwd_pkts_payload.max',
                    'bwd_pkts_payload.tot','bwd_pkts_payload.avg', 'bwd_pkts_payload.std', 'flow_pkts_payload.min',
                    'flow_pkts_payload.max', 'flow_pkts_payload.tot', 'flow_pkts_payload.avg', 'flow_pkts_payload.std',
                    'fwd_iat.min', 'fwd_iat.max', 'fwd_iat.tot', 'fwd_iat.avg', 'fwd_iat.std', 'bwd_iat.max',
                    'bwd_iat.tot', 'bwd_iat.avg', 'bwd_iat.std', 'flow_iat.max','flow_iat.tot', 'flow_iat.avg',
                    'flow_iat.std', 'payload_bytes_per_second', 'fwd_subflow_pkts', 'bwd_subflow_pkts', 'fwd_subflow_bytes',
                    'bwd_subflow_bytes', 'fwd_bulk_bytes','bwd_bulk_bytes', 'fwd_bulk_packets', 'bwd_bulk_packets',
                    'fwd_bulk_rate', 'bwd_bulk_rate', 'active.min', 'active.max', 'active.tot', 'active.avg',
                    'idle.tot', 'fwd_init_window_size', 'bwd_init_window_size', 'fwd_last_window_size', "label"]
    '''
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


def train_linear_regression(X, y, epochs=10000, learning_rate=0.001):
    print("linear regression training start")
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
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
            

    end_time = time.time()
    #print("Training time: {:.2f} seconds".format(end_time - start_time))

    # Extract weights
    weights = model.linear.weight.data.numpy().flatten()

    return weights
# return a dictionary of weight, key for features name and value for numberic weight

def LR_each_test_sample_preprocess(test_sample, scale_factors, features_list, heaviest_features):
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
    predict_score = 0
    for i in range(0,len(weights)):
        predict_score += weights[i] * test_sample[i]

    # 第四步：根据预测分数判断预判结果
    predict_label = 1 if predict_score >= -0.1 else 0

    # 第五步：返回预判结果
    return predict_label




def one_step_LR(Data, feature_list, epochs=10000, learning_rate=0.001):
    X_train, y_train, scale_factors, heaviest_features = LR_preprocess_data(Data, feature_list)
    weight_result = train_linear_regression(X_train, y_train, epochs, learning_rate)

    return scale_factors, weight_result, heaviest_features
    # scale_factors: 2D-array
    # weight_result: dictionary


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
