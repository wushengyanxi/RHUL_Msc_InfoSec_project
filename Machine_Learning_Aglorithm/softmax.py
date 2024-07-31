import numpy as np
from sklearn.preprocessing import StandardScaler

def Softmax_preprocess_training(training_set, features_list):
    
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
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    return X_train, y_train, scale_factors, heaviest_features

# Softmax函数
def softmax(z):
    if z.ndim == 1:
        z = np.expand_dims(z, axis=0)
    
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)

# 交叉熵损失函数
def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    logp = - np.log(y_pred[range(n_samples), y_true])
    loss = np.sum(logp) / n_samples
    return loss

# 计算梯度
def compute_gradient(X, y_true, y_pred):
    n_samples = X.shape[0]
    grad = X.T.dot(y_pred - y_true) / n_samples
    return grad

# 训练函数
def train(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    y = y.astype(int).ravel()    
    # 初始化权重
    W = np.random.randn(n_features, n_classes)
    
    # 将标签转换为one-hot编码
    y_one_hot = np.zeros((n_samples, n_classes))
    y_one_hot[np.arange(n_samples), y] = 1
    
    for epoch in range(epochs):
        # 计算预测值
        z = X.dot(W)
        y_pred = softmax(z)
        
        # 计算损失
        loss = cross_entropy_loss(y, y_pred)
        
        # 计算梯度
        dW = compute_gradient(X, y_one_hot, y_pred)
        
        # 更新权重
        W -= learning_rate * dW
        
        # if epoch % 100 == 0:
            # print(f'Epoch {epoch}, Loss: {loss}')
    
    return W

def each_test_sample_preprocess(test_sample, scale_factors, features_list, heaviest_features):
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
    
    testing_sample = np.array(testing_sample)
    
    return testing_sample

def softmax_predict(X, W):
    z = X.dot(W)
    y_pred = softmax(z)
    return np.argmax(y_pred, axis=1)

