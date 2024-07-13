import sys
from random import shuffle

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Data_Reader import Database_Reader
from Linear_Regression_1_1 import train_linear_regression
from Linear_Regression_1_1 import LR_preprocess_data
from Linear_Regression_1_1 import linear_regression_predict
from Linear_Regression_1_1 import one_step_LR

feature_list, data = Database_Reader("ALLFLOWMETER_HIKARI2021.csv", full_read=True, traffic_category=False)
'''
for i in range(0,len(data)):
    data[i][0] = 0
    data[i][1] = 0
print("length of features:",len(feature_list))
print("length of each sample:",len(data[0]))
print("Features:", feature_list)
print("Data Sample example:", data[150])
print("type of Data Sample example:", type(data[150]))

shuffle(data)

# Generate a larger dataset for performance testing
# Data = np.random.rand(550000, 81)  # Uncomment this line for actual testing

# Data preprocessing
X, y, numeric_features, scale_factors = preprocess_data(data, feature_list)
print("the scale of those factors are: ", scale_factors)
print("length of scale_factors:", len(scale_factors))
# Print the shape of the preprocessed data
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Train the model
weights = train_linear_regression(X, y, numeric_features, feature_list,5)

# Print weights
print("length of samples:", len(weights))
print("Weights:", weights)


sample_amount = len(data)
correct_predict = 0
for x in range(0, len(data)):
    predict_label = linear_regression_predict(feature_list, weights, data[x], scale_factors)
    if predict_label == data[x][-1]:
        correct_predict += 1
    if x%50000 == 0:
        print(x," samples has already predict")
        print("the predict label is: ",predict_label)
        print("the true label is: ", data[x][-1])
    elif x == len(data):
        print("all the sample has already predict")


correct_rate = correct_predict/sample_amount
print("the correct rate of predict is: ", correct_rate)
'''

s, w = one_step_LR(data, feature_list,300)
print(s)
print(w)