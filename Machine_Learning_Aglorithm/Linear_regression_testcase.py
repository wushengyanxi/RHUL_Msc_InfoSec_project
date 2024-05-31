import sys

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Data_Reader import Database_Reader
from Linear_Regression_1_1 import train_linear_regression
from Linear_Regression_1_1 import preprocess_data

feature_list, data = Database_Reader("ALLFLOWMETER_HIKARI2021.csv", full_read=True, traffic_category=False)
print("Features:", feature_list)
print("Data Sample:", data[0])

# Generate a larger dataset for performance testing
# Data = np.random.rand(550000, 81)  # Uncomment this line for actual testing

# Data preprocessing
X, y, numeric_features = preprocess_data(data)

# Print the shape of the preprocessed data
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Train the model
weights = train_linear_regression(X, y, numeric_features, feature_list)

# Print weights
print("Weights:", weights)