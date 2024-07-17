import sys

sys.path.append(r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser')
from Training_Set_Creator import Training_set_create
from SVM import train_svm
from SVM import preprocess_data

Features_name, Training_Data_Set, Testing_Data_Set = Training_set_create()
print("Features:", Features_name)
print("Data Sample:", Training_Data_Set[0])

# Generate a larger dataset for performance testing
# Data = np.random.rand(550000, 81)  # Uncomment this line for actual testing

# Data preprocessing
X, y, numeric_features, scale_factors = preprocess_data(Training_Data_Set, Features_name)

# Print the shape of the preprocessed data
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

a = train_svm(Features_name, X,100)
print(a)
print(type(a))
print(len(a))
