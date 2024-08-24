import pandas as pd

# Specifying the file path
file_path = r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser\ALLFLOWMETER_HIKARI2021.csv'

# Reading CSV Files
df = pd.read_csv(file_path)

# Change the values ​​of the first two features from numbers to strings
df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: str(x))
df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: str(x))

# Save the modified CSV file, overwriting the original file
df.to_csv(file_path, index=False)

print("File updated successfully")