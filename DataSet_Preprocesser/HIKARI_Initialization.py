import pandas as pd

# 指定文件路径
file_path = r'C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser\ALLFLOWMETER_HIKARI2021.csv'

# 读取CSV文件
df = pd.read_csv(file_path)

# 将前两个特征的值从数字改成字符串
df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: str(x))
df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: str(x))

# 保存修改后的CSV文件，覆盖原文件
df.to_csv(file_path, index=False)

print("文件已成功更新")