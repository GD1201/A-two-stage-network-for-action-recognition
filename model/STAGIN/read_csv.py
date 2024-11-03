import pandas as pd

# 输入CSV文件路径
input_csv_path = '/home/huaizhenhao/codes/GD/GAIN-master/data/pose_data.csv'
# 输出CSV文件路径
output_csv_path = '/home/huaizhenhao/codes/GD/GAIN-master/data/pose_data_processed.csv'

# 读取CSV文件，去掉第一行和第一列
df = pd.read_csv(input_csv_path)

# 去掉第一行和第一列
df_processed = df.iloc[1:800, 2:800]

# 保存处理后的数据到新的CSV文件
df_processed.to_csv(output_csv_path, index=False)

print(f"Processed data has been written to {output_csv_path}")
