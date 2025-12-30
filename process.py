import numpy as np
base_dir='./filt_global'
# 遍历文件夹下的所有文件

import os
import pandas as pd

# 创建一个列表来存储数据
shape_records = []

for file in os.listdir(base_dir):
    file_path = os.path.join(base_dir, file)
    if os.path.isfile(file_path):
        print(file_path)
        data = np.loadtxt(file_path, comments='#')
        # 输出数据
        print("数据形状:", data.shape)
        # 记录文件名和数据形状
        shape_records.append({
            'file_name': file,
            'rows': data.shape[0],
            'columns': data.shape[1]
        })

# 创建DataFrame并保存为CSV
df = pd.DataFrame(shape_records)
csv_path = os.path.join(os.path.dirname(base_dir), 'data_shapes.csv')
df.to_csv(csv_path, index=False)
print(f"数据形状信息已保存到: {csv_path}")


# print("数据内容:\n", data)

# # 如果是多列数据，可以按列访问
# if data.ndim > 1:
#     column_1 = data[:, 0]  # 第一列
#     print("第一列数据:", column_1)