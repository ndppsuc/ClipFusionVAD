import pandas as pd

# 假设你的数据存储在一个 CSV 文件中
file_path = 'ucf_CLIP_rgbtest.csv'

# 读取 CSV 文件
df = pd.read_csv(file_path)

# 确保第一列是路径列，这里假设第一列名称为 'path'
df['path'] = df['path'].str.replace('/home/xbgydx/Desktop/', '../')

# 将修改后的数据保存回 CSV 文件
df.to_csv('ucf_CLIP_rgbtest.csv', index=False)

print("路径替换完成，并已保存到 modified_file.csv")
