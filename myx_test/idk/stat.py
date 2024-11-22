# 统计数据集信息

import pandas as pd
import numpy as np
import os

# 读取数据集

data_path = './data/anomaly_data/idk/format'

files = os.listdir(data_path)
for file in files:
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_path, file), header=None)
        ld = df.to_numpy()
        label = ld[:, 0]
        data = ld[:, 1:]
        with open("./data/anomaly_data/README.md", "a+") as f:
            f.write(f"# {file.replace('.csv', '')}\n")
            f.write(f"nums:         \t{data.shape[0]}\n")
            f.write(f"features:     \t{data.shape[1]}\n")
            f.write(f"anomaly nums: \t{np.sum(label==1)}\n")
            f.write(f"anomaly ratio:\t{np.sum(label==1)/data.shape[0]:.4%}\n")
            f.write(f"\n")
print(files)