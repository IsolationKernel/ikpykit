# 统计数据集信息

import pandas as pd
import numpy as np
import os

# 读取数据集

data_path = './myx_test/ikdc/data/format'

files = os.listdir(data_path)
for file in files:
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_path, file), header=None)
        ld = df.to_numpy()
        label = ld[:, 0]
        data = ld[:, 1:]
        class_num = np.unique(label)
        with open("./myx_test/ikdc/README.md", "a+") as f:
            f.write(f"## {file.replace('.csv', '')}\n")
            f.write(f"nums:          \t{data.shape[0]}\n")
            f.write(f"features:      \t{data.shape[1]}\n")
            f.write(f"classes:       \t{class_num.shape[0]}\n")
            for k in class_num:
                f.write(f"class {k}:     \t{np.sum(label==k)}\n")
                f.write(f"class {k} rate:\t{np.sum(label==k)/data.shape[0]:.4%}\n")
            f.write(f"\n")
print(files)