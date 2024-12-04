# 统计数据集信息

import pandas as pd
import numpy as np
import os
from pathlib import Path

# 读取数据集

root_path = Path(__file__).resolve().parent


input_path = root_path.parent / "data" / "format"

files = os.listdir(input_path)
for file in files:
    if file.endswith('.csv'):
        df = pd.read_csv(input_path / file, header=None)
        ld = df.to_numpy()
        label = ld[:, 0]
        data = ld[:, 1:]
        with open(root_path.parent / "README.md", "a+") as f:
            f.write(f"## {file.replace('.csv', '')}\n")
            f.write(f"nums:         \t{data.shape[0]}\n")
            f.write(f"features:     \t{data.shape[1]}\n")
            f.write(f"anomaly nums: \t{np.sum(label==1)}\n")
            f.write(f"anomaly ratio:\t{np.sum(label==1)/data.shape[0]:.4%}\n")
            f.write(f"\n")
print(files)