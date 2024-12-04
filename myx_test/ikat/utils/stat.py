# 统计数据集信息

import numpy as np
import os
from pathlib import Path
import pickle

# 读取数据集

root_path = Path(__file__).resolve().parent


input_path = root_path.parent / "data" / "format"

files = os.listdir(input_path)
for file in files:
    if file.endswith('.pkl'):
        # pickle load
        with open(input_path / file, 'rb') as f:
            ld = pickle.load(f)
        data = ld['data']
        label = ld['label']
        label = np.array(label)
        dim = set()
        length = list()
        for traject in data:
            dim.add(len(traject[0]))
            length.append(len(traject))

        with open(root_path.parent / "README.md", "a+") as f:
            f.write(f"## {file.replace('.pkl', '')}\n")
            f.write(f"trajectory nums:       \t{len(data)}\n")
            f.write(f"min trajectory length: \t{min(length)}\n")
            f.write(f"max trajectory length: \t{max(length)}\n")
            f.write(f"mean trajectory length:\t{np.mean(length):.2f}\n")
            f.write(f"trajectory dimension:  \t{list(dim)[0]}\n")
            f.write(f"anomaly nums:          \t{np.sum(label==1)}\n")
            f.write(f"anomaly ratio:         \t{np.sum(label==1)/label.shape[0]:.4%}\n")
            f.write(f"\n")
print(files)