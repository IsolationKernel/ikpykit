import pandas as pd
import os
from pathlib import Path
import json
import itertools
import hdf5storage as h5


class DataLoader:
    def __init__(self, relative_path="../data/format"):
        self.data_root_path = Path(__file__).resolve().parent / relative_path
        self.files = os.listdir(self.data_root_path)
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.p < len(self.files):
            df = h5.loadmat(str(self.data_root_path / self.files[self.p]))
            data = df["data"]
            label = df["class"].reshape(-1)
            self.p += 1
            return {"data": data, "label": label, "info": {"name": self.files[self.p-1].replace(".mat", "")}}
        else:
            raise StopIteration


class ParaLoader:
    def __init__(self, relative_path="../para.json"):
        self.para: dict = json.load(
            open(Path(__file__).resolve().parent / relative_path, "r"))
        self.para_combinations = itertools.product(*self.para.values())

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return dict(zip(self.para.keys(), next(self.para_combinations)))
        except StopIteration:
            raise StopIteration
