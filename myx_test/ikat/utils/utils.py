import pandas as pd
import pickle
import os
from pathlib import Path
import json
import itertools


class DataLoader:
    def __init__(self, relative_path="../data/format"):
        self.data_root_path = Path(__file__).resolve().parent / relative_path
        self.files = os.listdir(self.data_root_path)
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.p < len(self.files):
            # pickle load
            with open(self.data_root_path / self.files[self.p], "rb") as f:
                ld = pickle.load(f)
            self.p += 1
            return {"data": ld["data"], "label": ld["label"], "info": {"name": self.files[self.p-1].replace(".pkl", "")}}
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

