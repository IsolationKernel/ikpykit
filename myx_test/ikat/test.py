from myx_test.ikat.utils.utils import DataLoader, ParaLoader
from isoml.trajectory.anomaly._ikat import IKAT
from sklearn.metrics import roc_auc_score
from myx_test.utils.logger import Logger
from myx_test.utils.timer import timer, get_time_str, get_params_str
from pathlib import Path
import json
import numpy as np

logger = Logger(__file__.replace(".py", ".log"))

data_loader = DataLoader()

for data_dict in data_loader:
    data = data_dict["data"]
    label = data_dict["label"]
    info = data_dict["info"]["name"]
    logger.info(f"Start to train IKAT on {info} dataset")
    logger.info(f"Data shape: {data.shape}, Label shape: {label.shape}")
    para_loader = ParaLoader()
    for para_dict in para_loader:
        logger.info(f"Start to train IKAT with para: {para_dict}")
        # try:
        if 1:
            with timer(logger):
                ikat = IKAT(n_estimators_1=para_dict["n_estimators"],
                            max_samples_1=para_dict["max_samples"],
                            n_estimators_2=para_dict["n_estimators"],
                            max_samples_2=para_dict["max_samples"],
                            method=para_dict["method"])
                predict = ikat.fit_predict(data)
                score = roc_auc_score(label, predict)
                logger.info(f"dataset: {info}, para: {para_dict}, score: {score}")

                time_str = get_time_str()
                result_path = Path(__file__).resolve().parent
                result_dict = {
                    "dataset": info,
                    "para": para_dict,
                    "score": score,
                    "time": time_str,
                }

                result_path.parent.mkdir(exist_ok=True, parents=True)

                with (result_path / "result.result").open("a+") as f:
                    f.write(
                        json.dumps(result_dict, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                with (result_path / "result" / f"{time_str}.result").open("a+") as f:
                    f.write(str(predict.tolist()))
        # except:
        #     continue
