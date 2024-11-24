from myx_test.ikdc.utils.utils import DataLoader, ParaLoader
from isoml.cluster._ikdc import IKDC
from sklearn.metrics import normalized_mutual_info_score
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
    logger.info(f"Start to train IDKD on {info} dataset")
    logger.info(f"Data shape: {data.shape}, Label shape: {label.shape}")
    para_loader = ParaLoader()
    for para_dict in para_loader:
        with timer(logger):
            k = np.unique(label).shape[0]
            kn = int(np.array(para_dict["kn"]) * len(data))
            ikdc = IKDC(
                n_estimators=para_dict["n_estimators"],
                max_samples=para_dict["max_samples"],
                method=para_dict["method"],
                k=k,
                kn=kn,
                v=para_dict["v"],
                n_init_samples=para_dict["n_init_samples"],
            )
            predict = ikdc.fit_predict(data)
            score = normalized_mutual_info_score(label, predict)
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
