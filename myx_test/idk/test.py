from myx_test.idk.utils.utils import DataLoader, ParaLoader
from isoml.anomaly._idkd import IDKD
from sklearn.metrics import roc_auc_score
from myx_test.utils.logger import Logger
from myx_test.utils.timer import timer, get_time_str, get_params_str
from pathlib import Path
import json

logger = Logger(__file__.replace(".py", ".log"))

data_loader = DataLoader()

for data_dict in data_loader:
    if "ALOI" in data_dict["info"]["name"]:
        continue
    data = data_dict["data"]
    label = data_dict["label"]
    info = data_dict["info"]["name"]
    logger.info(f"Start to train IDKD on {info} dataset")
    logger.info(f"Data shape: {data.shape}, Label shape: {label.shape}")
    para_loader = ParaLoader()
    for para_dict in para_loader:
        with timer(logger):
            idkd = IDKD(
                n_estimators=para_dict["n_estimators"], max_samples=para_dict["max_samples"])
            idkd.fit(data)
            predict = idkd.decision_function(data)
            pos_score = 1 - roc_auc_score(label, predict)
            neg_score = 1 - pos_score
            score = max(pos_score, neg_score)
            logger.info(
                f"dataset: {info}, para: {para_dict}, pos_score: {pos_score}, neg_score: {neg_score}, score: {score}")

            result_path = Path(__file__).resolve().parent / f"result"
            result_dict = {
                "dataset": info,
                "para": para_dict,
                "pos_score": pos_score,
                "neg_score": neg_score,
                "score": score,
                "time": get_time_str(),
                "predict": str(predict.tolist()),
            }

            result_path.parent.mkdir(exist_ok=True, parents=True)

            with result_path.open("a+") as f:
                f.write(json.dumps(result_dict, ensure_ascii=False,
                        separators=(',', ':')) + "\n")
