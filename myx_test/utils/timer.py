import time
from contextlib import contextmanager
from .logger import Logger


@contextmanager
def timer(logger: Logger, text=""):
    start_time = time.time()
    yield
    end_time = time.time()
    logger.info(f"{text} 执行时间: {end_time - start_time:.4f} 秒")


def get_time_str():
    millisecond = str(int((time.time() % 1) * 1000)).zfill(3)
    formattedtime = time.strftime("%Y%m%d%H%M%S", time.localtime())
    return f"{formattedtime}{millisecond}"


def get_params_str(params: dict):
    ans = []
    for k, v in params.items():
        ans.append(f"{k}_{v}")
    return "_".join(ans)
