import logging


class Logger(logging.Logger):
    def __init__(self, name, level=logging.DEBUG) -> None:
        super().__init__(__name__)

        # 创建文件处理器并设置编码为 UTF-8
        file_handler = logging.FileHandler(name, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 设置文件处理器的日志级别

        # 创建日志格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 将文件处理器添加到日志记录器
        self.addHandler(file_handler)

        # 记录不同级别的日志
        # logger.debug('这是调试信息')
        # logger.info('这是一般信息')
        # logger.warning('这是警告信息')
        # logger.error('这是错误信息')
        # logger.critical('这是严重错误信息')

