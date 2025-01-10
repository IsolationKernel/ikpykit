from datetime import datetime

def pytest_configure(config):
    config.option.log_cli = True
    config.option.log_cli_level = "WARNING"
    config.option.log_cli_date_format = "%Y-%m-%d %H:%M:%S"
    config.option.log_cli_format = "%(asctime)s (%(levelname)s) | %(filename)s:%(lineno)s | %(message)s"
    config.option.log_file = f"myx_test/log_dir/test_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    config.option.log_file_level = "INFO"
    config.option.log_file_date_format = "%Y-%m-%d %H:%M:%S"
    config.option.log_file_format = "%(asctime)s (%(levelname)s) %(filename)s:%(lineno)s | %(message)s"
    
def pytest_ignore_collect(path):
    if "merge_log.py" in str(path):
        return True
    return False