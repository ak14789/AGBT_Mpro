import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    输入日志保存目录，config
    """
    log_config = Path(log_config)
    if log_config.is_file():  # 判断是否是文件
        config = read_json(log_config)
        # 根据json修改logging
        for _, handler in config['handlers'].items():
            if 'filename' in handler:   # 修改json的filename文件名
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)    # 通过字典对logging进行初始化。所以json的格式差不多固定的
    # 如果没有输入json配置文件，则警告，将logging等级调为默认等级INFO
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
