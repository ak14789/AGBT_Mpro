import pandas as pd

from functools import wraps


# 过滤缺失值的装饰器
def decorator(func, file_name, label):
    @wraps(func)
    def wrapper(*args, **kwargs):
        df = pd.read_csv(file_name, sep=',').dropna(subset=label)

        # 原有函数放下面
        func(*args, **kwargs)

    return wrapper