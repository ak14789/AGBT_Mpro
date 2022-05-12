from functools import wraps
import time


def marker(old_func):
    """在函数开始和结束时打印标记"""
    @wraps(old_func)
    def new_func(*args, **kwargs):
        print('Start!!!\n')
        old_func(*args, **kwargs)
        print('End!!!')
    return new_func


def timer(old_func):
    """记录函数运行的时间并打印"""
    @wraps(old_func)
    def new_func(*args, **kwargs):
        start = time.time()
        res = old_func(*args, **kwargs)
        end = time.time()
        print(f'{old_func.__name__}()的运行时间是{end-start}s\n')
        return res

    return new_func
