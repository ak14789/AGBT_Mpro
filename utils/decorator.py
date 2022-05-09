from functools import wraps


# 过滤缺失值的装饰器
def start_end(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print('Start!!!\n')
        func(*args, **kwargs)
        print('End!!!')
    return wrapper
