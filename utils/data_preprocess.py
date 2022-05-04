import random
import pandas as pd
from scipy import stats

from copy import deepcopy


def minmax(data: pd.DataFrame, target: str):
    """
    对dataframe的target列进行归一化，返回min,max和归一化后的dataframe

    Parameters:
        data (pd.DataFrame): 要归一化的表格
        target (str): 归一化的目标列名

    Returns:
        最小值，最大值，归一化后的表格
    """
    min_ = data[target].min()
    max_ = data[target].max()

    temp = (data[target] - min_) / (max_ - min_)
    data = data.drop([target], axis=1)
    data[target] = temp

    return min_, max_, data


def std(data: pd.DataFrame, target: str):
    """
    对dataframe的target列进行标准化，返回mean,var和标准化后的dataframe

    Parameters:
        data (pd.DataFrame): 要标准化的表格
        target (str): 标准化的目标列名

    Returns:
        平均值，标准差，标准化后的表格
    """
    mean = data[target].mean()
    std_ = data[target].std()

    temp = (data[target] - mean) / std_
    data = data.drop([target], axis=1)
    data[target] = temp

    return mean, std_, data


def train_valid_split(dataset, train_size: float = 0.8, process_fn=None):
    """
    将Dataset类划分为训练集和验证集

    Parameters:
        dataset: 具有data和label属性的Dataset类，data和label都是Dataframe
        train_size: 训练集比例
        process_fn: 如果有特殊要求，那么就不按8：2划分了，把所有特殊要求都放验证集里。没有特殊要求就正常8：2划分

    Returns:
        train, valid (返回Dataset类，只改变了对象的data和label)
    """
    original_index = dataset.data.index.values  # 原始数据集的索引列表
    indexes = list(range(original_index[-1] + 1))
    random.seed(10)
    random.shuffle(indexes)

    # 特殊索引往后放
    if process_fn is not None:
        train_index, valid_index = process_fn(dataset)
    else:
        train_index_init = indexes[:int(len(indexes) * train_size)]
        valid_index_init = indexes[int(len(indexes) * train_size):]

        train_index = []
        valid_index = []
        for index in train_index_init:
            if index in original_index:
                train_index.append(index)
        for index in valid_index_init:
            if index in original_index:
                valid_index.append(index)

    # copy数据集，为训练集和测试集赋值。如果数据集没有该索引，则忽略
    train = deepcopy(dataset)
    train.data, train.label = dataset.data.loc[train_index], dataset.label.loc[train_index]
    valid = deepcopy(dataset)
    valid.data, valid.label = dataset.data.loc[valid_index], dataset.label.loc[valid_index]
    return train, valid


def is_normality(data: pd.DataFrame, target: str) -> bool:
    """
    判断dataframe的target列是否服从正态分布

    Parameters:
        data (pd.DataFrame): 数据表
        target (str): 要判断的列名

    Return:
        True or False
    """
    length = len(data)

    if length < 50:
        print('use shapiro:')
        p_value = stats.shapiro(data[target])[1]
        if p_value < 0.05:
            print(f'P = {p_value}\nP < 0.05\nFalse\n')
            return False
        else:
            print(f'P = {p_value}\nP >= 0.05\nTrue\n')
            return True

    elif 50 <= length <= 5000:
        mean = data[target].mean()
        std_ = data[target].std()
        p_value_ks = stats.kstest(data[target], 'norm', (mean, std_))[1]
        p_value_sh = stats.shapiro(data[target])[1]

        if p_value_ks >= p_value_sh:
            p_value = p_value_ks
            print('use kstest:')
        else:
            p_value = p_value_sh
            print('use shapiro:')

        if p_value < 0.05:
            print(f'P = {p_value}\nP < 0.05\nFalse\n')
            return False
        else:
            print(f'P = {p_value}\nP > 0.05\nTrue\n')
            return True

    else:
        mean = data[target].mean()
        std_ = data[target].std()
        p_value = stats.kstest(data[target], 'norm', (mean, std_))[1]
        print('use kstest:')
        if p_value < 0.05:
            print(f'P = {p_value}\nP < 0.05\nFalse\n')
            return False
        else:
            print(f'P = {p_value}\nP > 0.05\nTrue\n')
            return True
