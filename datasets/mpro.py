import os

import numpy as np
import pandas as pd

from typing import Tuple

import torch
from torch.utils.data import Dataset

import utils

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("TkAgg")


class Mpro(Dataset):
    """
    Mpro active data from COVID-Moonshot.

    Statistics:
        - #Molecule: 2062
        - #Valid molecules: 1926 ('f_avg_IC50' is not Null)
        - #Active Molecule: 915 (Divided by 10 μM)
        - #Regression valid Molecule: 1316 (IC50 less than 99 μM and docking without error)

    Parameters:
        path (str, optional): path to store the dataset
        use_classification (bool, optional): whether convert IC50 to active
        norm_type (str, optional): normalization method.[None, 'minmax', 'std']
        splitter (float, optional): bounds of active molecules(default 10) or regression valid molecules(default 99)
    """

    features = ['SMILES', 'CID', 'series']
    target = ['f_avg_IC50']

    url = "https://covid.postera.ai/covid/activity_data.csv"

    def __init__(self, path: str = 'data', use_classification: bool = True, **kwargs) -> None:
        path = os.path.expanduser(path)  # 将~/转化为绝对路径，没有则保持不变
        if not os.path.exists(path):  # 若该数据集存放的路径不存在，则创建路径
            os.makedirs(path)

        self.path = path  # path只是目录
        self.use_classification = use_classification
        self.npy = None  # 执行完load_npy函数后才有值

        file_name = utils.download(self.url, path)  # 若数据集未下载，则下载数据集。返回str(相对路径+文件名)

        # 定义不同模式下的预处理方法
        if self.use_classification:
            process_fn = self._load_classification_dataset
            self.splitter = kwargs['splitter'] if 'splitter' in kwargs.keys() else 10

        else:
            process_fn = self._load_regression_dataset
            self.splitter = kwargs['splitter'] if 'splitter' in kwargs.keys() else 99
            self.norm_type = kwargs['norm_type'] if 'splitter' in kwargs.keys() else None

        self.data, self.label = process_fn(file_name)  # 返回dataframe类型

    def _load_classification_dataset(self, file_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess dataset and extract features and label

        Parameters:
            file_name (str): Path of dataset

        Returns:
            data, label (pd.DataFrame, pd.DataFrame): Dataframe of features, Dataframe of preprocessed label
        """
        df = pd.read_csv(file_name, sep=',').dropna(subset=self.target).reset_index(drop=True)  # 删除fIC50的缺失值,并重置索引

        data = df[self.features]
        label = df[self.target]
        df['activity'] = [1 if x <= self.splitter else 0 for x in label.values.reshape(-1)]
        label = df[['activity']]

        return data, label

    def _load_regression_dataset(self, file_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess dataset and extract features and label

        Parameters:
            file_name (str): Path of dataset

        Returns:
            data, label (pd.DataFrame, pd.DataFrame): Dataframe of features, Dataframe of preprocessed label
        """
        df = pd.read_csv(file_name, sep=',').dropna(subset=self.target).reset_index(drop=True)  # 为了保证之后的数据集划分，不管以什么为阈值，只要划分方式没变，验证集和训练集永不相交
        df = df[df.f_avg_IC50 < self.splitter]  # 删除fIC50小于99的数据

        data = df[self.features]
        label = df[self.target]

        df['pIC50'] = -np.log10(label)
        label = df[['pIC50']]

        if self.norm_type == 'minmax':
            print('Using minmax normalization!\n')
            self.min, self.max, label = utils.minmax(label, target='pIC50')
        elif self.norm_type == 'std':
            print('Using Standardization!\n')
            self.mean, self.std, label = utils.std(label, target='pIC50')
        else:
            print('Normalization and standardization are not used!\n')

        return data, label

    def load_npy(self, data_stor_name: str):
        data_stor_name = f'{data_stor_name}'
        self.npy = np.load(data_stor_name)
        ...

    def __len__(self):
        return len(self.data)

    # 返回的是对应的分子指纹，再研究
    def __getitem__(self, idx):
        data = np.array(self.data)
        label = np.array(self.label)
        return data[idx], label[idx]


def extra_other_series(dataset):
    """ 将辉瑞和other结构放入验证集 """
    train_index = []
    valid_index = []

    for index, row in dataset.data.iterrows():
        if row['series'] is np.nan or row['CID'] == 'MAT-POS-7ed7af85-1':
            valid_index.append(index)
        else:
            train_index.append(index)

    return train_index, valid_index

