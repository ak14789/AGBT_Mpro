import os

import numpy as np
import pandas as pd

from typing import *

from torch.utils.data import Dataset

import utils

import matplotlib.pyplot as plt
import seaborn as sns


class Mpro(Dataset):
    """
    Mpro active data from COVID-Moonshot.

    Statistics:
        - #Molecule: 2062
        - #Valid molecules: 1926 ('f_avg_IC50' is not Null)
        - #Active Molecule: 915 (Divided by 10 μM)
        - #Regression valid Molecule: 1316 (IC50 less than 99 μM and docking without error)

    Parameters:
        data_dir (str, optional): data_dir to store the dataset
        use_classification (bool, optional): whether convert IC50 to activity
        norm_type (str, optional): normalization method.[None, 'minmax', 'std']
        splitter (float, optional): bounds of active molecules(default 10) or regression valid molecules(default 99)
    """

    features = ['SMILES', 'CID', 'series']
    target = ['f_avg_IC50']

    url = "https://covid.postera.ai/covid/activity_data.csv"

    def __init__(self, data_dir: str = 'data', use_classification: bool = True, **kwargs) -> None:
        data_dir = os.path.expanduser(data_dir)  # 将~/转化为绝对路径，没有则保持不变
        if not os.path.exists(data_dir):  # 若该数据集存放的路径不存在，则创建路径
            os.makedirs(data_dir)

        self.data_dir = data_dir  # path只是目录
        self.use_classification = use_classification

        self.feature = None  # 模型要输入的特征，可set_feature或其他自定义函数动态修改(ndarray格式)

        self.bt_npy = None
        self.ag_npy = None  # 回归专属
        self.agbt_npy = None  # 回归专属

        self.score = None  # docking_score,通过load_score函数读取csv设定(回归专属)
        self.docking_type = None  # docking的方式,在load_score时或其他时候设定(回归专属)

        self.pred = None  # 模型预测值，比较实验结果用

        file_name = utils.download(self.url, data_dir)  # 若数据集未下载，则下载数据集。返回str(相对路径+文件名)

        # 定义不同模式下的预处理方法
        if self.use_classification:
            process_fn = self._load_classification_dataset
            self.splitter = kwargs['splitter'] if 'splitter' in kwargs.keys() else 10

        else:
            process_fn = self._load_regression_dataset
            self.splitter = kwargs['splitter'] if 'splitter' in kwargs.keys() else 99
            self.norm_type = kwargs['norm_type'] if 'norm_type' in kwargs.keys() else None

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

    def set_feature(self, type_):
        """
        将feature设置为某种npy

        Parameter:
            type_: 必须是'bt', 'ag', 或'agbt'
        """
        types = ['bt', 'ag', 'agbt']
        assert type_ in types, 'parameter:type_ should be "bt","ag" or "agbt"'
        if type_ == 'bt':
            self.feature = self.bt_npy
        elif type_ == 'ag':
            self.feature = self.ag_npy
        else:
            self.feature = self.agbt_npy

    def load_npy(self, fpdata_dir: str, filename: str):
        """
        将某种npy导入进来

        Parameters:
            fpdata_dir: 模型数据保存目录
            filename: npy的文件名(不带.npy)
        """
        type_ = filename[filename.rfind('_')+1:]
        assert type_ in ['bt', 'ag', 'agbt'], 'filename should end with "bt","ag" or "agbt"'
        npy_stor_name = f'{fpdata_dir}/{filename}.npy'
        if type_ == 'bt':
            self.bt_npy = np.load(npy_stor_name)
        elif type_ == 'ag':
            self.ag_npy = np.load(npy_stor_name)
        else:
            self.agbt_npy = np.load(npy_stor_name)

    def load_score(self):
        """导入docking score"""
        # score.csv文件存放大self.path也就是data/中
        self.score = None

    def load_docking_type(self):
        """导入docking type"""
        # 也是通过score.csv设定
        self.docking_type = None

    def feature_add(self):
        """向feature中添加更多特征(dataframe)"""
        self.feature = None

    # 加入一些可视化统计函数

    def __len__(self):
        return len(self.data)

    # 返回类型可以是ndarray,dataloader字典转为tensor
    def __getitem__(self, idx):
        label = self.label.values
        return self.feature[idx], label[idx]


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

