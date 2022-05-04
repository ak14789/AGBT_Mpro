import os
import shutil
import datasets
from utils import train_valid_split, is_normality, generate_bt_input, create_preprocess_sh, create_train_sh
from utils import create_btfp_sh

from datasets import extra_other_series

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')


def main(config):
    print(f'config:{config}\n')

    # 导入数据集
    classification_dataset = datasets.Mpro(path='data', use_classification=True, splitter=10)
    regression_dataset = datasets.Mpro(path='data', use_classification=False, splitter=99, norm_type=None)

    # 判断pIC50是否服从正态分布
    # is_normality(regression_dataset.label, 'pIC50')

    # 数据集划分
    classification_train, classification_valid = train_valid_split(classification_dataset, process_fn=extra_other_series)  # 若需要将辉瑞/other结构的分子放入验证集，请指定函数
    regression_train, regression_valid = train_valid_split(regression_dataset, process_fn=extra_other_series)

    # 可视化选项
    # print(f'分类数据集的统计:\n{classification_dataset.label.value_counts()}\n')

    # sns.histplot(classification_valid.label, bins=100)
    # plt.show()

    # 数据集分割
    classification_data_stor_name = generate_bt_input(classification_train, data_type='train', split_type='extra_other_series')
    generate_bt_input(classification_valid, data_type='valid', split_type='extra_other_series')

    regression_data_stor_name = generate_bt_input(regression_train, data_type='train', split_type='extra_other_series')
    generate_bt_input(regression_valid, data_type='valid', split_type='extra_other_series')

    # 复制dict文件
    shutil.copyfile('models/dict.txt', classification_data_stor_name + f'/dict.txt')
    shutil.copyfile('models/dict.txt', classification_data_stor_name + f'/input0/dict.txt')

    shutil.copyfile('models/dict.txt', regression_data_stor_name + f'/dict.txt')
    shutil.copyfile('models/dict.txt', regression_data_stor_name + f'/input0/dict.txt')

    # BERT模型数据预处理
    # 自动生成preprocess_xxx.sh文件
    create_preprocess_sh(classification_data_stor_name)
    create_preprocess_sh(regression_data_stor_name)
    # 手动运行sh文件，运行完毕继续
    print('>>>>>>在agbt_pro目录下运行preprocess_xxx.sh文件>>>>>>')
    while True:
        inputs = input('进入下一步输入ok, 退出程序输入exit:')
        if inputs == 'ok':
            break
        elif inputs == 'exit':
            exit()
    print('')

    # 微调Bert模型
    # 创建模型存放的目录
    classification_model_stor_name = f'models{classification_data_stor_name[classification_data_stor_name.find("/"):]}'
    regression_model_stor_name = f'models{regression_data_stor_name[regression_data_stor_name.find("/"):]}'
    if not os.path.exists(classification_model_stor_name):
        os.mkdir(classification_model_stor_name)
    if not os.path.exists(regression_model_stor_name):
        os.mkdir(regression_model_stor_name)

    # 自动生成train_xxx.sh文件
    create_train_sh(classification_data_stor_name, classification_model_stor_name, len(classification_train))
    create_train_sh(regression_data_stor_name, regression_model_stor_name, len(regression_train))
    # 手动运行sh文件，运行完毕继续
    print('>>>>>>在agbt_pro目录下运行train_xxx.sh文件>>>>>>')
    while True:
        inputs = input('进入下一步输入ok, 退出程序输入exit:')
        if inputs == 'ok':
            break
        elif inputs == 'exit':
            exit()
    print('')

    # 自动生成generate_xxx.sh文件
    create_btfp_sh(classification_data_stor_name, classification_model_stor_name)
    create_btfp_sh(regression_data_stor_name, regression_model_stor_name)
    # 手动运行sh文件，运行完毕继续
    print('>>>>>>在agbt_pro目录下运行generate_xxx.sh文件>>>>>>')
    while True:
        inputs = input('进入下一步输入ok, 退出程序输入exit:')
        if inputs == 'ok':
            break
        elif inputs == 'exit':
            exit()
    print('')

    # 对分类任务直接进行下游任务训练（如果训练好直接进行下一步）
    # 展示分类模型的效果（tensorboard？）

    # 对回归任务挑选mol2文件
    # 生成ag分子指纹

    # fusion

    # 训练下游回归任务（是否加入score信息）（加入LDS和FDS）如果训练好了进入下一步
    # 展示回归效果


if __name__ == '__main__':
    print('start!\n')
    config = 0
    main(config)
    print('End!!!!')
