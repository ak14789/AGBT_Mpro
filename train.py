import argparse
import collections
import numpy as np

import torch

import dataloader.dataloader as module_dataloader
import model.downstream_model as module_downstream_model
import model.loss as model_loss
import model.metric as model_metric
from trainer import Trainer
from utils import is_normality, save_result_to_csv, prepare_device, marker, timer
from parse_config import ConfigParser

# 指定随机数状态
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
# 为了保证GPU训练时也是结果不变
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config):
    # 实例化一个记录器，记录器的名字为train,等级默认为2，最低等级DEBUG
    logger = config.get_logger('train', verbosity=2)

    # 数据模块
    dataloader = config.init_obj('data_loader', module_dataloader, fpdata_dir=config.save_dir.parent)
    valid_dataloader = dataloader.split_validation()

    # 回归模式下判断pIC50是否服从正态分布
    # is_normality(dataloader.dataset.label, 'pIC50')

    # 下游模型模块！！！！！！！！！！！！！！！！！！！！！！！！！！！model引入FDS
    model = config.init_obj('downstream_model', module_downstream_model,
                            input_dim=dataloader.train_dataset.feature.shape[1],
                            output_dim=dataloader.train_dataset.label.shape[1])
    logger.info(model)

    # 将模型传入GPU
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # 可以打印模型每层的参数
    # model.summary(input_dim=(1, dataloader.train_dataset.feature.shape[1]))

    # 损失与评估模块!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!loss引入LDS
    criterion = getattr(model_loss, config['loss'])
    metrics = [getattr(model_metric, met) for met in config['metrics']]

    # 优化器模块
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # 训练模型
    trainer = Trainer(model, criterion, metrics, optimizer, config=config, device=device, data_loader=dataloader,
                      valid_data_loader=valid_dataloader, lr_scheduler=lr_scheduler)
    trainer.train()

    # 预测结果保存
    save_result_to_csv(dataloader, path=config.save_dir.parent)


@marker
@timer
def cli_main():
    args = argparse.ArgumentParser(description='AGBT for Mpro')
    # (action='store'...)表示遇到-c参数时怎么处理，默认为将参数值保存
    # (nargs=1)表示参数的数量，默认为1。"?"表示可以不输入参数的值
    # (const=15)将变量赋值为常量,不可自行输入。一般与action='store_const'或nargs='?'配合使用
    # (choices=[1, 3, 5])允许参数的集合，不准输入集合以外的参数
    # (required=False)参数能否省略，默认为True
    # (dest='a')参数别名
    # (metavar)在说明中显示的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称
    args.add_argument('-c', '--config', default=None, type=str, help='config file data_dir (default: None)')

    # 命令行可以覆盖json中的参数
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    # options中的元素：CustomArgs(flags=['--lr', '--learning_rate'], type=<class 'float'>, target='optimizer;args;lr')
    options = [
        CustomArgs(flags=['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(flags=['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    main(config)


if __name__ == '__main__':
    cli_main()
