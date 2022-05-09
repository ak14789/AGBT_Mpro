import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    第二行以后父类是没有的
    """
    def __init__(self, model, criterion, metrics_ftns, optimizer, config,
                 device, data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super(Trainer, self).__init__(model, criterion, metrics_ftns, optimizer, config, data_loader, device)
        self.device = device
        self.dataloader = data_loader  # 训练集dataloader

        if len_epoch is None:
            self.len_epoch = len(self.dataloader)  # 一个epoch需要更新几次参数,一般根据数据量和batch_size自动算出，不需要指定
        else:
            self.dataloader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_dataloader = valid_data_loader
        self.do_validation = self.valid_dataloader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))  # 将bs开根号向下取整(多少次打印logging)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        每一个epoch的训练流程

        Parameter:
            epoch: 第几个epoch

        Return:
            log字典？？
        """
        self.model.train()  # 将模型调整到训练模式(BN层和Dropout层)
        self.train_metrics.reset()  # train度量矩阵置0
        for batch_idx, (data, target) in enumerate(self.dataloader):
            data = data.float()
            target = target.float()
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1)*self.len_epoch + batch_idx)  # 设置这是第几次更新参数
            self.train_metrics.update('loss', loss.item())  # 向tensorboard传入loss
            for met in self.metric_ftns:  # 向tensorboard传入评估结果
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                # 经过log_step次更新参数打印loss信息
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()  # 将loss和评估函数的平均值变成字典输出

        if self.do_validation:  # 每个epoch，train先迭代完，然后给valid迭代验证
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})  # log中加入验证参数

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """同train_epoch"""
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_dataloader):
                data = data.float()
                target = target.float()
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_dataloader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        # 模型参数的直方图添加到tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        # 返回每个epoch里第几轮，总共几轮，百分之多少了
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.dataloader, 'n_samples'):  # 判断对象是否有该属性(我是没写)
            current = batch_idx * self.dataloader.batch_size
            total = self.dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

