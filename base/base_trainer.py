import torch
from numpy import inf  # 正无穷
from abc import abstractmethod
from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers.

    Parameters:
        model: 模型
        criterion: 损失函数
        metric_ftns: 评估模块[多元素的列表]
        optimizer: 优化器
        config: 对象，config读取相当于有序字典
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, dataloader):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])  # 得到name和等级的记录器

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device

        cfg_trainer = config['trainer']  # train对应的字典
        self.epochs = cfg_trainer['epochs']
        # 监视器的作用是提前停止训练
        self.monitor = cfg_trainer.get('monitor', 'off')  # 在字典中查找monitor，找不到返回off

        # 配置监视器，保存最好的模型
        if self.monitor == 'off':  # 如果json里没设置monitor
            self.mnt_mode = 'off'
            self.mnt_best = 0  # min为正无穷,max为负无穷,不设置为0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max'], 'monitor第一个参数必须是min或max'
            # min的话初始损失在为正无穷
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)  # 没设置为正无穷，相当于永不早停
            if self.early_stop <= 0:   # 若early_stop设为负数，也不早停
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir  # 模型保存的路径

        # 设置Tensorboard
        # logger的作用只是输出警告， log_dir是数据保存路径, 最后一个参数不是true基本没啥用
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        每一个epoch训练的逻辑，子类要重写
        """
        raise NotImplementedError

    def train(self):
        """
        完整的训练逻辑
        """
        not_improved_count = 0  # 未提升计数
        best_log = {}
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)  # 输出的是字典

            # 记录第几轮，等信息。。。。
            log = {'epoch': epoch}
            log.update(result)  # 将本轮result传入log

            # 打印log内的信息
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))  # 15个字符的字符串

            # 评估模型并保存
            best = False
            if self.model != 'off':   # 为off就没有监视器
                try:
                    # 如果损失有下降，则improved为True
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                # 若损失函数的值没有在log中找到，则监视器失效
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    # 如果提升了，就更新最佳损失
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best_log = log
                    best = True
                else:
                    not_improved_count += 1

                # 若没提升的epoch超过了早停的步数，则退出训练
                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.\n".format(self.early_stop))
                    if best_log:
                        self.logger.info("best result:")
                        for key, value in best_log.items():
                            self.logger.info('    {:15s}: {}'.format(str(key), value))  # 15个字符的字符串
                    break

            # 如果是最好，将预测结果保存到train_dataset和valid_dataset
            if best:
                self._save_model()
                self.model.eval()
                with torch.no_grad():
                    self.dataloader.train_dataset.pred = self.model(torch.from_numpy(self.dataloader.train_dataset.feature).to(self.device).float())
                    self.dataloader.valid_dataset.pred = self.model(torch.from_numpy(self.dataloader.valid_dataset.feature).to(self.device).float())

            if epoch == self.epochs:
                if best_log:
                    self.logger.info("best result:")
                    for key, value in best_log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))  # 15个字符的字符串

    def _save_model(self):
        """
        保存模型
        """
        best_path = str(self.checkpoint_dir / 'model_best.pth')
        torch.save(self.model, best_path)
        self.logger.info("Saving current best: model_best.pth ...")
