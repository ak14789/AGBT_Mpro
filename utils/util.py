import pandas as pd
import torch.cuda
from itertools import repeat


def prepare_device(n_gpu_use):
    """
    设置GPU(如果可用，返回gpu索引:[0, 1, 2...])
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def inf_loop(data_loader):
    """
    创建一个无止境的生成器

    example:
        dataloader = in_loop(dataloader)
        for i in dataloader:
            print(i)
    无止境循环变量dataloader
    """
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    """
    创建行为*keys,列为total,counts和average的Dataframe,并将初始值全设为0
    """
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        """将self._data所有行所有列都设为0"""
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
