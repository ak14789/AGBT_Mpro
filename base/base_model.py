import torch.nn as nn
import numpy as np
from abc import abstractmethod  # 一种C++的工具包
from torchsummary import summary


class BaseModel(nn.Module):
    """
    Base class for all models.
    """
    @abstractmethod   # 添加该装饰器的类无法实例化，子类继承时必须重写该方法(类继承abc.ABC时才有效，等于这里没用)
    def forward(self, *inputs):
        raise NotImplementedError

    def summary(self, input_dim, **kwargs):
        summary(self, input_dim, **kwargs)

    # 魔法方法，str(obj)或print(obj)时调用
    def __str__(self):
        """
        在原有基础上打印可训练参数的总数量
        """
        # filter遍历迭代器丢入function，留下True的元素，返回新的迭代器
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())  # 模型可更新的参数
        # np.prod计算所有元素的乘积
        params = sum([np.prod(p.size()) for p in model_parameters])
        # super()中的没参数默认就是下面的参数
        return super(BaseModel, self).__str__() + f'\nTrainable parameters: {params}'
    #### !!!!!!!!!!!!!!!!!!!!上面可以加入打印模型结构那个包！！！！！！！！！！！！！！！！！！！
