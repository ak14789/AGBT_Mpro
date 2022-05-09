import importlib
from datetime import datetime


class TensorboardWriter:
    """
    可视化类

    Parameters:
        log_dir: log保存目录
        logger: 记录器对象
        enabled: true or false
    """

    def __init__(self, log_dir, logger, enabled: bool):
        # 若enabled为False, 则writer和selected_module都为空
        self.writer = None  # tensorboard的记录器
        self.selected_module = ""  # 选择哪种tensorboard

        if enabled:
            log_dir = str(log_dir)

            # 这俩导入一个就行
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                          "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                          "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'classification_config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        # 这是tensorboard.SummaryWriter的函数库
        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding',
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}  # 这些里面没有mode
        self.timer = datetime.now()  # 当前时间

    def set_step(self, step, mode='train'):
        """
        修改self的mode, step.
        """
        self.mode = mode  # mode可以表示train/valid
        self.step = step  # step表示第几轮吧
        # 将时间设置为当下时间。若step非0，则添加标量
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    # 动态返回对象的一个属性self.name
    def __getattr__(self, name):
        """

        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)  # add_data是tensorboard的name函数

            # 对add_data函数进行装饰
            # tag是数据的标签,data是数据, 剩下的是其他参数
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    if name not in self.tag_mode_exceptions:
                        tag = f'{tag}/{self.mode}'
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper

        else:
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr
