import os
import shutil
import logging
from pathlib import Path  # 专门操作路径的对象
from functools import reduce, partial  # 一些高级函数模块
from operator import getitem  # 运算符模块c实现，getitem(obj,k)得到obj[k]

from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    """
    Class to parse configuration json file. Handles hyper_parameters for training, initializations of modules,
    checkpoint saving and logging module.

    Parameters:
        config: Dict containing configurations, hyperparameters for training. contents of `classification_config.json` file for example.
        resume: String, data_dir to the checkpoint being loaded.
        modification: Dict keychain:value, specifying position values to be replaced from config dict.(修改参数的字典)
    """

    def __init__(self, config, resume=None, modification=None):
        self._config = _update_config(config, modification)  # json+更新后的参数(有序字典)
        self.resume = resume  # None或这检查点路径

        save_dir = Path(self.config['trainer']['save_dir'])  # 模型保存一级目录(就是data)
        exper_name = self.config['name']  # 模型保存二级目录

        self._save_dir = save_dir / exper_name / 'model'# 模型保存目录
        self._log_dir = save_dir / exper_name / 'log'  # log保存目录

        if os.path.exists(self._save_dir):
            shutil.rmtree(self._save_dir)
        if os.path.exists(self._log_dir):
            shutil.rmtree(self._log_dir)
        os.makedirs(self._save_dir)
        os.makedirs(self._log_dir)

        # 保存更新后的config到checkpoint目录
        write_json(self.config, self.save_dir / 'classification_config.json')

        # 导入logging
        setup_logging(self.log_dir)
        self.log_levers = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG  # 最低等级
        }

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            # *操作符将元组解包，*[a,b]等价于a,b
            # 对于key:value元组，使用**解包。**{a:1,b:2}等价于a=1,b=2
            args.add_argument(*opt.flags, default=None, type=opt.type)
        # 如果args是元组，那返回True
        if not isinstance(args, tuple):
            # 将参数实例化给args调用(一般都执行这句, 执行完后也不是元组)
            args = args.parse_args()

        # 若设置了device就改变环境变量（案例没有）
        if args.device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device

        if args.resume is not None:
            resume = Path(args.resume)
            # resume.parent返回父级目录
            cfg_fname = resume.parent / 'classification_config.json'  # 若从检查点开始训练，config文件与检查点在同一目录(不用指定-c了)
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c classification_config.json', for example."
            assert args.config is not None, msg_no_cfg   # 既不从检查点训练，又没指定config，则报错
            resume = None
            cfg_fname = Path(args.config)  # config文件路径(说明-c,-r参数二选一)

        config = read_json(cfg_fname)  # 返回.json的有序字典(笑死！！有序字典就是个特殊的列表)

        # 如果-r个-c都指定了，那么config按-c的来
        if args.config and resume:
            config.update(read_json(args.config))

        # getatter(对象, name)返回对象的属性
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        # modifaication是{'optimizer;args;lr': None, 'data_loader;args;batch_size': None}
        return cls(config, resume, modification)  # 返回该类实例化后的对象

    def init_obj(self, name, module, *args, **kwargs):
        """
        init_obj(name, module, a, b=1)相当于module.name(a, b=1)
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        # 若json文件不包括传入的参数kwargs，则报错
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        # 哦！！！这是字典的方法.更新相应key的参数
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        ...

    def __getitem__(self, name):
        """得到索引的值"""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        """
        得到相应名字和记录等级的记录器
        """
        msg_verbosity = f'verbosity option {verbosity} is invalid. Valid options are {self.log_levers.keys()}.'
        assert verbosity in self.log_levers, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levers[verbosity])
        return logger

    # 设置一些只读属性
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


def _get_opt_name(flags):
    """
    输入flags列表：['--lr', '--learning_rate'],返回第一个元素去--后的name
    """
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _update_config(config, modification):
    """
    输入config(有序字典),和修改字典。返回更新后的config
    """
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _set_by_path(tree, keys, value):
    """
    修改字典中key的value
    """
    keys = keys.split(';')  # 将key中的分号分开
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """
    在字典中找到keys对应的object
    """
    # reduce连续调用keys中的元素代入getitem，初始元素为tree
    # getitem(obj,k)得到obj[k]
    return reduce(getitem, keys, tree)
