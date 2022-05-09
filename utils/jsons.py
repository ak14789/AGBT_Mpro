import json
from pathlib import Path
from collections import OrderedDict  # 有序字典


def read_json(fname):
    """
    输入.json的Path或路径，返回反序列化后的有序字典
    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)  # 文件缩进为4，不对key进行排序
