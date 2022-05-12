import datasets
from torch.utils.data import DataLoader
from utils import train_valid_split, generate_bt_input, create_preprocess_sh, create_train_sh
from utils import create_btfp_sh, create_docking_csv


class MproDataLoader(DataLoader):
    """
    Mpro dataset的专属dataloader
    """
    def __init__(self, data_dir, use_classification, train_rate, fpdata_dir, batch_size, shuffle, num_workers,
                 npy_type='bt', **kwargs):
        self.data_dir = data_dir
        self.dataset = datasets.Mpro(data_dir, use_classification, **kwargs)  # npy和score等信息在子集里
        self.train_rate = train_rate

        if 0 < self.train_rate < 1:
            self.process_fn = None
        else:
            self.process_fn = datasets.extra_other_series

        # 数据集划分
        self.train_dataset, self.valid_dataset = train_valid_split(self.dataset, self.train_rate, self.process_fn)

        # 生成bt输入文件,模型保存文件创建。得到模型路径
        self.model_dir = generate_bt_input(self.train_dataset, self.valid_dataset, fpdata_dir)

        # 回归模式生成mpro_docking.csv文件(因为要docking)
        if not use_classification:
            create_docking_csv(self.data_dir)

        fpdata_dir = str(fpdata_dir)
        # 生成preprocess_xxx.sh文件
        create_preprocess_sh(fpdata_dir, wait=True)
        # 生成train_xxx.sh文件
        create_train_sh(fpdata_dir, self.model_dir, len(self.train_dataset), wait=True)
        # 生成generate_xxx.sh文件
        create_btfp_sh(fpdata_dir, self.model_dir, wait=True)

        # 加载bt_npy
        self.train_dataset.load_npy(fpdata_dir, filename='train_canonical_bt')
        self.valid_dataset.load_npy(fpdata_dir, filename='valid_canonical_bt')


        # 以下为回归模式额外的数据处理
        if not self.dataset.use_classification:
            ...
            # 去除docking有问题的分子

            # 挑选mol2文件
            #
            # # 生成ag分子指纹
            #
            # # 特征融合
            #
            # # 生成融合后的分子指纹
            #
            # 每种分子指纹导入dataset中

        # 将feature设置为某种npy
        self.train_dataset.set_feature(npy_type)
        self.valid_dataset.set_feature(npy_type)

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers
        }
        super().__init__(self.train_dataset, shuffle=shuffle, **self.init_kwargs)

    def split_validation(self):
        return DataLoader(self.valid_dataset, **self.init_kwargs)
