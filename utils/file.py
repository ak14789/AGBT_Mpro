import os
import shutil
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def download(url: str, path: str, save_file: str = None) -> str:
    """
    Download a file from the specified url.
    Skip the downloading step if there exists a file.

    Parameters:
        url (str): URL to download
        path (str): data_dir to store the downloaded file
        save_file (str, optional): name of save file. If not specified, infer the file name from the URL.

    Return:
        save_file (str): Relative data_dir of the file
    """
    from six.moves.urllib.request import urlretrieve

    if save_file is None:
        save_file = os.path.basename(url)
        if "?" in save_file:
            save_file = save_file[:save_file.find("?")]
    save_file = os.path.join(path, save_file)

    if not os.path.exists(save_file):
        logger.info("Downloading %s to %s" % (url, save_file))
        urlretrieve(url, save_file)
    return save_file


def generate_bt_input(train_dataset, valid_dataset, fpdata_dir: Path) -> str:
    """
    输入Dataset，在相应目录下生成bert的输入。返回文件夹名称。顺便把微调模型的目录也创建了,返回微调模型目录

    Parameters:
        train_dataset: 该Dataset应包含path,use_classification等属性,给文件夹起名用
        valid_dataset: 验证集Dataset
        fpdata_dir: fp预处理文件保存的路径

    Return:
        model_dir
    """
    if not os.path.exists(fpdata_dir):
        os.mkdir(fpdata_dir)
    if not os.path.exists(fpdata_dir / 'input0'):
        os.mkdir(fpdata_dir / 'input0')
        os.mkdir(fpdata_dir / 'label')

    if train_dataset.use_classification:
        column = 'activity'
    else:
        column = 'pIC50'
        train_dataset.label.to_csv(fpdata_dir / 'label/train.label', columns=[column], header=0, index=0)
        valid_dataset.label.to_csv(fpdata_dir / 'label/valid.label', columns=[column], header=0, index=0)

    train_dataset.data.to_csv(fpdata_dir / 'train.id', columns=['CID'], header=0, index=0)
    valid_dataset.data.to_csv(fpdata_dir / 'valid.id', columns=['CID'], header=0, index=0)
    train_dataset.data.to_csv(fpdata_dir / 'train_canonical.smi', columns=['SMILES'], header=0, index=0)
    valid_dataset.data.to_csv(fpdata_dir / 'valid_canonical.smi', columns=['SMILES'], header=0, index=0)
    train_dataset.label.to_csv(fpdata_dir / 'train.label', columns=[column], header=0, index=0)
    valid_dataset.label.to_csv(fpdata_dir / 'valid.label', columns=[column], header=0, index=0)
    np.save(fpdata_dir / 'train_y.npy', train_dataset.label.values)
    np.save(fpdata_dir / 'valid_y.npy', valid_dataset.label.values)

    shutil.copyfile('models/dict.txt', fpdata_dir / 'dict.txt')
    shutil.copyfile('models/dict.txt', fpdata_dir / 'input0/dict.txt')

    # 创建模型目录
    model_dir = 'models/' + str(fpdata_dir.name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    return model_dir


def create_preprocess_sh(fpdata_dir: str, wait: bool = False):
    """
    输入fpdata保存的目录，输出preprocess_xxx.sh。根据目录名自动判断分类还是回归，分类有两行要预处理
    """
    fpdata_dir = fpdata_dir.replace('\\', '/')
    preprocess_type = fpdata_dir[fpdata_dir.find('/') + 1: fpdata_dir.find('_')]  # regression or classification

    string1 = f'python preprocess.py --only-source --trainpref'
    string1 += f' ../{fpdata_dir}/train_canonical.smi'
    string1 += f' --validpref ../{fpdata_dir}/valid_canonical.smi'
    string1 += f' --destdir ../{fpdata_dir}/input0'
    string1 += f' --trainoutf train --validoutf valid'
    string1 += f' --workers 1 --file-format smiles'
    string1 += f' --srcdict ../models/dict.txt'

    string2 = '\n'
    if preprocess_type == 'classification':
        string2 += f'python preprocess.py --only-source --trainpref'
        string2 += f' ../{fpdata_dir}/train.label'
        string2 += f' --validpref ../{fpdata_dir}/valid.label'
        string2 += f' --destdir ../{fpdata_dir}/label'
        string2 += f' --trainoutf train --validoutf valid'
        string2 += f' --workers 1 --file-format smiles'

    with open(f"agbt_pro/preprocess_{fpdata_dir[fpdata_dir.find('/') + 1:]}.sh", 'w') as f:
        f.write(string1)
        f.write(string2)

    if wait:
        wait_for_sh('preprocess')


def create_train_sh(fpdata_dir: str, model_dir: str, len_data=int, wait: bool = False):
    """
    输入目录，和训练数据长度，输出train_xxx.sh。根据目录判断分类还是回归，输出方式略有差别
    """
    fpdata_dir = fpdata_dir.replace('\\', '/')
    model_dir = model_dir.replace('\\', '/')

    preprocess_type = fpdata_dir[fpdata_dir.find('/') + 1: fpdata_dir.find('_')]  # regression or classification

    string = f'train_data_len={len_data}\nnum_epoch=50\nnum_sent_pergpu=16\nupdata_freq=1\n'
    string += 'num_warmup=`expr $num_epoch \* $train_data_len / ${num_sent_pergpu} / $updata_freq / 10 `\n'
    string += 'max_num_update=100000\n'
    string += 'CUDA_VISIBLE_DEVICES=0\n'
    string += f'python train.py ../{fpdata_dir} '
    string += f'--save-dir ../{model_dir} '
    string += '--train-subset train --valid-subset valid \\\n'
    string += '--restore-file ../models/checkpoint_best.pt \\\n'
    string += '--task sentence_prediction \\\n'
    if preprocess_type == 'classification':
        string += '--num-classes 2 --init-token 0 --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \\\n'
    else:
        string += '--num-classes 1 --regression-target --init-token 0 --best-checkpoint-metric loss \\\n'
    string += '--arch roberta_base --bpe smi --encoder-attention-heads 8 --encoder-embed-dim 512 --encoder-ffn-embed-dim 1024 \\\n'
    string += '--encoder-layers 8 --dropout 0.1 --attention-dropout 0.1 --criterion sentence_prediction --max-positions 256 \\\n'
    string += '--truncate-sequence --skip-invalid-size-inputs-valid-test --optimizer adam --adam-betas "(0.9,0.999)" \\\n'
    string += '--adam-eps 1e-6 --clip-norm 0.0 --lr-scheduler polynomial_decay --lr 0.0001 \\\n'
    string += '--warmup-updates ${num_warmup} --total-num-update  ${max_num_update} --max-update ${max_num_update} --max-epoch ${num_epoch} --weight-decay 0.1 --log-format simple \\\n'
    string += '--reset-optimizer --reset-dataloader --reset-meters --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state --find-unused-parameters \\\n'
    string += '--log-interval 5 --max-sentences ${num_sent_pergpu} --update-freq ${updata_freq} --required-batch-size-multiple 1 --ddp-backend no_c10d'

    with open(f'agbt_pro/train_{fpdata_dir[fpdata_dir.find("/") + 1:]}.sh', 'w') as f:
        f.write(string)

    if wait:
        wait_for_sh('train')


def create_btfp_sh(fpdata_dir: str, model_dir: str, wait: bool = False, test: bool = False):
    """
    输入数据和模型的目录,是否是测试集，输出generate_xxx.sh（包括两行，train一行valid一行）
    """
    fpdata_dir = fpdata_dir.replace('\\', '/')
    model_dir = model_dir.replace('\\', '/')

    string = 'python generate_bt_fps.py \\\n'
    string += f'--model_name_or_path ../{model_dir}/ \\\n'
    string += '--checkpoint_file checkpoint_best.pt \\\n'
    string += f'--data_name_or_path ../{fpdata_dir}/ \\\n'
    string += f'--dict_file ../{fpdata_dir}/dict.txt \\\n'

    if not test:
        string1 = f'--target_file ../{fpdata_dir}/train_canonical.smi \\\n'
        string1 += f'--save_feature_path ../{fpdata_dir}/train_canonical_bt.npy\n'

        string2 = f'--target_file ../{fpdata_dir}/valid_canonical.smi \\\n'
        string2 += f'--save_feature_path ../{fpdata_dir}/valid_canonical_bt.npy\n'

        string = string + string1 + string + string2
    # !!!!!!!!!!!!!!!!!!!!!!!下面可能有问题
    else:
        string += f'--target_file ../{fpdata_dir}/test_canonical.smi \\\n'
        string += f'--save_feature_path ../{fpdata_dir}/test_canonical_bt.npy\n'

    with open(f'agbt_pro/generate_{fpdata_dir[fpdata_dir.find("/") + 1:]}.sh', 'w') as f:
        f.write(string)

    if wait:
        wait_for_sh('generate')


def wait_for_sh(sh_name):
    print(f'>>>>>>退出程序,在agbt_pro目录下运行{sh_name}_xxx.sh文件,然后进入下一步>>>>>>')
    while True:
        inputs = input('进入下一步输入ok, 退出程序输入exit:')
        if inputs == 'ok':
            break
        elif inputs == 'exit':
            exit()
    print('')


def save_result_to_csv(dataloader, path: Path):
    train_result = pd.concat([dataloader.train_dataset.data, dataloader.train_dataset.label], axis=1)
    valid_result = pd.concat([dataloader.valid_dataset.data, dataloader.valid_dataset.label], axis=1)
    train_result['pred'] = dataloader.train_dataset.pred.to('cpu').numpy()
    valid_result['pred'] = dataloader.valid_dataset.pred.to('cpu').numpy()

    train_result.to_csv(path / 'train_result.csv', index=False)
    valid_result.to_csv(path / 'valid_result.csv', index=False)
