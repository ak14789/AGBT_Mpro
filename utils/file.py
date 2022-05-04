import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


def download(url: str, path: str, save_file: str = None) -> str:
    """
    Download a file from the specified url.
    Skip the downloading step if there exists a file.

    Parameters:
        url (str): URL to download
        path (str): path to store the downloaded file
        save_file (str, optional): name of save file. If not specified, infer the file name from the URL.

    Return:
        save_file (str): Relative path of the file
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


def generate_bt_input(dataset, data_type: str, split_type: str = 'norm') -> str:
    """
    输入Dataset，在相应目录下生成bert的输入。返回文件夹名称

    Parameters:
        dataset: 该Dataset应包含path,use_classification等属性,给文件夹起名用
        data_type (str): 'train' or 'valid'
        split_type (str): 数据集划分的方式，如正常的8:2或者什么特殊的方法

    Return:
        stor_name (str): 文件夹名称
    """
    assert data_type in ['train', 'valid'], 'data_type should be "train" or "valid"'

    stor_name = dataset.path + '/'
    if dataset.use_classification:
        stor_name += f'classification_{dataset.splitter}_{split_type}'
        column = 'activity'
    else:
        stor_name += f'regression_{dataset.splitter}_{split_type}_{dataset.norm_type}'
        column = 'pIC50'

    if not os.path.exists(stor_name):
        os.mkdir(stor_name)
        os.mkdir(stor_name + '/input0')
        os.mkdir(stor_name + '/label')

    dataset.data.to_csv(stor_name + f'/{data_type}.id', columns=['CID'], header=0, index=0)
    dataset.data.to_csv(stor_name + f'/{data_type}_canonical.smi', columns=['SMILES'], header=0, index=0)
    dataset.label.to_csv(stor_name + f'/{data_type}.label', columns=[column], header=0, index=0)
    np.save(stor_name + f'/{data_type}_y.npy', dataset.label.values)

    if not dataset.use_classification:
        dataset.label.to_csv(stor_name + f'/label/{data_type}.label', columns=[column], header=0, index=0)

    return stor_name


def create_preprocess_sh(data_stor_name: str):
    """
    输入data目录，输出preprocess_xxx.sh。根据目录名自动判断分类还是回归，分类有两行要预处理
    """
    preprocess_type = data_stor_name[data_stor_name.find('/') + 1: data_stor_name.find('_')]  # regression or classification

    string1 = f'python preprocess.py --only-source --trainpref'
    string1 += f' ../{data_stor_name}/train_canonical.smi'
    string1 += f' --validpref ../{data_stor_name}/valid_canonical.smi'
    string1 += f' --destdir ../{data_stor_name}/input0'
    string1 += f' --trainoutf train --validoutf valid'
    string1 += f' --workers 1 --file-format smiles'
    string1 += f' --srcdict ../models/dict.txt'

    string2 = '\n'
    if preprocess_type == 'classification':
        string2 += f'python preprocess.py --only-source --trainpref'
        string2 += f' ../{data_stor_name}/train.label'
        string2 += f' --validpref ../{data_stor_name}/valid.label'
        string2 += f' --destdir ../{data_stor_name}/label'
        string2 += f' --trainoutf train --validoutf valid'
        string2 += f' --workers 1 --file-format smiles'

    with open(f'agbt_pro/preprocess_{data_stor_name[data_stor_name.find("/") + 1:]}.sh', 'w') as f:
        f.write(string1)
        f.write(string2)


def create_train_sh(data_stor_name: str, model_stor_name: str, length=int):
    """
    输入目录，和训练数据长度，输出train_xxx.sh。根据目录判断分类还是回归，输出方式略有差别
    """
    preprocess_type = data_stor_name[data_stor_name.find('/') + 1: data_stor_name.find('_')]  # regression or classification

    string = f'train_data_len={length}\nnum_epoch=50\nnum_sent_pergpu=16\nupdata_freq=1\n'
    string += 'num_warmup=`expr $num_epoch \* $train_data_len / ${num_sent_pergpu} / $updata_freq / 10 `\n'
    string += 'max_num_update=100000\n'
    string += 'CUDA_VISIBLE_DEVICES=0\n'
    string += f'python train.py ../{data_stor_name} '
    string += f'--save-dir ../{model_stor_name} '
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

    with open(f'agbt_pro/train_{data_stor_name[data_stor_name.find("/") + 1:]}.sh', 'w') as f:
        f.write(string)


def create_btfp_sh(data_stor_name: str, model_stor_name: str, test: bool = False):
    """
    输入数据和模型的目录,是否是测试集，输出generate_xxx.sh（包括两行，train一行valid一行）
    """
    string = 'python generate_bt_fps.py \\\n'
    string += f'--model_name_or_path ../{model_stor_name}/ \\\n'
    string += '--checkpoint_file checkpoint_best.pt \\\n'
    string += f'--data_name_or_path ../{data_stor_name}/ \\\n'
    string += f'--dict_file ../{data_stor_name}/dict.txt \\\n'

    if not test:
        string1 = f'--target_file ../{data_stor_name}/train_canonical.smi \\\n'
        string1 += f'--save_feature_path ../{data_stor_name}/train_canonical.smi\n'

        string2 = f'--target_file ../{data_stor_name}/valid_canonical.smi \\\n'
        string2 += f'--save_feature_path ../{data_stor_name}/valid_canonical.smi\n'

        string = string + string1 + string + string2
    # !!!!!!!!!!!!!!!!!!!!!!!下面可能有问题
    else:
        string += f'--target_file ../{data_stor_name}/test_canonical.smi \\\n'
        string += f'--save_feature_path ../{data_stor_name}/test_canonical.smi\n'

    with open(f'agbt_pro/generate_{data_stor_name[data_stor_name.find("/") + 1:]}.sh', 'w') as f:
        f.write(string)
