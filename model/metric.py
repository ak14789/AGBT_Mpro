from sklearn.metrics import r2_score


def accuracy(output, target):
    """
    output和target都是张量
    """
    acc_num = 0
    for i in range(len(output)):
        output[i] = 1 if output[i] >= 0.5 else 0
        if output[i] == target[i]:
            acc_num += 1
    acc = acc_num / len(output) * 100
    return acc


def recall(output, target):
    """
    output和target都是张量
    """
    recall_num = 0  # 查全个数
    recall_all = 0  # 查全分母
    for i in range(len(output)):
        output[i] = 1 if output[i] >= 0.5 else 0
        if output[i] == target[i] == 1:
            recall_num += 1
        if target[i] == 1:
            recall_all += 1
    if recall_all == 0:
        return 1
    recall_ = recall_num / recall_all * 100
    return recall_


def precision(output, target):
    """
    output和target都是张量
    """
    precision_num = 0  # 查准个数
    precision_all = 0 # 查准分母
    for i in range(len(output)):
        output[i] = 1 if output[i] >= 0.5 else 0
        if output[i] == target[i] == 1:
            precision_num += 1
        if output[i] == 1:
            precision_all += 1
    if precision_all == 0:
        return 1
    precision_ = precision_num / precision_all * 100
    return precision_


def f1_score(output, target):
    """
    output和target都是张量
    """
    precision_num = 0  # 查准个数
    precision_all = 0  # 查准分母
    recall_all = 0  # 查全分母
    for i in range(len(output)):
        output[i] = 1 if output[i] >= 0.5 else 0
        if output[i] == target[i] == 1:
            precision_num += 1
        if output[i] == 1:
            precision_all += 1
        if target[i] == 1:
            recall_all += 1
    if precision_all == 0:
        precision_ = 1
    else:
        precision_ = precision_num / precision_all * 100
    if recall_all == 0:
        recall_ = 1
    else:
        recall_ = precision_num / recall_all * 100
    f1_score_ = 2 * precision_ * recall_ / (precision_ + recall_)
    return f1_score_


def r2(output, target):
    """都是张量"""
    output = output.detach().cpu()
    target = target.detach().cpu()
    return r2_score(target, output)
