import copy
import torch
import torch.nn as nn

from .det_db_loss import DBLoss

def build_loss(config):
    support_dict = ['DBLoss']
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('loss only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class