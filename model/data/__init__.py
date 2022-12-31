import os
import sys
import numpy as np
import skimage
import torch
import signal
import random

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import copy
from torch.utils.data import DataLoader, BatchSampler

from model.data.imaug import transform, create_operators
from model.data.simple_dataset import SimpleDataSet

__all__ = ['build_dataloader', 'transform', 'create_operators']

def term_mp(sig_num, frame):
    """ kill all child processes
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)


def build_dataloader(config, mode, device, logger, seed=None):
    config = copy.deepcopy(config)
    # 支持数据集字典
    support_dict = ['SimpleDataSet']
    module_name = config[mode]['dataset']['name']
    assert module_name in support_dict, Exception(
        'DataSet only support {}'.format(support_dict))
    assert mode in ['Train', 'Eval', 'Test'
                    ], "Mode should be Train, Eval or Test."

    # eval获取对应对象
    dataset = eval(module_name)(config, mode, logger, seed)
    # 从config中获取
    loader_config = config[mode]['loader']
    batch_size = loader_config['batch_size_per_card']
    drop_last = loader_config['drop_last']
    shuffle = loader_config['shuffle']
    num_workers = loader_config['num_workers']

    # Distribute data to single card

    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)

    # support exit using ctrl+c
    signal.signal(signal.SIGINT, term_mp)
    signal.signal(signal.SIGTERM, term_mp)

    return data_loader
