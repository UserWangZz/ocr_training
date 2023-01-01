import logging
import os
import imghdr
import cv2
import random
import numpy as np
import torch


def print_dict(d, logger, delimiter=0):
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, logger, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, logger, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ", k, v))


def set_seed(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    # 设置 CPU 生成随机数的 种子
    torch.manual_seed(seed)
    # 设置当前GPU的随机数生成种子
    torch.cuda.manual_seed(seed)
    # 设置所有GPU的随机数生成种子
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
