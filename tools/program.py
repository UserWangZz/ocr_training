import os
import sys
import platform
import yaml
import time
import datetime

import torch
from tqdm import tqdm
import cv2
import numpy as np
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from model.utils.logging import get_logger
from model.utils.utility import print_dict
from model.utils.loggers import Loggers


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument('-c', '--config', help="configuration file to use")
        self.add_argument('-o', '--opt', nargs='+', help="set configuration options")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config


def load_config(file_path):
    # 加载config文件
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config


def merge_config(config, opts):
    # 更新来自命令行的配置
    for key, value in opts.items():
        if '.' not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split('.')
            assert (sub_keys[0] in config), "the sub_keys can only be one of global_config: {}, but get: " \
                                            "{}, please check your running command".format(
                config.keys(), sub_keys[0])
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config


def preprocess(is_train=False):
    # 解析命令行配置
    FLAGS = ArgsParser().parse_args()
    # 根据命令行地址加载配置文件
    config = load_config(FLAGS.config)
    # 更新来自命令行的配置
    config = merge_config(config, FLAGS.opt)

    if is_train:
        # 保存config
        save_model_dir = config["Global"]['save_model_dir']
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, 'config.yml'), "w") as f:
            yaml.dump(dict(config), f, default_flow_style=False, sort_keys=False)
        log_file = '{}/train.log'.format(save_model_dir)
    else:
        log_file = None
    # 初始化logger
    logger = get_logger(log_file=log_file)

    use_gpu = config['Global'].get('use_gpu', False)

    alg = config['Architecture']['algorithm']
    assert alg in ['DB']

    device = 'cuda:{}'.format(config['Global']['gpu_id']) if use_gpu else 'cpu'

    loggers = []

    log_writer = None
    print_dict(config, logger)

    if loggers:
        log_writer = Loggers(loggers)
    else:
        log_writer = None

    logger.info('train with pytorch {} and device {}'.format(torch.__version__,
                                                             device))
    return config, device, logger, log_writer
