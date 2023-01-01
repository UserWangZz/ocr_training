import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

import yaml
import torch
import torch.distributed as dist

from model.data import build_dataloader
from model.modeling.architectures import build_model
from model.losses import build_loss
from model.metrics import build_metric
from model.utils.utility import set_seed
from model.postprocess import build_post_process
import tools.program as program


def main(config, device, logger, log_writer):
    global_config = config['Global']
    # 构建dataloader
    train_dataloader = build_dataloader(config, 'Train', device, logger, config['Global']['seed'])
    print(len(train_dataloader))
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n" +
            "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            +
            "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    # 构建验证集dataloader
    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', device, logger, config['Global']['seed'])
    else:
        valid_dataloader = None

    # build post process
    post_porcess_class = build_post_process(config['PostProcess'], global_config)

    # build model

    model = build_model(config['Architecture'])
    # build loss
    loss_class = build_loss(config['Loss'])

    eval_class = build_metric(config['Metric'])
    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info('valid dataloader has {} iters'.format(
            len(valid_dataloader)))

    # start train
    program.train(config, train_dataloader, valid_dataloader, device,
                  model, loss_class, post_porcess_class, eval_class,
                  logger, config['Global']['use_tensorboard'])


if __name__ == '__main__':
    # 预处理，读取config文件，构造config，log以及log_writer
    config, device, logger, log_writer = program.preprocess(is_train=True)
    # 设置随机种子，方便结果复现
    seed = config['Global']['seed'] if 'seed' in config['Global'] else 1024
    set_seed(seed)
    # 训练
    main(config, device, logger, log_writer)
