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
from model.utils.utility import print_dict, AverageMeter
from model.utils.loggers import Loggers
from model.utils.stats import TrainingStats
from model.utils.schedulers import WarmupPolyLR


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


def get_lr(config, name, module, *args, **kwargs):
    module_name = config[name]['type']
    module_args = config[name]['args']
    assert all([k not in module_args for k in kwargs])
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)


def train(config,
          train_dataloader,
          valid_dataloader,
          device,
          model,
          loss_class,
          post_process_class,
          eval_class,
          logger,
          use_tensorboard=False,
          scaler=None
          ):
    log_smooth_window = config['Global']['log_smooth_window']
    epoch_num = config['Global']['epoch_num']
    print_batch_step = config['Global']['print_batch_step']
    eval_batch_step = config['Global']['eval_batch_step']

    global_step = 0
    start_eval_step = 0
    if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0]
        eval_batch_step = eval_batch_step[1]
        if len(valid_dataloader) == 0:
            logger.info(
                'No Images in eval dataset, evaluation during training ' \
                'will be disabled'
            )
            start_eval_step = 1e111
        logger.info(
            "During the training process, after the {}th iteration, " \
            "an evaluation is run every {} iterations".
            format(start_eval_step, eval_batch_step)
        )
    save_epoch_step = config['Global']['save_epoch_step']
    save_model_dir = config['Global']['save_model_dir']
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    # 主要评价参数
    main_indicator = eval_class.main_indicator
    best_model_dict = {main_indicator: 0}
    train_stats = TrainingStats(log_smooth_window, ['lr'])
    model_average = False
    model.train()
    model.to(device)

    if use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tensor_log_dir = save_model_dir + '/tensor_log'
        if not os.path.exists(tensor_log_dir):
            os.makedirs(tensor_log_dir)
        writer = SummaryWriter(tensor_log_dir)
        try:
            img_mode = config['Train']['dataset']['transforms']
            for idx in range(len(img_mode)):
                if img_mode[idx].get('DecodeImage') is not None:
                    img_mode = img_mode[idx]['DecodeImage']['img_mode']
                    break
            in_channels = 3 if img_mode != 'GRAY' else 1
            dummy_input = torch.zeros(1, in_channels, 640, 640).to(device)
            writer.add_graph(model, dummy_input, use_strict_trace=False)
            torch.cuda.empty_cache()
        except:
            import traceback
            logger.error(traceback.format_exc())
            logger.warn('add graph to tensorboard failed')

    algorithm = config['Architecture']['algorithm']

    start_epoch = best_model_dict[
        'start_epoch'] if 'start_epoch' in best_model_dict else 1

    total_samples = 0
    train_batch_cost = 0.0
    reader_start = time.time()
    eta_meter = AverageMeter()
    warmup_iters = config['Optimizer']['lr_scheduler']['args']['warmup_epoch'] * len(train_dataloader)
    optimizer = get_lr(config, 'Optimizer', torch.optim, model.parameters())
    scheduler = WarmupPolyLR(optimizer, max_iters=epoch_num * len(train_dataloader),
                             warmup_iters=warmup_iters, **config['Optimizer']['lr_scheduler']['args'])

    max_iter = len(train_dataloader) - 1 if platform.system() == "Windows" else len(train_dataloader)

    for epoch in range(start_epoch, epoch_num + 1):
        for idx, batch in enumerate(train_dataloader):
            if idx >= max_iter:
                break
            lr = optimizer.param_groups[0]['lr']
            for idx, value in enumerate(batch):
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[idx] = value.to(device)
            images = batch[0]
            # model_type == det
            model_type = config['Architecture']['model_type']
            preds = model(images)
            loss = loss_class(preds, batch)
            avg_loss = loss['loss']
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()

            if config['Optimizer']['lr_scheduler']['type'] == 'WarmupPolyLR':
                scheduler.step()

            train_batch_time = time.time() - reader_start
            train_batch_cost += train_batch_time
            eta_meter.update(train_batch_time)
            global_step += 1
            total_samples += len(images)

            # logger and tensorboard
            stats = {}
            for k, v in loss.items():
                v = v.to('cpu')
                stats[k] = v.detach().numpy().mean()
            stats['lr'] = lr
            train_stats.update(stats)

            if (global_step > 0 and global_step % print_batch_step == 0) or (idx >= len(train_dataloader) - 1):
                logs = train_stats.log()
                eta_sec = ((epoch_num + 1 - epoch) * \
                           len(train_dataloader) - idx - 1) * eta_meter.avg
                eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))
                strs = 'epoch: [{}/{}], global_step: {}, {}, \n' \
                       ' \t\t\t\t\t\t\t\t ips: {:.5f} samples/s, eta: {}'.format(
                    epoch, epoch_num, global_step, logs,
                    total_samples / train_batch_cost, eta_sec_format)
                logger.info(strs)

                total_samples = 0
                train_batch_cost = 0.0

            # eval
            if global_step > start_eval_step and (global_step - start_eval_step) % eval_batch_step == 0:
                cur_metric = eval(
                    model,
                    valid_dataloader,
                    post_process_class,
                    eval_class
                )
                cur_metric_str = 'cur metric, {}'.format(', '.join(
                    ['{}: {}'.format(k, v) for k, v in cur_metric.items()]
                ))
                logger.info(cur_metric_str)
            reader_start = time.time()


def eval(model,
         valid_dataloader,
         post_process_class,
         eval_class):
    model.eval()
    with torch.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(
            total=len(valid_dataloader),
            desc='eval model:',
            position=0,
            leave=True
        )
        max_iter = len(valid_dataloader) - 1 if platform.system() == 'Windows' else len(valid_dataloader)
        sum_images = 0
        for idx, batch in enumerate(valid_dataloader):
            if idx >= max_iter:
                break
            model.to('cpu')
            images = batch[0]
            start = time.time()

            preds = model(images)

            batch_numpy = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    batch_numpy.append(item.detach().numpy())
                else:
                    batch_numpy.append(item)

            total_time += time.time() - start
            post_result = post_process_class(preds, batch_numpy[1])
            eval_class(post_result, batch_numpy)

            pbar.update(1)
            total_frame += len(images)
            sum_images += 1
        metric = eval_class.get_metric()

    pbar.close()
    model.train()
    metric['fps'] = total_frame / total_time
    return metric
