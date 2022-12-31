import numpy as np
import os
import json
import random
import traceback
from torch.utils.data import Dataset
from .imaug import transform, create_operators


class SimpleDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        # 构造函数
        super(SimpleDataSet, self).__init__()
        # logger和训练模式
        self.logger = logger
        self.mode = mode

        # 全局配置、数据集配置、加载器配置
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        # 设置图片路径与标签分隔符
        self.delimiter = dataset_config.get('delimiter', '\t')
        # 获取标签地址
        label_file_list = dataset_config.pop('label_file_list')
        # 数据源个数
        data_source_num = len(label_file_list)
        # 获取数据源比例（即每数据集抽取比例）
        ratio_list = dataset_config.get('ratio_list', 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(ratio_list) == data_source_num, "The length of ratio_list should be the same as the file_list."
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']
        self.seed = seed
        logger.info('Initialize indexs of datasets:%s' % label_file_list)
        # 获取数据行
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        if self.mode == 'Train' and self.do_shuffle:
            self.shuffle_data_random()
        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx")

        self.need_reset = True in [x < 1 for x in ratio_list]

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                # 读取label文件
                lines = f.readlines()
                if self.mode == 'Train' or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    # 打乱
                    lines = random.sample(lines, round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def _try_parse_filename_list(self, file_name):
        if len(file_name) > 0 and file_name[0] == "[":
            try:
                info = json.loads(file_name)
                file_name = random.choice(info)
            except:
                pass
        return file_name

    def get_ext_data(self):
        # 没用到，暂时没看
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, 'ext_data_num'):
                ext_data_num = getattr(op, 'ext_data_num')
                break
        load_data_ops = self.ops[:self.ext_op_transform_idx]
        ext_data = []
        while len(ext_data) < ext_data_num:
            file_idx = self.data_idx_order_list[np.random.randint(self.__len__())]
            data_line = self.data_lines[file_idx]
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                continue
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data = transform(data, load_data_ops)

            if data is None:
                continue
            if 'polys' in data.keys():
                if data['polys'].shape[1] != 4:
                    continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
        # 获取图片
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip('\n').split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} dose not exist!".format(img_path))
            with open(data['img_path'], "rb") as f:
                img = f.read()
                data['image'] = img
            data['ext_data'] = self.get_ext_data()
            # 对图像进行预处理
            outs = transform(data, self.ops)
        except:
            self.logger.error(
                "When parsing line {}, error happened with msg:{}".format(data_line, traceback.format_exc()))
            outs = None
        if outs is None:
            # eval阶段，需要修复idx在多次推理中得到相同的结果
            rnd_idx = np.random.randint(self.__len__()) if self.mode == 'Train' else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)
