import pickle
import logging
import datetime
import random
from typing import List, Dict, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time as t
import numpy as np
from easydict import EasyDict


def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x


def set_seed(seed):
    if seed == -1:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def tensor_to_device(tensor_dict: dict, dev):
    for key, obj in tensor_dict.items():
        if torch.is_tensor(obj):
            tensor_dict[key] = obj.to(dev)


def load_pickle(file_path):
    with open(file_path, 'rb') as fr:
        return pickle.load(fr)


def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def pkl_to_txt(dataset='beauty'):
    data_dir = '../../dataset'
    file_path = os.path.join(data_dir, f'{dataset}/{dataset}_seq.pkl')
    record = load_pickle(file_path)

    target_file = os.path.join(data_dir, f'{dataset}/{dataset}_seq.txt')
    with open(target_file, 'w') as fw:
        for seq in record:
            seq = list(map(str, seq))
            seq_str = ' '.join(seq) + '\n'
            fw.write(seq_str)


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


class HyperParamDict(EasyDict):
    def __init__(self, description=None):
        super(HyperParamDict, self).__init__({})
        self.description = description
        self.attr_registered = []

    def add_argument(self, param_name, type=object, default=None, action=None, choices=None, help=None):
        param_name = self._parse_param_name(param_name)
        if default and type:
            try:
                default = type(default)
            except Exception:
                assert isinstance(default, type), f'KeyError. Type of param {param_name} should be {type}.'
        if choices:
            assert isinstance(choices, List), f'choices should be a list.'
            assert default in choices, f'KeyError. Please choose {param_name} from {choices}. ' \
                                       f'Now {param_name} = {default}.'
        if action:
            default = self._parse_action(action)
        if help:
            assert isinstance(help, str), f'help should be a str.'
        self.attr_registered.append(param_name)
        self.__setattr__(param_name, default)

    @staticmethod
    def _parse_param_name(param_name: str):
        index = param_name.rfind('-')
        return param_name[index + 1:]

    @staticmethod
    def _parse_action(action):
        action_infos = action.split('_')
        assert action_infos[0] == 'store' and action_infos[-1] in ['false', 'true'], \
            f"Wrong action format: {action}. Please choose from ['store_false', 'store_true']."
        res = False if action_infos[-1] == 'true' else True
        return res

    def keys(self):
        return self.attr_registered

    def values(self):
        return [self.get(key) for key in self.attr_registered]

    def items(self):
        return [(key, self.get(key)) for key in self.attr_registered]

    def __str__(self):
        info_str = 'HyperParamDict{'
        param_list = []
        for key, value in self.items():
            param_list.append(f'({key}: {value})')
        info_str += ', '.join(param_list) + '}'
        return info_str


def get_gpu_usage(device=None):

    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    return '{:.2f} G/{:.2f} G'.format(reserved, total)


if __name__ == '__main__':
    datasets = ['ml-10m']
    for dataset in datasets:
        pkl_to_txt(dataset)
