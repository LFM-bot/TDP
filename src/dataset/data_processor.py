import codecs
import copy
import logging
import math
import pickle

import pandas as pd
import scipy.sparse as sp
import numpy as np
import torch
import os
from scipy import sparse

from src.dataset.dataset import load_specified_dataset
from src.utils.utils import save_pickle, load_pickle


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.model_name = config.model
        self.dataset = config.dataset
        self.data_aug = config.data_aug

        self.graph_type_list = [g_type.upper() for g_type in config.graph_type]
        self.valid_graphs = ['GGNN', 'BIPARTITE', 'TRANSITION', 'HYPER']

        self.data_path = None
        self.train_data = None
        self.eval_data = None
        self.test_data = None
        self.do_data_split = True

        self.split_type = config.split_type
        self.split_mode = config.split_mode
        self.eval_ratio = 0.1

        self.statistic = None

        self._init_data_processer()

    def _init_data_processer(self):
        if self.split_mode == 'PS':
            self.do_data_split = False
            self.eval_ratio = self.config.eval_ratio
        elif 'LS_R' == self.split_mode.split('@')[0]:
            self.eval_ratio = float(self.split_mode.split('@')[-1])
            self.split_mode = 'LS_R'
        self._set_data_path()

    def prepare_data(self):
        if self.do_data_split:
            seq_data_list = self._load_row_data()
            self._train_test_split(seq_data_list)
        else:
            self._load_pre_split_data()

        data_dict = {'train': self.train_data,
                     'eval': self.eval_data,
                     'test': self.test_data}

        extra_data_dict = self._prepare_additional_data()

        return data_dict, extra_data_dict

    def _prepare_additional_data(self):
        cur_dataset_cls = load_specified_dataset(self.model_name, self.config)
        return cur_dataset_cls.load_additional_data(self.config, self.image_data_path)

    def _set_data_path(self):
        csv_root = '/mnt1/userhome/tangpang/shichenglong/proj/MORE-FND/dataset'
        image_root = '/mnt1/userhome/tangpang/shichenglong/proj/MMFakeNewDetector/dataset'
        self.data_path = os.path.join(csv_root, self.dataset)
        self.image_data_path = os.path.join(image_root, self.dataset)
        self.config.data_path = self.data_path
        self.config.image_data_path = self.image_data_path

    def _read_csv_data(self, file_path):
        df = pd.read_csv(file_path)
        columns = ['image', 'text', 'label', 'event']
        for extra in ['fg_label', 'fake_type_id', 'confidence']:
            if extra in df.columns:
                columns.append(extra)
        selected_df = df[columns]
        return selected_df

    def _load_row_data(self):
        file_path = os.path.join(self.data_path, f'{self.dataset}.seq')
        seq_data_list = self._read_csv_data(file_path)
        self._set_statistic(seq_data_list)

        return seq_data_list

    def _set_statistic(self):
        statistic = {}

        all_df = pd.concat([self.train_data, self.test_data], ignore_index=True)
        all_df = pd.concat([all_df, self.eval_data], ignore_index=True) if self.eval_data is not None else all_df

        assert all_df['label'].isin([0, 1]).all(), "Label values must be 0 or 1 !"


        statistic['num_event'] = all_df['event'].nunique()
        statistic['num_image'] = all_df['image'].nunique()
        statistic['num_total_fake'] = all_df['label'].sum()
        statistic['num_total_real'] = len(all_df) - all_df['label'].sum()
        statistic['num_total'] = len(all_df)

        statistic['num_train'] = len(self.train_data)
        statistic['num_train_fake'] = self.train_data['label'].sum()
        statistic['num_train_real'] = len(self.train_data) - self.train_data['label'].sum()

        statistic['num_eval'] = len(self.eval_data) if self.eval_data is not None else 0
        statistic['num_eval_fake'] = self.eval_data['label'].sum() if self.eval_data is not None else 0
        statistic['num_eval_real'] = len(self.eval_data) - self.eval_data[
            'label'].sum() if self.eval_data is not None else 0

        statistic['num_test'] = len(self.test_data) if self.test_data is not None else 0
        statistic['num_test_fake'] = self.test_data['label'].sum() if self.test_data is not None else 0
        statistic['num_test_real'] = len(self.test_data) - self.test_data[
            'label'].sum() if self.test_data is not None else 0

        self.config.pos_weight = statistic["num_train_real"] / statistic["num_train_fake"]

        self.statistic = statistic

    def _load_pre_split_data(self):
        _suffix = getattr(self.config, 'train_csv_suffix', '')
        train_file = os.path.join(self.data_path, f'{self.dataset}_train{_suffix}.csv')
        test_file = os.path.join(self.data_path, f'{self.dataset}_test.csv')
        val_file = os.path.join(self.data_path, f'{self.dataset}_val.csv')

        train_df = self._read_csv_data(train_file)
        test_df = self._read_csv_data(test_file)
        self.train_data = self.clean_data(train_df)

        _train_ratio = getattr(self.config, 'train_ratio', 1.0)
        if _train_ratio < 1.0 and isinstance(self.train_data, pd.DataFrame):
            _unit = getattr(self.config, 'subsample_unit', 'row')
            if _unit == 'image':
                img_keys = self.train_data.drop_duplicates('image')
                sampled_imgs = (img_keys.sort_values('image').reset_index(drop=True)
                                .groupby('label', group_keys=False)
                                .sample(frac=_train_ratio, random_state=42))
                self.train_data = (self.train_data[
                                       self.train_data['image'].isin(sampled_imgs['image'])]
                                   .reset_index(drop=True))
            else:
                self.train_data = (self.train_data.sort_values('image').reset_index(drop=True)
                                   .groupby('label', group_keys=False)
                                   .sample(frac=_train_ratio, random_state=42))

        has_val = os.path.exists(val_file)
        eval_df = self._read_csv_data(val_file) if has_val else test_df
        self.eval_data = self.clean_data(eval_df)

        if self.split_type == 'valid_and_test':
            self.test_data = self.clean_data(test_df) if has_val else self.eval_data

        self._set_statistic()

    def clean_data(self, data):
        data['text'] = data['text'].fillna(' ')
        data['image'] = data['image'].str.lower()

        return data
    def _train_test_split(self, seq_data_list):
        if self.split_type == 'valid_only':
            train_x, train_y, eval_x, eval_y = self._leave_one_out_split(seq_data_list)
        else:
            if self.split_mode == 'LS':
                train_x = [item_seq[:-3] for item_seq in seq_data_list if len(item_seq) > 3]
                train_y = [item_seq[-3] for item_seq in seq_data_list if len(item_seq) > 3]
                eval_x = [item_seq[:-2] for item_seq in seq_data_list if len(item_seq) > 2]
                eval_y = [item_seq[-2] for item_seq in seq_data_list if len(item_seq) > 2]
                test_x = [item_seq[:-1] for item_seq in seq_data_list if len(item_seq) > 1]
                test_y = [item_seq[-1] for item_seq in seq_data_list if len(item_seq) > 1]
            else:
                train_x, train_y, test_x, test_y = self._leave_one_out_split(seq_data_list)
                eval_x, eval_y, test_x, test_y = self._split_by_ratio(test_x, test_y)
            self.test_data = (test_x, test_y)

        self.row_train_data = (copy.deepcopy(train_x), copy.deepcopy(train_y))
        self._data_augmentation(train_x, train_y)

        self.train_data = (train_x, train_y)
        self.eval_data = (eval_x, eval_y)

    def data_log_verbose(self, order):
        logging.info(f'[{order}] Data Statistic '.ljust(47, '-'))
        logging.info(f'dataset: {self.dataset}')
        logging.info(f'event number: {self.statistic["num_event"]}')
        logging.info(f'image number: {self.statistic["num_image"]}')
        logging.info(
            f'total size: {self.statistic["num_total"]} (fake:{self.statistic["num_total_fake"]}, real:{self.statistic["num_total_real"]})')
        if self.data_aug:
            logging.info(f'data after augmentation:')
            if self.split_type == 'valid_only':
                logging.info(f'train samples: {len(self.train_data)}\teval samples: {len(self.eval_data)}')
            else:
                logging.info(f'train samples: {len(self.train_data)}\teval samples: {len(self.eval_data)}\ttest '
                             f'samples: {len(self.test_data)}')
        else:
            logging.info(f'data without augmentation:')
            if self.split_type == 'valid_only':
                logging.info(f'train samples: {len(self.train_data)}\teval samples: {len(self.eval_data)}')
            else:
                logging.info(
                    f'train size: {len(self.train_data)}(fake:{self.statistic["num_train_fake"]}, real:{self.statistic["num_train_real"]}) ' + \
                    f'eval size: {len(self.eval_data)}(fake:{self.statistic["num_eval_fake"]}, real:{self.statistic["num_eval_real"]}) ' + \
                    f'test size: {len(self.test_data)}(fake:{self.statistic["num_test_fake"]}, real:{self.statistic["num_test_real"]})')
