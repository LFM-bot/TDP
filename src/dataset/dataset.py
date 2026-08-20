import logging
import os
import pickle
import random

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, default_collate
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

chinese_datasets = ["weibo_CN", "weibo21"]

def load_specified_dataset(model_name, config):
    if model_name == 'TDP_FND':
        return TDP_FNDDataset
    raise NotImplementedError(
        f'Unsupported model: {model_name}. Only TDP_FND is available.'
    )

class BaseMultiModalityFakeNewsDataset(Dataset):
    chinese_datasets = ['weibo_CN', 'weibo21']

    def __init__(self, config, data, additional_data_dict=None, mode='train'):
        super(BaseMultiModalityFakeNewsDataset, self).__init__()
        assert mode in ['train', 'eval', 'test'], 'Mode must be train, eval or test !'
        self.mode = mode
        self.config = config
        self.data_path = config.data_path
        _image_data_path = getattr(config, 'image_data_path', config.data_path)
        self.img_root = os.path.join(_image_data_path, 'images')
        self.dataset = config.dataset
        self.max_text_len = config.max_text_len
        key_column = ['image', 'text', 'label', 'event']

        for column in key_column:
            assert column in data.columns, f"Column '{column}' not found in {mode} data !"

        self.data = data
        self.additional_data = additional_data_dict
        self.batch_dict = {}

    def clean_data(self, data):
        data['text'] = data['text'].fillna(' ')
        data['image'] = data['image'].str.lower()

        return data

    def __len__(self):
        return len(self.data)

    def update(self):
        pass

    @staticmethod
    def load_additional_data(config, data_path):
        return {}

class MMFakeNewsDataset(BaseMultiModalityFakeNewsDataset):
    def __init__(self, config, data, additional_data_dict=None, mode='train'):
        super(MMFakeNewsDataset, self).__init__(config, data, additional_data_dict, mode)
        self.model_name = config.model
        self.image_size = config.image_size
        self.data_path = config.data_path
        self.tokenizer = additional_data_dict['tokenizer']
        self.img_name2tensor = additional_data_dict['img_name2tensor']

    @staticmethod
    def language_is_chinese(dataset):
        is_chinese = dataset in chinese_datasets
        return is_chinese

    def select_image(self, images: str):
        imgs = images.split("|")
        random.shuffle(imgs)

        for img in imgs:
            if img in self.img_name2tensor:
                return img
        raise ValueError(f'No valid image found for {images}')

class MIMoE_FNDDataset(MMFakeNewsDataset):
    def __init__(self, config, data, additional_data_dict=None, mode='train'):
        super(MIMoE_FNDDataset, self).__init__(config, data, additional_data_dict, mode)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = self.select_image(row['image'])
        image, text, label = self.img_name2tensor[image], row['text'], row['label']

        cur_tensors = (image,
                       text,
                       torch.tensor(label, dtype=torch.long),
                       torch.tensor(idx, dtype=torch.long))

        return cur_tensors

    def collate_fn(self, x):
        if MIMoE_FNDDataset.language_is_chinese(self.dataset):
            return self.collate_fn_chinese(x)
        return self.collate_fn_english(x)

    def collate_fn_chinese(self, data):
        image, text, label, sample_id = default_collate(data)

        token_chinese = self.tokenizer
        token_data = token_chinese.batch_encode_plus(
            batch_text_or_text_pairs=text,
            truncation=True,
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt",
            return_length=True,
        )

        self.batch_dict['image'] = image
        self.batch_dict['input_ids'] = token_data["input_ids"]
        self.batch_dict['attention_mask'] = token_data["attention_mask"]
        self.batch_dict['token_type_ids'] = token_data["token_type_ids"]
        self.batch_dict['clip_inputs'] = None
        self.batch_dict['label'] = label
        self.batch_dict['sample_id'] = sample_id

        return self.batch_dict

    def collate_fn_english(self, data):
        image, text, label, sample_id = default_collate(data)

        token_english = self.tokenizer
        token_data = token_english.batch_encode_plus(
            batch_text_or_text_pairs=text,
            truncation=True,
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt",
            return_length=True,
        )

        self.batch_dict['image'] = image
        self.batch_dict['input_ids'] = token_data["input_ids"]
        self.batch_dict['attention_mask'] = token_data["attention_mask"]
        self.batch_dict['token_type_ids'] = token_data["token_type_ids"]
        self.batch_dict['clip_inputs'] = None
        self.batch_dict['label'] = label
        self.batch_dict['sample_id'] = sample_id

        return self.batch_dict

    @staticmethod
    def load_image(data_path):
        img_tensor_dict_path = f'{data_path}/img_name2tensor_only_resize.pkl'

        if os.path.exists(img_tensor_dict_path):
            img_name2tensor = pickle.load(open(img_tensor_dict_path, 'rb'))
            logging.info(f'Loaded {len(img_name2tensor)} image tensors from {img_tensor_dict_path}.')
        else:
            data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            img_name2tensor = {}
            image_dir = os.path.join(data_path, 'images')
            for img_name in tqdm(os.listdir(image_dir), desc='Loading images'):
                im = Image.open(os.path.join(image_dir, img_name)).convert('RGB')
                im = data_transforms(im)
                try:
                    img_name_prefix, img_type = img_name.split('/')[-1].split(".")
                    img_name_lower = img_name_prefix.lower() + '.' + img_type
                    img_name2tensor[img_name_lower] = im
                except Exception as e:
                    print(e)
                    print(f'invalid image: {img_name}')

            pickle.dump(img_name2tensor, open(img_tensor_dict_path, 'wb'))
            logging.info(f'Saved {len(img_name2tensor)} image tensors to {img_tensor_dict_path}.')

        return img_name2tensor

    @staticmethod
    def load_additional_data(config, data_path):
        additional_data_dict = {}
        lan_type = 'chinese' if MIMoE_FNDDataset.language_is_chinese(config.dataset) else 'uncased'
        additional_data_dict['tokenizer'] = AutoTokenizer.from_pretrained(
            f'/mnt1/userhome/tangpang/shichenglong/proj/LLMs/bert-base-{lan_type}')
        additional_data_dict['img_name2tensor'] = MIMoE_FNDDataset.load_image(data_path)

        return additional_data_dict

class TDP_FNDDataset(MIMoE_FNDDataset):

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = self.select_image(row['image'])
        image, text, label = self.img_name2tensor[image], row['text'], row['label']
        if 'fake_type_id' in self.data.columns:
            fg_val = row['fake_type_id']
        elif 'fg_label' in self.data.columns:
            fg_val = row['fg_label']
        else:
            fg_val = None
        fg_label = int(fg_val) if (fg_val is not None and fg_val == fg_val) else (0 if label == 0 else 1)
        if 'confidence' in self.data.columns:
            conf_val = row['confidence']
            confidence = float(conf_val) if (conf_val == conf_val) else 0.0
        else:
            confidence = 0.0
        cur_tensors = (
            image,
            text,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(fg_label, dtype=torch.long),
            torch.tensor(confidence, dtype=torch.float),
            torch.tensor(idx, dtype=torch.long),
        )
        return cur_tensors

    def collate_fn(self, x):
        image, text, label, fg_label, confidence, sample_id = default_collate(x)
        token_data = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            truncation=True,
            padding='max_length',
            max_length=self.max_text_len,
            return_tensors='pt',
            return_length=True,
        )
        self.batch_dict['image'] = image
        self.batch_dict['input_ids'] = token_data['input_ids']
        self.batch_dict['attention_mask'] = token_data['attention_mask']
        self.batch_dict['token_type_ids'] = token_data['token_type_ids']
        self.batch_dict['clip_inputs'] = None
        self.batch_dict['label'] = label
        self.batch_dict['fg_label'] = fg_label
        self.batch_dict['confidence'] = confidence
        self.batch_dict['sample_id'] = sample_id
        return self.batch_dict
