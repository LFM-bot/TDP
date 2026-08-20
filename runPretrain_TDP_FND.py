import argparse
import copy
import logging
import math
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import src.model as model
from src.dataset.dataset import load_specified_dataset
from src.dataset.data_processor import DataProcessor
from src.evaluation.estimator import Estimator
from src.train.config import config_override
from src.utils.utils import set_seed, tensor_to_device
from train_utils import (
    build_optimizer, save_pretrain_checkpoint, setup_logging,
)

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='TDP_FND', type=str)
    parser.add_argument('--model_type', default='General', choices=['General', 'LLM-based'], type=str)
    parser.add_argument('--warmup_epochs', default=0, type=int)
    parser.add_argument('--num_expert', default=2, type=int)
    parser.add_argument('--beta', default=0.7, type=float)

    parser.add_argument('--tdp_stage', default='source', choices=['source', 'target'], type=str)
    parser.add_argument('--tdp_source_loss', default='multiclass', choices=['multiclass', 'binary'], type=str,
                        help='source-stage loss: multiclass=6-way CE on prototypes; binary=CE on bin_logits (real vs fake)')
    parser.add_argument('--tdp_adapter_rank', default=64, type=int)
    parser.add_argument('--tdp_fusion_layers', default=2, type=int)
    parser.add_argument('--tdp_fusion_heads', default=8, type=int)
    parser.add_argument('--tdp_moe_rank', default=64, type=int)
    parser.add_argument('--tdp_expert_type', default='adapter', choices=['adapter', 'transformer'], type=str,
                        help="MoE expert: 'adapter' (default) or 'transformer' (timm Block, param-scaling variant)")
    parser.add_argument('--tdp_num_patterns', default=64, type=int)
    parser.add_argument('--tdp_cluster_interval', default=1, type=int)
    parser.add_argument('--tdp_cluster_chunk_size', default=4096, type=int)
    parser.add_argument('--tdp_temperature', default=0.1, type=float)
    parser.add_argument('--tdp_text_anchor_mode', default='live', choices=['live', 'frozen'], type=str,
                        help='live: recompute type-text anchors each forward; frozen: cache from pretrained encoder')
    parser.add_argument('--lambda_pcl', default=0.5, type=float)
    parser.add_argument('--lambda_moe', default=0.1, type=float)
    parser.add_argument('--eta_moe', default=0.1, type=float)
    parser.add_argument('--beta_div', default=0.01, type=float)

    parser.add_argument('--pretrain_datasets', default=['AMG'], nargs='+', type=str,
                        help='source datasets (must carry fake_type_id/fg_label for 6-way supervision)')
    parser.add_argument('--temperature_alpha', default=0.5, type=float)
    parser.add_argument('--pretrain_epoch', default=10, type=int)
    parser.add_argument('--steps_per_epoch', default=0, type=int)
    parser.add_argument('--pretrain_ckpt_name', default='tdp_pretrain', type=str)
    parser.add_argument('--pretrain_save_dir', default='save/pretrain', type=str)
    parser.add_argument('--overwrite_pretrain_ckpt', action='store_true')
    parser.add_argument('--save_interval', default=5, type=int,
                        help='Save an epoch checkpoint every N epochs. 0 disables interval saving.')
    parser.add_argument('--eval_dataset', default=None, type=str,
                        help='dataset for per-epoch val/test evaluation (default: first pretrain dataset)')

    parser.add_argument('--dataset', default='AMG', type=str)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--max_text_len', default=197, type=int)

    parser.add_argument('--train_batch', default=24, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--encoder_learning_rate', default=1e-5, type=float)
    parser.add_argument('--num_worker', default=4, type=int)
    parser.add_argument('--l2', default=0.0, type=float)
    parser.add_argument('--seed', default=2027, type=int)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--mark', default='pretrain_tdp_text_anchor_live', type=str)
    parser.add_argument('--log_save', default='log', type=str)

    parser.add_argument('--eval_batch', default=24, type=int)
    parser.add_argument('--split_type', default='valid_and_test', choices=['valid_only', 'valid_and_test'])
    parser.add_argument('--split_mode', default='PS', type=str)
    parser.add_argument('--metric', default=['acc', 'precision', 'recall', 'f1'], type=str, nargs='+')
    parser.add_argument('--valid_metric', default='acc')
    parser.add_argument('--test_device', default='cpu', type=str)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--epoch_num', default=10, type=int)

    return parser

def build_model_config(cmd_config):
    model_config = getattr(model, f'{cmd_config.model}_config')()
    config = config_override(model_config, cmd_config)
    config.model_type = config.model_type.upper()
    config.graph_type = [g_type.upper() for g_type in config.graph_type]
    return config

def prepare_dataset_loader(base_config, dataset_name):
    dataset_config = copy.deepcopy(base_config)
    dataset_config.dataset = dataset_name
    processor = DataProcessor(dataset_config)
    data_dict, additional_data_dict = processor.prepare_data()
    specified_dataset = load_specified_dataset(base_config.model, dataset_config)
    train_dataset = specified_dataset(dataset_config, data_dict['train'], additional_data_dict)
    train_loader = DataLoader(
        train_dataset,
        batch_size=base_config.train_batch,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    fg_col = None
    for col in ['fake_type_id', 'fg_label']:
        if col in data_dict['train'].columns:
            fg_col = col
            break
    num_fake = int((data_dict['train']['label'] == 0).sum())
    num_real = int((data_dict['train']['label'] == 1).sum())
    return {
        'dataset': train_dataset,
        'loader': train_loader,
        'num_fake': num_fake,
        'num_real': num_real,
        'has_fg_label': fg_col is not None,
        'fg_col': fg_col,
    }

def cycle_next(loader_iters, loaders, dataset_name):
    try:
        return next(loader_iters[dataset_name])
    except StopIteration:
        loader_iters[dataset_name] = iter(loaders[dataset_name])
        return next(loader_iters[dataset_name])

def prepare_eval_loaders(base_config, dataset_name, batch_size):
    dataset_config = copy.deepcopy(base_config)
    dataset_config.dataset = dataset_name
    processor = DataProcessor(dataset_config)
    data_dict, additional_data_dict = processor.prepare_data()
    specified_dataset = load_specified_dataset(base_config.model, dataset_config)
    eval_dataset = specified_dataset(dataset_config, data_dict['eval'], additional_data_dict, mode='eval')
    test_dataset = specified_dataset(dataset_config, data_dict['test'], additional_data_dict, mode='test')
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=eval_dataset.collate_fn,
        shuffle=False, num_workers=0, drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn,
        shuffle=False, num_workers=0, drop_last=False,
    )
    return eval_loader, test_loader

@torch.no_grad()
def evaluate_fine_grained(model, loader, device, log_prefix, desc='attr eval'):
    model.eval()
    attr_preds, attr_labels = [], []
    for batch in tqdm(loader, desc=desc):
        tensor_to_device(batch, device)
        out = model.inner_forward(batch)
        attr_pred = out['sim'].argmax(dim=-1)
        attr_preds.extend(attr_pred.cpu().tolist())
        attr_labels.extend(batch['fg_label'].cpu().tolist())
    attr_preds = np.array(attr_preds)
    attr_labels = np.array(attr_labels)

    six = {
        'acc': accuracy_score(attr_labels, attr_preds),
        'macro_f1': f1_score(attr_labels, attr_preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(attr_labels, attr_preds, average='weighted', zero_division=0),
    }
    mask = attr_labels > 0
    if mask.sum() > 0:
        five = {
            'acc': accuracy_score(attr_labels[mask], attr_preds[mask]),
            'macro_f1': f1_score(attr_labels[mask], attr_preds[mask], average='macro',
                                 labels=[1, 2, 3, 4, 5], zero_division=0),
            'weighted_f1': f1_score(attr_labels[mask], attr_preds[mask], average='weighted',
                                    labels=[1, 2, 3, 4, 5], zero_division=0),
        }
    else:
        five = {k: 0.0 for k in ['acc', 'macro_f1', 'weighted_f1']}

    report = classification_report(attr_labels, attr_preds, labels=[0, 1, 2, 3, 4, 5],
                                   digits=4, zero_division=0)
    logging.info(f'{log_prefix} attribution 6-way report:\n{report}')
    logging.info(f'{log_prefix} attribution 6-way: acc={six["acc"]:.4f} macro_f1={six["macro_f1"]:.4f} '
                 f'weighted_f1={six["weighted_f1"]:.4f}')
    logging.info(f'{log_prefix} attribution 5-way (fake): acc={five["acc"]:.4f} macro_f1={five["macro_f1"]:.4f} '
                 f'weighted_f1={five["weighted_f1"]:.4f}')
    return six, five

def main():
    cmd_config = build_parser().parse_args()
    set_seed(cmd_config.seed)
    setup_logging(cmd_config.log_save, cmd_config.model, cmd_config.mark)
    config = build_model_config(cmd_config)
    device = torch.device(config.device)

    dataset_infos = {}
    total_real, total_fake = 0, 0
    for dataset_name in config.pretrain_datasets:
        logging.info(f'Preparing dataset: {dataset_name}')
        dataset_infos[dataset_name] = prepare_dataset_loader(config, dataset_name)
        total_real += dataset_infos[dataset_name]['num_real']
        total_fake += dataset_infos[dataset_name]['num_fake']
        fg_flag = dataset_infos[dataset_name]['fg_col'] or 'binary(pseudo)'
        logging.info(
            f'{dataset_name}: train={len(dataset_infos[dataset_name]["dataset"])} '
            f'fake={dataset_infos[dataset_name]["num_fake"]} real={dataset_infos[dataset_name]["num_real"]} '
            f'label={fg_flag}'
        )

    config.pos_weight = total_real / max(total_fake, 1)
    dataset_id_map = {name: idx for idx, name in enumerate(config.pretrain_datasets)}
    dataset_sizes = np.array(
        [len(dataset_infos[d]['dataset']) for d in config.pretrain_datasets], dtype=np.float64
    )
    sample_probs = dataset_sizes ** config.temperature_alpha
    sample_probs = sample_probs / sample_probs.sum()
    logging.info(f'Dataset id map: {dataset_id_map}')
    logging.info(f'Temperature sampling probabilities: {dict(zip(config.pretrain_datasets, sample_probs.tolist()))}')

    Model = getattr(model, config.model)
    training_model = Model(config, {}).to(device)
    optimizer = build_optimizer(training_model, config.encoder_learning_rate, config.learning_rate, config.l2)

    loaders = {name: dataset_infos[name]['loader'] for name in config.pretrain_datasets}
    loader_iters = {name: iter(loaders[name]) for name in config.pretrain_datasets}
    total_samples = sum(len(dataset_infos[d]['dataset']) for d in config.pretrain_datasets)
    steps_per_epoch = (
        config.steps_per_epoch
        if config.steps_per_epoch > 0
        else math.ceil(total_samples / config.train_batch)
    )
    largest_dataset = config.pretrain_datasets[int(dataset_sizes.argmax())]
    logging.info(
        f'Effective batch={config.train_batch}, total_samples={total_samples}, '
        f'steps/epoch={steps_per_epoch}, discovery_loader={largest_dataset}'
    )

    eval_dataset_name = config.eval_dataset or config.pretrain_datasets[0]
    eval_loader, test_loader = prepare_eval_loaders(config, eval_dataset_name, config.eval_batch)
    estimator = Estimator(config)
    logging.info(
        f'Per-epoch evaluation on {eval_dataset_name}: '
        f'val={len(eval_loader.dataset)} test={len(test_loader.dataset)}'
    )

    logging.info(f'Start TDP source pretraining for {config.pretrain_epoch} epochs.')
    best_val_acc = -1.0
    best_info = None
    for epoch in range(config.pretrain_epoch):
        logging.info(f'Discovering pattern centers before pretrain epoch {epoch}.')
        training_model.discover_patterns(loaders[largest_dataset], device, epoch=epoch)

        training_model.train()
        epoch_loss = 0.0
        loss_parts = defaultdict(float)
        dataset_loss_sum = defaultdict(float)
        dataset_step_count = defaultdict(int)
        progress = tqdm(range(steps_per_epoch), desc=f'tdp pretrain epoch {epoch}')
        for _ in progress:
            dataset_name = random.choices(config.pretrain_datasets, weights=sample_probs.tolist(), k=1)[0]
            batch_dict = cycle_next(loader_iters, loaders, dataset_name)
            batch_size = batch_dict['label'].size(0)
            batch_dict['dataset_id'] = torch.full(
                (batch_size,), dataset_id_map[dataset_name], dtype=torch.long
            )
            batch_dict['dataset_name'] = dataset_name
            tensor_to_device(batch_dict, device)

            loss = training_model.calc_loss(batch_dict)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(training_model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_value = float(loss.item())
            epoch_loss += loss_value
            dataset_loss_sum[dataset_name] += loss_value
            dataset_step_count[dataset_name] += 1
            if hasattr(training_model, '_last_loss_dict'):
                for key, value in training_model._last_loss_dict.items():
                    loss_parts[key] += value
            progress.set_postfix({'loss': f'{loss_value:.4f}', 'dataset': dataset_name})

        logging.info(f'Epoch {epoch} average loss: {epoch_loss / max(steps_per_epoch, 1):.4f}')
        for key, value in loss_parts.items():
            logging.info(f'Epoch {epoch} average {key}: {value / max(steps_per_epoch, 1):.4f}')
        for dataset_name in config.pretrain_datasets:
            if dataset_step_count[dataset_name] > 0:
                logging.info(
                    f'Epoch {epoch} {dataset_name} average loss: '
                    f'{dataset_loss_sum[dataset_name] / dataset_step_count[dataset_name]:.4f} '
                    f'({dataset_step_count[dataset_name]} steps)'
                )

        epoch_num = epoch + 1
        if config.save_interval > 0 and epoch_num % config.save_interval == 0:
            save_pretrain_checkpoint(config, training_model, dataset_id_map, sample_probs, epoch=epoch_num)

        val_metrics, val_loss = estimator.evaluate(eval_loader, training_model)
        val_metrics = dict(val_metrics)
        test_metrics, test_loss = estimator.evaluate(test_loader, training_model)
        test_metrics = dict(test_metrics)
        logging.info(
            f'Epoch {epoch} eval on {eval_dataset_name}: '
            f'val_loss={val_loss:.4f} val_acc={val_metrics.get("acc", 0):.4f} '
            f'val_p={val_metrics.get("precision", 0):.4f} val_r={val_metrics.get("recall", 0):.4f} '
            f'val_f1={val_metrics.get("f1", 0):.4f} || '
            f'test_loss={test_loss:.4f} test_acc={test_metrics.get("acc", 0):.4f} '
            f'test_p={test_metrics.get("precision", 0):.4f} test_r={test_metrics.get("recall", 0):.4f} '
            f'test_f1={test_metrics.get("f1", 0):.4f}'
        )
        logging.info(f'Epoch {epoch} val metrics: {val_metrics}')
        logging.info(f'Epoch {epoch} test metrics: {test_metrics}')

        attr_six, attr_five = evaluate_fine_grained(training_model, test_loader, device,
                                                    log_prefix=f'epoch {epoch} test', desc=f'attr epoch {epoch}')

        val_acc = val_metrics.get('acc', 0.0)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_info = {
                'epoch': epoch,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'attr_six': attr_six,
                'attr_five': attr_five,
            }

        training_model.train()

    save_pretrain_checkpoint(config, training_model, dataset_id_map, sample_probs)

    if best_info is not None:
        bv = best_info['val_metrics']
        bt = best_info['test_metrics']
        attr_str = ''
        if best_info.get('attr_six') is not None:
            attr_str = (f'attr6_wf1={best_info["attr_six"]["weighted_f1"]:.4f} '
                        f'attr5_wf1={best_info["attr_five"]["weighted_f1"]:.4f}')
        logging.info(
            f'Best source pretrain result (selected by val acc): '
            f'epoch={best_info["epoch"]} '
            f'val_acc={bv.get("acc", 0):.4f} '
            f'test_acc={bt.get("acc", 0):.4f} test_p={bt.get("precision", 0):.4f} '
            f'test_r={bt.get("recall", 0):.4f} test_f1={bt.get("f1", 0):.4f} {attr_str}'
        )
        logging.info(f'Best val metrics: {bv}')
        logging.info(f'Best test metrics: {bt}')
    else:
        logging.info('No evaluation was performed; no best result to report.')

if __name__ == '__main__':
    main()
