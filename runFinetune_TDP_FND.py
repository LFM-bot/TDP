import argparse
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.model as model
from src.dataset.dataset import load_specified_dataset
from src.dataset.data_processor import DataProcessor
from src.evaluation.estimator import Estimator
from src.train.config import config_override
from src.utils.utils import set_seed, tensor_to_device
from train_utils import build_optimizer, freeze_module, setup_logging

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='TDP_FND', type=str)
    parser.add_argument('--model_type', default='General', choices=['General', 'LLM-based'], type=str)
    parser.add_argument('--agr_threshold', type=float, default=0.3)
    parser.add_argument('--sem_threshold', type=float, default=0.3)
    parser.add_argument('--warmup_epochs', default=0, type=int)
    parser.add_argument('--num_expert', default=2, type=int)
    parser.add_argument('--beta', default=0.7, type=float)

    parser.add_argument('--tdp_stage', default='target', choices=['source', 'target'], type=str)
    parser.add_argument('--tdp_source_loss', default='multiclass', choices=['multiclass', 'binary'], type=str,
                        help='source-stage loss (only used when tdp_stage=source)')
    parser.add_argument('--tdp_adapter_rank', default=64, type=int)
    parser.add_argument('--tdp_fusion_layers', default=2, type=int)
    parser.add_argument('--tdp_fusion_heads', default=8, type=int)
    parser.add_argument('--tdp_moe_rank', default=64, type=int)
    parser.add_argument('--tdp_expert_type', default='adapter', choices=['adapter', 'transformer'], type=str,
                        help="MoE expert: 'adapter' (default) or 'transformer' (timm Block, param-scaling variant). "
                             'Must match the source ckpt expert_type for the experts to load.')
    parser.add_argument('--tdp_num_patterns', default=64, type=int)
    parser.add_argument('--tdp_cluster_interval', default=1, type=int)
    parser.add_argument('--tdp_cluster_chunk_size', default=4096, type=int)
    parser.add_argument('--tdp_temperature', default=0.1, type=float)
    parser.add_argument('--tdp_input_dropout', default=0.0, type=float)
    parser.add_argument('--tdp_fusion_drop', default=0.0, type=float)
    parser.add_argument('--tdp_fusion_dropout', default=0.2, type=float,
                        help='dropout inside the fusion MLP (after the first Linear+ReLU)')
    parser.add_argument('--tdp_deep_dropout', default=0.0, type=float,
                        help='dropout on the deep post-pattern representation g before prototype scoring')
    parser.add_argument('--tdp_pattern_ffn_dropout', default=0.0, type=float,
                        help='dropout between the two Linears of the pattern FFN')
    parser.add_argument('--tdp_pattern_ffn_act', default='gelu', choices=['gelu', 'relu'], type=str,
                        help='activation in the pattern FFN (gelu or relu)')
    parser.add_argument('--tdp_text_anchor_mode', default='live', choices=['live', 'frozen'], type=str,
                        help='live: recompute type-text anchors each forward; frozen: cache from pretrained encoder')
    parser.add_argument('--lambda_pcl', default=0.5, type=float)
    parser.add_argument('--lambda_moe', default=0.1, type=float)
    parser.add_argument('--eta_moe', default=0.1, type=float)
    parser.add_argument('--beta_div', default=0.01, type=float)
    parser.add_argument('--use_llm_weak', action='store_true',
                        help='enable confidence-weighted L_weak + L_route^t on LLM-annotated target posts')
    parser.add_argument('--eta_w', default=0.1, type=float,
                        help='weight for L_weak (confidence-weighted fine-grained classification)')
    parser.add_argument('--llm_conf_threshold', default=0.5, type=float,
                        help='gamma: retain LLM labels with confidence >= gamma')
    parser.add_argument('--weak_rebalance', default='none',
                        choices=['none', 'unk_mean', 'unk_max', 'unk_cap'], type=str,
                        help='downsample over-represented unk (fake_type_id==6) weak '
                             'annotations so unk no longer dominates L_weak (and stops '
                             'p_unk collapsing into a universal attractor). The dropped '
                             'rows STAY in the train set for L_bin (confidence zeroed, '
                             'fake_type_id cleared); only the confidence-gated L_weak / '
                             'L_route^t skip them. unk_mean: cap unk to the mean count of '
                             'known-type (fti 0..5) retained rows; unk_max: cap to the max '
                             'known-type count; unk_cap: cap to --weak_unk_cap.')
    parser.add_argument('--weak_unk_cap', default=0, type=int,
                        help='target unk count when --weak_rebalance=unk_cap (0 = disabled)')
    parser.add_argument('--weak_rebalance_seed', default=2026, type=int,
                        help='seed for the deterministic unk downsampling')
    parser.add_argument('--llm_annot_cap', default=0, type=int,
                        help='cap the number of retained LLM annotations to N (0 = no '
                             'cap). Keeps a deterministic random subset of N retained '
                             'rows; the rest are zeroed (confidence=0, fake_type_id '
                             'cleared) so they still drive L_bin but skip L_weak / '
                             'L_route^t. For the annotation-quantity sweep.')
    parser.add_argument('--train_csv_suffix', default='', type=str,
                        help="suffix appended to the train CSV name, e.g. '_1k' -> "
                             "'{dataset}_train_1k.csv' (default '' = '{dataset}_train.csv'). "
                             'Lets the annotation-quantity sweep load the _1k annotated set.')
    parser.add_argument('--train_ratio', default=1.0, type=float,
                        help='use only this fraction of train data (stratified by label, seed 42); 1.0 = full set')
    parser.add_argument('--subsample_unit', default='row', type=str,
                        choices=['row', 'image'],
                        help="subsampling unit for --train_ratio: 'row' = per sample "
                             "(original behaviour); 'image' = per unique image, then expand "
                             "back to all its rows. Use 'image' for datasets with heavily "
                             "duplicated images (e.g. twitter) so per-ratio image coverage "
                             "grows smoothly instead of jumping.")

    parser.add_argument('--dataset', default='pheme', type=str)
    parser.add_argument('--pretrained_ckpt_path', default='save/pretrain/tdp_source_AMG_ep10_live_epoch10.pth', type=str)
    parser.add_argument('--freeze_encoders', action='store_true', default=True,
                        help='freeze both text and image encoders (paper default for target stage)')
    parser.add_argument('--finetune_encoders', action='store_true', default=False,
                        help='unfreeze BOTH text and image encoders for target-stage tuning')
    parser.add_argument('--finetune_text_encoder', action='store_true', default=False,
                        help='unfreeze the text encoder (overrides --freeze_encoders for text)')
    parser.add_argument('--finetune_image_encoder', action='store_true', default=False,
                        help='unfreeze the image encoder (overrides --freeze_encoders for image)')
    parser.add_argument('--freeze_text_encoder', action='store_true', default=False,
                        help='freeze only the text encoder')
    parser.add_argument('--freeze_image_encoder', action='store_true', default=False,
                        help='freeze only the image encoder')
    parser.add_argument('--finetune_save_dir', default='save/finetune', type=str)
    parser.add_argument('--finetune_ckpt_name', default=None, type=str)

    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--max_text_len', default=197, type=int)

    parser.add_argument('--epoch_num', default=10, type=int)
    parser.add_argument('--train_batch', default=24, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--encoder_learning_rate', default=1e-5, type=float)
    parser.add_argument('--num_worker', default=4, type=int)
    parser.add_argument('--l2', default=0.01, type=float)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--seed', default=2026, type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mark', default='finetune_tdp', type=str)
    parser.add_argument('--log_save', default='log', type=str)

    parser.add_argument('--eval_batch', default=24, type=int)
    parser.add_argument('--split_type', default='valid_and_test', choices=['valid_only', 'valid_and_test'])
    parser.add_argument('--split_mode', default='PS', type=str)
    parser.add_argument('--metric', default=['acc', 'precision', 'recall', 'f1'], type=str, nargs='+')
    parser.add_argument('--valid_metric', default='acc')
    parser.add_argument('--test_device', default='cpu', type=str)

    return parser

def build_model_config(cmd_config):
    model_config = getattr(model, f'{cmd_config.model}_config')()
    config = config_override(model_config, cmd_config)
    config.model_type = config.model_type.upper()
    config.graph_type = [g_type.upper() for g_type in config.graph_type]
    return config

def load_shape_compatible_state_dict(training_model, state_dict):
    model_state = training_model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append((key, tuple(value.shape),
                            tuple(model_state[key].shape) if key in model_state else None))
    missing, unexpected = training_model.load_state_dict(filtered, strict=False)
    for key, src_shape, tgt_shape in skipped:
        logging.info(f'skip (shape mismatch) {key}: src={src_shape} tgt={tgt_shape}')
    return missing, unexpected

def rebalance_weak_annotations(train_df, config):
    mode = getattr(config, 'weak_rebalance', 'none')
    if mode == 'none':
        return train_df
    if 'confidence' not in train_df.columns or 'fake_type_id' not in train_df.columns:
        logging.info('[weak-rebalance] no fake_type_id/confidence columns; skip.')
        return train_df

    thr = config.llm_conf_threshold
    unk_idx = 6
    df = train_df.copy()
    fti = pd.to_numeric(df['fake_type_id'], errors='coerce')
    conf = pd.to_numeric(df['confidence'], errors='coerce').fillna(0.0)
    label = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

    consistent = ((label == 0) & (fti == 0)) | ((label == 1) & (fti > 0))
    retained = (conf >= thr) & consistent & (fti >= 0) & (fti < 7)
    unk = retained & (fti == unk_idx)
    known = retained & (fti != unk_idx)
    known_counts = fti[known].astype(int).value_counts().to_dict()
    n_unk = int(unk.sum())
    n_retained = int(retained.sum())

    if n_unk == 0:
        logging.info(f'[weak-rebalance] no retained unk rows; skip. known={known_counts}')
        return df

    if mode == 'unk_cap':
        target = int(getattr(config, 'weak_unk_cap', 0))
    elif mode == 'unk_mean':
        target = int(round(sum(known_counts.values()) / len(known_counts))) if known_counts else 0
    elif mode == 'unk_max':
        target = int(max(known_counts.values())) if known_counts else 0
    else:
        target = n_unk
    target = max(0, min(target, n_unk))

    before_unk_pct = n_unk / max(n_retained, 1) * 100
    if target >= n_unk:
        logging.info(f'[weak-rebalance] mode={mode} unk={n_unk} <= target={target}; '
                     f'no downsample. known={known_counts}')
        return df

    rng = np.random.RandomState(getattr(config, 'weak_rebalance_seed', 2026))
    unk_indices = df.index[unk].to_numpy()
    drop_idx = rng.choice(unk_indices, size=n_unk - target, replace=False)
    df.loc[drop_idx, 'confidence'] = 0.0
    df.loc[drop_idx, 'fake_type_id'] = np.nan

    after_unk_pct = target / max(n_retained - (n_unk - target), 1) * 100
    logging.info(
        f'[weak-rebalance] mode={mode} seed={config.weak_rebalance_seed} '
        f'unk {n_unk}->{target} (unk% {before_unk_pct:.0f}->{after_unk_pct:.0f}); '
        f'dropped {n_unk - target} from L_weak (kept for L_bin). known={known_counts}'
    )
    return df

def cap_weak_annotations(train_df, config):
    n_cap = int(getattr(config, 'llm_annot_cap', 0) or 0)
    if n_cap <= 0:
        return train_df
    if 'confidence' not in train_df.columns or 'fake_type_id' not in train_df.columns:
        logging.info('[weak-cap] no fake_type_id/confidence columns; skip.')
        return train_df

    thr = config.llm_conf_threshold
    df = train_df.copy()
    fti = pd.to_numeric(df['fake_type_id'], errors='coerce')
    conf = pd.to_numeric(df['confidence'], errors='coerce').fillna(0.0)
    label = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

    consistent = ((label == 0) & (fti == 0)) | ((label == 1) & (fti > 0))
    retained = (conf >= thr) & consistent & (fti >= 0) & (fti < 7)
    n_retained = int(retained.sum())
    if n_retained <= n_cap:
        logging.info(f'[weak-cap] retained={n_retained} <= cap={n_cap}; no cap.')
        return df

    rng = np.random.RandomState(getattr(config, 'weak_rebalance_seed', 2026))
    retained_idx = df.index[retained].to_numpy()
    keep_idx = set(rng.choice(retained_idx, size=n_cap, replace=False))
    drop_idx = [i for i in retained_idx if i not in keep_idx]
    df.loc[drop_idx, 'confidence'] = 0.0
    df.loc[drop_idx, 'fake_type_id'] = np.nan

    logging.info(f'[weak-cap] cap={n_cap} seed={config.weak_rebalance_seed} '
                 f'retained {n_retained}->{n_cap} (dropped {n_retained - n_cap} '
                 f'from L_weak, kept for L_bin).')
    return df

def main():
    cmd_config = build_parser().parse_args()
    set_seed(cmd_config.seed)
    setup_logging(cmd_config.log_save, cmd_config.dataset, cmd_config.finetune_ckpt_name or cmd_config.mark)
    config = build_model_config(cmd_config)
    device = torch.device(config.device)

    processor = DataProcessor(config)
    data_dict, additional_data_dict = processor.prepare_data()
    specified_dataset = load_specified_dataset(config.model, config)

    train_df = data_dict['train']
    has_conf = 'confidence' in train_df.columns
    has_fti = 'fake_type_id' in train_df.columns
    if has_conf:
        n_annot = int(train_df['confidence'].notna().sum())
        n_retained = int((train_df['confidence'].fillna(0).astype(float) >= config.llm_conf_threshold).sum())
    else:
        n_annot = n_retained = 0
    raw_cols = []
    try:
        import pandas as _pd
        _suffix = getattr(config, 'train_csv_suffix', '')
        csv_path = os.path.join(config.data_path, f'{config.dataset}_train{_suffix}.csv')
        raw_cols = list(_pd.read_csv(csv_path, nrows=0).columns)
        logging.info(f'[weak-supervision] csv_file={csv_path}')
        logging.info(f'[weak-supervision] raw_csv_columns={raw_cols}')
    except Exception as e:
        logging.info(f'[weak-supervision] could not read raw csv header: {e}')
    logging.info(
        f'[weak-supervision] use_llm_weak={config.use_llm_weak} '
        f'has_fake_type_id={has_fti} has_confidence={has_conf} '
        f'annotated_rows={n_annot} retained(>={config.llm_conf_threshold})={n_retained} '
        f'(if has_confidence=False, the annotated CSV is not at the csv_file path above)'
    )
    data_dict['train'] = cap_weak_annotations(data_dict['train'], config)
    data_dict['train'] = rebalance_weak_annotations(data_dict['train'], config)
    train_dataset = specified_dataset(config, data_dict['train'], additional_data_dict)
    eval_dataset = specified_dataset(config, data_dict['eval'], additional_data_dict, mode='eval')
    test_dataset = specified_dataset(config, data_dict['test'], additional_data_dict, mode='test')
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch, collate_fn=train_dataset.collate_fn,
                              shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=config.eval_batch, collate_fn=eval_dataset.collate_fn,
                             shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch, collate_fn=test_dataset.collate_fn,
                             shuffle=False, num_workers=0)

    Model = getattr(model, config.model)
    training_model = Model(config, additional_data_dict).to(device)

    if config.pretrained_ckpt_path and config.pretrained_ckpt_path.lower() != 'none':
        checkpoint = torch.load(config.pretrained_ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        missing_keys, unexpected_keys = load_shape_compatible_state_dict(training_model, state_dict)
        logging.info(f'Loaded pretrained checkpoint: {config.pretrained_ckpt_path}')
        logging.info(f'Missing keys (incl. fresh p_unk params): {missing_keys}')
        logging.info(f'Unexpected keys: {unexpected_keys}')
    else:
        logging.info('from-scratch: no pretrained checkpoint loaded (--pretrained_ckpt_path none).')

    text_trainable = config.finetune_encoders or config.finetune_text_encoder
    image_trainable = config.finetune_encoders or config.finetune_image_encoder
    freeze_text = (not text_trainable) and (config.freeze_encoders or config.freeze_text_encoder)
    freeze_image = (not image_trainable) and (config.freeze_encoders or config.freeze_image_encoder)
    if freeze_text:
        freeze_module(getattr(training_model, 'text_model', None))
        logging.info('Frozen text_model encoder.')
    else:
        logging.info('Text encoder TRAINABLE (encoder_learning_rate=%s).', config.encoder_learning_rate)
    if freeze_image:
        freeze_module(getattr(training_model, 'image_model', None))
        logging.info('Frozen image_model encoder.')
    else:
        logging.info('Image encoder TRAINABLE (encoder_learning_rate=%s).', config.encoder_learning_rate)

    optimizer = build_optimizer(training_model, config.encoder_learning_rate, config.learning_rate, config.l2)
    estimator = Estimator(config)
    os.makedirs(config.finetune_save_dir, exist_ok=True)
    ckpt_name = config.finetune_ckpt_name or f'{config.model}-{config.dataset}-finetune'
    best_model_path = os.path.join(config.finetune_save_dir, f'{ckpt_name}.pth')
    best_score = -1.0
    patience_left = int(config.patience)

    logging.info('Start TDP target adaptation...')
    for epoch in range(config.epoch_num):
        logging.info(f'Discovering pattern centers before finetune epoch {epoch}.')
        training_model.discover_patterns(train_loader, device, epoch=epoch)

        training_model.train()
        total_loss = 0.0
        loss_parts = defaultdict(float)
        train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'tdp finetune epoch {epoch}')
        for step, batch_dict in train_iter:
            batch_dict['epoch'] = epoch
            batch_dict['step'] = step
            tensor_to_device(batch_dict, device)
            loss = training_model.calc_loss(batch_dict)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(training_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            if hasattr(training_model, '_last_loss_dict'):
                for key, value in training_model._last_loss_dict.items():
                    loss_parts[key] += value
            train_iter.set_postfix({'loss': f'{loss.item():.4f}'})

        eval_metric_result, eval_loss = estimator.evaluate(eval_loader, training_model)
        core_score = eval_metric_result[config.valid_metric]
        avg_train_loss = total_loss / max(len(train_loader), 1)
        parts_str = ' '.join(f'{k}={v / max(len(train_loader), 1):.4f}' for k, v in loss_parts.items())
        logging.info(
            f'Epoch {epoch}: train_loss={avg_train_loss:.4f} [{parts_str}] '
            f'eval_loss={eval_loss:.4f} metrics={eval_metric_result}'
        )
        if core_score > best_score:
            best_score = core_score
            patience_left = int(config.patience)
            torch.save(training_model.state_dict(), best_model_path)
            logging.info(f'Saved best finetuned model to: {best_model_path}')
        else:
            patience_left -= 1
            logging.info(f'EarlyStopping Counter: {int(config.patience) - patience_left} out of {config.patience}.')
            if patience_left <= 0:
                break

    training_model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metric_result = estimator.test(test_loader, training_model)
    if isinstance(test_metric_result, tuple):
        test_metric_result = test_metric_result[0]
    logging.info(f'Best eval {config.valid_metric}: {best_score:.4f}')
    logging.info(f'Test metrics: {test_metric_result}')

if __name__ == '__main__':
    main()
