import logging
import os

import torch

def build_optimizer(training_model, encoder_lr, fast_lr, weight_decay):
    optim_params_normal, optim_params_fast = [], []
    for name, param in training_model.named_parameters():
        if not param.requires_grad:
            continue
        if 'image_model' in name or 'text_model' in name:
            optim_params_normal.append(param)
        else:
            optim_params_fast.append(param)

    param_groups = []
    if optim_params_normal:
        param_groups.append({'params': optim_params_normal, 'lr': encoder_lr})
    if optim_params_fast:
        param_groups.append({'params': optim_params_fast, 'lr': fast_lr})
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=weight_decay)

def setup_logging(log_root, model_name, ckpt_name):
    log_dir = os.path.join(log_root, model_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{ckpt_name}.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()],
    )
    logging.info(f'log save at: {log_path}')

def save_pretrain_checkpoint(config, training_model, dataset_id_map, sample_probs, epoch=None):
    if epoch is None:
        checkpoint_path = os.path.join(config.pretrain_save_dir, f'{config.pretrain_ckpt_name}.pth')
    else:
        checkpoint_path = os.path.join(config.pretrain_save_dir, f'{config.pretrain_ckpt_name}_epoch{epoch}.pth')

    if os.path.exists(checkpoint_path) and not config.overwrite_pretrain_ckpt:
        raise FileExistsError(
            f'{checkpoint_path} already exists. Use --overwrite_pretrain_ckpt to overwrite it.'
        )

    torch.save({
        'model_state_dict': training_model.state_dict(),
        'model': config.model,
        'config': vars(config),
        'pretrain_datasets': list(config.pretrain_datasets),
        'dataset_id_map': dataset_id_map,
        'temperature_alpha': config.temperature_alpha,
        'sample_probs': dict(zip(config.pretrain_datasets, sample_probs.tolist())),
        'epoch': epoch,
    }, checkpoint_path)
    logging.info(f'Saved pretrained checkpoint to: {checkpoint_path}')

def freeze_module(module):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = False
