# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
import random
import re
from datetime import timedelta
from typing import Optional

import hydra

import numpy as np
import omegaconf
import torch
import torch.distributed as dist
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf


def multiply_all(*args):
    return np.prod(np.array(args)).item()


def collect_dict_keys(config):
    """This function recursively iterates through a dataset configuration, and collect all the dict_key that are defined"""
    val_keys = []
    # If the this config points to the collate function, then it has a key
    if "_target_" in config and re.match(r".*collate_fn.*", config["_target_"]):
        val_keys.append(config["dict_key"])
    else:
        # Recursively proceed
        for v in config.values():
            if isinstance(v, type(config)):
                val_keys.extend(collect_dict_keys(v))
            elif isinstance(v, omegaconf.listconfig.ListConfig):
                for item in v:
                    if isinstance(item, type(config)):
                        val_keys.extend(collect_dict_keys(item))
    return val_keys


class Phase:
    TRAIN = "train"
    VAL = "val"


def register_omegaconf_resolvers():
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    OmegaConf.register_new_resolver("get_class", hydra.utils.get_class)
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)
    OmegaConf.register_new_resolver("times", multiply_all)
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y)
    OmegaConf.register_new_resolver("pow", lambda x, y: x**y)
    OmegaConf.register_new_resolver("subtract", lambda x, y: x - y)
    OmegaConf.register_new_resolver("range", lambda x: list(range(x)))
    OmegaConf.register_new_resolver("int", lambda x: int(x))
    OmegaConf.register_new_resolver("ceil_int", lambda x: int(math.ceil(x)))
    OmegaConf.register_new_resolver("merge", lambda *x: OmegaConf.merge(*x))


def setup_distributed_backend(backend, timeout_mins):
    """
    Initialize torch.distributed and set the CUDA device.
    Expects environment variables to be set as per
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    along with the environ variable "LOCAL_RANK" which is used to set the CUDA device.
    """
    # enable TORCH_NCCL_ASYNC_ERROR_HANDLING to ensure dist nccl ops time out after timeout_mins
    # of waiting
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    logging.info(f"Setting up torch.distributed with a timeout of {timeout_mins} mins")
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_mins))
    return dist.get_rank()


def get_machine_local_and_dist_rank():
    """
    Get the distributed and local rank of the current gpu.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", None))
    distributed_rank = int(os.environ.get("RANK", None))
    assert (
        local_rank is not None and distributed_rank is not None
    ), "Please the set the RANK and LOCAL_RANK environment variables."
    return local_rank, distributed_rank


def print_cfg(cfg):
    """
    Supports printing both Hydra DictConfig and also the AttrDict config
    """
    logging.info("Training with config:")
    logging.info(OmegaConf.to_yaml(cfg))


def set_seeds(seed_value, max_epochs, dist_rank):
    """
    Set the python random, numpy and torch seed for each gpu. Also set the CUDA
    seeds if the CUDA is available. This ensures deterministic nature of the training.
    """
    # Since in the pytorch sampler, we increment the seed by 1 for every epoch.
    seed_value = (seed_value + dist_rank) * max_epochs
    logging.info(f"MACHINE SEED: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    if not g_pathmgr.exists(dir_path):
        g_pathmgr.mkdirs(dir_path)
    return True


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_amp_type(amp_type: Optional[str] = None):
    if amp_type is None:
        return None
    assert amp_type in ["bfloat16", "float16"], "Invalid Amp type."
    if amp_type == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float16


def log_env_variables():
    env_keys = sorted(list(os.environ.keys()))
    st = ""
    for k in env_keys:
        v = os.environ[k]
        st += f"{k}={v}\n"
    logging.info("Logging ENV_VARIABLES")
    logging.info(st)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, device, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self._allow_updates = True

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class MemMeter:
    """Computes and stores the current, avg, and max of peak Mem usage per iteration"""

    def __init__(self, name, device, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0  # Per iteration max usage
        self.avg = 0  # Avg per iteration max usage
        self.peak = 0  # Peak usage for lifetime of program
        self.sum = 0
        self.count = 0
        self._allow_updates = True

    def update(self, n=1, reset_peak_usage=True):
        self.val = torch.cuda.max_memory_allocated() // 1e9
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
        self.peak = max(self.peak, self.val)
        if reset_peak_usage:
            torch.cuda.reset_peak_memory_stats()

    def __str__(self):
        fmtstr = (
            "{name}: {val"
            + self.fmt
            + "} ({avg"
            + self.fmt
            + "}/{peak"
            + self.fmt
            + "})"
        )
        return fmtstr.format(**self.__dict__)


def human_readable_time(time_seconds):
    time = int(time_seconds)
    minutes, seconds = divmod(time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days:02}d {hours:02}h {minutes:02}m"


class DurationMeter:
    def __init__(self, name, device, fmt=":f"):
        self.name = name
        self.device = device
        self.fmt = fmt
        self.val = 0

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def add(self, val):
        self.val += val

    def __str__(self):
        return f"{self.name}: {human_readable_time(self.val)}"


class ProgressMeter:
    def __init__(self, num_batches, meters, real_meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.real_meters = real_meters
        self.prefix = prefix

    def display(self, batch, enable_print=False):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += [
            " | ".join(
                [
                    f"{os.path.join(name, subname)}: {val:.4f}"
                    for subname, val in meter.compute().items()
                ]
            )
            for name, meter in self.real_meters.items()
        ]
        logging.info(" | ".join(entries))
        if enable_print:
            print(" | ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_resume_checkpoint(checkpoint_save_dir):
    if not g_pathmgr.isdir(checkpoint_save_dir):
        return None
    ckpt_file = os.path.join(checkpoint_save_dir, "checkpoint.pt")
    if not g_pathmgr.isfile(ckpt_file):
        return None

    return ckpt_file


def focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Target masks [B, 1, H, W] 
        alpha: Balancing factor for rare class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean' or 'sum')
    
    Returns:
        Focal loss value
    """
    # Apply sigmoid to get probabilities
    pred_sigmoid = torch.sigmoid(pred)
    
    # Calculate cross entropy
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    
    # Calculate p_t
    p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
    
    # Calculate alpha_t
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    
    # Calculate focal weight
    focal_weight = alpha_t * (1 - p_t) ** gamma
    
    # Apply focal weight
    focal_loss = focal_weight * ce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss


def dice_loss(pred, target, smooth=1e-6):
    """
    Dice Loss for segmentation tasks.
    
    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Target masks [B, 1, H, W]
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice loss value (1 - Dice coefficient)
    """
    # Apply sigmoid to get probabilities
    pred_sigmoid = torch.sigmoid(pred)
    
    # Flatten the tensors
    pred_flat = pred_sigmoid.view(pred_sigmoid.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)
    
    # Calculate Dice coefficient
    dice_coeff = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Return Dice loss (1 - Dice coefficient)
    return 1 - dice_coeff.mean()


def combined_focal_dice_loss(
    pred,
    target,
    focal_alpha=0.25,
    focal_gamma=2.0,
    focal_weight=1.0,
    dice_weight=1.0,
    bce_weight=0.0,
):
    """
    Combined Focal Loss + Dice Loss.
    
    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Target masks [B, 1, H, W]
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss  
        focal_weight: Weight for focal loss term
        dice_weight: Weight for dice loss term
        bce_weight: Weight for plain BCE-with-logits loss term
        
    Returns:
        Combined loss value
    """
    total_loss = pred.new_tensor(0.0)
    if focal_weight != 0.0:
        total_loss = total_loss + focal_weight * focal_loss(pred, target, alpha=focal_alpha, gamma=focal_gamma)
    if dice_weight != 0.0:
        total_loss = total_loss + dice_weight * dice_loss(pred, target)
    if bce_weight != 0.0:
        total_loss = total_loss + bce_weight * F.binary_cross_entropy_with_logits(pred, target, reduction="mean")
    
    return total_loss


def compute_feature_similarity_loss(y, x, balance_factor, use_softmax=False):
    total_similarity_loss = 0.0
    num_features = 0
    
    for i, (y_feat, x_feat) in enumerate(zip(y, x)):
        y_feat = y_feat.detach()
        
        if isinstance(y_feat, torch.Tensor) and isinstance(x_feat, torch.Tensor):
            if y_feat.shape == x_feat.shape:
                if len(y_feat.shape) == 4:  # [B, C, H, W]
                    B, C, H, W = y_feat.shape
                    
                    if use_softmax:
                        y_feat_flat = y_feat.view(B, C, -1)  # [B, C, H*W]
                        x_feat_flat = x_feat.view(B, C, -1)  # [B, C, H*W]
                        
                        y_feat_norm = F.softmax(y_feat_flat, dim=1)  # [B, C, H*W]
                        x_feat_norm = F.softmax(x_feat_flat, dim=1)  # [B, C, H*W]
                        
                        y_feat_norm = y_feat_norm.view(B, C, H, W)
                        x_feat_norm = x_feat_norm.view(B, C, H, W)
                    else:
                        y_feat_norm = y_feat
                        x_feat_norm = x_feat
                    
                    y_mean = y_feat_norm.mean(dim=(2, 3))  # [B, C]
                    y_var = y_feat_norm.var(dim=(2, 3))    # [B, C]
                    x_mean = x_feat_norm.mean(dim=(2, 3))  # [B, C]
                    x_var = x_feat_norm.var(dim=(2, 3))    # [B, C]
                    
                    mean_l1 = torch.abs(y_mean - x_mean).mean()
                    var_l1 = torch.abs(y_var - x_var).mean()
                    
                    feature_similarity = (mean_l1 + var_l1) / 2.0
                    total_similarity_loss += feature_similarity
                    num_features += 1
                
                elif len(y_feat.shape) == 3:
                    if use_softmax:
                        y_feat_norm = F.softmax(y_feat, dim=1)
                        x_feat_norm = F.softmax(x_feat, dim=1)
                    else:
                        y_feat_norm = y_feat
                        x_feat_norm = x_feat
                    
                    y_mean = y_feat_norm.mean(dim=-1)  # [B, C]
                    y_var = y_feat_norm.var(dim=-1)    # [B, C]
                    x_mean = x_feat_norm.mean(dim=-1)  # [B, C]
                    x_var = x_feat_norm.var(dim=-1)    # [B, C]
                    
                    mean_l1 = torch.abs(y_mean - x_mean).mean()
                    var_l1 = torch.abs(y_var - x_var).mean()
                    
                    feature_similarity = (mean_l1 + var_l1) / 2.0
                    total_similarity_loss += feature_similarity
                    num_features += 1
    
    if num_features > 0:
        avg_similarity_loss = total_similarity_loss / num_features
        return balance_factor * avg_similarity_loss
    else:
        if len(y) > 0 and isinstance(y[0], torch.Tensor):
            return torch.tensor(0.0, device=y[0].device)
        elif len(x) > 0 and isinstance(x[0], torch.Tensor):
            return torch.tensor(0.0, device=x[0].device)
        else:
            return torch.tensor(0.0)


def compute_feature_cos_similarity_loss(y, x, balance_factor, use_zscore=False):
    total_similarity_loss = 0.0
    
    for y_feat, x_feat in zip(y, x):
        y_feat = y_feat.detach()
        
        B, C, H, W = y_feat.shape
        
        if use_zscore:
            y_mean = y_feat.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            y_std = y_feat.std(dim=(2, 3), keepdim=True) + 1e-8  # [B, C, 1, 1]
            x_mean = x_feat.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            x_std = x_feat.std(dim=(2, 3), keepdim=True) + 1e-8  # [B, C, 1, 1]
            
            y_feat_norm = (y_feat - y_mean) / y_std  # [B, C, H, W]
            x_feat_norm = (x_feat - x_mean) / x_std  # [B, C, H, W]
        else:
            y_feat_norm = y_feat
            x_feat_norm = x_feat
        
        y_flat = y_feat_norm.reshape(B, -1) # [B, C*H*W]
        x_flat = x_feat_norm.reshape(B, -1) # [B, C*H*W]
        
        target = torch.ones(B, dtype=torch.float, device=y_feat.device)

        cosine_loss = torch.nn.CosineEmbeddingLoss(reduction='mean')
        feature_similarity_loss = cosine_loss(y_flat, x_flat, target)
        total_similarity_loss += feature_similarity_loss
                
    avg_similarity_loss = total_similarity_loss / len(y)
    return balance_factor * avg_similarity_loss


def compute_feature_consistency_loss(
    y,
    x,
    balance_factor,
    mode="legacy_cosine",
    eps=1e-6,
):
    """
    Compute weak/strong feature consistency loss with selectable branches.

    Args:
        y: Target feature list. It is detached inside this function.
        x: Trainable feature list.
        balance_factor: External loss weight.
        mode: Feature loss branch.
            - legacy_cosine: original flattened CosineEmbeddingLoss behavior.
            - legacy_cosine_zscore: original flattened cosine after per-channel z-score.
            - spatial_cosine: channel-direction cosine at each spatial location.
            - spatial_cosine_zscore: spatial cosine after per-channel z-score.
        eps: Numerical epsilon for normalization.

    Returns:
        Weighted feature consistency loss.
    """
    if balance_factor <= 0:
        ref = None
        for feats in (x, y):
            if feats:
                ref = next((feat for feat in feats if isinstance(feat, torch.Tensor)), None)
            if ref is not None:
                break
        if ref is not None:
            return ref.new_tensor(0.0)
        return torch.tensor(0.0)

    if mode in ("legacy_cosine", "cosine"):
        return compute_feature_cos_similarity_loss(y, x, balance_factor, use_zscore=False)
    if mode in ("legacy_cosine_zscore", "cosine_zscore"):
        return compute_feature_cos_similarity_loss(y, x, balance_factor, use_zscore=True)
    if mode not in ("spatial_cosine", "spatial_cosine_zscore"):
        raise ValueError(
            f"Unknown feature consistency mode: {mode}. "
            "Expected one of legacy_cosine, legacy_cosine_zscore, "
            "spatial_cosine, spatial_cosine_zscore."
        )

    total_loss = None
    num_features = 0
    use_zscore = mode == "spatial_cosine_zscore"

    for y_feat, x_feat in zip(y, x):
        if not isinstance(y_feat, torch.Tensor) or not isinstance(x_feat, torch.Tensor):
            continue
        if y_feat.shape != x_feat.shape or y_feat.ndim != 4:
            continue

        y_feat = y_feat.detach().float()
        x_feat = x_feat.float()

        if use_zscore:
            y_mean = y_feat.mean(dim=(2, 3), keepdim=True)
            y_std = y_feat.std(dim=(2, 3), keepdim=True).clamp_min(eps)
            x_mean = x_feat.mean(dim=(2, 3), keepdim=True)
            x_std = x_feat.std(dim=(2, 3), keepdim=True).clamp_min(eps)
            y_feat = (y_feat - y_mean) / y_std
            x_feat = (x_feat - x_mean) / x_std

        y_norm = F.normalize(y_feat, p=2, dim=1, eps=eps)
        x_norm = F.normalize(x_feat, p=2, dim=1, eps=eps)
        layer_loss = (1.0 - (y_norm * x_norm).sum(dim=1)).mean()

        total_loss = layer_loss if total_loss is None else total_loss + layer_loss
        num_features += 1

    if num_features == 0:
        ref = None
        for feats in (x, y):
            if feats:
                ref = next((feat for feat in feats if isinstance(feat, torch.Tensor)), None)
            if ref is not None:
                break
        if ref is not None:
            return ref.new_tensor(0.0)
        return torch.tensor(0.0)

    return balance_factor * (total_loss / num_features)
