import argparse
import csv
import json
import logging
import math
import os
from typing import List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from models.medsam2_spda_model import build_sam2_adapt_model
from utils import (
	build_logger,
	collect_files_by_directory,
	set_seeds,
)
from utils.image_processing import preprocess_fine, preprocess_fine_normalized
from utils import (
    decompose_multiclass_gts, find_key_slice_for_class,
)
from utils.statistics import (
    calculate_hierarchical_directory_stats,
    calculate_final_metric_summary,
    export_final_metric_summary,
    log_final_metric_summary,
    print_directory_tree_stats,
    export_tta_directory_tree_stats_json
)
from utils.experiment_done import (
    check_experiment_done,
    create_experiment_done_file
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="SPDA: Efficient Online Test-Time Adaptation for Promptable Medical Segmentation", allow_abbrev=False)
	# Model args
	parser.add_argument("--model_cfg", default="configs/sam2/sam2.1_hiera_t512.yaml")
	parser.add_argument("--checkpoint", default="path to checkpoint")
	parser.add_argument("--image_size", type=int, default=512)

	# TTA args
	parser.add_argument("--num_steps", type=int, default=1)
	parser.add_argument("--lr", type=float, default=1e-5)
	parser.add_argument("--cosine_lr", action='store_true', default=True)
	parser.add_argument("--decoder_lr", type=float, default=0.0)
	parser.add_argument("--weight_decay", type=float, default=1e-3)
	parser.add_argument("--grad_clip", type=float, default=1.0)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--reset_adapters", choices=["no", "subtask", "sample"], default="subtask")
	parser.add_argument("--reset_freq", type=int, default=0, help="reset interval within a subtask.")
	parser.add_argument("--reset_ratio", type=float, default=1.0)
	parser.add_argument("--momentum_update", action='store_true', default=False, help="Blend parameters between samples.")
	parser.add_argument("--momentum", type=float, default=0.9, help="Weight of initial parameters in momentum blending.")

	parser.add_argument("--device", default="cuda")
	parser.add_argument("--seed", type=int, default=1314)

	# Input args
	parser.add_argument("--data_dir", default="path to dataset")

	# Output args
	parser.add_argument('--output_dir', type=str, default='path to output')
	parser.add_argument('--skip_if_done', action='store_true', default=True, help="Skip when output_dir contains a completed run.")
	parser.add_argument('--save_predictions', action='store_true', default=False)
	parser.add_argument('--save_adapted_model', action='store_true', default=False)

	# Prompt args
	parser.add_argument("--prompt_type", choices=["box", "point"], default="box")
	parser.add_argument("--num_points", type=int, default=2)
	parser.add_argument("--point_strategy", choices=["center", "random", "boundary"], default="random")
	
	# Subsampled prompt args
	parser.add_argument("--subsampled_prompt_type", choices=["point", "box"], default="point")
	parser.add_argument("--subsampled_num_points", type=int, default=1)
	parser.add_argument("--subsampled_point_strategy", choices=["center", "random", "boundary"], default="random")
	parser.add_argument("--box_expand_ratio", type=float, default=0.005)
	
	# Fill hole args
	parser.add_argument("--adapt_fillhole", type=int, default=0, help="Pseudo-label hole area threshold (0 disables filling).")
	parser.add_argument("--eval_fillhole", type=int, default=0, help="Adapted-mask hole area threshold (0 disables filling).")
	
	# Loss balance args
	parser.add_argument("--gamma", type=float, default=0.1, help="Legacy feature-loss weight overridden by --feature_loss_weight.")
	parser.add_argument("--weak_mask_loss_weight", type=float, default=0.5)
	parser.add_argument("--strong_mask_loss_weight", type=float, default=0.5)
	parser.add_argument("--feature_loss_weight", type=float, default=0.0)
	parser.add_argument("--feature_loss_mode", type=str, default="legacy_cosine",
	                   choices=["legacy_cosine", "legacy_cosine_zscore", "spatial_cosine", "spatial_cosine_zscore"],
	                   help="Flattened cosine in legacy modes or per-location channel cosine in spatial modes.")
	parser.add_argument("--mask_focal_loss_weight", type=float, default=0.5)
	parser.add_argument("--mask_dice_loss_weight", type=float, default=1.0)
	parser.add_argument("--mask_bce_loss_weight", type=float, default=0.1)
	parser.add_argument("--pseudo_target_mode", choices=["hard", "soft"], default="soft")
	parser.add_argument("--pseudo_temperature", type=float, default=0.75)
	parser.add_argument("--pseudo_confidence_threshold", type=float, default=0.0)
	parser.add_argument("--pseudo_confidence_power", type=float, default=0.0)
	parser.add_argument("--pseudo_min_keep_ratio", type=float, default=0.01)
	
	# Frequency augmentation args
	parser.add_argument("--augmentation", type=str, default='spda', choices=['weak', 'freq', 'spda'])
	parser.add_argument("--freq_aug_bands", type=int, default=3)
	parser.add_argument("--freq_aug_prob", type=float, default=1.0)
	parser.add_argument("--spda_K", type=int, default=24, help="Total radial bands.")
	parser.add_argument("--spda_K_low", type=int, default=8, help="Number of perturbed bands.")
	parser.add_argument("--spda_K_low_position", type=str, default='front', choices=['front', 'middle', 'back'])
	parser.add_argument("--spda_gamma", type=float, default=1.7, help="Radial band-edge exponent.")
	parser.add_argument("--spda_eta", type=float, default=0.1, help="EWMA/EWVar update rate.")
	parser.add_argument("--spda_beta", type=float, default=1.0, help="Perturbation scale.")
	parser.add_argument("--spda_lambda_inter", type=float, default=0.7)
	parser.add_argument("--spda_lambda_intra", type=float, default=0.3)
	parser.add_argument("--spda_rho_band", type=float, default=0.2, help="AR(1) correlation between adjacent bands.")
	parser.add_argument("--spda_exclude_dc", action='store_true', default=True)
	parser.add_argument("--spda_zeta_dc", type=float, default=0.8)
	parser.add_argument("--spda_eps", type=float, default=1e-12)
	parser.add_argument("--spda_rescale", action='store_true', default=True, help="Match the original mean/std.")
	parser.add_argument("--spda_sigma_max", type=float, default=1.0, help="Max band perturbation std.")
	parser.add_argument("--spda_log_alpha_clip", type=float, default=0.35, help="Symmetric log-amplitude clipping.")
	parser.add_argument("--spda_alpha_min", type=float, default=0.70, help="Min amplitude multiplier.")
	parser.add_argument("--spda_alpha_max", type=float, default=1.30, help="Max amplitude multiplier.")
	parser.add_argument("--spda_alpha_smooth_sigma", type=float, default=0.0, help="Log-amplitude Gaussian smooth.")
	parser.add_argument("--spda_alpha_taper_bands", type=int, default=0, help="Tapered bands at edges.")
	parser.add_argument("--spda_max_slice_batch_size", type=int, default=64, help="FFT slice-batch cap.")
	parser.add_argument("--spda_fft_dtype", type=str, default='float32', choices=['float32', 'float64'])
	parser.add_argument("--spda_lambda_xchan", type=float, default=0.1)
	parser.add_argument("--spda_channel_mode", type=str, default='shared', choices=['shared', 'per_channel_jitter'])
	parser.add_argument("--spda_beta_xchan", type=float, default=0.0, help="Channel jitter strength in per_channel_jitter mode.")
	parser.add_argument("--spda_per_channel_rescale", action='store_true', default=True)
	parser.add_argument("--spda_norm_mode", type=str, default='clip', choices=['minmax', 'clip', 'percentile', 'softclip', 'none'])
	parser.add_argument("--spda_percentile_low", type=float, default=1.0)
	parser.add_argument("--spda_percentile_high", type=float, default=99.0)
	parser.add_argument("--spda_percentile_sample_size", type=int, default=262144, help="Sampled pixels per channel for percentiles.")
	parser.add_argument("--spda_softclip_temperature", type=float, default=0.25)
	parser.add_argument("--spda_blend", type=float, default=1.0, help="Augmented-view weight, from 0 (original) to 1 (fully augmented).")
	
	# Slice control args
	parser.add_argument("--num_slices", type=int, default=4, help="Number of slices for adaptation")
	parser.add_argument('--num_keyslices', type=int, default=0)
	
	# Evaluation frequency control
	parser.add_argument("--eval_freq", type=int, default=1, help="Number of samples to adapt before evaluation")
	
	# Reproducibility control
	parser.add_argument("--sort", action='store_true', default=True)
	
	args = parser.parse_args()
	if args.feature_loss_weight is None:
		args.feature_loss_weight = args.gamma
	else:
		args.gamma = args.feature_loss_weight
	for attr in (
		"weak_mask_loss_weight", "strong_mask_loss_weight", "feature_loss_weight",
		"mask_focal_loss_weight", "mask_dice_loss_weight", "mask_bce_loss_weight",
	):
		if getattr(args, attr) < 0:
			parser.error(f"--{attr} must be non-negative.")
	if args.pseudo_temperature <= 0:
		parser.error("--pseudo_temperature must be positive.")
	if not 0.0 <= args.pseudo_confidence_threshold <= 1.0:
		parser.error("--pseudo_confidence_threshold must be in [0, 1].")
	if args.pseudo_confidence_power < 0:
		parser.error("--pseudo_confidence_power must be non-negative.")
	if not 0.0 <= args.pseudo_min_keep_ratio <= 1.0:
		parser.error("--pseudo_min_keep_ratio must be in [0, 1].")
	return args


def _maybe_apply_cosine_lr_schedule(model, args, step_idx: int, total_steps: int, logger=None) -> None:
	"""Apply cosine-annealed LR to optimizer param groups (in-place) when enabled."""
	if not getattr(args, "cosine_lr", False):
		return
	if getattr(model, "optimizer", None) is None:
		return
	denom = max(1, int(total_steps))
	scale = 0.5 * (1.0 + math.cos(math.pi * step_idx / denom))
	group_lrs = []
	for group in model.optimizer.param_groups:
		group_name = group.get('name', '')
		base_lr = args.decoder_lr if (group_name == 'mask_decoder' and args.decoder_lr > 0) else args.lr
		group['lr'] = base_lr * scale
		group_lrs.append(group['lr'])
	if logger is not None:
		logger.debug(f"Learning rates: {group_lrs}")


def _resize_mask_nearest(mask: np.ndarray, size: int) -> np.ndarray:
	"""Resize binary/label mask to (size, size) using nearest neighbor."""
	img = Image.fromarray(mask.astype(np.uint8))
	img = img.resize((size, size), resample=Image.NEAREST)
	return np.array(img)


def _prepare_images(raw_imgs: np.ndarray) -> torch.Tensor:
	"""
	Convert raw images.
	
	Args:
		raw_imgs: Raw image data
		
	Returns:
		imgs: Tensor of shape [D, 3, 512, 512]
	"""
	device = torch.device('cpu')  # Use CPU for preprocessing, move to device later
	
	with torch.no_grad():
		imgs = preprocess_fine(raw_imgs, device)  # D,3,512,512 or 3,512,512
	if imgs.ndim == 3:
		imgs = imgs.unsqueeze(0)  # T,C,H,W
	
	return imgs


def _analyze_masks_slices(masks: dict, num_slices: int = -1) -> tuple[dict, dict]:
	key_slices = {}
	valid_slices = {}
	
	for class_id, binary_mask in masks.items():
		key_slice_idx = find_key_slice_for_class(binary_mask)
		key_slices[class_id] = key_slice_idx
		
		if binary_mask.ndim == 2:
			if np.any(binary_mask):
				valid_slices[class_id] = (0, 0)
			else:
				valid_slices[class_id] = None
		elif binary_mask.ndim == 3:
			valid_indices = []
			for i in range(binary_mask.shape[0]):
				if np.any(binary_mask[i]):  
					valid_indices.append(i)
			
			if valid_indices:
				start_idx = min(valid_indices)
				end_idx = max(valid_indices)
				valid_slices[class_id] = (start_idx, end_idx)
			else:
				valid_slices[class_id] = None 
		else:
			raise ValueError(f"Unsupported mask shape for class {class_id}: {binary_mask.shape}")
	
	if num_slices >= 0:
		for class_id in valid_slices:
			if valid_slices[class_id] is not None:
				key_slice_idx = key_slices[class_id]
				original_start, original_end = valid_slices[class_id]
				
				if num_slices == 0:
					valid_slices[class_id] = (key_slice_idx, key_slice_idx)
				else:
					limited_start = max(original_start, key_slice_idx - num_slices)
					limited_end = min(original_end, key_slice_idx + num_slices)
					valid_slices[class_id] = (limited_start, limited_end)
	
	return key_slices, valid_slices


def _prepare_input(raw_imgs: np.ndarray, gts: np.ndarray, image_size: int, num_slices: int = -1, augmentation: str = "weak") -> dict:
	"""
	Convert raw npz inputs to 3D volumes and masks.
	"""
	imgs = _prepare_images(raw_imgs)  # D,3,512,512
	imgs_for_aug = None
	if augmentation == "spda":
		with torch.no_grad():
			imgs_for_aug = preprocess_fine_normalized(raw_imgs, torch.device('cpu'))
		if imgs_for_aug.ndim == 3:
			imgs_for_aug = imgs_for_aug.unsqueeze(0)

	masks_raw = decompose_multiclass_gts(gts)  # Dict{class_id: np.ndarray[original_size]}
	
	# Ensure masks have D dimension for consistency (D=1 for 2D cases)
	masks = {}
	for class_id, mask in masks_raw.items():
		if mask.ndim == 3:
			# 3D mask: keep as is
			masks[class_id] = mask
		elif mask.ndim == 2:
			# 2D mask: add D=1 dimension for consistency
			masks[class_id] = mask[None]  # Add D=1 dimension: (H, W) -> (1, H, W)
		else:
			raise ValueError(f"Unsupported mask shape: {mask.shape}")
	
	# Create resized masks to align with imgs spatial dimensions
	resized_masks = {}
	for class_id, mask in masks.items():
		if mask.ndim == 3:
			# 3D mask: resize each slice
			resized_mask = np.zeros((mask.shape[0], image_size, image_size), dtype=mask.dtype)
			for i in range(mask.shape[0]):
				resized_mask[i] = _resize_mask_nearest(mask[i], image_size)
		else:
			raise ValueError(f"Unsupported mask shape: {mask.shape}")
		# Convert to PyTorch tensor for compatibility with augmentation functions
		resized_masks[class_id] = torch.from_numpy(resized_mask).float()
	
	key_slices, valid_slices = _analyze_masks_slices(masks, num_slices)
	class_ids = sorted(masks.keys())
	
	return {
		'imgs': imgs,  # [D,C,H,W]
		'imgs_for_aug': imgs_for_aug,  # [D,C,H,W] in [0,1]
		'masks': masks,  # Dict{class_id: np.ndarray[D,H,W]}
		'resized_masks': resized_masks,  # Dict{class_id: Tensor[D,H_size,W_size]}
		'keyframes': key_slices,
		'valid_slices': valid_slices,  # Dict{class_id: (start_idx, end_idx)}
		'class_ids': class_ids
	}


def _compute_tta_volume_metrics(inputs: dict, adapted_results: dict, logger=None) -> dict:
	"""Average adapted DSC equally across foreground classes."""
	class_metrics = {
		class_id: {'mean_dsc': float(adapted_results[class_id]['dsc'])}
		for class_id in inputs['class_ids']
	}
	mean_dsc = sum(metric['mean_dsc'] for metric in class_metrics.values()) / len(class_metrics)
	if logger is not None:
		for class_id, metric in class_metrics.items():
			logger.debug(f"Class {class_id}: mean_dsc={metric['mean_dsc']:.6f}")
	return {
		'class_metrics': class_metrics,
		'num_classes': len(class_metrics),
		'mean_dsc': float(mean_dsc),
	}


def initialize_model_and_logging(args) -> tuple:
	"""
	Initialize model, logging, and perform basic setup.
	
	Args:
		args: Parsed command line arguments
		
	Returns:
		tuple: (model, logger, device)
	"""
	logger = build_logger(args.output_dir, args=args, logger_name="medsam2_spda", log_args=False)
	logger.info("===== SPDA for 2D/3D medical images start =====")
	logger.info(f"PID={os.getpid()} working_dir={os.getcwd()}")

	# Initialize SAM2Adapt model directly
	device = torch.device(args.device)
	
	# Build augmentation configuration based on augmentation type
	augmentation_args = {}
	if args.augmentation == 'freq':
		# augmentation configuration
		augmentation_args = {
			'num_freq_bands': 12,  # Total frequency bands
			'kappa': 16.0,
			'epsilon': 1e-12,
			'cold_start_sigma': 0.05
		}
	elif args.augmentation == 'spda':
		augmentation_args = {
			'K': args.spda_K,
			'K_low': args.spda_K_low,
			'K_low_position': args.spda_K_low_position,
			'gamma': args.spda_gamma,
			'eta': args.spda_eta,
			'beta': args.spda_beta,
			'lambda_inter': args.spda_lambda_inter,
			'lambda_intra': args.spda_lambda_intra,
			'lambda_xchan': args.spda_lambda_xchan,
			'rho_band': args.spda_rho_band,
			'exclude_dc': args.spda_exclude_dc,
			'zeta_dc': args.spda_zeta_dc,
			'eps': args.spda_eps,
			'rescale_to_original': args.spda_rescale,
			'channel_weights': None,
			'channel_mode': args.spda_channel_mode,
			'beta_xchan': args.spda_beta_xchan,
			'per_channel_rescale': args.spda_per_channel_rescale,
			'sigma_max': args.spda_sigma_max,
			'log_alpha_clip': args.spda_log_alpha_clip,
			'alpha_min': args.spda_alpha_min,
			'alpha_max': args.spda_alpha_max,
			'alpha_smooth_sigma': args.spda_alpha_smooth_sigma,
			'alpha_taper_bands': args.spda_alpha_taper_bands,
			'max_slice_batch_size': args.spda_max_slice_batch_size,
			'fft_dtype': args.spda_fft_dtype,
			'spda_norm_mode': args.spda_norm_mode,
			'spda_percentile_low': args.spda_percentile_low,
			'spda_percentile_high': args.spda_percentile_high,
			'spda_percentile_sample_size': args.spda_percentile_sample_size,
			'spda_softclip_temperature': args.spda_softclip_temperature,
			'spda_blend': args.spda_blend
		}
	
	model = build_sam2_adapt_model(
		args.model_cfg, args.checkpoint, device, args.image_size, args.lr, args.decoder_lr, args.weight_decay, args.seed,
		offload_state_to_cpu=False, adapt_fillhole=args.adapt_fillhole, gamma=args.gamma,
		feature_loss_mode=args.feature_loss_mode,
		weak_mask_loss_weight=args.weak_mask_loss_weight,
		strong_mask_loss_weight=args.strong_mask_loss_weight,
		mask_focal_loss_weight=args.mask_focal_loss_weight,
		mask_dice_loss_weight=args.mask_dice_loss_weight,
		mask_bce_loss_weight=args.mask_bce_loss_weight,
		pseudo_target_mode=args.pseudo_target_mode,
		pseudo_temperature=args.pseudo_temperature,
		pseudo_confidence_threshold=args.pseudo_confidence_threshold,
		pseudo_confidence_power=args.pseudo_confidence_power,
		pseudo_min_keep_ratio=args.pseudo_min_keep_ratio,
		batch_size=args.batch_size, eval_fillhole=args.eval_fillhole,
		augmentation_type=args.augmentation, augmentation_args=augmentation_args,
	)
	
	# Configure gradient clipping
	if args.grad_clip > 0:
		model.grad_clip_norm = args.grad_clip
	
	logger.info(f"SAM2Adapt model initialized on {device}")
	logger.info("TTA configuration loaded from args.txt")
	logger.info(
		"Pseudo target: "
		f"mode={args.pseudo_target_mode}, "
		f"temperature={args.pseudo_temperature}, "
		f"confidence_threshold={args.pseudo_confidence_threshold}, "
		f"confidence_power={args.pseudo_confidence_power}, "
		f"min_keep_ratio={args.pseudo_min_keep_ratio}"
	)
	
	# Log adapter reset configuration
	logger.info(f"Adapter Reset Strategy: {args.reset_adapters}")
	if args.reset_adapters == "no":
		logger.info("Adapters will NOT be reset - state will accumulate across all tasks")
	elif args.reset_adapters == "subtask":
		logger.info("Adapters will be reset to initial values when starting each new subtask")
	elif args.reset_adapters == "sample":
		logger.info("Adapters will be reset to initial values for each individual sample")
	
	# Log evaluation frequency configuration
	logger.info(f"Evaluation Frequency: {args.eval_freq}")
	if args.eval_freq == 1:
		logger.info("Standard mode: adapt and evaluate each sample individually")
	else:
		logger.info(f"Batch mode: adapt on {args.eval_freq} samples, then evaluate all {args.eval_freq} samples")
	
	# Log momentum update configuration
	if args.momentum_update:
		logger.info(f"Momentum Update: ENABLED with momentum={args.momentum}")
		logger.info(f"  Formula: params_{{t+1}} = {args.momentum:.2f} * params_t^{{init}} + {1-args.momentum:.2f} * params_t^{{updated}}")
		if args.eval_freq > 1:
			logger.warning("  Momentum update is disabled in batch mode (eval_freq > 1)")
	else:
		logger.info("Momentum Update: DISABLED")
	
	return model, logger, device


def reset_adapters_if_needed(model, reset_strategy: str, context: str, logger, reset_ratio: float = 1.0) -> None:
	if reset_strategy == "no":
		return
	
	if hasattr(model, 'compute_adapter_l1_norm'):
		pre_reset_norms = model.compute_adapter_l1_norm()
		logger.debug(f"Pre-reset adapter L1 norm: {pre_reset_norms.get('total_l1', 0):.6f}")
	
	reset_params = model.reset_adapters(reset_ratio=reset_ratio)
	
	if reset_ratio >= 1.0:
		model.initialize_optimizer()
		logger.debug(f"Full reset at {context}")
	else:
		if hasattr(model, 'clear_optimizer_momentum_for_params'):
			model.clear_optimizer_momentum_for_params(reset_params)
		logger.debug(f"Partial reset ({reset_ratio*100:.1f}%) at {context} - cleared momentum for {len(reset_params)} parameters")
	
	if hasattr(model, 'compute_adapter_l1_norm'):
		post_reset_norms = model.compute_adapter_l1_norm()
		logger.debug(f"Post-reset adapter L1 norm: {post_reset_norms.get('total_l1', 0):.6f}")
		

def prepare_dataset(args, logger) -> tuple:
	"""
	Prepare dataset files and log statistics.
	
	Args:
		args: Parsed command line arguments
		logger: Logger instance
		
	Returns:
		tuple: (files_by_dir, total_files)
	"""
	# Dataset directory processing
	files_by_dir = collect_files_by_directory(args.data_dir)
	
	# Sort subdirectories and files if requested for reproducibility
	if args.sort:
		logger.info("Sorting subdirectories and files for reproducibility")
		# Sort files within each directory
		for dir_path in files_by_dir:
			files_by_dir[dir_path] = sorted(files_by_dir[dir_path])
		
		# Convert to sorted OrderedDict to ensure directory order is also deterministic
		files_by_dir = OrderedDict(sorted(files_by_dir.items()))

	total_files = sum(len(fs) for fs in files_by_dir.values())
	logger.info(f"Total files to process: {total_files}")
	logger.info(f"Directories found: {len(files_by_dir)}")
	for dir_path, files in files_by_dir.items():
		rel_path = os.path.relpath(dir_path, args.data_dir)
		logger.info(f"  {rel_path}: {len(files)} files")
	
	return files_by_dir, total_files


def _read_npz_arrays(file_path: str) -> tuple[np.ndarray, np.ndarray]:
	"""
	Read complete image and label arrays, then close the NPZ archive.
	"""
	with np.load(file_path, allow_pickle=True) as data:
		return data['imgs'], data['gts']


def _prepare_device_inputs(inputs, device, augmentation):
	"""
	Upload complete image volumes once, keeping CPU masks for prompt sampling.
	"""
	device_inputs = dict(inputs)
	if augmentation != 'spda':
		# Preserve the CPU RNG stream used by weak/freq augmentation.
		device_inputs['_cpu_augmentation_imgs'] = inputs.get('_cpu_augmentation_imgs', inputs['imgs'])
	device_inputs['imgs'] = inputs['imgs'].to(device=device, dtype=torch.float32)
	if augmentation == 'spda' and 'imgs_for_aug' in inputs:
		device_inputs['imgs_for_aug'] = inputs['imgs_for_aug'].to(device=device, dtype=torch.float32)
	return device_inputs


def process_single_sample(file_path: str, model, args, device, logger) -> dict:
	"""
	Process a single sample (npz file) through adaptation and inference.
	
	Args:
		file_path: Path to the .npz file
		model: SAM2Adapt model
		args: Command line arguments
		device: PyTorch device
		logger: Logger instance
		
	Returns:
		dict: Sample metadata, mean DSC and optional predictions
	"""
	# Load data
	raw_imgs, gts = _read_npz_arrays(file_path)
	logger.debug(f"raw_imgs.shape={getattr(raw_imgs, 'shape', None)}, gts.shape={getattr(gts, 'shape', None)}")
	
	# Extract volume name for logging and saving
	volume_name = os.path.splitext(os.path.basename(file_path))[0]
	logger.debug(f"Processing volume: {volume_name}")
	
	# Reset adapter parameters for each sample if requested
	if args.reset_adapters == "sample":
		reset_adapters_if_needed(model, "sample", "new sample", logger, reset_ratio=1.0)
	
	# Prepare 3D volume and masks (multi-class mode)
	inputs = _prepare_input(raw_imgs, gts, args.image_size, num_slices=args.num_slices, augmentation=args.augmentation)
	
	logger.debug(f"3D Volume shape: imgs={inputs['imgs'].shape} (slices, channels, height, width), num_classes={len(inputs['class_ids'])}")
	
	# Log num_slices parameter effect
	if args.num_slices >= 0:
		logger.debug(f"num_slices={args.num_slices}: Limiting adaptation slices around key slices")
		for class_id in inputs['class_ids']:
			key_slice = inputs['keyframes'][class_id]
			valid_range = inputs['valid_slices'][class_id]
			if valid_range is not None:
				start_idx, end_idx = valid_range
				total_slices = end_idx - start_idx + 1
				logger.debug(f"  Class {class_id}: key_slice={key_slice}, adapt_slices=[{start_idx}:{end_idx}] ({total_slices} slices)")
			else:
				logger.debug(f"  Class {class_id}: no valid slices")
	else:
		logger.debug(f"num_slices={args.num_slices}: Using all valid slices for adaptation")
	
	
	# Prepare prompt arguments for adaptation and inference
	full_prompt_args = {
		'prompt_type': args.prompt_type,
		'num_points': args.num_points,
		'points_type': args.point_strategy
	}
	
	# Subsampled prompts use specified configuration
	subsampled_prompt_args = {
		'prompt_type': args.subsampled_prompt_type,
		'num_points': args.subsampled_num_points,
		'points_type': args.subsampled_point_strategy,
		'box_expand_ratio': args.box_expand_ratio
	}
	
	logger.debug("Starting adaptation")
	inputs = _prepare_device_inputs(inputs, device, args.augmentation)
	original_image_encoder_cache = None
	for step in range(args.num_steps):
		_maybe_apply_cosine_lr_schedule(model, args, step_idx=step, total_steps=args.num_steps, logger=logger)
		adaptation_results, cache = model.adapt_one_imgs(
			input=inputs,
			full_prompt_args=full_prompt_args,
			subsampled_prompt_args=subsampled_prompt_args,
			augmentation=args.augmentation,
			freq_aug_bands=args.freq_aug_bands,
			freq_aug_prob=args.freq_aug_prob,
		)
		if step == 0 and cache is not None:
			original_image_encoder_cache = cache
		del adaptation_results, cache

	logger.debug("Starting inference")
	original_results, adapted_results = model.infer_after_adapt(
		inputs=inputs,
		prompt_args=full_prompt_args,
		original_image_encoder_cache=original_image_encoder_cache,
		num_keyslices=args.num_keyslices,
	)
	metrics = _compute_tta_volume_metrics(inputs, adapted_results, logger=logger)
	results = {
		'volume_name': volume_name,
		**metrics,
		'num_slices': inputs['imgs'].shape[0],
		'predictions': {'original': original_results, 'adapted': adapted_results} if args.save_predictions else {},
	}
	logger.debug(f"Sample {volume_name}: mean_dsc={results['mean_dsc']:.6f}")
	if args.save_predictions:
		save_predictions(results, volume_name, args.output_dir, logger)
	return results


def _load_sample_data(file_path: str, args):
	"""
	Load and prepare sample data from file.
	The returned CPU input is reusable across adaptation steps and inference.
	
	Args:
		file_path: Path to .npz file
		args: Command line arguments
		
	Returns:
		Tuple of (volume_name, inputs)
	"""
	# Load data
	raw_imgs, gts = _read_npz_arrays(file_path)
	
	# Extract volume name
	volume_name = os.path.splitext(os.path.basename(file_path))[0]
	
	# Prepare 3D volume and masks
	inputs = _prepare_input(raw_imgs, gts, args.image_size, num_slices=args.num_slices, augmentation=args.augmentation)
	
	return volume_name, inputs


def process_batch_adapt_then_eval(file_paths: List[str], model, args, device, logger) -> List[dict]:
	"""
	Process a batch of samples.

	Returns:
		List[dict]: Processing results for each sample
	"""
	batch_results = []
	
	# Prepare prompt arguments
	full_prompt_args = {
		'prompt_type': args.prompt_type,
		'num_points': args.num_points,
		'points_type': args.point_strategy
	}
	
	subsampled_prompt_args = {
		'prompt_type': args.subsampled_prompt_type,
		'num_points': args.subsampled_num_points,
		'points_type': args.subsampled_point_strategy,
		'box_expand_ratio': args.box_expand_ratio
	}
	
	batch_samples = [_load_sample_data(file_path, args) for file_path in file_paths]
	
	logger.debug(f"Adapting {len(file_paths)} preloaded samples")
	batch_samples = [
		(name, _prepare_device_inputs(inputs, device, args.augmentation))
		for name, inputs in batch_samples
	]
	all_original_caches = [None] * len(file_paths)
	for step in range(args.num_steps):
		_maybe_apply_cosine_lr_schedule(model, args, step_idx=step, total_steps=args.num_steps, logger=logger)
		for sample_idx, (volume_name, inputs) in enumerate(batch_samples):
			adaptation_results, cache = model.adapt_one_imgs(
				input=inputs,
				full_prompt_args=full_prompt_args,
				subsampled_prompt_args=subsampled_prompt_args,
				augmentation=args.augmentation,
				freq_aug_bands=args.freq_aug_bands,
				freq_aug_prob=args.freq_aug_prob,
			)
			if step == 0 and cache is not None:
				all_original_caches[sample_idx] = cache
			del adaptation_results, cache

	logger.debug(f"Evaluating {len(file_paths)} preloaded samples")
	for sample_idx, (volume_name, inputs) in enumerate(batch_samples):
		original_results, adapted_results = model.infer_after_adapt(
			inputs=inputs,
			prompt_args=full_prompt_args,
			original_image_encoder_cache=all_original_caches[sample_idx],
			num_keyslices=args.num_keyslices,
		)
		metrics = _compute_tta_volume_metrics(inputs, adapted_results, logger=logger)
		sample_results = {
			'volume_name': volume_name,
			**metrics,
			'num_slices': inputs['imgs'].shape[0],
			'predictions': {'original': original_results, 'adapted': adapted_results} if args.save_predictions else {},
		}
		batch_results.append(sample_results)
		logger.debug(f"Sample {volume_name}: mean_dsc={sample_results['mean_dsc']:.6f}")
		del inputs, original_results, adapted_results, volume_name
		batch_samples[sample_idx] = None
		all_original_caches[sample_idx] = None

	del all_original_caches, batch_samples
	return batch_results


def initialize_csv_log(output_dir: str, logger=None) -> str:
	"""
	Initialize CSV log file for tracking TTA results.
	
	Args:
		output_dir: Output directory path
		logger: Logger instance
		
	Returns:
		str: Path to the CSV log file
	"""
	os.makedirs(output_dir, exist_ok=True)
	csv_path = os.path.join(output_dir, "tta_results_log.csv")
	
	# Create CSV header for average results across all classes
	header = [
		'subtask', 'volume_name', 'num_classes', 'num_slices', 'mean_dsc'
	]
	
	# Write header to CSV file
	with open(csv_path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(header)
	
	if logger:
		logger.info(f"Initialized CSV log file: {csv_path}")
	
	return csv_path


def append_to_csv_log(csv_path: str, subtask: str, results: dict, logger=None):
	"""
	Append sample results to CSV log file.
	
	Args:
		csv_path: Path to CSV log file
		subtask: Subtask name (relative path from data_dir)
		results: Processing results from process_single_sample
		logger: Logger instance
	"""
	with open(csv_path, 'a', newline='', encoding='utf-8') as f:
		writer = csv.writer(f)
			
		volume_name = results['volume_name']
			
		# Write one row
		row = [
			subtask,
			volume_name,
			results['num_classes'],
			results.get('num_slices', 1),  # Default to 1 for 2D cases
			f"{results['mean_dsc']:.6f}"
		]

		writer.writerow(row)


def display_dsc_for_progress(results: dict) -> tuple[str, float]:
	return "mean_dsc", results['mean_dsc']


def build_final_metric_results(all_tta_results: list) -> list:
	return [
		{'directory': result['directory'], 'mean_dsc': result['mean_dsc']}
		for result in all_tta_results
	]


def save_predictions(results, volume_name, output_dir, logger=None):
	os.makedirs(output_dir, exist_ok=True)
	output_file = os.path.join(output_dir, f"{volume_name}_tta_results.npz")
	save_data = {'mean_dsc': results['mean_dsc']}
	for branch, class_results in results['predictions'].items():
		for class_id, result in class_results.items():
			save_data[f'pred_{branch}_{class_id}'] = result['predictions']
	np.savez_compressed(output_file, **save_data)
	if logger:
		logger.info(f"Saved predictions to: {output_file}")


def save_adapted_model(model, output_dir, logger=None):
	"""
	Save the adapted model parameters after TTA is complete.
	"""
	os.makedirs(output_dir, exist_ok=True)
	
	# Create model checkpoint path
	model_save_path = os.path.join(output_dir, "adapted_model.pt")
	
	# Save model state dict
	# If model has a save method, use it; otherwise save state_dict directly
	if hasattr(model, 'save_checkpoint'):
		model.save_checkpoint(model_save_path)
		if logger:
			logger.info(f"Saved adapted model checkpoint to: {model_save_path}")
	else:
		# Save the entire model state dict
		torch.save(model.state_dict(), model_save_path)
		if logger:
			logger.info(f"Saved adapted model state dict to: {model_save_path}")
		
	# Also save adapter parameters separately if available
	if hasattr(model, 'get_adapter_state_dict'):
		adapter_save_path = os.path.join(output_dir, "adapted_adapters.pt")
		adapter_state = model.get_adapter_state_dict()
		torch.save(adapter_state, adapter_save_path)
		if logger:
			logger.info(f"Saved adapter parameters to: {adapter_save_path}")

	return model_save_path


def _is_timestamp_dir_name(dir_name: str) -> bool:
	"""Check whether a directory name matches the timestamp format YYYYMMDD_HHMMSS."""
	return (
		len(dir_name) == 15
		and dir_name[8] == "_"
		and dir_name[:8].isdigit()
		and dir_name[9:].isdigit()
	)


def _find_completed_timestamp_run_dir(base_output_dir: str) -> Optional[str]:
	"""
	Find the latest completed timestamp run directory under the base output directory.

	A completed run is identified by the presence of `experiment_done.txt`.
	"""
	if not os.path.isdir(base_output_dir):
		return None

	candidates = sorted(os.listdir(base_output_dir), reverse=True)

	for name in candidates:
		if not _is_timestamp_dir_name(name):
			continue
		run_dir = os.path.join(base_output_dir, name)
		if not os.path.isdir(run_dir):
			continue
		if os.path.exists(os.path.join(run_dir, "experiment_done.txt")):
			return run_dir

	return None


def main():
	# Parse arguments
	args = parse_args()
	
	base_output_dir = args.output_dir
	# Skip experiment if a completed timestamped run already exists.
	if args.skip_if_done:
		completed_run_dir = _find_completed_timestamp_run_dir(base_output_dir)
		if completed_run_dir is not None:
			done_file_path = os.path.join(completed_run_dir, "experiment_done.txt")
			print(f"Experiment already completed! Done file found: {done_file_path}")
			print(f"Skipping execution. Existing run directory: {completed_run_dir}")
			print("Remove the done file to re-run the experiment.")
			return

	# Create time-based subdirectory under the specified output directory
	import datetime
	timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	args.output_dir = os.path.join(base_output_dir, timestamp)
	
	# Check if experiment is already completed.
	if check_experiment_done(args.output_dir):
		return
	
	# Set random seeds for reproducibility, i prefer 1314
	set_seeds(args.seed)
	
	# Initialize model, logging, and basic setup
	model, logger, device = initialize_model_and_logging(args)
	
	# Prepare dataset files
	files_by_dir, total_files = prepare_dataset(args, logger)
	
	# Initialize CSV log file for tracking results.
	csv_log_path = initialize_csv_log(args.output_dir, logger=logger)
	

	# Process all files with nested progress bars
	processed = 0
	current_subtask_dir = None 
	all_tta_results = []
	
	if total_files == 0:
		tqdm.write("No files found to process!")
		logger.info("===== SPDA for 2D/3D medical images end =====")
		return
	
	# Create outer progress bar for subtasks
	subtask_pbar = tqdm(
		total=len(files_by_dir), 
		desc="Subtasks", 
		position=0,
		leave=True,
		ascii=True,
		bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}"
	)
	
	for subtask_idx, (dir_path, files) in enumerate(files_by_dir.items()):
		# Handle subtask-level adapter reset
		if current_subtask_dir != dir_path:
			current_subtask_dir = dir_path
			rel_path = os.path.relpath(dir_path, args.data_dir)
			
			# Update outer progress bar description with fixed-width formatting
			subtask_display = f"{rel_path[:35]:<35}"  
			subtask_pbar.set_description(f"Subtask: {subtask_display}")
			subtask_pbar.set_postfix_str(f"Files: {len(files)}")
			
			# Initialize sample counter for this subtask (for partial reset)
			subtask_sample_counter = 0
			
			# Reset adapter parameters for new subtask if requested (always full reset)
			if args.reset_adapters == "subtask":
				reset_adapters_if_needed(model, "subtask", "new subtask", logger, reset_ratio=1.0)
				# Reset augmentation handler statistics when switching subtasks
				model.reset_augmentation_handler()
				logger.info(f"  Augmentation handler statistics reset for new subtask")
			else:
				tqdm.write("Adapter reset disabled - continuing with current state")
		
		# Create inner progress bar for files in current subtask
		file_pbar = tqdm(
			total=len(files),
			desc=f"Files",
			position=1,
			leave=False,
			ascii=True,
			bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}"
		)
		
		# Process files in batches based on eval_freq
		if args.eval_freq == 1:
			# Original behavior: process each file individually
			for file_idx, file_path in enumerate(files):
				# Update file progress bar with fixed-width formatting
				volume_name = os.path.splitext(os.path.basename(file_path))[0]
				# Display the last part of volume_name for better differentiation, right-aligned
				if len(volume_name) > 30:
					display_name = f"...{volume_name[-27:]}"  # Show last 27 chars with "..." prefix
				else:
					display_name = volume_name
				volume_display = f"{display_name:>30}"  # Right-align volume name, pad to 30 chars
				file_pbar.set_description(f"Processing: {volume_display}")
				
				# Check for partial reset within subtask (before processing sample)
				if args.reset_freq > 0 and subtask_sample_counter > 0 and subtask_sample_counter % args.reset_freq == 0:
					reset_adapters_if_needed(model, "subtask", f"partial reset (sample {subtask_sample_counter})", logger, reset_ratio=args.reset_ratio)
				
				# Save adapter params before processing (for momentum update)
				if args.momentum_update:
					model.save_adapter_params_for_momentum()
				
				# Process single sample (includes sample-level reset if needed)
				results = process_single_sample(file_path, model, args, device, logger)
				
				# Apply momentum update after processing (if enabled)
				if args.momentum_update:
					model.apply_momentum_update(args.momentum)
					logger.debug(f"Applied momentum update with momentum={args.momentum}")
				
				# Increment sample counter
				subtask_sample_counter += 1

				subtask_name = os.path.relpath(dir_path, args.data_dir)

				# Update progress bar with results
				dsc_label, avg_dsc = display_dsc_for_progress(results)
				file_pbar.set_postfix_str(f"{dsc_label}: {avg_dsc:.6f}")
				
				# Append results to CSV log
				append_to_csv_log(csv_log_path, subtask_name, results, logger)
				
				tta_result = {
					'directory': subtask_name,
					'mean_dsc': results['mean_dsc'],
					'num_slices': results['num_slices'],
				}
				all_tta_results.append(tta_result)
				
				processed += 1
				file_pbar.update(1)
		else:
			if args.momentum_update and subtask_idx == 0:
				logger.warning("Momentum update is disabled in batch mode (eval_freq > 1)")
			
			batch_size = args.eval_freq
			num_batches = (len(files) + batch_size - 1) // batch_size
			
			for batch_idx in range(num_batches):
				start_idx = batch_idx * batch_size
				end_idx = min(start_idx + batch_size, len(files))
				batch_files = files[start_idx:end_idx]
				
				# Update progress bar to show batch processing
				file_pbar.set_description(f"Batch {batch_idx + 1}/{num_batches} ({len(batch_files)} samples)")
				
				if args.reset_freq > 0 and subtask_sample_counter > 0 and subtask_sample_counter % args.reset_freq == 0:
					reset_adapters_if_needed(model, "subtask", f"partial reset (sample {subtask_sample_counter})", logger, reset_ratio=args.reset_ratio)
				
				if args.reset_adapters == "sample":
					reset_adapters_if_needed(model, "sample", "new batch", logger, reset_ratio=1.0)
				
				batch_results = process_batch_adapt_then_eval(batch_files, model, args, device, logger)
				
				# Increment sample counter by the number of samples in this batch
				subtask_sample_counter += len(batch_files)
				
				for file_path, results in zip(batch_files, batch_results):
					# Update progress bar
					if file_path == batch_files[-1]:
						dsc_label, avg_dsc = display_dsc_for_progress(results)
						file_pbar.set_postfix_str(f"Batch {dsc_label}: {avg_dsc:.6f}")
					
					# Append results to CSV log
					subtask_name = os.path.relpath(dir_path, args.data_dir)
					append_to_csv_log(csv_log_path, subtask_name, results, logger)
					
					tta_result = {
						'directory': subtask_name,
						'mean_dsc': results['mean_dsc'],
						'num_slices': results['num_slices'],
					}
					all_tta_results.append(tta_result)
					
					if args.save_predictions and results.get('predictions'):
						save_predictions(results, results['volume_name'], args.output_dir, logger)
					
					processed += 1
				
				# Update progress bar by the number of files processed in this batch
				file_pbar.update(len(batch_files))
		
		# Close file progress bar and update subtask progress
		file_pbar.close()
		subtask_pbar.update(1)
	
	# Close outer progress bar
	subtask_pbar.close()

	if all_tta_results:
		hierarchical_stats = calculate_hierarchical_directory_stats(all_tta_results, args.data_dir)
		print_directory_tree_stats(hierarchical_stats, args.data_dir, logger)
		json_output_path = os.path.join(args.output_dir, f"tta_hierarchical_stats_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
		export_tta_directory_tree_stats_json(hierarchical_stats, args.data_dir, json_output_path)
		logger.info(f"Directory summary JSON saved to: {json_output_path}")

	if all_tta_results:
		final_prediction = 'adapted'
		final_metric_results = build_final_metric_results(
			all_tta_results,
		)
		final_metric_summary = calculate_final_metric_summary(
			final_metric_results,
			args.data_dir,
			metric_names=('mean_dsc',),
			metadata={
				'script': os.path.basename(__file__),
				'result_type': 'tta_final_prediction',
				'final_prediction': final_prediction,
				'metric_source': 'mean_dsc in tta_results_log.csv',
			},
		)
		final_summary_json_path, final_summary_csv_path = export_final_metric_summary(
			final_metric_summary,
			args.output_dir,
		)
		log_final_metric_summary(final_metric_summary, logger, title='FINAL TTA METRIC SUMMARY')
		logger.info(f"Final summary JSON saved to: {final_summary_json_path}")
		logger.info(f"Final summary CSV saved to: {final_summary_csv_path}")

	tqdm.write("SPDA evaluation completed.")
	tqdm.write(f"Results logged to CSV file: {csv_log_path}")
	
	if args.save_adapted_model:
		logger.info("Saving adapted model parameters...")
		tqdm.write("Saving adapted model parameters...")
		model_path = save_adapted_model(model, args.output_dir, logger)
		if model_path:
			tqdm.write(f"Adapted model saved to: {model_path}")
	
	logger.info("===== SPDA for 2D/3D medical images end =====")
	
	# Create done file to mark experiment completion
	create_experiment_done_file(
		output_dir=args.output_dir,
		csv_log_path=csv_log_path,
		all_tta_results=all_tta_results,
		logger=logger
	)


if __name__ == "__main__":
	main()
