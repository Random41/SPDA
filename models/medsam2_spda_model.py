import math
import random

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import torch.optim as optim
from collections import OrderedDict
from scipy import ndimage

from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
    MLP,
)

from sam2.utils.misc import fill_holes_in_mask_scores

from utils.medsam2_spda_augmentation_utils import build_augmentation_handler
from training.utils.train_utils import (
    compute_feature_consistency_loss,
    combined_focal_dice_loss
)
from utils.evaluation_metrics import dice_coefficient


# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


def build_confidence_aware_pseudo_target(
    teacher_logits,
    target_mode="hard",
    temperature=1.0,
    confidence_threshold=0.0,
    confidence_power=0.0,
    min_keep_ratio=0.01,
    eps=1e-6,
):
    if target_mode not in ("hard", "soft"):
        raise ValueError(f"Unsupported pseudo target mode: {target_mode}")
    if temperature <= 0:
        raise ValueError("pseudo temperature must be positive")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("pseudo confidence threshold must be in [0, 1]")
    if confidence_power < 0:
        raise ValueError("pseudo confidence power must be non-negative")
    if not 0.0 <= min_keep_ratio <= 1.0:
        raise ValueError("pseudo minimum keep ratio must be in [0, 1]")

    teacher_prob = torch.sigmoid(teacher_logits.detach() / float(temperature))
    if target_mode == "hard":
        pseudo_target = (teacher_prob >= 0.5).to(teacher_prob.dtype)
    else:
        pseudo_target = teacher_prob

    confidence = torch.abs(2.0 * teacher_prob - 1.0)
    flat_confidence = confidence.reshape(-1, confidence.shape[-2] * confidence.shape[-1])
    flat_keep = flat_confidence >= float(confidence_threshold)

    min_keep_count = int(math.ceil(float(min_keep_ratio) * flat_confidence.shape[1]))
    if min_keep_count > 0:
        min_keep_count = min(min_keep_count, flat_confidence.shape[1])
        for mask_idx in range(flat_keep.shape[0]):
            if int(flat_keep[mask_idx].sum().item()) >= min_keep_count:
                continue
            top_indices = torch.topk(
                flat_confidence[mask_idx], k=min_keep_count, largest=True, sorted=False
            ).indices
            flat_keep[mask_idx, top_indices] = True

    keep_mask = flat_keep.reshape_as(confidence)
    pixel_weight = keep_mask.to(confidence.dtype)
    if confidence_power > 0:
        pixel_weight = pixel_weight * confidence.pow(float(confidence_power))

    return pseudo_target.detach(), pixel_weight.detach()


def confidence_weighted_focal_dice_loss(
    pred,
    target,
    pixel_weight,
    focal_alpha=0.25,
    focal_gamma=2.0,
    focal_weight=1.0,
    dice_weight=1.0,
    bce_weight=0.0,
    eps=1e-6,
):
    """Apply the existing mask-loss components with normalized pixel weights."""
    target = target.to(device=pred.device, dtype=pred.dtype)
    pixel_weight = pixel_weight.to(device=pred.device, dtype=pred.dtype)
    if target.shape != pred.shape or pixel_weight.shape != pred.shape:
        raise ValueError(
            "pred, target, and pixel_weight must have identical shapes, got "
            f"{pred.shape}, {target.shape}, and {pixel_weight.shape}"
        )

    normalizer = pixel_weight.sum().clamp_min(eps)
    pred_prob = torch.sigmoid(pred)
    total_loss = pred.new_tensor(0.0)

    if focal_weight != 0.0:
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = pred_prob * target + (1.0 - pred_prob) * (1.0 - target)
        alpha_t = focal_alpha * target + (1.0 - focal_alpha) * (1.0 - target)
        focal_map = alpha_t * (1.0 - p_t).pow(focal_gamma) * ce_loss
        total_loss = total_loss + float(focal_weight) * (
            focal_map * pixel_weight
        ).sum() / normalizer

    if dice_weight != 0.0:
        pred_flat = pred_prob.reshape(pred_prob.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        weight_flat = pixel_weight.reshape(pixel_weight.shape[0], -1)
        intersection = (weight_flat * pred_flat * target_flat).sum(dim=1)
        denominator = (
            (weight_flat * pred_flat).sum(dim=1)
            + (weight_flat * target_flat).sum(dim=1)
        )
        dice_coeff = (2.0 * intersection + eps) / (denominator + eps)
        total_loss = total_loss + float(dice_weight) * (1.0 - dice_coeff.mean())

    if bce_weight != 0.0:
        bce_map = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        total_loss = total_loss + float(bce_weight) * (
            bce_map * pixel_weight
        ).sum() / normalizer

    return total_loss


def sample_points_from_mask(mask, num_points=1, strategy='center', max_shift=20):
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D")
    
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0:
        h, w = mask.shape
        return np.array([[w//2, h//2]]), np.array([1])
    
    points = []
    
    if strategy == 'center':
        center_x = np.mean(x_indices)
        center_y = np.mean(y_indices)
        points.append([center_x, center_y])
        
        if num_points > 1:
            additional_points = np.random.choice(len(x_indices), 
                                               min(num_points - 1, len(x_indices)), 
                                               replace=False)
            for idx in additional_points:
                points.append([x_indices[idx], y_indices[idx]])
                
    elif strategy == 'random':
        sampled_indices = np.random.choice(len(x_indices), 
                                         min(num_points, len(x_indices)), 
                                         replace=False)
        for idx in sampled_indices:
            points.append([x_indices[idx], y_indices[idx]])
            
    elif strategy == 'boundary':
        boundary = mask - ndimage.binary_erosion(mask)
        by_indices, bx_indices = np.where(boundary > 0)
        
        if len(by_indices) > 0:
            boundary_samples = min(num_points, len(bx_indices))
            sampled_boundary = np.random.choice(len(bx_indices), boundary_samples, replace=False)
            for idx in sampled_boundary:
                points.append([bx_indices[idx], by_indices[idx]])
            
            if len(points) < num_points:
                inner_mask = mask - boundary
                iy_indices, ix_indices = np.where(inner_mask > 0)
                if len(iy_indices) > 0:
                    remaining = num_points - len(points)
                    sampled_inner = np.random.choice(len(ix_indices), 
                                                   min(remaining, len(ix_indices)), 
                                                   replace=False)
                    for idx in sampled_inner:
                        points.append([ix_indices[idx], iy_indices[idx]])
        else:
            sampled_indices = np.random.choice(len(x_indices), 
                                             min(num_points, len(x_indices)), 
                                             replace=False)
            for idx in sampled_indices:
                points.append([x_indices[idx], y_indices[idx]])
    
    while len(points) < num_points and len(x_indices) > 0:
        idx = np.random.randint(len(x_indices))
        points.append([x_indices[idx], y_indices[idx]])
    
    points = np.array(points[:num_points])
    labels = np.ones(len(points), dtype=int)
    
    if max_shift > 0:
        h, w = mask.shape
        for i in range(len(points)):
            shift_x = np.random.randint(-max_shift, max_shift + 1)
            shift_y = np.random.randint(-max_shift, max_shift + 1)
            
            points[i, 0] = np.clip(points[i, 0] + shift_x, 0, w - 1)
            points[i, 1] = np.clip(points[i, 1] + shift_y, 0, h - 1)
    
    return points, labels


def generate_bbox(mask, max_shift=20):
    """
    Generate a box prompt from a binary mask.
    """
    mask_np = np.asarray(mask)
    if mask_np.ndim == 2:
        y_indices, x_indices = np.where(mask_np > 0)
        if len(y_indices) == 0:
            h, w = mask_np.shape
            return np.array([0, 0, w - 1, h - 1])

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        h, w = mask_np.shape
        bbox_shift = np.random.randint(0, max_shift + 1, 1)[0] if max_shift > 0 else 0
        x_min = max(0, x_min - bbox_shift)
        x_max = min(w - 1, x_max + bbox_shift)
        y_min = max(0, y_min - bbox_shift)
        y_max = min(h - 1, y_max + bbox_shift)
        return np.array([x_min, y_min, x_max, y_max])

    if mask_np.ndim == 3:
        z_indices, y_indices, x_indices = np.where(mask_np > 0)
        if len(z_indices) == 0:
            d, h, w = mask_np.shape
            return np.array([0, 0, 0, w - 1, h - 1, d - 1])

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        z_min, z_max = np.min(z_indices), np.max(z_indices)
        d, h, w = mask_np.shape
        bbox_shift = np.random.randint(0, max_shift + 1, 1)[0] if max_shift > 0 else 0
        x_min = max(0, x_min - bbox_shift)
        x_max = min(w - 1, x_max + bbox_shift)
        y_min = max(0, y_min - bbox_shift)
        y_max = min(h - 1, y_max + bbox_shift)
        z_min = max(0, z_min)
        z_max = min(d - 1, z_max)
        return np.array([x_min, y_min, z_min, x_max, y_max, z_max])

    raise ValueError("Input mask must be 2D or 3D")


def _generate_full_prompt(ground_truth_mask, prompt_args, device):
    """
    Generate full prompt.
    """
    prompt_type = prompt_args['prompt_type']
    
    # Convert to numpy for processing
    if isinstance(ground_truth_mask, torch.Tensor):
        mask_np = ground_truth_mask.cpu().numpy().astype(np.uint8)
    else:
        mask_np = ground_truth_mask.astype(np.uint8)
    
    if prompt_type in ['point', 'points']:
        num_points = prompt_args.get('num_points', 2)
        points_type = prompt_args.get('points_type', 'center')
        points, labels = sample_points_from_mask(mask_np, num_points=num_points, strategy=points_type)
        return {
            'points': torch.tensor(points, dtype=torch.float32, device=device),
            'labels': torch.tensor(labels, dtype=torch.int32, device=device)
        }
        
    elif prompt_type == 'box':
        # Generate bounding box
        bbox = generate_bbox(mask_np)
        return {
            'box': torch.tensor(bbox, dtype=torch.float32, device=device)
        }
        
    elif prompt_type == 'mask':
        raise NotImplementedError("Direct mask prompts are not implemented in this function.")
        
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")


def _generate_subsampled_prompt(full_prompt, subsampled_prompt_args, ground_truth_mask, device):
    """
    Generate subsampled prompt.
    """
    prompt_type = subsampled_prompt_args.get('prompt_type', 'point')
    
    # Handle box-type subsampled prompts
    if prompt_type == 'box':
        box_expand_ratio = subsampled_prompt_args.get('box_expand_ratio', 0.1)
        H, W = ground_truth_mask.shape
        
        if 'box' in full_prompt:
            # Expand existing box
            box = full_prompt['box'].cpu().numpy()
            x_min, y_min, x_max, y_max = box
            
            # Calculate box dimensions
            width = x_max - x_min
            height = y_max - y_min
            
            # Expand box outward
            new_x_min = x_min - width * box_expand_ratio
            new_x_max = x_max + width * box_expand_ratio
            new_y_min = y_min - height * box_expand_ratio
            new_y_max = y_max + height * box_expand_ratio
            
            # Clip to image bounds
            new_x_min = max(0, new_x_min)
            new_y_min = max(0, new_y_min)
            new_x_max = min(W, new_x_max)
            new_y_max = min(H, new_y_max)
            
            return {
                'box': torch.tensor([new_x_min, new_y_min, new_x_max, new_y_max], 
                                   dtype=torch.float32, device=device)
            }
            
        elif 'points' in full_prompt:
            # Cannot generate semantically meaningful box from points
            raise NotImplementedError(
                "Cannot generate box-based subsampled prompt from point-based full prompt. "
                "It is difficult to generate a semantically meaningful box from points."
            )
        else:
            raise ValueError("Full prompt must contain 'points' or 'box'")
    
    # Handle point-type subsampled prompts
    elif prompt_type == 'point':
        num_points = subsampled_prompt_args['num_points']
        points_type = subsampled_prompt_args.get('points_type', 'random')
        
        if num_points <= 0:
            # Return minimal point prompt
            H, W = ground_truth_mask.shape
            return {
                'points': torch.tensor([[W//2, H//2]], dtype=torch.float32, device=device),
                'labels': torch.tensor([1], dtype=torch.int32, device=device)
            }
        
        if 'box' in full_prompt:
            # Sample center point(s) from box
            box = full_prompt['box'].cpu().numpy()
            x_min, y_min, x_max, y_max = box
            
            # Generate points within the box
            points = []
            labels = []
            
            # Always include center point
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            points.append([center_x, center_y])
            labels.append(1)
            
            for _ in range(num_points - 1):
                rand_x = np.random.uniform(x_min, x_max)
                rand_y = np.random.uniform(y_min, y_max)
                points.append([rand_x, rand_y])
                labels.append(1)
            
            points = np.array(points[:num_points])
            
            return {
                'points': torch.tensor(points, dtype=torch.float32, device=device),
                'labels': torch.tensor(labels[:num_points], dtype=torch.int32, device=device)
            }
            
        elif 'points' in full_prompt:
            # Subsample from existing points
            coords = full_prompt['points']
            point_labels = full_prompt['labels']
            coords_np = coords.cpu().numpy()
            labels_np = point_labels.cpu().numpy()
            
            # Randomly select subset of points
            num_available = len(coords_np)
            num_to_sample = min(num_points, num_available)
            
            if num_to_sample == 0:
                H, W = ground_truth_mask.shape
                return {
                    'points': torch.tensor([[W//2, H//2]], dtype=torch.float32, device=device),
                    'labels': torch.tensor([1], dtype=torch.int32, device=device)
                }
            
            selected_indices = np.random.choice(
                num_available, size=num_to_sample, replace=False
            )
            
            selected_coords = coords_np[selected_indices]
            selected_labels = labels_np[selected_indices]
            
            return {
                'points': torch.tensor(selected_coords, dtype=torch.float32, device=device),
                'labels': torch.tensor(selected_labels, dtype=torch.int32, device=device)
            }
            
        else:
            raise ValueError("Full prompt must contain 'points' or 'box'")
    
    else:
        raise ValueError(f"Unknown subsampled prompt_type: {prompt_type}. Must be 'point' or 'box'.")


def _generate_prompts_for_masks(masks, keyframes, class_ids, full_prompt_args, subsampled_prompt_args, device):
    full_prompts = {}
    subsampled_prompts = {}
    
    for class_id in class_ids:
        keyframe_idx = keyframes[class_id]
        class_masks = masks[class_id]  # Shape: [D, H, W]
        keyframe_mask = class_masks[keyframe_idx]  # Shape: [H, W]
        full_prompt = _generate_full_prompt(
            ground_truth_mask=keyframe_mask,
            prompt_args=full_prompt_args,
            device='cpu'
        )
        
        subsampled_prompt = _generate_subsampled_prompt(
            full_prompt=full_prompt,
            subsampled_prompt_args=subsampled_prompt_args,
            ground_truth_mask=keyframe_mask,
            device='cpu'
        )
        full_prompts[class_id] = {key: value.to(device) for key, value in full_prompt.items()}
        subsampled_prompts[class_id] = {key: value.to(device) for key, value in subsampled_prompt.items()}
    
    return full_prompts, subsampled_prompts


class ImageEncoderAdapter(torch.nn.Module):
    """
    Adapter module for image encoder features.
    """
    def __init__(self, channels, enable_adapter=True):
        super().__init__()
        self.enable_adapter = enable_adapter
        if enable_adapter:
            self.adapter_conv = torch.nn.Conv2d(
                channels, channels, kernel_size=1, bias=False
            )
            torch.nn.init.zeros_(self.adapter_conv.weight)
            
    def forward(self, x):
        if not self.enable_adapter:
            return x
        
        identity = x
        adapted = self.adapter_conv(x)
        
        out = identity + adapted
        
        return out


class SAM2Adapt(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        # Optimizer parameters
        optimizer_lr=1e-4,
        decoder_lr=0.0,
        optimizer_weight_decay=1e-2,
        optimizer_betas=(0.9, 0.999),
        optimizer_eps=1e-8,
        # Random seed
        seed=2025,
        # adapt parameters
        adapt_device=None,
        offload_state_to_cpu=False,
        adapt_fillhole=0,
        gamma=0.0,
        feature_loss_mode='legacy_cosine',
        weak_mask_loss_weight=1.0,
        strong_mask_loss_weight=1.0,
        mask_focal_loss_weight=1.0,
        mask_dice_loss_weight=1.0,
        mask_bce_loss_weight=0.0,
        pseudo_target_mode='hard',
        pseudo_temperature=1.0,
        pseudo_confidence_threshold=0.0,
        pseudo_confidence_power=0.0,
        pseudo_min_keep_ratio=0.01,
        batch_size=4,
        # infer_after_adapt parameters
        eval_fillhole=0,
        # Augmentation handler configuration
        augmentation_type='weak',
        augmentation_args=None,
        **kwargs,
    ):
        super().__init__()
        
        # Initialize all SAM2Adapter functionality
        self._init_sam2_adapter(
            image_encoder=image_encoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            **kwargs
        )     
        
        # Random number generator with configurable seed
        self.rng = np.random.default_rng(seed=seed)
        
        # Independent random number generator for adapter parameter reset
        self.reset_rng = random.Random(seed)
        
        augmentation_config = {
            'type': augmentation_type,
            'seed': seed,
            'aug_args': augmentation_args if augmentation_args is not None else {}
        }
        self.augmentation_handler = build_augmentation_handler(augmentation_config)
        
        # Optimizer parameters (stored for later initialization)
        self.optimizer_lr = optimizer_lr
        self.decoder_lr = decoder_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_betas = optimizer_betas
        self.optimizer_eps = optimizer_eps
        self.optimizer = None  # Will be created later via initialize_optimizer()
        
        # adapt parameters
        if isinstance(adapt_device, str) and adapt_device != "None":
            import torch
            self.adapt_device = torch.device(adapt_device)
        elif adapt_device == "None" or adapt_device is None:
            self.adapt_device = None
        else:
            self.adapt_device = adapt_device
        self.offload_state_to_cpu = offload_state_to_cpu
        self.adapt_fillhole = adapt_fillhole
        self.gamma = gamma
        self.feature_loss_mode = feature_loss_mode
        self.weak_mask_loss_weight = weak_mask_loss_weight
        self.strong_mask_loss_weight = strong_mask_loss_weight
        self.mask_focal_loss_weight = mask_focal_loss_weight
        self.mask_dice_loss_weight = mask_dice_loss_weight
        self.mask_bce_loss_weight = mask_bce_loss_weight
        self.pseudo_target_mode = pseudo_target_mode
        self.pseudo_temperature = pseudo_temperature
        self.pseudo_confidence_threshold = pseudo_confidence_threshold
        self.pseudo_confidence_power = pseudo_confidence_power
        self.pseudo_min_keep_ratio = pseudo_min_keep_ratio
        self.batch_size = batch_size
        
        # infer_after_adapt parameters
        self.eval_fillhole = eval_fillhole
        

    def _init_sam2_adapter(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,
        image_size=512,
        backbone_stride=16,
        sigmoid_scale_for_mem_enc=1.0,
        sigmoid_bias_for_mem_enc=0.0,
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,
        max_cond_frames_in_attn=-1,
        directly_add_no_mem_embed=False,
        use_high_res_features_in_sam=False,
        multimask_output_in_sam=False,
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=False,
        use_multimask_token_for_obj_ptr: bool = False,
        iou_prediction_use_sigmoid=False,
        memory_temporal_stride_for_eval=1,
        non_overlap_masks_for_mem_enc=False,
        use_obj_ptrs_in_encoder=False,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=False,
        use_signed_tpos_enc_to_obj_ptrs=False,
        only_obj_ptrs_in_the_past_for_eval=False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        fixed_no_obj_ptr: bool = False,
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        no_obj_embed_spatial: bool = False,
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
        enable_image_encoder_adapter: bool = True,
    ):
        """Initialize all SAM2Adapter functionality."""
        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Initialize image encoder adapters for different feature levels
        self.enable_image_encoder_adapter = enable_image_encoder_adapter
        if enable_image_encoder_adapter:
            # We create adapters dynamically based on the actual feature channels
            # This will be initialized in the first forward pass when we know the actual dimensions
            self.image_encoder_adapters = None
            self._adapters_initialized = False
        else:
            self.image_encoder_adapters = None
            self._adapters_initialized = True

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # Model compilation
        if compile_image_encoder:
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )
        
        # Initialize adapters using a dummy forward pass during construction
        if self.enable_image_encoder_adapter:
            self._initialize_adapters()

    @property
    def device(self):
        return next(self.parameters()).device

    def initialize_optimizer(self):    
        parameter_groups = self.get_parameter_groups_for_optimization()
        
        if not parameter_groups:
            raise ValueError("No parameters found for optimization. Check if adapters are enabled.")
        
        self.optimizer = optim.AdamW(
            parameter_groups,
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps
        )
            
    def get_adapter_parameters(self):
        if not self.enable_image_encoder_adapter or self.image_encoder_adapters is None:
            return []
        
        adapter_params = []
        for adapter in self.image_encoder_adapters:
            adapter_params.extend(list(adapter.parameters()))
        
        return adapter_params

    def get_parameter_groups_for_optimization(self):
        parameter_groups = []
        
        adapter_params = self.get_adapter_parameters()
        if adapter_params:
            parameter_groups.append({
                'params': adapter_params,
                'name': 'adapters'
            })
        
        if self.decoder_lr > 0:
            decoder_params = list(self.sam_mask_decoder.parameters())
            if decoder_params:
                parameter_groups.append({
                    'params': decoder_params,
                    'lr': self.decoder_lr,
                    'name': 'mask_decoder'
                })
        
        return parameter_groups


    def clear_memory_cache(self):
        """
        Clear various memory caches to free up memory.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Force garbage collection
        import gc
        gc.collect()
    
    def _set_training_mode(self):
        # Set the entire model to training mode
        self.train()
        
        # Get adapter parameters
        adapter_params = set(self.get_adapter_parameters())
        
        decoder_params = set()
        if self.decoder_lr > 0:
            decoder_params = set(self.sam_mask_decoder.parameters())
        
        # Combine all trainable parameters
        trainable_params = adapter_params | decoder_params
        
        # Disable gradients for all parameters first
        for param in self.parameters():
            param.requires_grad = False
        
        # Enable gradients only for trainable parameters (adapters + optional decoder)
        for param in trainable_params:
            param.requires_grad = True      

    def _set_evaluation_mode(self):
        """
        Set the model to evaluation mode and restore original gradient requirements.
        """
        self.eval()

    def adapt_one_imgs(self, input, full_prompt_args, subsampled_prompt_args,
                      augmentation='weak', freq_aug_bands=3, freq_aug_prob=0.8):
        if self.offload_state_to_cpu:
            storage_device = torch.device("cpu")
        else:
            storage_device = self.adapt_device if self.adapt_device is not None else self.device
        
        self._set_training_mode()
        
        weak_imgs = input['imgs']
        imgs_for_aug = input.get('imgs_for_aug', None)
        original_masks = input['masks']
        resized_masks = input['resized_masks']
        
        compute_device = self.adapt_device if self.adapt_device is not None else self.device
        
        aug_runtime_args = {}
        
        if augmentation == 'freq':
            aug_runtime_args = {
                'num_low_freq_bands': freq_aug_bands,
                'augmentation_probability': freq_aug_prob,
                'single_pass': True
            }
        elif augmentation == 'spda':
            aug_runtime_args = {
                'storage_device': storage_device,
                'compute_device': compute_device
            }
        
        if augmentation == 'spda' and imgs_for_aug is not None:
            aug_input_imgs = imgs_for_aug
        else:
            aug_input_imgs = input.get('_cpu_augmentation_imgs', input['imgs'])
        if augmentation == 'spda':
            aug_input_imgs = aug_input_imgs.to(device=compute_device, dtype=torch.float32)
        strong_imgs = self.augmentation_handler.apply_augmentation(
            aug_input_imgs,
            args=aug_runtime_args
        )
        
        weak_image_encoder_cache = self._precompute_image_encoder_cache(
            weak_imgs, storage_device, compute_device, batch_size=self.batch_size
        )
        strong_image_encoder_cache = self._precompute_image_encoder_cache(
            strong_imgs, storage_device, compute_device, batch_size=self.batch_size
        )

        full_prompts_weak, subsampled_prompts_weak = _generate_prompts_for_masks(
            original_masks, input['keyframes'], input['class_ids'], full_prompt_args, subsampled_prompt_args, compute_device
        )
        
        for class_id in input['class_ids']:
            keyframe_idx = input['keyframes'][class_id]
            orig_mask = original_masks[class_id][keyframe_idx]
            resized_mask = resized_masks[class_id][keyframe_idx]
            
            H_orig, W_orig = orig_mask.shape
            H_resize, W_resize = resized_mask.shape
            
            scale_h = H_resize / H_orig
            scale_w = W_resize / W_orig
            
            full_prompts_weak[class_id] = self._scale_prompt_coordinates(
                full_prompts_weak[class_id], scale_w, scale_h, full_prompt_args['prompt_type']
            )
            
            subsampled_prompts_weak[class_id] = self._scale_prompt_coordinates(
                subsampled_prompts_weak[class_id], scale_w, scale_h, subsampled_prompt_args['prompt_type']
            )

        adaptation_results = {}
        
        for class_id in input['class_ids']:
            if self.optimizer is None:
                raise RuntimeError("Optimizer not initialized. Please call initialize_optimizer() after moving model to device.")
            self.optimizer.zero_grad(set_to_none=True)
            num_frames = input['imgs'].shape[0]
            
            self._current_storage_device = storage_device
            
            total_loss = 0.0
            
            
            valid_slice_range = input.get('valid_slices', {}).get(class_id, None)
            if valid_slice_range is not None:
                start_idx, end_idx = valid_slice_range
                keyframe_idx = input['keyframes'][class_id]
                forward_frames = max(0, min(end_idx + 1, num_frames) - keyframe_idx - 1)
                backward_frames = max(0, keyframe_idx - max(start_idx, 0))
                total_frames_to_process = 1 + forward_frames + backward_frames
            else:
                total_frames_to_process = num_frames
            
            
            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {}
            }
            
            keyframe_idx = input['keyframes'][class_id]
            weak_backbone_out_keyframe, weak_backbone_out_adapt_keyframe = self._get_image_features(weak_imgs, keyframe_idx, weak_image_encoder_cache)
            strong_backbone_out_keyframe, strong_backbone_out_adapt_keyframe = self._get_image_features(strong_imgs, keyframe_idx, strong_image_encoder_cache)
            full_point_box_inputs = self._get_prompt_features(full_prompts_weak[class_id], full_prompt_args['prompt_type'])
            subsampled_point_inputs = self._get_prompt_features(subsampled_prompts_weak[class_id], subsampled_prompt_args['prompt_type'])
            
            compact_out_keyframe, loss_keyframe = self._run_single_frame_adapt(
                frame_idx=keyframe_idx,
                is_init_cond_frame=True,
                weak_backbone_out=weak_backbone_out_adapt_keyframe,
                strong_backbone_out=strong_backbone_out_adapt_keyframe,
                full_point_inputs=full_point_box_inputs,
                subsampled_point_inputs=subsampled_point_inputs,
                output_dict=output_dict,
                num_frames=num_frames,
                run_mem_encoder=True,
                reverse=False,
                weak_backbone_out_orig=weak_backbone_out_keyframe,
            )
            
            output_dict["cond_frame_outputs"][keyframe_idx] = compact_out_keyframe
            total_loss += loss_keyframe

            valid_slice_range = input.get('valid_slices', {}).get(class_id, None)
            if valid_slice_range is not None:
                start_idx, end_idx = valid_slice_range
                forward_end = min(end_idx + 1, num_frames)
            else:
                forward_end = num_frames
            
            for frame_idx in range(keyframe_idx + 1, forward_end):
                weak_backbone_out_frame, weak_backbone_out_adapt_frame = self._get_image_features(weak_imgs, frame_idx, weak_image_encoder_cache)
                strong_backbone_out_frame, strong_backbone_out_adapt_frame = self._get_image_features(strong_imgs, frame_idx, strong_image_encoder_cache)
                
                compact_out_frame, loss_frame = self._run_single_frame_adapt(
                    frame_idx=frame_idx,
                    is_init_cond_frame=False,
                    weak_backbone_out=weak_backbone_out_adapt_frame,
                    strong_backbone_out=strong_backbone_out_adapt_frame,
                    full_point_inputs=full_point_box_inputs,
                    subsampled_point_inputs=subsampled_point_inputs,
                    output_dict=output_dict,
                    num_frames=num_frames,
                    run_mem_encoder=True,
                    reverse=False,
                    weak_backbone_out_orig=weak_backbone_out_frame,
                )
                
                output_dict["non_cond_frame_outputs"][frame_idx] = compact_out_frame
                total_loss += loss_frame
                
            if valid_slice_range is not None:
                start_idx, end_idx = valid_slice_range
                backward_start = max(start_idx - 1, -1)
            else:
                backward_start = -1
            
            for frame_idx in range(keyframe_idx - 1, backward_start, -1):
                weak_backbone_out_frame, weak_backbone_out_adapt_frame = self._get_image_features(weak_imgs, frame_idx, weak_image_encoder_cache)
                strong_backbone_out_frame, strong_backbone_out_adapt_frame = self._get_image_features(strong_imgs, frame_idx, strong_image_encoder_cache)
                
                compact_out_frame, loss_frame = self._run_single_frame_adapt(
                    frame_idx=frame_idx,
                    is_init_cond_frame=False,
                    weak_backbone_out=weak_backbone_out_adapt_frame,
                    strong_backbone_out=strong_backbone_out_adapt_frame,
                    full_point_inputs=full_point_box_inputs,
                    subsampled_point_inputs=subsampled_point_inputs,
                    output_dict=output_dict,
                    num_frames=num_frames,
                    run_mem_encoder=True,
                    reverse=True,
                    weak_backbone_out_orig=weak_backbone_out_frame,
                )
                
                output_dict["non_cond_frame_outputs"][frame_idx] = compact_out_frame
                total_loss += loss_frame
                
            
            adaptation_results[class_id] = {
                'total_loss': total_loss,
                'avg_loss_per_frame': total_loss / num_frames,
                'num_frames': num_frames,
                'keyframe_idx': keyframe_idx,
            }
            
            if hasattr(self, '_current_storage_device'):
                delattr(self, '_current_storage_device')
            
        original_image_encoder_cache = weak_image_encoder_cache
        
        return adaptation_results, original_image_encoder_cache

    def _precompute_image_encoder_cache(self, images, storage_device, compute_device, batch_size=4):
        cache = {}
        num_frames = images.shape[0]
        
        with torch.no_grad():
            images_on_device = images.to(device=compute_device, dtype=torch.float32).contiguous()
            for batch_start in range(0, num_frames, batch_size):
                batch_end = min(batch_start + batch_size, num_frames)
                batch_indices = list(range(batch_start, batch_end))
                current_batch_size = len(batch_indices)
                
                batch_images = images_on_device[batch_start:batch_end]
                
                batch_backbone_out = self.image_encoder(batch_images)
                
                for i, frame_idx in enumerate(batch_indices):
                    single_backbone_out = {
                        "backbone_fpn": [feat[i:i+1] for feat in batch_backbone_out["backbone_fpn"]],
                        "vision_pos_enc": [pos_enc[i:i+1] for pos_enc in batch_backbone_out["vision_pos_enc"]]
                    }
                    
                    # Keep features resident unless CPU offload was explicitly requested.
                    cached_backbone_out = {}
                    cached_backbone_out["backbone_fpn"] = [
                        feat.to(storage_device, non_blocking=True) for feat in single_backbone_out["backbone_fpn"]
                    ]
                    cached_backbone_out["vision_pos_enc"] = [
                        pos_enc.to(storage_device, non_blocking=True) for pos_enc in single_backbone_out["vision_pos_enc"]
                    ]
                    
                    cache[frame_idx] = cached_backbone_out
                
        return cache

    def _get_image_features(self, images, frame_idx, image_encoder_cache=None):
        if image_encoder_cache is not None and frame_idx in image_encoder_cache:
            cached_backbone_out = image_encoder_cache[frame_idx]
            
            backbone_out = {}
            backbone_out["backbone_fpn"] = [
                feat.to(self.device, non_blocking=True) for feat in cached_backbone_out["backbone_fpn"]
            ]
            backbone_out["vision_pos_enc"] = [
                pos_enc.to(self.device, non_blocking=True) for pos_enc in cached_backbone_out["vision_pos_enc"]
            ]
            
            backbone_out_copy, backbone_out_adapt = self.forward_adapters(backbone_out)
            
            return backbone_out_copy, backbone_out_adapt
        else:
            images_on_device = images.to(device=self.device, dtype=torch.float32)
            image = images_on_device[frame_idx].unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
            backbone_out, backbone_out_adapt = self.forward_image_with_adapters(image)
            return backbone_out, backbone_out_adapt
    
    
    def _scale_prompt_coordinates(self, prompt, scale_w, scale_h, prompt_type):
        """
        Scale prompt coordinates from original mask size to resized mask size.
            
        Returns:
            scaled_prompt: prompt with scaled coordinates
        """
        if prompt_type in ['point', 'points']:
            scaled_prompt = {}
            
            if 'labels' in prompt:
                points_data = prompt['points']
                labels_data = prompt['labels']
            else:
                points_data, labels_data = prompt['points']
            
            if isinstance(points_data, torch.Tensor):
                scaled_points = points_data.clone()
            else:
                scaled_points = torch.tensor(points_data, dtype=torch.float32)
            
            scaled_points[..., 0] = scaled_points[..., 0] * scale_w  # scale x
            scaled_points[..., 1] = scaled_points[..., 1] * scale_h  # scale y
            
            # Preserve the original format
            if 'labels' in prompt:
                scaled_prompt['points'] = scaled_points
                scaled_prompt['labels'] = labels_data
            else:
                scaled_prompt['points'] = (scaled_points, labels_data)
                
            return scaled_prompt
            
        elif prompt_type == 'box':
            if isinstance(prompt, dict) and 'box' in prompt:
                box_data = prompt['box']
            else:
                box_data = prompt
            
            if isinstance(box_data, torch.Tensor):
                scaled_box = box_data.clone()
            else:
                scaled_box = torch.tensor(box_data, dtype=torch.float32)
            
            scaled_box[0] = scaled_box[0] * scale_w  # x_min
            scaled_box[1] = scaled_box[1] * scale_h  # y_min
            scaled_box[2] = scaled_box[2] * scale_w  # x_max
            scaled_box[3] = scaled_box[3] * scale_h  # y_max
            
            if isinstance(prompt, dict) and 'box' in prompt:
                return {'box': scaled_box}
            else:
                return scaled_box
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")
    
    def _get_prompt_features(self, prompt, prompt_type):
        if prompt_type in ['point', 'points']:
            if 'labels' in prompt:
                points_data = prompt['points']
                labels_data = prompt['labels']
            else:
                points_data, labels_data = prompt['points']
            
            if not isinstance(points_data, torch.Tensor):
                points_data = torch.tensor(points_data, dtype=torch.float32)
            if not isinstance(labels_data, torch.Tensor):
                labels_data = torch.tensor(labels_data, dtype=torch.int32)

            if points_data.dim() == 2:
                points_data = points_data.unsqueeze(0)
            if labels_data.dim() == 1:
                labels_data = labels_data.unsqueeze(0)

            points = points_data.to(self.device)
            labels = labels_data.to(self.device)
            
        elif prompt_type == 'box':
            if isinstance(prompt, dict) and 'box' in prompt:
                box_prompt = prompt['box']
            else:
                box_prompt = prompt
            
            if not isinstance(box_prompt, torch.Tensor):
                box_prompt = torch.tensor(box_prompt, dtype=torch.float32)

            box_coords = box_prompt.reshape(1, 2, 2)  
            box_labels = torch.tensor([2, 3], dtype=torch.int32)
            box_labels = box_labels.reshape(1, 2)
            
            points = box_coords.to(self.device)
            labels = box_labels.to(self.device)
            
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}. Supported types are 'point', 'points', and 'box'")

        return {"point_coords": points, "point_labels": labels}

    def _detach_pos_enc(self, maskmem_pos_enc):
        if maskmem_pos_enc is None:
            return None
        elif isinstance(maskmem_pos_enc, list):
            return [x.detach() if x is not None else None for x in maskmem_pos_enc]
        else:
            return maskmem_pos_enc.detach()

    def _run_single_frame_adapt(
        self,
        frame_idx,
        is_init_cond_frame,
        weak_backbone_out,
        strong_backbone_out,
        full_point_inputs,
        subsampled_point_inputs,
        output_dict,
        num_frames,
        run_mem_encoder=True,
        reverse=False,
        prev_sam_mask_logits=None,
        weak_backbone_out_orig=None,
    ):
        """
        forward, calculate loss and backward
        """
        
        
        assert full_point_inputs is not None or subsampled_point_inputs is not None
        
        (
            _,
            weak_current_vision_feats,
            weak_current_vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(weak_backbone_out)
        
        (
            _,
            strong_current_vision_feats,
            strong_current_vision_pos_embeds,
            _,
        ) = self._prepare_backbone_features(strong_backbone_out)
        
        
        weak_full_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=weak_current_vision_feats,
            current_vision_pos_embeds=weak_current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=full_point_inputs,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        
        
        # 3. get weak_subsampled prediction
        weak_subsampled_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=weak_current_vision_feats,
            current_vision_pos_embeds=weak_current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=subsampled_point_inputs,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        
        
        # 4. get strong_full prediction
        strong_full_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=strong_current_vision_feats,
            current_vision_pos_embeds=strong_current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=full_point_inputs,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        
        # 5. Build a detached pseudo target and optional confidence weights.
        pseudo_label_logits = weak_full_out["pred_masks"].detach()
        
        # Apply fill hole processing if enabled
        if self.adapt_fillhole > 0:
            pseudo_label_logits = fill_holes_in_mask_scores(pseudo_label_logits, self.adapt_fillhole)
        
        use_legacy_pseudo_loss = (
            self.pseudo_target_mode == "hard"
            and float(self.pseudo_temperature) == 1.0
            and float(self.pseudo_confidence_threshold) == 0.0
            and float(self.pseudo_confidence_power) == 0.0
        )
        if use_legacy_pseudo_loss:
            pseudo_prob = torch.sigmoid(pseudo_label_logits)
            pseudo_label = (pseudo_prob > 0.5).float()
            pseudo_pixel_weight = None
        else:
            pseudo_label, pseudo_pixel_weight = build_confidence_aware_pseudo_target(
                pseudo_label_logits,
                target_mode=self.pseudo_target_mode,
                temperature=self.pseudo_temperature,
                confidence_threshold=self.pseudo_confidence_threshold,
                confidence_power=self.pseudo_confidence_power,
                min_keep_ratio=self.pseudo_min_keep_ratio,
            )
        
        # 6. calculate consistency losses using Focal Loss + Dice Loss
        # Consistency loss between weak_subsampled and pseudo label
        weak_subsampled_pred = weak_subsampled_out["pred_masks"]
        if use_legacy_pseudo_loss:
            consistency_loss_weak = combined_focal_dice_loss(
                weak_subsampled_pred, pseudo_label,
                focal_alpha=0.25, focal_gamma=2.0,
                focal_weight=self.mask_focal_loss_weight,
                dice_weight=self.mask_dice_loss_weight,
                bce_weight=self.mask_bce_loss_weight,
            )
        else:
            consistency_loss_weak = confidence_weighted_focal_dice_loss(
                weak_subsampled_pred, pseudo_label, pseudo_pixel_weight,
                focal_alpha=0.25, focal_gamma=2.0,
                focal_weight=self.mask_focal_loss_weight,
                dice_weight=self.mask_dice_loss_weight,
                bce_weight=self.mask_bce_loss_weight,
            )
        
        # Consistency loss between strong_full and pseudo label
        strong_full_pred = strong_full_out["pred_masks"]  
        if use_legacy_pseudo_loss:
            consistency_loss_strong = combined_focal_dice_loss(
                strong_full_pred, pseudo_label,
                focal_alpha=0.25, focal_gamma=2.0,
                focal_weight=self.mask_focal_loss_weight,
                dice_weight=self.mask_dice_loss_weight,
                bce_weight=self.mask_bce_loss_weight,
            )
        else:
            consistency_loss_strong = confidence_weighted_focal_dice_loss(
                strong_full_pred, pseudo_label, pseudo_pixel_weight,
                focal_alpha=0.25, focal_gamma=2.0,
                focal_weight=self.mask_focal_loss_weight,
                dice_weight=self.mask_dice_loss_weight,
                bce_weight=self.mask_bce_loss_weight,
            )
        
        # Calculate feature consistency loss between weak and strong backbone features if enabled.
        # The default mode keeps the historical flattened cosine behavior.
        cos_similarity_loss = 0.0
        if self.gamma > 0.0:
            cos_similarity_loss = compute_feature_consistency_loss(
                y=weak_backbone_out["backbone_fpn"],
                x=strong_backbone_out["backbone_fpn"],
                balance_factor=self.gamma,
                mode=self.feature_loss_mode,
            )
        
        weighted_consistency_loss_weak = self.weak_mask_loss_weight * consistency_loss_weak
        weighted_consistency_loss_strong = self.strong_mask_loss_weight * consistency_loss_strong
        if isinstance(cos_similarity_loss, torch.Tensor):
            weighted_cos_similarity_loss = cos_similarity_loss
        else:
            weighted_cos_similarity_loss = consistency_loss_weak.new_tensor(float(cos_similarity_loss))

        # Total loss
        total_loss = (
            weighted_consistency_loss_weak
            + weighted_consistency_loss_strong
            + weighted_cos_similarity_loss
        )
        
        # 7. backward and optimizer step
        total_loss.backward()
        
        if hasattr(self, 'grad_clip_norm') and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm)
        
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized. Please call initialize_optimizer() after moving model to device.")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        
        
        # 8. prepare compact output for memory storage
        
        pred_masks_gpu = weak_full_out["pred_masks"]
        storage_device = getattr(self, '_current_storage_device', pred_masks_gpu.device)
        
        maskmem_features = weak_full_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        
        maskmem_pos_enc = weak_full_out.get("maskmem_pos_enc", None)
        if maskmem_pos_enc is not None:
            if isinstance(maskmem_pos_enc, list):
                maskmem_pos_enc = [x.to(storage_device, non_blocking=True) if x is not None else None for x in maskmem_pos_enc]
            else:
                maskmem_pos_enc = maskmem_pos_enc.to(storage_device, non_blocking=True)
        
        obj_ptr = weak_full_out["obj_ptr"]
        object_score_logits = weak_full_out.get("object_score_logits", None)
        
        compact_current_out = {
            "maskmem_features": maskmem_features.detach() if maskmem_features is not None else None,
            "maskmem_pos_enc": self._detach_pos_enc(maskmem_pos_enc),
            "pred_masks": pred_masks.detach(),
            "obj_ptr": obj_ptr.detach(),
            "object_score_logits": object_score_logits.detach() if object_score_logits is not None else None,
        }
        
        del weak_full_out, weak_subsampled_out, strong_full_out
        
        
        return compact_current_out, total_loss.item()

    def init_state(
        self,
        images,
        video_height,
        video_width,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device
        inference_state = {}
        inference_state["images"] = images.to(device=compute_device, dtype=torch.float32)
        inference_state["num_frames"] = len(images)
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}

        return inference_state

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()

    def reset_state(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
    ):
        batch_size = len(inference_state["obj_idx_to_id"])
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.image_size // 4
            consolidated_mask_key = "pred_masks"

        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: torch.full(
                size=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.hidden_dim),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                fill_value=10.0,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
        }
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(
                            inference_state, frame_idx
                        )
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = empty_mask_ptr
                continue
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]
            consolidated_out["object_score_logits"][obj_idx : obj_idx + 1] = out[
                "object_score_logits"
            ]

        if run_mem_encoder:
            device = inference_state["device"]
            high_res_masks = torch.nn.functional.interpolate(
                consolidated_out["pred_masks"].to(device, non_blocking=True),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.non_overlap_masks_for_mem_enc:
                high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                inference_state=inference_state,
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                object_score_logits=consolidated_out["object_score_logits"],
                is_mask_from_pts=True,
            )
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc

        return consolidated_out

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        # if self.non_overlap_masks:
        #     video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks


    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        image, backbone_out = inference_state["cached_features"].get(frame_idx, (None, None))
        
        if backbone_out is None:
            # Select a frame from the volume uploaded during state initialization.
            image = inference_state["images"][frame_idx].unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
            
            backbone_out = self.forward_image(image)
            
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        expanded_image = image.expand(batch_size, -1, -1, -1)  # (1,C,H,W) -> (batch_size,C,H,W)
        
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        
        features = (expanded_image,) + features
        
        return features

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc


    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        assert point_inputs is None or mask_inputs is None

        for i in range(len(current_vision_feats)):
            current_vision_feats[i] = current_vision_feats[i].to(self.device, non_blocking=True)
            current_vision_pos_embeds[i] = current_vision_pos_embeds[i].to(self.device, non_blocking=True)
        
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        storage_device = inference_state["storage_device"]
        
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        
        pred_masks_gpu = current_out["pred_masks"]
        
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        
        return compact_current_out, pred_masks_gpu

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # This is a new object id not sent to the server before. We only allow adding
        # new objects *before* the tracking starts.
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."\
            )

    def concat_points(self, old_point_inputs, new_points, new_labels):
        """Add new points and labels to previous point inputs (add at the end)."""
        if old_point_inputs is None:
            points, labels = new_points, new_labels
        else:
            points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
            labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

        return {"point_coords": points, "point_labels": labels}

    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        
        if points.dim() == 2:
            points = points.unsqueeze(0)  # (N, 2) -> (1, N, 2)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # (N,) -> (1, N)

        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            
            box_coords = box.reshape(1, 2, 2)  
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        
        points = points * self.image_size
        
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        
        point_inputs = self.concat_points(point_inputs, points, labels)
        
        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        prev_sam_mask_logits = None
        
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            device = inference_state["device"]
            prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        obj_ids = inference_state["obj_ids"]
        
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        
        return frame_idx, obj_ids, video_res_masks

    def propagate_in_video_preflight(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Tracking has started and we don't allow adding new objects until session is reset.
        inference_state["tracking_has_started"] = True
        batch_size = len(inference_state["obj_idx_to_id"])

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        # "consolidated_frame_inds" contains indices of those frames where consolidated
        # temporary outputs have been added (either in this call or any previous calls
        # to `propagate_in_video_preflight`).
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in [False, True]:
            # Separately consolidate conditioning and non-conditioning temp outputs
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # Find all the frames that contain temporary outputs for any objects
            # (these should be the frames that have just received clicks for mask inputs
            # via `add_new_points_or_box` or `add_new_mask`)
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # consolidate the temporary output across all objects on this frame
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True
                )
                # merge them into "output_dict" and also create per-object slices
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, frame_idx, consolidated_out, storage_key
                )
                # clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                #     self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                # )
                # if clear_non_cond_mem:
                #     # clear non-conditioning memory of the surrounding frames
                #     self._clear_non_cond_mem_around_input(inference_state, frame_idx)

            # clear temporary outputs in `temp_output_dict_per_obj`
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # edge case: if an output is added to "cond_frame_outputs", we remove any prior
        # output on the same frame in "non_cond_frame_outputs"
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # Make sure that the frame indices in "consolidated_frame_inds" are exactly those frames
        # with either points or mask inputs (which should be true under a correct workflow).
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

    def _add_output_per_object(
        self, inference_state, frame_idx, current_out, storage_key
    ):
        """
        Split a multi-object output into per-object output slices and add them into
        `output_dict_per_obj`. The resulting slices share the same tensor storage.
        """
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        output_dict_per_obj = inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)

        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = len(inference_state["obj_idx_to_id"])
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = False

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in processing_order:
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
                output_dict[storage_key][frame_idx] = current_out
            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            yield frame_idx, obj_ids, video_res_masks

    def infer_after_adapt(self, inputs, prompt_args, original_image_encoder_cache=None, num_keyslices=0):
        assert 'imgs' in inputs and 'masks' in inputs and 'keyframes' in inputs and 'class_ids' in inputs
        assert len(inputs['class_ids']) > 0, "No valid class IDs provided"
        
        self.eval()
        
        storage_device = self.device
        images_on_device = inputs['imgs'].to(device=self.device, dtype=torch.float32)
        if original_image_encoder_cache is not None:
            image_encoder_cache = original_image_encoder_cache
        else:
            image_encoder_cache = self._precompute_image_encoder_cache(
                images_on_device, storage_device, self.device, batch_size=self.batch_size
            )
        mask_example = next(iter(inputs['masks'].values()))
        original_inference_state = self.init_state(images_on_device, mask_example.shape[1], mask_example.shape[2])
        adapted_inference_state = self.init_state(images_on_device, mask_example.shape[1], mask_example.shape[2])

        with torch.no_grad():
            for frame_idx, backbone_out in image_encoder_cache.items():
                image = images_on_device[frame_idx].unsqueeze(0)
                backbone_out, backbone_out_adapt = self._get_image_features(images_on_device, frame_idx, image_encoder_cache)
                original_inference_state["cached_features"][frame_idx] = (image, backbone_out)
                adapted_inference_state["cached_features"][frame_idx] = (image, backbone_out_adapt)

        original_inference_features = original_inference_state["cached_features"]
        adapted_inference_features = adapted_inference_state["cached_features"]

        original_results = {}
        adapted_results = {}

        with torch.no_grad():
            for class_id in inputs['class_ids']:
                keyframe_idx = inputs['keyframes'][class_id]
                num_frames = inputs['imgs'].shape[0]
                
                start_slice = max(0, keyframe_idx - num_keyslices)
                end_slice = min(num_frames - 1, keyframe_idx + num_keyslices)
                prompt_slices = list(range(start_slice, end_slice + 1))
                
                original_mask = inputs['masks'][class_id]
                
                self.reset_state(original_inference_state)
                original_inference_state["cached_features"] = original_inference_features
                self.reset_state(adapted_inference_state)
                adapted_inference_state["cached_features"] = adapted_inference_features

                original_class_seg = torch.zeros(mask_example.shape, dtype=torch.uint8, device=self.device)
                
                for slice_idx in prompt_slices:
                    slice_mask = original_mask[slice_idx]
                    if np.sum(slice_mask) == 0:
                        continue
                    
                    prompt = self._generate_single_prompt(slice_mask, prompt_args)
                    
                    if prompt_args['prompt_type'] == 'box':
                        frame_idx, obj_ids, mask_logits = self.add_new_points_or_box(original_inference_state, slice_idx, 1, box=prompt)
                    else:
                        frame_idx, obj_ids, mask_logits = self.add_new_points_or_box(original_inference_state, slice_idx, 1, points=prompt['points'], labels=prompt['labels'])
                
                original_forward_masks = {}
                for frame_idx, obj_ids, mask_logits in self.propagate_in_video(original_inference_state, start_frame_idx=prompt_slices[0]):
                    original_class_seg[frame_idx, (mask_logits[0] > 0.0)[0]] = 1
                    original_forward_masks[frame_idx] = mask_logits[0]
                
                self.reset_state(original_inference_state)
                original_inference_state["cached_features"] = original_inference_features
                
                for slice_idx in reversed(prompt_slices):
                    slice_mask = original_mask[slice_idx]
                    if np.sum(slice_mask) == 0:
                        continue
                    
                    prompt = self._generate_single_prompt(slice_mask, prompt_args)
                    
                    if prompt_args['prompt_type'] == 'box':
                        frame_idx, obj_ids, mask_logits = self.add_new_points_or_box(original_inference_state, slice_idx, 1, box=prompt)
                    else:
                        frame_idx, obj_ids, mask_logits = self.add_new_points_or_box(original_inference_state, slice_idx, 1, points=prompt['points'], labels=prompt['labels'])
                
                original_backward_masks = {}
                for frame_idx, obj_ids, mask_logits in self.propagate_in_video(original_inference_state, start_frame_idx=prompt_slices[-1], reverse=True):
                    original_backward_masks[frame_idx] = mask_logits[0]
                
                self.reset_state(original_inference_state)
                for frame_idx in original_forward_masks.keys():
                    if frame_idx in original_backward_masks:
                        avg_mask_logits = (original_forward_masks[frame_idx] + original_backward_masks[frame_idx]) / 2.0
                        original_class_seg[frame_idx, (avg_mask_logits > 0.0)[0]] = 1
                    else:
                        original_class_seg[frame_idx, (original_forward_masks[frame_idx] > 0.0)[0]] = 1
                for frame_idx in original_backward_masks.keys():
                    if frame_idx not in original_forward_masks:
                        original_class_seg[frame_idx, (original_backward_masks[frame_idx] > 0.0)[0]] = 1

                original_class_seg = original_class_seg.cpu().numpy()
                del original_forward_masks, original_backward_masks
                original_dsc = dice_coefficient(original_mask, original_class_seg)
                original_results[class_id] = {
                    'dsc': original_dsc,
                    'predictions': original_class_seg,
                    'num_frames': num_frames
                }

                adapted_class_seg = torch.zeros(mask_example.shape, dtype=torch.uint8, device=self.device)

                for slice_idx in prompt_slices:
                    slice_mask = original_mask[slice_idx]
                    if np.sum(slice_mask) == 0:
                        continue
                    
                    prompt = self._generate_single_prompt(slice_mask, prompt_args)
                    
                    if prompt_args['prompt_type'] == 'box':
                        frame_idx, obj_ids, mask_logits = self.add_new_points_or_box(adapted_inference_state, slice_idx, 1, box=prompt)
                    else:
                        frame_idx, obj_ids, mask_logits = self.add_new_points_or_box(adapted_inference_state, slice_idx, 1, points=prompt['points'], labels=prompt['labels'])
                
                adapted_forward_masks = {}
                for frame_idx, obj_ids, mask_logits in self.propagate_in_video(adapted_inference_state, start_frame_idx=prompt_slices[0]):
                    adapted_forward_masks[frame_idx] = mask_logits[0]
                
                self.reset_state(adapted_inference_state)
                adapted_inference_state["cached_features"] = adapted_inference_features
                
                for slice_idx in reversed(prompt_slices):
                    slice_mask = original_mask[slice_idx]
                    if np.sum(slice_mask) == 0:
                        continue
                    
                    prompt = self._generate_single_prompt(slice_mask, prompt_args)
                    
                    if prompt_args['prompt_type'] == 'box':
                        frame_idx, obj_ids, mask_logits = self.add_new_points_or_box(adapted_inference_state, slice_idx, 1, box=prompt)
                    else:
                        frame_idx, obj_ids, mask_logits = self.add_new_points_or_box(adapted_inference_state, slice_idx, 1, points=prompt['points'], labels=prompt['labels'])
                
                adapted_backward_masks = {}
                for frame_idx, obj_ids, mask_logits in self.propagate_in_video(adapted_inference_state, start_frame_idx=prompt_slices[-1], reverse=True):
                    adapted_backward_masks[frame_idx] = mask_logits[0]

                
                self.reset_state(adapted_inference_state)

                for frame_idx in adapted_forward_masks.keys():
                    if frame_idx in adapted_backward_masks:
                        avg_mask_logits = (adapted_forward_masks[frame_idx] + adapted_backward_masks[frame_idx]) / 2.0
                        adapted_class_seg[frame_idx, (avg_mask_logits > 0.0)[0]] = 1
                    else:
                        adapted_class_seg[frame_idx, (adapted_forward_masks[frame_idx] > 0.0)[0]] = 1
                for frame_idx in adapted_backward_masks.keys():
                    if frame_idx not in adapted_forward_masks:
                        adapted_class_seg[frame_idx, (adapted_backward_masks[frame_idx] > 0.0)[0]] = 1
                adapted_class_seg = adapted_class_seg.cpu().numpy()
                del adapted_forward_masks, adapted_backward_masks
                adapted_dsc = dice_coefficient(original_mask, adapted_class_seg)
                adapted_results[class_id] = {
                    'dsc': adapted_dsc,
                    'predictions': adapted_class_seg,
                    'num_frames': num_frames
                }

        
        return original_results, adapted_results
    
    def _generate_single_prompt(self, mask, prompt_args):
        if prompt_args['prompt_type'] in ['point', 'points']:
            points, labels = sample_points_from_mask(mask, num_points=prompt_args.get('num_points', 1), strategy=prompt_args.get('points_type', 'center'))
            return {
                'points': points,
                'labels': labels
            }
        
        elif prompt_args['prompt_type'] == 'box':
            box = generate_bbox(mask)
            return box
        
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_args['prompt_type']}. Supported types are 'point', 'points', and 'box'")

    def forward(self, *args, **kwargs):
        """
        Empty forward function. Use adapt() method for training with pseudo-labeling and online optimization.
        
        This method is kept empty to maintain compatibility with PyTorch nn.Module interface.
        For actual functionality, use the adapt() method which provides:
        - Memory-efficient frame-by-frame processing
        - Pseudo-labeling strategy using weak_full predictions
        - Online optimization with gradient updates
        - Consistency loss computation
        
        Args:
            *args: Variable arguments (not used)
            **kwargs: Keyword arguments (not used)
            
        Returns:
            None
        """
        pass

    def freeze_adapters(self):
        if self.enable_image_encoder_adapter and self.image_encoder_adapters is not None:
            for adapter in self.image_encoder_adapters:
                for param in adapter.parameters():
                    param.requires_grad = False

    def unfreeze_adapters(self):
        if self.enable_image_encoder_adapter and self.image_encoder_adapters is not None:
            for adapter in self.image_encoder_adapters:
                for param in adapter.parameters():
                    param.requires_grad = True

    def reset_adapters(self, reset_ratio=1.0):
        if not self.enable_image_encoder_adapter or self.image_encoder_adapters is None:
            return []
        
        if not hasattr(self, 'original_adapter_params') or self.original_adapter_params is None:
            return []
        
        reset_count = 0
        total_params = 0
        reset_params = []
        
        resetable_params = []
        for i, adapter in enumerate(self.image_encoder_adapters):
            if i not in self.original_adapter_params:
                continue
                
            original_params = self.original_adapter_params[i]
            
            if hasattr(adapter, 'adapter_conv') and adapter.adapter_conv is not None and 'conv_weight' in original_params:
                resetable_params.append(('conv_weight', i, adapter))
                total_params += 1
            
            if hasattr(adapter, 'adapter_conv') and adapter.adapter_conv is not None and adapter.adapter_conv.bias is not None and 'conv_bias' in original_params:
                resetable_params.append(('conv_bias', i, adapter))
                total_params += 1
        
        if reset_ratio >= 1.0:
            params_to_reset = resetable_params
        else:
            num_to_reset = max(1, int(total_params * reset_ratio))
            params_to_reset = self.reset_rng.sample(resetable_params, num_to_reset)
        
        for param_type, adapter_idx, adapter in params_to_reset:
            original_params = self.original_adapter_params[adapter_idx]
            
            if param_type == 'conv_weight':
                current_conv_weight = adapter.adapter_conv.weight
                original_conv_weight = original_params['conv_weight']
                
                if current_conv_weight.device != original_conv_weight.device:
                    original_conv_weight = original_conv_weight.to(current_conv_weight.device)
                
                current_conv_weight.data.copy_(original_conv_weight)
                reset_params.append(current_conv_weight)
                reset_count += 1
                
            elif param_type == 'conv_bias':
                current_conv_bias = adapter.adapter_conv.bias
                original_conv_bias = original_params['conv_bias']
                
                if current_conv_bias.device != original_conv_bias.device:
                    original_conv_bias = original_conv_bias.to(current_conv_bias.device)
                
                current_conv_bias.data.copy_(original_conv_bias)
                reset_params.append(current_conv_bias)
                reset_count += 1
        
        return reset_params

    def save_adapter_params_for_momentum(self):
        if not self.enable_image_encoder_adapter or self.image_encoder_adapters is None:
            self.momentum_saved_params = None
            return
        
        self.momentum_saved_params = {}
        for i, adapter in enumerate(self.image_encoder_adapters):
            self.momentum_saved_params[i] = {}
            # Store conv weights
            if hasattr(adapter, 'adapter_conv') and adapter.adapter_conv is not None and adapter.adapter_conv.weight is not None:
                self.momentum_saved_params[i]['conv_weight'] = adapter.adapter_conv.weight.data.clone().detach()
            
            # Store conv bias if exists
            if hasattr(adapter, 'adapter_conv') and adapter.adapter_conv is not None and adapter.adapter_conv.bias is not None:
                self.momentum_saved_params[i]['conv_bias'] = adapter.adapter_conv.bias.data.clone().detach()

    def apply_momentum_update(self, momentum: float):
        if not self.enable_image_encoder_adapter or self.image_encoder_adapters is None:
            return
        
        if not hasattr(self, 'momentum_saved_params') or self.momentum_saved_params is None:
            return
        
        for i, adapter in enumerate(self.image_encoder_adapters):
            if i not in self.momentum_saved_params:
                continue
            saved = self.momentum_saved_params[i]
            
            # Update conv weight
            if 'conv_weight' in saved:
                current = adapter.adapter_conv.weight.data
                saved_weight = saved['conv_weight']
                # Ensure device consistency
                if current.device != saved_weight.device:
                    saved_weight = saved_weight.to(current.device)
                # Apply momentum update in-place
                adapter.adapter_conv.weight.data.copy_(
                    momentum * saved_weight + (1 - momentum) * current
                )
            
            # Update conv bias
            if 'conv_bias' in saved:
                current = adapter.adapter_conv.bias.data
                saved_bias = saved['conv_bias']
                # Ensure device consistency
                if current.device != saved_bias.device:
                    saved_bias = saved_bias.to(current.device)
                # Apply momentum update in-place
                adapter.adapter_conv.bias.data.copy_(
                    momentum * saved_bias + (1 - momentum) * current
                )

    def reset_augmentation_handler(self):
        self.augmentation_handler.reset_state()

    def clear_optimizer_momentum_for_params(self, reset_params):
        """
        Clear optimizer momentum (state) for specific parameters.
        This is useful when only a subset of parameters are reset to avoid
        stale momentum from affecting the reset parameters.
        
        Args:
            reset_params (list): List of parameter tensors that were reset
        """
        if self.optimizer is None or not reset_params:
            return
        
        # Get parameter IDs for the reset parameters
        reset_param_ids = {id(param) for param in reset_params}
        
        # Clear momentum for reset parameters
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if id(param) in reset_param_ids:
                    self.optimizer.state[param]['exp_avg'].zero_()
                    self.optimizer.state[param]['exp_avg_sq'].zero_()
        

    def _store_original_adapter_params(self):
        """
        Store a copy of original adapter parameters after initialization.
        This copy can be used for regularization and parameter change analysis.
        """
        if not self.enable_image_encoder_adapter or self.image_encoder_adapters is None:
            self.original_adapter_params = None
            return
        
        self.original_adapter_params = {}
        for i, adapter in enumerate(self.image_encoder_adapters):
            self.original_adapter_params[i] = {}
            # Store conv weights
            if hasattr(adapter, 'adapter_conv') and adapter.adapter_conv is not None and adapter.adapter_conv.weight is not None:
                self.original_adapter_params[i]['conv_weight'] = adapter.adapter_conv.weight.data.clone().detach()
            
            # Store conv bias if exists
            if hasattr(adapter, 'adapter_conv') and adapter.adapter_conv is not None and adapter.adapter_conv.bias is not None:
                self.original_adapter_params[i]['conv_bias'] = adapter.adapter_conv.bias.data.clone().detach()
            
            # Store scale parameters (if they exist)
            if hasattr(adapter, 'adapter_scale'):
                self.original_adapter_params[i]['scale'] = adapter.adapter_scale.data.clone().detach()

    def compute_adapter_l1_norm(self):
        if not self.enable_image_encoder_adapter or self.image_encoder_adapters is None or self.original_adapter_params is None:
            return {
                'total_l1': 0.0,
                'per_adapter_l1': []
            }
        
        total_l1 = 0.0
        per_adapter_l1 = []
        
        for i, adapter in enumerate(self.image_encoder_adapters):
            if i not in self.original_adapter_params:
                continue
                
            adapter_l1 = 0.0
            original_params = self.original_adapter_params[i]
            
            # Compute L1 norm for conv weights
            if hasattr(adapter, 'adapter_conv') and 'conv_weight' in original_params:
                current_conv_weight = adapter.adapter_conv.weight.data
                original_conv_weight = original_params['conv_weight']
                
                # Ensure device consistency
                if current_conv_weight.device != original_conv_weight.device:
                    original_conv_weight = original_conv_weight.to(current_conv_weight.device)
                
                conv_diff = torch.abs(current_conv_weight - original_conv_weight).sum()
                adapter_l1 += conv_diff.item()

            
            per_adapter_l1.append(adapter_l1)
            total_l1 += adapter_l1
        
        return {
            'total_l1': total_l1,
            'per_adapter_l1': per_adapter_l1
        }

    def get_adapter_parameter_groups(self):
        if not self.enable_image_encoder_adapter or self.image_encoder_adapters is None:
            return {'conv_params': [], 'scale_params': [], 'all_params': []}
        
        conv_params = []
        scale_params = []
        all_params = []
        
        for adapter in self.image_encoder_adapters:
            # Get convolution parameters
            if hasattr(adapter, 'adapter_conv'):
                conv_params.extend(list(adapter.adapter_conv.parameters()))
            
            # Get scale parameters
            if hasattr(adapter, 'adapter_scale'):
                scale_params.append(adapter.adapter_scale)
            
            # Get all parameters
            all_params.extend(list(adapter.parameters()))
        
        return {
            'conv_params': conv_params,
            'scale_params': scale_params, 
            'all_params': all_params
        }

    def _build_sam_heads(self):
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(self, backbone_features, point_inputs=None, mask_inputs=None, high_res_features=None, multimask_output=False):
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Args:
            backbone_features: Backbone feature maps
            high_res_features: High resolution feature maps
            mask_inputs: Input mask prompts
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch: torch.Tensor):
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        return backbone_out

    def forward_image_with_adapters(self, img_batch: torch.Tensor):
        # Get backbone features directly from image encoder (similar to forward_image)
        backbone_out = self.image_encoder(img_batch)
        
        # Create adapted version of backbone output
        backbone_adapter_out = {
            "backbone_fpn": [],
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy()  # positional encodings remain unchanged
        }
        
        # Ensure position encodings are on the correct device (same as feature maps)
        if len(backbone_out["backbone_fpn"]) > 0:
            target_device = backbone_out["backbone_fpn"][0].device
            backbone_adapter_out["vision_pos_enc"] = [
                pos_enc.to(target_device) if pos_enc.device != target_device else pos_enc
                for pos_enc in backbone_adapter_out["vision_pos_enc"]
            ]
        
        # Apply adapters to each feature map
        if self.enable_image_encoder_adapter and self._adapters_initialized:
            for i, feature_map in enumerate(backbone_out["backbone_fpn"]):
                if i < len(self.image_encoder_adapters):
                    # Apply adapter to the feature map
                    adapted_feature_map = self.image_encoder_adapters[i](feature_map)
                    backbone_adapter_out["backbone_fpn"].append(adapted_feature_map)
                else:
                    # If no adapter for this feature map, use original
                    backbone_adapter_out["backbone_fpn"].append(feature_map)
        else:
            # If adapters are disabled, use original features
            backbone_adapter_out["backbone_fpn"] = backbone_out["backbone_fpn"].copy()
        
        # Apply use_high_res_features_in_sam logic to both backbone_out and backbone_adapter_out
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
            
            backbone_adapter_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_adapter_out["backbone_fpn"][0])
            backbone_adapter_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_adapter_out["backbone_fpn"][1])
        
        return backbone_out, backbone_adapter_out

    def forward_adapters(self, backbone_out):
        # Create a copy of backbone_out to avoid modifying the cached input
        backbone_out_copy = {
            "backbone_fpn": [feat.detach().clone() for feat in backbone_out["backbone_fpn"]],
            "vision_pos_enc": [pos_enc.detach().clone() for pos_enc in backbone_out["vision_pos_enc"]]
        }

        backbone_features_with_grad = []
        for feat in backbone_out["backbone_fpn"]:
            feat_with_grad = feat.detach().clone().requires_grad_(True)
            backbone_features_with_grad.append(feat_with_grad)
        
        backbone_adapter_out = {
            "backbone_fpn": [],
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy()  # positional encodings remain unchanged
        }
        
        # Ensure position encodings are on the correct device (same as feature maps)
        if len(backbone_out["backbone_fpn"]) > 0:
            target_device = backbone_out["backbone_fpn"][0].device
            backbone_adapter_out["vision_pos_enc"] = [
                pos_enc.to(target_device) if pos_enc.device != target_device else pos_enc
                for pos_enc in backbone_adapter_out["vision_pos_enc"]
            ]
            backbone_out_copy["vision_pos_enc"] = [
                pos_enc.to(target_device) if pos_enc.device != target_device else pos_enc
                for pos_enc in backbone_out_copy["vision_pos_enc"]
            ]
        
        # Apply adapters to each feature map
        if self.enable_image_encoder_adapter and self._adapters_initialized:
            for i, feature_map in enumerate(backbone_features_with_grad):
                if i < len(self.image_encoder_adapters):
                    # Apply adapter to the feature map with gradients
                    adapted_feature_map = self.image_encoder_adapters[i](feature_map)
                    backbone_adapter_out["backbone_fpn"].append(adapted_feature_map)
                else:
                    # If no adapter for this feature map, use original with gradients
                    backbone_adapter_out["backbone_fpn"].append(feature_map)
        else:
            # If adapters are disabled, use original features with gradients
            backbone_adapter_out["backbone_fpn"] = backbone_features_with_grad
        
        # Apply use_high_res_features_in_sam logic to both outputs
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            
            # For backbone_out_copy (original features)
            backbone_out_copy["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out_copy["backbone_fpn"][0])
            backbone_out_copy["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out_copy["backbone_fpn"][1])
            
            # For backbone_adapter_out (adapter processed features)
            backbone_adapter_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_adapter_out["backbone_fpn"][0])
            backbone_adapter_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_adapter_out["backbone_fpn"][1])
        
        return backbone_out_copy, backbone_adapter_out

    def _initialize_adapters(self, backbone_fpn=None):
        """
        Args:
            backbone_fpn: List of feature maps from the backbone, each of shape [B, C, H, W]. If None, will create a dummy forward pass to get features.
        """
        if not self.enable_image_encoder_adapter or self._adapters_initialized:
            return
        
        # If no backbone features provided, run a dummy forward pass to get them
        if backbone_fpn is None:
            # Temporarily set device to CPU for initialization (to avoid GPU memory issues)
            original_device = next(self.parameters()).device
            temp_device = torch.device('cpu')
            
            # Create a dummy input to initialize adapters
            dummy_img = torch.randn(1, 3, self.image_size, self.image_size, device=temp_device)
            
            # Temporarily move image_encoder to CPU for dummy forward pass
            self.image_encoder = self.image_encoder.to(temp_device)
            
            with torch.no_grad():
                # Run forward pass to get backbone features
                backbone_out = self.image_encoder(dummy_img)
                backbone_fpn = backbone_out["backbone_fpn"]
            
            # Move image_encoder back to original device
            self.image_encoder = self.image_encoder.to(original_device)
            
            print_construction_message = True
        else:
            print_construction_message = False
            
        # Initialize adapters based on backbone features
        if self.image_encoder_adapters is None:
            self.image_encoder_adapters = torch.nn.ModuleList()
            
            for i, feature_map in enumerate(backbone_fpn):
                # Get the actual channel dimension from the feature map
                channels = feature_map.shape[1]  # [B, C, H, W] -> C
                
                adapter = ImageEncoderAdapter(
                    channels=channels,
                    enable_adapter=self.enable_image_encoder_adapter
                )
                
                # Move adapter to the same device as the feature map
                adapter = adapter.to(feature_map.device)
                self.image_encoder_adapters.append(adapter)
            
            # If we did dummy forward pass during construction, move adapters to original device
            if print_construction_message:
                original_device = next(self.parameters()).device
                for adapter in self.image_encoder_adapters:
                    adapter.to(original_device)
        
        self._adapters_initialized = True
        
        self._store_original_adapter_params()

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        # Ensure all tensors are on the same device as feature_maps
        device = feature_maps[0].device
        vision_pos_embeds = [x.to(device) for x in vision_pos_embeds]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, output_dict, num_frames, track_in_reverse=False):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                # Convert from bfloat16 back to float32 to match model weights dtype
                if feats.dtype == torch.bfloat16:
                    feats = feats.to(torch.float32)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(self, current_vision_feats, feat_sizes, pred_masks_high_res, object_score_logits, is_mask_from_pts):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

    def _track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs, output_dict, num_frames, track_in_reverse, prev_sam_mask_logits):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(self, current_vision_feats, feat_sizes, point_inputs, run_mem_encoder, high_res_masks, object_score_logits, current_out):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs, output_dict, num_frames, track_in_reverse=False, run_mem_encoder=True, prev_sam_mask_logits=None):
        """
        Args:
            frame_idx: Current frame index
            is_init_cond_frame: Whether this is an initial conditioning frame
            current_vision_feats: Current frame visual features
            current_vision_pos_embeds: Current frame positional embeddings
            feat_sizes: Feature map sizes
            point_inputs: Point input prompts
            mask_inputs: Mask input prompts
            output_dict: Output dictionary
            num_frames: Total number of frames
            track_in_reverse: Tracking in reverse time order (for demo usage)
            run_mem_encoder: Whether to run the memory encoder on the predicted masks. Sometimes we might want to skip the memory encoder with `run_mem_encoder=False`. For example, in demo we might call `track_step` multiple times for each user click, and only encode the memory when the user finalizes their clicks. And in ablation settings like SAM training on static images, we don't need the memory encoder.
            prev_sam_mask_logits: The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        """
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks


def build_sam2_adapt_model(model_cfg, checkpoint_path, device, image_size=512, lr=1e-4, decoder_lr=0.0, weight_decay=1e-4, seed=2025, 
                          offload_state_to_cpu=False, adapt_fillhole=0, gamma=0.0,
                          feature_loss_mode='legacy_cosine',
                          weak_mask_loss_weight=1.0, strong_mask_loss_weight=1.0,
                          mask_focal_loss_weight=1.0, mask_dice_loss_weight=1.0,
                          mask_bce_loss_weight=0.0,
                          pseudo_target_mode='hard', pseudo_temperature=1.0,
                          pseudo_confidence_threshold=0.0, pseudo_confidence_power=0.0,
                          pseudo_min_keep_ratio=0.01,
                          batch_size=4, eval_fillhole=0, 
                          augmentation_type='weak', augmentation_args=None):
    # Configure hydra overrides to use SAM2Adapt instead of SAM2Base
    hydra_overrides = [
        "++model._target_=models.medsam2_spda_model.SAM2Adapt",
        # Match build_sam2_video_predictor_npz_float32 inference behavior.
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        "++model.binarize_mask_from_pts_for_mem_enc=true",
        # Add optimizer parameters as model arguments
        f"++model.optimizer_lr={lr}",
        f"++model.decoder_lr={decoder_lr}",
        f"++model.optimizer_weight_decay={weight_decay}",
        "++model.optimizer_betas=[0.9,0.999]",
        f"++model.optimizer_eps=1e-8",
        # Enable adapter features
        "++model.enable_image_encoder_adapter=True",
        # Set image size
        f"++model.image_size={image_size}",
        # Set seed
        f"++model.seed={seed}",
        # Add adapt parameters
        f"++model.adapt_device='{str(device)}'",
        f"++model.offload_state_to_cpu={offload_state_to_cpu}",
        f"++model.adapt_fillhole={adapt_fillhole}",
        f"++model.gamma={gamma}",
        f"++model.feature_loss_mode={feature_loss_mode}",
        f"++model.weak_mask_loss_weight={weak_mask_loss_weight}",
        f"++model.strong_mask_loss_weight={strong_mask_loss_weight}",
        f"++model.mask_focal_loss_weight={mask_focal_loss_weight}",
        f"++model.mask_dice_loss_weight={mask_dice_loss_weight}",
        f"++model.mask_bce_loss_weight={mask_bce_loss_weight}",
        f"++model.pseudo_target_mode={pseudo_target_mode}",
        f"++model.pseudo_temperature={pseudo_temperature}",
        f"++model.pseudo_confidence_threshold={pseudo_confidence_threshold}",
        f"++model.pseudo_confidence_power={pseudo_confidence_power}",
        f"++model.pseudo_min_keep_ratio={pseudo_min_keep_ratio}",
        f"++model.batch_size={batch_size}",
        f"++model.eval_fillhole={eval_fillhole}",
        # Add augmentation configuration
        f"++model.augmentation_type={augmentation_type}",
    ]
    
    # Read config and instantiate model
    cfg = compose(config_name=model_cfg, overrides=hydra_overrides)
    
    # Add augmentation_args to config if provided (can't be easily passed via overrides)
    # Need to disable struct mode temporarily to add new fields
    if augmentation_args is not None:
        OmegaConf.set_struct(cfg.model, False)
        cfg.model.augmentation_args = augmentation_args
        OmegaConf.set_struct(cfg.model, True)
    
    OmegaConf.resolve(cfg)
    
    model = instantiate(cfg.model, _recursive_=True)
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
        
        # Load weights with strict=False to allow for new adapter parameters
        model.load_state_dict(sd, strict=False)
    
    model = model.to(device)
    model.eval()  # Set to eval mode initially
    model.initialize_optimizer()
    
    return model
