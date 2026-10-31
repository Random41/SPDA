import torch
import numpy as np
from typing import Dict, List, Tuple, Union
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
from bbox_utils import generate_bbox, mask2bbox
from point_sampling import sample_points_from_mask


def generate_full_prompt(ground_truth_mask: torch.Tensor, 
                        prompt_args: Dict,
                        device: torch.device) -> Dict:
    prompt_type = prompt_args['prompt_type']

    if isinstance(ground_truth_mask, torch.Tensor):
        mask_np = ground_truth_mask.cpu().numpy().astype(np.uint8)
    else:
        mask_np = ground_truth_mask.astype(np.uint8)

    if prompt_type in ['point', 'points']:
        num_points = prompt_args.get('num_points', 2)
        points_type = prompt_args.get('points_type', 'center')
        points, labels = sample_points_from_mask(mask_np, num_points=num_points, strategy=points_type, max_shift=0)
        return {
            'points': torch.tensor(points, dtype=torch.float32, device=device),
            'labels': torch.tensor(labels, dtype=torch.int32, device=device)
        }

    elif prompt_type == 'box':
        bbox = generate_bbox(mask_np, max_shift=0)
        return {
            'box': torch.tensor(bbox, dtype=torch.float32, device=device)
        }

    elif prompt_type == 'mask':
        raise NotImplementedError("Direct mask prompts are not implemented in this function.")

    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")


def generate_subsampled_prompt(full_prompt: Dict, 
                              subsampled_prompt_args: Dict,
                              ground_truth_mask: torch.Tensor,
                              device: torch.device) -> Dict:
    prompt_type = subsampled_prompt_args.get('prompt_type', 'point')

    if prompt_type == 'box':
        box_expand_ratio = subsampled_prompt_args.get('box_expand_ratio', 0.1)
        H, W = ground_truth_mask.shape

        if 'box' in full_prompt:
            box = full_prompt['box'].cpu().numpy()
            x_min, y_min, x_max, y_max = box

            width = x_max - x_min
            height = y_max - y_min

            new_x_min = x_min - width * box_expand_ratio
            new_x_max = x_max + width * box_expand_ratio
            new_y_min = y_min - height * box_expand_ratio
            new_y_max = y_max + height * box_expand_ratio

            new_x_min = max(0, new_x_min)
            new_y_min = max(0, new_y_min)
            new_x_max = min(W, new_x_max)
            new_y_max = min(H, new_y_max)

            return {
                'box': torch.tensor([new_x_min, new_y_min, new_x_max, new_y_max], 
                                   dtype=torch.float32, device=device)
            }

        elif 'mask' in full_prompt:
            mask_np = full_prompt['mask'].cpu().numpy().astype(np.uint8)
            bbox = mask2bbox(mask_np)
            x_min, y_min, x_max, y_max = bbox

            width = x_max - x_min
            height = y_max - y_min

            new_x_min = x_min - width * box_expand_ratio
            new_x_max = x_max + width * box_expand_ratio
            new_y_min = y_min - height * box_expand_ratio
            new_y_max = y_max + height * box_expand_ratio

            new_x_min = max(0, new_x_min)
            new_y_min = max(0, new_y_min)
            new_x_max = min(W, new_x_max)
            new_y_max = min(H, new_y_max)

            return {
                'box': torch.tensor([new_x_min, new_y_min, new_x_max, new_y_max], 
                                   dtype=torch.float32, device=device)
            }

        elif 'points' in full_prompt:
            raise NotImplementedError(
                "Cannot generate box-based subsampled prompt from point-based full prompt. "
                "It is difficult to generate a semantically meaningful box from points."
            )
        else:
            raise ValueError("Full prompt must contain 'points', 'box', or 'mask'")

    elif prompt_type == 'point':
        num_points = subsampled_prompt_args['num_points']
        points_type = subsampled_prompt_args.get('points_type', 'random')

        if num_points <= 0:
            H, W = ground_truth_mask.shape
            return {
                'points': torch.tensor([[W//2, H//2]], dtype=torch.float32, device=device),
                'labels': torch.tensor([1], dtype=torch.int32, device=device)
            }

        if 'mask' in full_prompt:
            mask_np = full_prompt['mask'].cpu().numpy().astype(np.uint8)
            points, labels = sample_points_from_mask(
                mask_np, num_points=num_points, strategy=points_type
            )
            return {
                'points': torch.tensor(points, dtype=torch.float32, device=device),
                'labels': torch.tensor(labels, dtype=torch.int32, device=device)
            }

        elif 'box' in full_prompt:
            box = full_prompt['box'].cpu().numpy()
            x_min, y_min, x_max, y_max = box

            points = []
            labels = []

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            points.append([center_x, center_y])
            labels.append(1)

            for _ in range(num_points - 1):
                rand_x = np.random.uniform(x_min, x_max)
                rand_y = np.random.uniform(y_min, y_max)
                points.append([rand_x, rand_y])
                labels.append(1)

            return {
                'points': torch.tensor(points[:num_points], dtype=torch.float32, device=device),
                'labels': torch.tensor(labels[:num_points], dtype=torch.int32, device=device)
            }

        elif 'points' in full_prompt:
            coords = full_prompt['points']
            point_labels = full_prompt['labels']
            coords_np = coords.cpu().numpy()
            labels_np = point_labels.cpu().numpy()

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
            raise ValueError("Full prompt must contain 'points', 'box', or 'mask'")

    else:
        raise ValueError(f"Unknown subsampled prompt_type: {prompt_type}. Must be 'point' or 'box'.")


def generate_prompts_for_masks(masks: Dict[int, torch.Tensor], 
                              keyframes: Dict[int, int], 
                              class_ids: List[int], 
                              full_prompt_args: Dict, 
                              subsampled_prompt_args: Dict,
                              device: torch.device) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    full_prompts = {}
    subsampled_prompts = {}

    for class_id in class_ids:
        keyframe_idx = keyframes[class_id]
        class_masks = masks[class_id]

        keyframe_mask = class_masks[keyframe_idx]

        full_prompt = generate_full_prompt(
            ground_truth_mask=keyframe_mask,
            prompt_args=full_prompt_args,
            device=device
        )

        subsampled_prompt = generate_subsampled_prompt(
            full_prompt=full_prompt,
            subsampled_prompt_args=subsampled_prompt_args,
            ground_truth_mask=keyframe_mask,
            device=device
        )

        full_prompts[class_id] = full_prompt
        subsampled_prompts[class_id] = subsampled_prompt

    return full_prompts, subsampled_prompts


def normalize_coordinates_to_01(prompt: Dict, height: int, width: int) -> Dict:
    normalized_prompt = prompt.copy()

    if 'points' in prompt:
        coords, labels = prompt['points']
        normalized_coords = coords.clone()
        normalized_coords[:, 0] = normalized_coords[:, 0] / width
        normalized_coords[:, 1] = normalized_coords[:, 1] / height
        normalized_prompt['points'] = (normalized_coords, labels)

    if 'box' in prompt:
        box = prompt['box'].clone()
        box[0] = box[0] / width
        box[1] = box[1] / height
        box[2] = box[2] / width
        box[3] = box[3] / height
        normalized_prompt['box'] = box


    return normalized_prompt
