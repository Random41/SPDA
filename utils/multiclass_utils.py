import numpy as np


def decompose_multiclass_gts(gts):
    unique_classes = np.unique(gts)
    foreground_classes = unique_classes[unique_classes > 0]

    binary_gts = {}
    for class_id in foreground_classes:
        binary_gt = (gts == class_id).astype(np.uint8)
        binary_gts[int(class_id)] = binary_gt

    return binary_gts


def find_key_slice_for_class(binary_gt):
    if binary_gt.ndim == 3:
        slice_pixel_counts = np.sum(binary_gt > 0, axis=(1, 2))
        key_slice_idx = np.argmax(slice_pixel_counts)
        return key_slice_idx
    else:
        return 0


def merge_class_segmentations(class_segs):
    if not class_segs:
        return None

    first_seg = list(class_segs.values())[0]

    max_class_id = max(class_segs.keys())
    if max_class_id > 255:
        merged_seg = np.zeros_like(first_seg, dtype=np.uint16)
    else:
        merged_seg = np.zeros_like(first_seg, dtype=np.uint8)

    for class_id in sorted(class_segs.keys()):
        class_seg = class_segs[class_id]
        merged_seg[class_seg > 0] = class_id

    return merged_seg


def calculate_multiclass_metrics(gts, segs, class_results):
    overall_gt = (gts > 0).astype(np.uint8)
    overall_seg = (segs > 0).astype(np.uint8)

    from .evaluation_metrics import dice_coefficient
    overall_dsc = dice_coefficient(overall_gt, overall_seg)

    class_dscs = [result['dsc'] for result in class_results.values()]
    mean_class_dsc = np.mean(class_dscs) if class_dscs else 0.0

    unique_gt_classes = set(np.unique(gts)[np.unique(gts) > 0])
    detected_classes = set(class_results.keys())
    class_coverage = len(detected_classes) / len(unique_gt_classes) if unique_gt_classes else 1.0

    metrics = {
        'overall_dsc': float(overall_dsc),
        'mean_dsc': float(mean_class_dsc),
        'class_coverage': float(class_coverage),
        'num_gt_classes': len(unique_gt_classes),
        'num_detected_classes': len(detected_classes),
        'class_dscs': class_dscs
    }

    return metrics
