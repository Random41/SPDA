import os
import argparse
import csv
from collections import defaultdict
import random
import time
import multiprocessing as mp
from multiprocessing import Queue, Process
from multiprocessing.connection import wait
from contextlib import ExitStack, nullcontext

import tqdm
import torch
import numpy as np

from sam2.build_sam import build_sam2_video_predictor_npz_float32
from utils import (
    # Common utilities
    set_seeds, build_model, build_logger,
    # Image processing
    preprocess, getLargestCC,
    # Bbox utilities
    generate_bbox,
    # Evaluation metrics
    dice_coefficient,
    # Point sampling
    sample_points_from_mask,
    # File utilities
    collect_files_by_directory,
    # Statistics
    calculate_directory_stats, calculate_hierarchical_directory_stats,
    calculate_final_metric_summary,
    export_final_metric_summary,
    log_final_metric_summary,
    # Evaluation functions
    calculate_and_display_statistics,
    # Multi-class utilities
    decompose_multiclass_gts, find_key_slice_for_class,
    merge_class_segmentations, calculate_multiclass_metrics,
)


def configure_fp32_execution():
    """
    Force full FP32 execution.
    """
    torch.set_float32_matmul_precision('highest')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        if hasattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction'):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def fp32_inference_context():
    if torch.cuda.is_available():
        return torch.autocast("cuda", enabled=False)
    return nullcontext()


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Evaluation',
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # data args
    parser.add_argument('--data_dir', type=str, required=True)

    # model args
    parser.add_argument('--model_cfg', type=str, default='configs/sam2/sam2.1_hiera_t512.yaml')
    parser.add_argument('--checkpoint', type=str, default='path to checkpoint')

    # prompt args
    parser.add_argument('--prompt_type', type=str, default='box', choices=['box', 'point'])
    parser.add_argument('--num_points', type=int, default=2)
    parser.add_argument('--point_strategy', type=str, default='random', choices=['center', 'random', 'boundary'])

    # output args
    parser.add_argument('--output_dir', type=str, default='path to output')
    parser.add_argument('--output_prediction', action='store_true', default=False)

    # other args
    parser.add_argument('--cc', action='store_true', default=False, help='Connected-component postprocessing.')
    parser.add_argument('--multi_class', action='store_true', default=True)

    # sample order
    parser.add_argument('--shuffle', action='store_true', default=False)
    
    # multi-gpu args
    parser.add_argument('--stream', action='store_true', default=False,
                       help='Sequential single-GPU evaluation.')
    parser.add_argument('--gpus', type=int, nargs='*', default=None)
    parser.add_argument('--max_workers', type=int, default=None)
    parser.add_argument('--result_timeout', type=int, default=3600, help='Worker result timeout in seconds.')
    
    # Random seed args
    parser.add_argument('--seed', type=int, default=1314)
    
    # Key slices args
    parser.add_argument('--num_keyslices', type=int, default=0)

    args = parser.parse_args()
    return args


def truncate_display_names(dir_name, filename, max_dir_len=20, max_file_len=25):
    if len(dir_name) > max_dir_len:
        truncated_dir = dir_name[:max_dir_len-3] + "..."
    else:
        truncated_dir = dir_name.ljust(max_dir_len)
    
    if len(filename) > max_file_len:
        truncated_file = "..." + filename[-(max_file_len-3):]
    else:
        truncated_file = filename.ljust(max_file_len)
    
    return truncated_dir, truncated_file


def initialize_csv_log(output_dir: str, logger=None) -> str:
    """
    Initialize CSV log file for tracking evaluation results.
    
    Args:
        output_dir: Output directory path
        logger: Logger instance
        
    Returns:
        str: Path to the CSV log file
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results.csv")
    
    # Create CSV header for evaluation results
    header = [
        'subtask', 'filename', 'num_classes', 'num_slices', 'mean_dsc', 'gpu_id'
    ]
    
    # Write header to CSV file
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    if logger:
        logger.info(f"Initialized CSV log file: {csv_path}")
    
    return csv_path


def append_to_csv_log(csv_path: str, subtask: str, filename: str, results: dict, gpu_id: int = -1, logger=None):
    row = [
        subtask, filename, results.get('num_classes', 1),
        results.get('num_slices', 1), f"{results['mean_dsc']:.6f}", gpu_id,
    ]
    import fcntl
    with open(csv_path, 'a', newline='', encoding='utf-8') as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        csv.writer(file).writerow(row)


def build_final_metric_results(file_results):
    return [
        {'directory': result['directory'], 'mean_dsc': result['mean_dsc']}
        for result in file_results
    ]


def get_available_gpus():
    if not torch.cuda.is_available():
        return []
    
    gpu_count = torch.cuda.device_count()
    available_gpus = []
    
    for i in range(gpu_count):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            test_tensor = torch.zeros(10, device=i)
            del test_tensor
            available_gpus.append(i)
    
    return available_gpus


def select_stream_gpu(args, available_gpus, logger):
    if args.max_workers is not None and args.max_workers != 1:
        raise ValueError("--stream only supports one worker; omit --max_workers or set --max_workers 1")

    if args.gpus is None:
        gpu_id = available_gpus[0]
        logger.info(f"--stream enabled without --gpus; using first available GPU {gpu_id}")
        return gpu_id

    if len(args.gpus) != 1:
        raise ValueError("--stream requires exactly one GPU, e.g. --stream --gpus 0")

    gpu_id = args.gpus[0]
    if gpu_id not in available_gpus:
        raise ValueError(f"Requested stream GPU {gpu_id} is not available. Available GPUs: {available_gpus}")
    return gpu_id


def build_file_result(rel_file_path, rel_dir, result, multi_class=True):
    """
    Build a compact JSON/statistics record without a per-sample path.
    """
    record = {
        'filename': os.path.basename(rel_file_path),
        'directory': rel_dir,
        'mean_dsc': result['mean_dsc'],
        'num_classes': result.get('num_classes', 1),
        'num_slices': result.get('num_slices', 1),
        'gpu_id': result.get('gpu_id', -1),
    }
    if multi_class:
        record['class_results'] = {
            class_id: {'mean_dsc': values['dsc']}
            for class_id, values in result['class_results'].items()
        }
    return record


def worker_process(gpu_id, task_queue, result_connection, args):
    """
    Process samples on one GPU.
    """
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    configure_fp32_execution()
    predictor = build_sam2_video_predictor_npz_float32(
        config_file=args.model_cfg,
        ckpt_path=args.checkpoint,
        device=device,
    )

    with result_connection:
        for file_path in iter(task_queue.get, None):
            result = process_single_file(file_path, predictor, args)
            validate_sample_result(file_path, result)
            result['gpu_id'] = gpu_id
            result_connection.send((file_path, result))
        result_connection.send(None)


def validate_sample_result(file_path, result):
    if not isinstance(result, dict):
        raise TypeError(f"Invalid result format for {file_path}: {type(result).__name__}")
    if 'error' in result:
        raise RuntimeError(f"Error processing {file_path}: {result['error']}")


def receive_worker_result(connections, workers, timeout):
    """
    Wait for a result.
    """
    deadline = time.monotonic() + timeout
    while True:
        for worker in workers:
            if worker.exitcode not in (None, 0):
                raise ChildProcessError(
                    f"Evaluation worker {worker.pid} exited with code {worker.exitcode}; "
                    "see its traceback above."
                )
        if not connections:
            raise EOFError("All evaluation workers finished before returning every result")
        active_workers = [worker for worker in workers if worker.exitcode is None]
        ready = wait(
            [*connections, *(worker.sentinel for worker in active_workers)],
            timeout=max(0.0, deadline - time.monotonic()),
        )
        for worker in workers:
            if worker.exitcode not in (None, 0):
                raise ChildProcessError(
                    f"Evaluation worker {worker.pid} exited with code {worker.exitcode}; "
                    "see its traceback above."
                )
        for connection in tuple(connections):
            if connection in ready:
                result = connection.recv()
                if result is None:
                    connections.remove(connection)
                else:
                    return result
        if not ready:
            raise TimeoutError(f"No evaluation result received within {timeout} seconds")


def stop_evaluation_workers(workers):
    """Release only the processes started by this evaluation."""
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        worker.join(timeout=5)
        if worker.is_alive():
            worker.kill()
            worker.join()
        worker.close()


def close_task_queue(task_queue):
    # Pending tasks no longer have readers after a worker failure.
    task_queue.cancel_join_thread()
    task_queue.close()


def process_single_file(file_path, predictor, args):
    data = np.load(file_path, 'r', allow_pickle=True)
    raw_imgs = data['imgs']
        
    if raw_imgs.ndim == 2:
        # 2D grayscale image (h, w)
        video_height, video_width = raw_imgs.shape
        segs = np.zeros_like(raw_imgs, dtype=np.uint8)
        num_slices = 1
    elif raw_imgs.ndim == 3:
        if raw_imgs.shape[2] == 3:
            # 2D color image (h, w, 3)
            video_height, video_width = raw_imgs.shape[:2]
            segs = np.zeros((video_height, video_width), dtype=np.uint8)
            num_slices = 1
        else:
            # 3D grayscale volume (d, h, w)
            num_slices, video_height, video_width = raw_imgs.shape
            segs = np.zeros_like(raw_imgs, dtype=np.uint8)
        
    imgs = preprocess(raw_imgs, predictor.device)
    gts = data['gts']  # D x orig_h x orig_w
        
    if gts.ndim == 3:
        if args.multi_class:
            binary_gts = decompose_multiclass_gts(gts)  # D x orig_h x orig_w
                
            class_results = {}
            class_segs = {}
                
            shared_inference_state = None
            shared_cached_features = None
                
            # Keep the baseline in full FP32 for fair comparison against TTA methods.
            with fp32_inference_context():
                shared_inference_state = predictor.init_state(
                    images=imgs,
                    video_height=video_height,
                    video_width=video_width,
                    offload_video_to_cpu=False,
                    offload_state_to_cpu=False
                )
                shared_cached_features = shared_inference_state["cached_features"]
                    
                for class_id, binary_gt in binary_gts.items():
                    key_slice_idx = find_key_slice_for_class(binary_gt)
                    key_slice_gt = binary_gt[key_slice_idx]
                        
                    if np.sum(key_slice_gt) == 0:
                        continue
                        
                    start_slice = max(0, key_slice_idx - args.num_keyslices)
                    end_slice = min(num_slices - 1, key_slice_idx + args.num_keyslices)
                    prompt_slices = list(range(start_slice, end_slice + 1))
                        
                    class_seg = np.zeros_like(gts, dtype=np.uint8)
                        
                    predictor.reset_state(shared_inference_state)
                    shared_inference_state["cached_features"] = shared_cached_features
                        
                    for slice_idx in prompt_slices:
                        slice_gt = binary_gt[slice_idx]

                        if np.sum(slice_gt) == 0:
                            continue

                        if args.prompt_type == 'box':
                            bbox2d = generate_bbox(slice_gt)
                            frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                                shared_inference_state, slice_idx, 0, box=bbox2d)
                        else:  # point
                            points, point_labels = sample_points_from_mask(
                                slice_gt, 
                                num_points=args.num_points, 
                                strategy=args.point_strategy
                            )
                            frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                                shared_inference_state, slice_idx, 0, points=points, labels=point_labels)

                    forward_masks = {}
                    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                        shared_inference_state, start_frame_idx=prompt_slices[0]
                    ):
                        forward_masks[frame_idx] = mask_logits[0].cpu()
                    predictor.reset_state(shared_inference_state)

                    shared_inference_state["cached_features"] = shared_cached_features

                    for slice_idx in reversed(prompt_slices):
                        slice_gt = binary_gt[slice_idx]

                        if np.sum(slice_gt) == 0:
                            continue

                        if args.prompt_type == 'box':
                            bbox2d = generate_bbox(slice_gt)
                            frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                                shared_inference_state, slice_idx, 1, box=bbox2d)
                        else:  # point
                            points, point_labels = sample_points_from_mask(
                                slice_gt, 
                                num_points=args.num_points, 
                                strategy=args.point_strategy
                            )
                            frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                                shared_inference_state, slice_idx, 1, points=points, labels=point_labels)

                    backward_masks = {}
                    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                        shared_inference_state, start_frame_idx=prompt_slices[-1], reverse=True
                    ):
                        backward_masks[frame_idx] = mask_logits[0].cpu()
                    predictor.reset_state(shared_inference_state)

                    for frame_idx in forward_masks.keys():
                        if frame_idx in backward_masks:
                            avg_mask_logits = (forward_masks[frame_idx] + backward_masks[frame_idx]) / 2.0
                            class_seg[frame_idx, (avg_mask_logits > 0.0).numpy()[0]] = 1
                        else:
                            class_seg[frame_idx, (forward_masks[frame_idx] > 0.0).numpy()[0]] = 1

                    for frame_idx in backward_masks.keys():
                        if frame_idx not in forward_masks:
                            class_seg[frame_idx, (backward_masks[frame_idx] > 0.0).numpy()[0]] = 1

                    if args.cc and np.max(class_seg) > 0:
                        class_seg = getLargestCC(class_seg)
                        class_seg = np.clip(class_seg, 0, 255).astype(np.uint8)

                    class_dsc = dice_coefficient(binary_gt, class_seg)
                    class_results[class_id] = {
                        'dsc': float(class_dsc),
                        'key_slice_idx': int(key_slice_idx)
                    }
                    class_segs[class_id] = class_seg
                
            segs = merge_class_segmentations(class_segs)
            if segs is None:
                segs = np.zeros_like(raw_imgs, dtype=np.uint8)
                
            metrics = calculate_multiclass_metrics(gts, segs, class_results)
            mean_dsc = metrics['mean_dsc']
                
            return {
                'mean_dsc': float(mean_dsc),
                'class_results': class_results,
                'num_classes': len(class_results),
                'num_slices': num_slices,
                'segs': segs if args.output_prediction else None,
            }

        else:
            slice_pixel_counts = np.sum(gts > 0, axis=(1, 2))
            key_slice_idx = np.argmax(slice_pixel_counts)

            start_slice = max(0, key_slice_idx - args.num_keyslices)
            end_slice = min(num_slices - 1, key_slice_idx + args.num_keyslices)
            prompt_slices = list(range(start_slice, end_slice + 1))

            # Keep the baseline in full FP32 for fair comparison against TTA methods.
            with fp32_inference_context():
                inference_state = predictor.init_state(imgs, video_height, video_width)

                for slice_idx in prompt_slices:
                    slice_gt = gts[slice_idx]

                    if np.sum(slice_gt) == 0:
                        continue

                    if args.prompt_type == 'box':
                        bbox2d = generate_bbox(slice_gt)
                        frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                            inference_state, slice_idx, 1, box=bbox2d)
                    else:  # point
                        points, point_labels = sample_points_from_mask(
                            slice_gt,
                            num_points=args.num_points,
                            strategy=args.point_strategy
                        )
                        frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                            inference_state, slice_idx, 1, points=points, labels=point_labels)

                forward_masks = {}
                for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                    inference_state, start_frame_idx=prompt_slices[0]
                ):
                    forward_masks[frame_idx] = mask_logits[0].cpu()
                predictor.reset_state(inference_state)

                for slice_idx in reversed(prompt_slices):
                    slice_gt = gts[slice_idx]

                    if np.sum(slice_gt) == 0:
                        continue

                    if args.prompt_type == 'box':
                        bbox2d = generate_bbox(slice_gt)
                        frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                            inference_state, slice_idx, 1, box=bbox2d)
                    else:  # point
                        points, point_labels = sample_points_from_mask(
                            slice_gt,
                            num_points=args.num_points,
                            strategy=args.point_strategy
                        )
                        frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                            inference_state, slice_idx, 1, points=points, labels=point_labels)

                backward_masks = {}
                for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                    inference_state, start_frame_idx=prompt_slices[-1], reverse=True
                ):
                    backward_masks[frame_idx] = mask_logits[0].cpu()
                predictor.reset_state(inference_state)

                for frame_idx in forward_masks.keys():
                    if frame_idx in backward_masks:
                        avg_mask_logits = (forward_masks[frame_idx] + backward_masks[frame_idx]) / 2.0
                        segs[frame_idx, (avg_mask_logits > 0.0).numpy()[0]] = 1
                    else:
                        segs[frame_idx, (forward_masks[frame_idx] > 0.0).numpy()[0]] = 1

                for frame_idx in backward_masks.keys():
                    if frame_idx not in forward_masks:
                        segs[frame_idx, (backward_masks[frame_idx] > 0.0).numpy()[0]] = 1

            if args.cc and np.max(segs) > 0:
                segs = getLargestCC(segs)
                segs = np.clip(segs, 0, 255).astype(np.uint8)

            # calculate metrics
            dsc = dice_coefficient(gts, segs)

            return {
                'mean_dsc': float(dsc),
                'num_slices': num_slices,
                'segs': segs if args.output_prediction else None,
            }
        
    else: # 2D
        if args.multi_class:
            binary_gts = decompose_multiclass_gts(gts)
                
            class_results = {}
            class_segs = {}
                
            shared_inference_state = None
            shared_cached_features = None
            key_slice = 0
                
            with fp32_inference_context():
                shared_inference_state = predictor.init_state(imgs.unsqueeze(0), video_height, video_width)
                shared_cached_features = shared_inference_state["cached_features"]
                    
                for class_id, binary_gt in binary_gts.items():
                    if np.sum(binary_gt) == 0:
                        continue
                        
                    if args.prompt_type == 'box':
                        bbox2d = generate_bbox(binary_gt)
                        points, point_labels = None, None
                    else:  # point
                        points, point_labels = sample_points_from_mask(
                            binary_gt,
                            num_points=args.num_points,
                            strategy=args.point_strategy
                        )
                        bbox2d = None
                        
                    if raw_imgs.ndim == 3 and raw_imgs.shape[2] == 3:
                        class_seg = np.zeros((video_height, video_width), dtype=np.uint8)
                    else:
                        class_seg = np.zeros_like(raw_imgs, dtype=np.uint8)
                        
                    predictor.reset_state(shared_inference_state)
                    shared_inference_state["cached_features"] = shared_cached_features
                        
                    if args.prompt_type == 'box':
                        frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                            shared_inference_state, key_slice, 1, box=bbox2d)
                    else:  # point
                        frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                            shared_inference_state, key_slice, 1, points=points, labels=point_labels)

                    class_seg[(mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
                    predictor.reset_state(shared_inference_state)

                    if args.cc and np.max(class_seg) > 0:
                        class_seg = getLargestCC(class_seg)
                        class_seg = np.clip(class_seg, 0, 255).astype(np.uint8)

                    class_dsc = dice_coefficient(binary_gt, class_seg)
                    class_results[class_id] = {
                        'dsc': float(class_dsc),
                    }
                    class_segs[class_id] = class_seg

            segs = merge_class_segmentations(class_segs)
            if segs is None:
                segs = np.zeros_like(raw_imgs, dtype=np.uint8)

            metrics = calculate_multiclass_metrics(gts, segs, class_results)
            mean_dsc = metrics['mean_dsc']

            return {
                'mean_dsc': float(mean_dsc),
                'class_results': class_results,
                'num_classes': len(class_results),
                'num_slices': num_slices,
                'segs': segs if args.output_prediction else None,
            }
                
        else:
            if args.prompt_type == 'box':
                bbox2d = generate_bbox(gts)
                points, point_labels = None, None
            else:  # point
                points, point_labels = sample_points_from_mask(
                    gts,
                    num_points=args.num_points,
                    strategy=args.point_strategy
                )
                bbox2d = None

            key_slice = 0

            with fp32_inference_context():
                inference_state = predictor.init_state(imgs.unsqueeze(0), video_height, video_width)

                if args.prompt_type == 'box':
                    frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                        inference_state, key_slice, 1, box=bbox2d)
                else:  # point
                    frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                        inference_state, key_slice, 1, points=points, labels=point_labels)

                segs[(mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
                predictor.reset_state(inference_state)

            if args.cc and np.max(segs) > 0:
                segs = getLargestCC(segs)
                segs = np.clip(segs, 0, 255).astype(np.uint8)

            # calculate metrics
            dsc = dice_coefficient(gts, segs)
                
            return {
                'mean_dsc': float(dsc),
                'num_slices': num_slices,
                'segs': segs if args.output_prediction else None,
            }


def run_evaluation_stream(args, logger):
    """Evaluate samples sequentially on one GPU."""
    available_gpus = get_available_gpus()
    if not available_gpus:
        logger.error("No available GPUs found!")
        raise RuntimeError("No available GPUs found!")

    gpu_id = select_stream_gpu(args, available_gpus, logger)
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    configure_fp32_execution()

    logger.info(f"Stream mode enabled: using GPU {gpu_id}, max_workers=1")

    predictor = build_sam2_video_predictor_npz_float32(
        config_file=args.model_cfg,
        ckpt_path=args.checkpoint,
        device=device
    )

    files_by_dir = collect_files_by_directory(args.data_dir)
    total_files = sum(len(files) for files in files_by_dir.values())
    logger.info(f'Total files to process: {total_files}')
    logger.info(f'Directories found: {len(files_by_dir)}')

    for dir_path, files in files_by_dir.items():
        rel_path = os.path.relpath(dir_path, args.data_dir)
        logger.info(f'  {rel_path}: {len(files)} files')

    prediction_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)

    csv_log_path = initialize_csv_log(
        args.output_dir,
        logger=logger,
    )

    all_files = []
    file_to_dir = {}
    for dir_path, file_paths in files_by_dir.items():
        for file_path in file_paths:
            file_to_dir[file_path] = dir_path
            all_files.append(file_path)

    if args.shuffle:
        random.shuffle(all_files)
        logger.info("Shuffle enabled: Processing files in random order")
    else:
        logger.info("Shuffle disabled: Processing files in original order")

    file_results = []
    all_dsc_scores = []
    dir_results = defaultdict(list)

    with tqdm.tqdm(total=total_files, desc=f"Stream GPU {gpu_id}", unit="sample", ascii=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}") as pbar:
        for file_path in all_files:
            rel_dir = os.path.relpath(file_to_dir[file_path], args.data_dir)
            rel_file_path = os.path.relpath(file_path, args.data_dir)
            pbar.set_postfix_str(os.path.basename(file_path)[:30])

            result = process_single_file(file_path, predictor, args)

            validate_sample_result(file_path, result)
            result['gpu_id'] = gpu_id

            file_result = build_file_result(
                rel_file_path=rel_file_path,
                rel_dir=rel_dir,
                result=result,
                multi_class=args.multi_class,
            )
            file_results.append(file_result)
            all_dsc_scores.append(result['mean_dsc'])
            dir_results[file_to_dir[file_path]].append(file_result)

            append_to_csv_log(
                csv_path=csv_log_path,
                subtask=rel_dir,
                filename=os.path.basename(file_path),
                results=result,
                gpu_id=gpu_id,
                logger=logger,
            )

            if args.output_prediction and result['segs'] is not None:
                rel_save_path = os.path.relpath(file_path, args.data_dir)
                save_path = os.path.join(prediction_dir, rel_save_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez_compressed(save_path, segs=result['segs'])

            pbar.set_postfix({
                'mean_dsc': f"{result['mean_dsc']:.6f}",
            })
            pbar.update(1)

    logger.info(f"Stream mode successfully processed all {total_files} files")

    dir_stats = {}
    for dir_path, results in dir_results.items():
        dsc_scores = [r['mean_dsc'] for r in results]
        dir_stats[dir_path] = calculate_directory_stats(dsc_scores)

    hierarchical_stats = calculate_hierarchical_directory_stats(file_results, args.data_dir)

    return dir_stats, file_results, all_dsc_scores, dir_results, hierarchical_stats


def run_evaluation_multigpu(args, logger):
    available_gpus = get_available_gpus()
    if not available_gpus:
        logger.error("No available GPUs found!")
        raise RuntimeError("No available GPUs found!")
    
    if args.gpus is not None:
        use_gpus = []
        for gpu_id in args.gpus:
            if gpu_id in available_gpus:
                use_gpus.append(gpu_id)
            else:
                logger.warning(f"GPU {gpu_id} is not available, skipping")
        if not use_gpus:
            logger.error("None of the specified GPUs are available!")
            raise RuntimeError("None of the specified GPUs are available!")
    else:
        use_gpus = available_gpus
    
    num_workers = args.max_workers if args.max_workers is not None else len(use_gpus)
    num_workers = min(num_workers, len(use_gpus))
    if num_workers < 1:
        raise ValueError("--max_workers must be positive")
    
    logger.info(f"Using {num_workers} GPUs: {use_gpus[:num_workers]}")
    
    files_by_dir = collect_files_by_directory(args.data_dir)
    
    total_files = sum(len(files) for files in files_by_dir.values())
    logger.info(f'Total files to process: {total_files}')
    logger.info(f'Directories found: {len(files_by_dir)}')
    
    for dir_path, files in files_by_dir.items():
        rel_path = os.path.relpath(dir_path, args.data_dir)
        logger.info(f'  {rel_path}: {len(files)} files')

    prediction_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)

    # Initialize CSV log file for tracking evaluation results
    csv_log_path = initialize_csv_log(args.output_dir, logger=logger)

    all_files = []
    file_to_dir = {}
    for dir_path, file_paths in files_by_dir.items():
        for file_path in file_paths:
            file_to_dir[file_path] = dir_path
            all_files.append(file_path)
    
    if args.shuffle:
        random.shuffle(all_files)
        logger.info(f"Shuffle enabled: Processing files in random order")
    else:
        logger.info(f"Shuffle disabled: Processing files in original order")
    
    file_results = []
    all_dsc_scores = []
    dir_results = defaultdict(list)

    with ExitStack() as cleanup:
        task_queue = Queue()
        cleanup.callback(close_task_queue, task_queue)
        # Separate pipes expose EOF if a worker dies during a result transfer.
        result_pipes = [mp.Pipe(duplex=False) for _ in range(num_workers)]
        for result_reader, result_writer in result_pipes:
            cleanup.callback(result_reader.close)
            cleanup.callback(result_writer.close)
        result_readers = [reader for reader, _ in result_pipes]
        workers = []
        cleanup.callback(stop_evaluation_workers, workers)

        for file_path in all_files:
            task_queue.put(file_path)
        for _ in range(num_workers):
            task_queue.put(None)

        for gpu_id, (_, result_writer) in zip(use_gpus[:num_workers], result_pipes):
            worker = Process(
                target=worker_process,
                args=(gpu_id, task_queue, result_writer, args),
            )
            worker.start()
            workers.append(worker)
            result_writer.close()

        with tqdm.tqdm(total=total_files, desc="Multi-GPU Processing", unit="file", ascii=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}") as pbar:
            for _ in range(total_files):
                file_path, result = receive_worker_result(
                    result_readers, workers, args.result_timeout,
                )
                validate_sample_result(file_path, result)
                rel_dir = os.path.relpath(file_to_dir[file_path], args.data_dir)
                rel_file_path = os.path.relpath(file_path, args.data_dir)

                file_result = build_file_result(
                    rel_file_path, rel_dir, result, multi_class=args.multi_class,
                )

                file_results.append(file_result)
                all_dsc_scores.append(result['mean_dsc'])
                dir_results[file_to_dir[file_path]].append(file_result)

                append_to_csv_log(
                    csv_path=csv_log_path,
                    subtask=rel_dir,
                    filename=os.path.basename(file_path),
                    results=result,
                    gpu_id=result.get('gpu_id', -1),
                    logger=logger,
                )

                if args.output_prediction and result['segs'] is not None:
                    save_path = os.path.join(prediction_dir, rel_file_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.savez_compressed(save_path, segs=result['segs'])

                pbar.update(1)
                pbar.set_postfix({
                    'GPU': result['gpu_id'],
                    'Current': os.path.basename(file_path)[:20],
                })

        for worker in workers:
            worker.join(timeout=args.result_timeout)
            if worker.is_alive():
                raise TimeoutError(f"Evaluation worker {worker.pid} did not exit")
            if worker.exitcode != 0:
                raise ChildProcessError(
                    f"Evaluation worker {worker.pid} exited with code {worker.exitcode}"
                )

    logger.info(f"Successfully processed all {total_files} files")
    
    dir_stats = {}
    for dir_path, results in dir_results.items():
        dsc_scores = [r['mean_dsc'] for r in results]
        dir_stats[dir_path] = calculate_directory_stats(dsc_scores)
    
    hierarchical_stats = calculate_hierarchical_directory_stats(file_results, args.data_dir)
    
    return dir_stats, file_results, all_dsc_scores, dir_results, hierarchical_stats


if __name__ == '__main__':
    args = parse_args()
    configure_fp32_execution()
    set_seeds(args.seed)
    logger = build_logger(args.output_dir, args, log_args=False)
    
    mp.set_start_method('spawn', force=True)
    
    if args.stream:
        logger.info("Starting Single-GPU Stream MedSAM2 Evaluation")
    else:
        logger.info("Starting Multi-GPU MedSAM2 Evaluation")
    logger.info("Precision mode: FP32 baseline (autocast disabled, TF32 disabled)")
    
    if args.stream:
        dir_stats, file_results, all_dsc_scores, dir_results, hierarchical_stats = run_evaluation_stream(args, logger)
    else:
        dir_stats, file_results, all_dsc_scores, dir_results, hierarchical_stats = run_evaluation_multigpu(args, logger)
    
    calculate_and_display_statistics(args, logger, dir_stats, file_results, all_dsc_scores, hierarchical_stats)

    final_metric_results = build_final_metric_results(file_results)
    final_metric_summary = calculate_final_metric_summary(
        final_metric_results,
        args.data_dir,
        metric_names=('mean_dsc',),
        metadata={
            'script': os.path.basename(__file__),
            'result_type': 'baseline_evaluation',
            'execution_mode': 'stream' if args.stream else 'multigpu',
        },
    )
    final_summary_json_path, final_summary_csv_path = export_final_metric_summary(
        final_metric_summary,
        args.output_dir,
    )
    log_final_metric_summary(final_metric_summary, logger, title='FINAL BASELINE METRIC SUMMARY')
    logger.info(f"Final summary JSON saved to: {final_summary_json_path}")
    logger.info(f"Final summary CSV saved to: {final_summary_csv_path}")
