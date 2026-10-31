import os
import sys
import logging
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor_npz


def set_seeds(seed=2025):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_cfg, checkpoint_path):
    predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint_path)
    return predictor


def build_logger(output_dir, args=None, logger_name='medsam2_eval', resume=False, log_args=True):
    os.makedirs(output_dir, exist_ok=True)

    file_mode = 'a' if resume else 'w'

    if args is not None:
        args_file_mode = 'a' if resume else 'w'
        with open(os.path.join(output_dir, "args.txt"), mode=args_file_mode, encoding="utf-8") as f:
            if resume:
                f.write("\n" + "="*60 + "\n")
                f.write("RESUMED RUN\n")
                f.write("="*60 + "\n")
            args_dict = vars(args)
            for key, value in sorted(args_dict.items()):
                f.write(f"{key}: {value}\n")

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    debug_handler = logging.FileHandler(os.path.join(output_dir, "debug.log"), mode=file_mode)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(debug_handler)

    info_handler = logging.FileHandler(os.path.join(output_dir, "info.log"), mode=file_mode)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(info_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    logger.propagate = False

    if args is not None and log_args:
        args_dict = vars(args)
        args_lines = []
        for key, value in sorted(args_dict.items()):
            args_lines.append(f"{key}: {value}")
        args_str = "\n".join(args_lines)
        logger.debug(f"Arguments:\n{args_str}")

    return logger
