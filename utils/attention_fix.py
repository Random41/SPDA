import warnings
import torch
import torch.nn.functional as F
from functools import wraps
import logging


def suppress_attention_warnings():
    warnings.filterwarnings("ignore", message=".*Memory efficient kernel not used.*")
    warnings.filterwarnings("ignore", message=".*Memory Efficient attention has been runtime disabled.*")
    warnings.filterwarnings("ignore", message=".*Flash attention kernel not used.*")
    warnings.filterwarnings("ignore", message=".*Expected query, key and value to all be of dtype.*")
    warnings.filterwarnings("ignore", message=".*CuDNN attention kernel not used.*")
    warnings.filterwarnings("ignore", message=".*Flash Attention kernel failed.*")


def patch_attention_with_mixed_precision():
    original_sdpa = F.scaled_dot_product_attention

    @wraps(original_sdpa)
    def mixed_precision_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        original_dtype = query.dtype

        if (original_dtype == torch.float32 and 
            query.is_cuda and 
            torch.cuda.get_device_properties(query.device).major >= 8):

            query_fp16 = query.half()
            key_fp16 = key.half()
            value_fp16 = value.half()

            attn_mask_fp16 = attn_mask.half() if attn_mask is not None else None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = original_sdpa(
                    query_fp16, key_fp16, value_fp16,
                    attn_mask=attn_mask_fp16,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale
                )

            return result.to(original_dtype)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return original_sdpa(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale
            )

    F.scaled_dot_product_attention = mixed_precision_sdpa


def enable_efficient_attention():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


def apply_attention_fix(logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Applying attention mechanism optimizations...")

    suppress_attention_warnings()

    enable_efficient_attention()

    patch_attention_with_mixed_precision()

    logger.info("Attention mechanism optimizations applied successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    apply_attention_fix()
