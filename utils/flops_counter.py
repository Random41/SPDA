import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging


class FLOPsCounter:

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.reset()

    def reset(self):
        self.total_flops = 0
        self.adapt_flops = 0
        self.infer_flops = 0
        self.current_phase = None
        self._hooks = []
        self.region_flops = {}
        self.current_region = None
        self.region_start_flops = {}

    def start_adapt_phase(self):
        self.current_phase = 'adapt'
        self._phase_start_flops = self.total_flops

    def end_adapt_phase(self):
        if self.current_phase == 'adapt':
            self.adapt_flops = self.total_flops - self._phase_start_flops
            self.current_phase = None

    def start_infer_phase(self):
        self.current_phase = 'infer'
        self._phase_start_flops = self.total_flops

    def end_infer_phase(self):
        if self.current_phase == 'infer':
            self.infer_flops = self.total_flops - self._phase_start_flops
            self.current_phase = None

    def start_region(self, region_name: str):
        self.current_region = region_name
        self.region_start_flops[region_name] = self.total_flops

    def end_region(self, region_name: str):
        if region_name in self.region_start_flops:
            region_cost = self.total_flops - self.region_start_flops[region_name]
            if region_name not in self.region_flops:
                self.region_flops[region_name] = 0
            self.region_flops[region_name] += region_cost
            del self.region_start_flops[region_name]
            if self.current_region == region_name:
                self.current_region = None

    @staticmethod
    def count_conv2d_flops(module: nn.Conv2d, input: torch.Tensor, output: torch.Tensor) -> int:
        batch_size = input[0].size(0)
        output_height = output.size(2)
        output_width = output.size(3)

        kernel_height, kernel_width = module.kernel_size
        in_channels = module.in_channels
        out_channels = module.out_channels
        groups = module.groups

        kernel_flops = kernel_height * kernel_width * (in_channels // groups)
        output_size = output_height * output_width
        total_flops = batch_size * kernel_flops * output_size * out_channels

        if module.bias is not None:
            total_flops += batch_size * output_size * out_channels

        return int(total_flops)

    @staticmethod
    def count_linear_flops(module: nn.Linear, input: torch.Tensor, output: torch.Tensor) -> int:
        batch_size = input[0].size(0)
        in_features = module.in_features
        out_features = module.out_features

        total_flops = 2 * batch_size * in_features * out_features

        if module.bias is not None:
            total_flops += batch_size * out_features

        return int(total_flops)

    @staticmethod
    def count_matmul_flops(input: torch.Tensor, other: torch.Tensor) -> int:
        if input.dim() == 2 and other.dim() == 2:
            m, k = input.shape
            n = other.shape[1]
            return int(2 * m * k * n)
        elif input.dim() == 3 and other.dim() == 3:
            b, m, k = input.shape
            n = other.shape[2]
            return int(2 * b * m * k * n)
        elif input.dim() >= 2 and other.dim() >= 2:
            batch_size = max(input.shape[:-2], other.shape[:-2], default=[1])
            if isinstance(batch_size, tuple):
                batch_size = 1
                for dim in batch_size:
                    batch_size *= dim
            m = input.shape[-2] if input.dim() >= 2 else 1
            k = input.shape[-1]
            n = other.shape[-1] if other.dim() >= 2 else 1
            return int(2 * batch_size * m * k * n)
        else:
            return 0

    @staticmethod
    def count_norm_flops(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> int:
        num_elements = input[0].numel()
        return int(4 * num_elements)

    @staticmethod
    def count_activation_flops(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> int:
        num_elements = input[0].numel()
        return int(num_elements)

    def _make_hook(self):
        def hook(module, input, output):
            flops = 0

            if isinstance(module, nn.Conv2d):
                flops = self.count_conv2d_flops(module, input, output)
            elif isinstance(module, nn.Conv1d):
                flops = self.count_conv2d_flops(module, input, output)

            elif isinstance(module, nn.Linear):
                flops = self.count_linear_flops(module, input, output)

            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                flops = self.count_norm_flops(module, input, output)

            elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.Softmax)):
                flops = self.count_activation_flops(module, input, output)

            self.total_flops += flops

        return hook

    def register_hooks(self):
        self._remove_hooks()
        hook_fn = self._make_hook()

        for module in self.model.modules():
            if len(list(module.children())) == 0:
                handle = module.register_forward_hook(hook_fn)
                self._hooks.append(handle)

    def _remove_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    def get_stats(self, num_samples: int = 1, num_slices: int = 1) -> Dict[str, float]:
        stats = {
            'total_flops': self.total_flops,
            'adapt_flops': self.adapt_flops,
            'infer_flops': self.infer_flops,
            'flops_per_sample': self.total_flops / max(num_samples, 1),
            'flops_per_slice': self.total_flops / max(num_slices, 1),
            'adapt_flops_per_sample': self.adapt_flops / max(num_samples, 1),
            'adapt_flops_per_slice': self.adapt_flops / max(num_slices, 1),
            'infer_flops_per_sample': self.infer_flops / max(num_samples, 1),
            'infer_flops_per_slice': self.infer_flops / max(num_slices, 1),
        }

        for region, flops in self.region_flops.items():
            stats[f'region_{region}_flops'] = flops
            stats[f'region_{region}_percentage'] = (flops / max(self.total_flops, 1)) * 100

        return stats

    def format_flops(self, flops: float) -> str:
        if flops >= 1e12:
            return f"{flops / 1e12:.2f} TFLOPs"
        elif flops >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            return f"{flops / 1e6:.2f} MFLOPs"
        elif flops >= 1e3:
            return f"{flops / 1e3:.2f} KFLOPs"
        else:
            return f"{flops:.0f} FLOPs"

    def print_stats(self, num_samples: int = 1, num_slices: int = 1, logger: Optional[logging.Logger] = None):
        stats = self.get_stats(num_samples, num_slices)

        log_fn = logger.info if logger else print

        log_fn("=" * 70)
        log_fn("📊 FLOPs Statistics:")
        log_fn("-" * 70)
        log_fn(f"  Total FLOPs:              {self.format_flops(stats['total_flops'])}")
        log_fn(f"    ├─ Adaptation FLOPs:    {self.format_flops(stats['adapt_flops'])}")
        log_fn(f"    └─ Inference FLOPs:     {self.format_flops(stats['infer_flops'])}")
        log_fn(f"  FLOPs per Sample:         {self.format_flops(stats['flops_per_sample'])}")
        log_fn(f"  FLOPs per Slice:          {self.format_flops(stats['flops_per_slice'])}")
        log_fn("-" * 70)
        log_fn(f"  Samples: {num_samples}, Slices: {num_slices}")
        log_fn("=" * 70)

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_hooks()


class SimpleFLOPsTracker:

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_flops = 0
        self.adapt_flops = 0
        self.infer_flops = 0
        self.adapt_phase_active = False
        self.infer_phase_active = False
        self.region_flops = {}
        self.current_region = None
        self.region_start_flops = {}

    def start_adapt_phase(self):
        self.adapt_phase_active = True
        self._adapt_start = self.total_flops

    def end_adapt_phase(self):
        if self.adapt_phase_active:
            self.adapt_flops += (self.total_flops - self._adapt_start)
            self.adapt_phase_active = False

    def start_infer_phase(self):
        self.infer_phase_active = True
        self._infer_start = self.total_flops

    def end_infer_phase(self):
        if self.infer_phase_active:
            self.infer_flops += (self.total_flops - self._infer_start)
            self.infer_phase_active = False

    def start_region(self, region_name: str):
        self.current_region = region_name
        self.region_start_flops[region_name] = self.total_flops

    def end_region(self, region_name: str):
        if region_name in self.region_start_flops:
            region_cost = self.total_flops - self.region_start_flops[region_name]
            if region_name not in self.region_flops:
                self.region_flops[region_name] = 0
            self.region_flops[region_name] += region_cost
            del self.region_start_flops[region_name]
            if self.current_region == region_name:
                self.current_region = None

    def add_flops(self, flops: int, description: str = ""):
        self.total_flops += flops
        if self.current_region and self.current_region in self.region_start_flops:
            pass

    def estimate_image_encoder_flops(self, batch_size: int, image_size: int = 512, 
                                     channels: int = 3, hidden_dim: int = 768) -> int:
        patch_size = 16
        num_patches = (image_size // patch_size) ** 2

        patch_embed_flops = batch_size * channels * (patch_size ** 2) * num_patches * hidden_dim * 2

        num_blocks = 12

        qkv_flops = 3 * batch_size * num_patches * hidden_dim * hidden_dim * 2
        attn_flops = batch_size * num_patches * num_patches * hidden_dim * 2
        attn_v_flops = batch_size * num_patches * num_patches * hidden_dim * 2
        out_proj_flops = batch_size * num_patches * hidden_dim * hidden_dim * 2

        sa_flops = qkv_flops + attn_flops + attn_v_flops + out_proj_flops

        mlp_hidden = hidden_dim * 4
        mlp_flops = batch_size * num_patches * hidden_dim * mlp_hidden * 2
        mlp_flops += batch_size * num_patches * mlp_hidden * hidden_dim * 2

        transformer_flops = num_blocks * (sa_flops + mlp_flops)

        total_flops = patch_embed_flops + transformer_flops

        return int(total_flops)

    def estimate_mask_decoder_flops(self, batch_size: int, num_queries: int = 1, 
                                    output_size: int = 256) -> int:
        hidden_dim = 256

        cross_attn_flops = batch_size * num_queries * (output_size ** 2) * hidden_dim * 2

        upsample_flops = batch_size * (output_size ** 2) * hidden_dim * 64 * 2

        pred_head_flops = batch_size * (output_size ** 2) * 64 * num_queries * 2

        total_flops = cross_attn_flops + upsample_flops + pred_head_flops

        return int(total_flops)

    def estimate_sam2_forward_flops(self, batch_size: int, height: int = 512, width: int = 512) -> int:
        encoder_flops = self.estimate_image_encoder_flops(
            batch_size=batch_size,
            image_size=max(height, width)
        )

        decoder_flops = self.estimate_mask_decoder_flops(
            batch_size=batch_size,
            num_queries=1
        )

        total_flops = encoder_flops + decoder_flops

        return int(total_flops)

    def estimate_fft2_flops(self, batch_size: int, channels: int, height: int, width: int) -> int:
        import math
        if height <= 0 or width <= 0:
            return 0
        log_cost = (math.log2(max(height, 1)) + math.log2(max(width, 1)))
        per_image_channel = 5.0 * height * width * log_cost
        complex_to_real = 10.0
        total = batch_size * channels * per_image_channel * complex_to_real
        return int(total)

    def estimate_ifft2_flops(self, batch_size: int, channels: int, height: int, width: int) -> int:
        return self.estimate_fft2_flops(batch_size, channels, height, width)

    def estimate_freq_filter_flops(self, batch_size: int, channels: int, height: int, width: int, bands: int) -> int:
        per_image_channel = height * width * max(bands, 1)
        ops_per_element = 4
        total = batch_size * channels * per_image_channel * ops_per_element
        return int(total)

    def estimate_weak_aug_flops(self, batch_size: int, channels: int, height: int, width: int) -> int:
        ops_per_pixel_channel = 20
        total = batch_size * channels * height * width * ops_per_pixel_channel
        return int(total)

    def estimate_sbct_flops(self, batch_size: int, channels: int, height: int, width: int) -> int:
        ops_per_pixel = 20
        total_pixels = batch_size * channels * height * width
        return int(total_pixels * ops_per_pixel)

    def estimate_loss_flops(self, batch_size: int, height: int, width: int, num_scales: int = 2) -> int:
        ops_per_pixel = 4
        total_pixels = 0

        total_pixels += batch_size * height * width

        low_res = 64
        total_pixels += batch_size * low_res * low_res

        return int(total_pixels * ops_per_pixel * num_scales)

    def estimate_backward_flops(self, forward_flops: int) -> int:
        return int(forward_flops * 2.5)

    def estimate_optimizer_step_flops(self, num_params: int) -> int:
        ops_per_param = 10
        return int(num_params * ops_per_param)

    def estimate_ema_update_flops(self, num_params: int) -> int:
        ops_per_param = 3
        return int(num_params * ops_per_param)

    def estimate_preprocess_flops(self, num_slices: int, height: int, width: int, target_size: int = 512) -> int:

        ops_per_pixel = 21
        total_pixels = num_slices * target_size * target_size * 3

        return int(total_pixels * ops_per_pixel)

    def estimate_dice_computation_flops(self, num_classes: int, num_slices: int, height: int, width: int) -> int:
        ops_per_pixel = 4
        total_pixels = num_classes * num_slices * height * width

        return int(total_pixels * ops_per_pixel)

    def estimate_mask_decompose_flops(self, num_slices: int, height: int, width: int, num_classes: int) -> int:
        ops_per_pixel = num_classes
        total_pixels = num_slices * height * width

        return int(total_pixels * ops_per_pixel)

    def get_stats(self, num_samples: int = 1, num_slices: int = 1) -> Dict[str, float]:
        stats = {
            'total_flops': self.total_flops,
            'adapt_flops': self.adapt_flops,
            'infer_flops': self.infer_flops,
            'flops_per_sample': self.total_flops / max(num_samples, 1),
            'flops_per_slice': self.total_flops / max(num_slices, 1),
            'adapt_flops_per_sample': self.adapt_flops / max(num_samples, 1),
            'adapt_flops_per_slice': self.adapt_flops / max(num_slices, 1),
            'infer_flops_per_sample': self.infer_flops / max(num_samples, 1),
            'infer_flops_per_slice': self.infer_flops / max(num_slices, 1),
        }

        for region, flops in self.region_flops.items():
            stats[f'region_{region}_flops'] = flops
            stats[f'region_{region}_percentage'] = (flops / max(self.total_flops, 1)) * 100

        return stats

    def format_flops(self, flops: float) -> str:
        if flops >= 1e12:
            return f"{flops / 1e12:.2f} TFLOPs"
        elif flops >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            return f"{flops / 1e6:.2f} MFLOPs"
        elif flops >= 1e3:
            return f"{flops / 1e3:.2f} KFLOPs"
        else:
            return f"{flops:.0f} FLOPs"

