import math
import numpy as np
import torch
import torch.nn.functional as F


class Image3DAugmentationHandler:

    def __init__(self, seed=42, aug_args=None):
        self.rng = np.random.default_rng(seed=seed)
        self.aug_args = aug_args if aug_args is not None else {}

    def reset_state(self):
        pass

    def apply_augmentation(self, imgs, args=None):
        raise NotImplementedError("Subclasses must implement apply_augmentation()")


class WeakAugmentationHandler(Image3DAugmentationHandler):

    def __init__(self, seed=42, aug_args=None):
        super().__init__(seed, aug_args)

    def apply_augmentation(self, imgs, args=None):
        weak_aug_types = ['intensity', 'noise']
        aug_type = self.rng.choice(weak_aug_types)

        if aug_type == 'intensity':
            return self._apply_intensity_augmentation(imgs)
        elif aug_type == 'noise':
            return self._apply_noise_augmentation(imgs)

    def _apply_intensity_augmentation(self, imgs):
        intensity_types = ['brightness', 'contrast', 'gamma']
        intensity_type = self.rng.choice(intensity_types)

        augmented_imgs = imgs.clone()

        if intensity_type == 'brightness':
            brightness_factor = self.rng.uniform(-0.1, 0.1)
            augmented_imgs = torch.clamp(augmented_imgs + brightness_factor, 0, 1)

        elif intensity_type == 'contrast':
            contrast_factor = self.rng.uniform(0.9, 1.1)
            mean_val = augmented_imgs.mean(dim=[2, 3], keepdim=True)
            augmented_imgs = torch.clamp(
                (augmented_imgs - mean_val) * contrast_factor + mean_val, 0, 1
            )

        elif intensity_type == 'gamma':
            gamma = self.rng.uniform(0.9, 1.1)
            augmented_imgs = torch.pow(torch.clamp(augmented_imgs, 1e-8, 1), gamma)

        return augmented_imgs

    def _apply_noise_augmentation(self, imgs):
        noise_types = ['gaussian', 'uniform']
        noise_type = self.rng.choice(noise_types)

        augmented_imgs = imgs.clone()

        if noise_type == 'gaussian':
            noise_std = 0.01
            noise = torch.randn_like(augmented_imgs) * noise_std
            augmented_imgs = torch.clamp(augmented_imgs + noise, 0, 1)

        elif noise_type == 'uniform':
            noise_range = 0.02
            noise = (torch.rand_like(augmented_imgs) - 0.5) * 2 * noise_range
            augmented_imgs = torch.clamp(augmented_imgs + noise, 0, 1)

        return augmented_imgs


class FrequencyAugmentationHandler(Image3DAugmentationHandler):

    def __init__(self, seed=42, aug_args=None):
        super().__init__(seed, aug_args)

        self.num_freq_bands = self.aug_args.get('num_freq_bands', 12)
        self.kappa = self.aug_args.get('kappa', 16.0)
        self.epsilon = self.aug_args.get('epsilon', 1e-12)
        self.cold_start_sigma = self.aug_args.get('cold_start_sigma', 0.05)

        self.W = 0.0
        self.mu = None
        self.M2_between = None

        self.WithinM2 = None
        self.DF_within = 0.0

        self.freq_bands_cache = {}

    def reset_state(self):
        self.W = 0.0
        self.mu = None
        self.M2_between = None

        self.WithinM2 = None
        self.DF_within = 0.0

        self.freq_bands_cache = {}

    def apply_augmentation(self, imgs, args=None):
        if args is None:
            args = {}

        num_low_freq_bands = args.get('num_low_freq_bands', 3)
        augmentation_probability = args.get('augmentation_probability', 1.0)
        single_pass = args.get('single_pass', True)

        if self.rng.random() > augmentation_probability:
            return imgs.clone()

        D, C, H, W = imgs.shape
        device = imgs.device

        band_masks = self._create_frequency_bands(H, W)
        B = len(band_masks)

        n_s = 0
        mu_s = torch.zeros(B, dtype=torch.float32, device=device)
        M2_s = torch.zeros(B, dtype=torch.float32, device=device)

        augmented_imgs = imgs.clone()

        if single_pass:
            for d in range(D):
                slice_img = imgs[d]

                with torch.no_grad():
                    fft_result = torch.fft.fft2(slice_img[0])
                    fft_shifted = torch.fft.fftshift(fft_result)

                    z = self._compute_rapsd_per_band(fft_shifted, band_masks)

                alpha = n_s / (n_s + self.kappa) if n_s > 0 else 0.0

                if self.mu is not None:
                    mu_global = self.mu.to(device)
                else:
                    mu_global = torch.zeros(B, dtype=torch.float32, device=device)

                mu_tilde = alpha * mu_s + (1 - alpha) * mu_global

                sigma_within = self._get_sigma_within().to(device)

                augmented_imgs[d] = self._apply_frequency_rescaling(
                    slice_img, band_masks, num_low_freq_bands, mu_tilde, sigma_within
                )

                n_s, mu_s, M2_s = self._update_within_sample_stats(z, n_s, mu_s, M2_s)

        else:
            for d in range(D):
                slice_img = imgs[d]

                with torch.no_grad():
                    fft_result = torch.fft.fft2(slice_img[0])
                    fft_shifted = torch.fft.fftshift(fft_result)

                    z = self._compute_rapsd_per_band(fft_shifted, band_masks)

                n_s, mu_s, M2_s = self._update_within_sample_stats(z, n_s, mu_s, M2_s)

            alpha = n_s / (n_s + self.kappa)
            if self.mu is not None:
                mu_global = self.mu.to(device)
            else:
                mu_global = mu_s.clone()

            mu_tilde = alpha * mu_s + (1 - alpha) * mu_global

            sigma_within = self._get_sigma_within().to(device)

            for d in range(D):
                slice_img = imgs[d]
                augmented_imgs[d] = self._apply_frequency_rescaling(
                    slice_img, band_masks, num_low_freq_bands, mu_tilde, sigma_within
                )

        if n_s > 0:
            self._merge_between_sample_stats(n_s, mu_s, M2_s)

        return augmented_imgs

    def _create_frequency_bands(self, H, W, num_bands=None):
        if num_bands is None:
            num_bands = self.num_freq_bands

        cache_key = (H, W, num_bands)
        if cache_key in self.freq_bands_cache:
            return self.freq_bands_cache[cache_key]

        cy, cx = H // 2, W // 2

        y = torch.arange(H, dtype=torch.float32)
        x = torch.arange(W, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        radius = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
        max_radius = min(cy, cx)

        band_masks = []
        for b in range(num_bands):
            r_min = (b * max_radius) / num_bands
            r_max = ((b + 1) * max_radius) / num_bands
            mask = (radius >= r_min) & (radius < r_max)
            band_masks.append(mask)

        self.freq_bands_cache[cache_key] = band_masks

        return band_masks

    def _compute_rapsd_per_band(self, fft_spectrum, band_masks):
        power_spectrum = torch.abs(fft_spectrum) ** 2

        num_bands = len(band_masks)
        rapsd = torch.zeros(num_bands, dtype=torch.float32, device=fft_spectrum.device)

        for b, mask in enumerate(band_masks):
            mask = mask.to(fft_spectrum.device)
            if mask.sum() > 0:
                rapsd[b] = power_spectrum[mask].mean()
            else:
                rapsd[b] = self.epsilon

        rapsd_log = torch.log(rapsd + self.epsilon)

        return rapsd_log

    def _update_within_sample_stats(self, z, n_s, mu_s, M2_s):
        n_s_new = n_s + 1
        delta = z - mu_s
        mu_s_new = mu_s + delta / n_s_new
        M2_s_new = M2_s + delta * (z - mu_s_new)

        return n_s_new, mu_s_new, M2_s_new

    def _merge_between_sample_stats(self, n_s, mu_s, M2_s):
        B = mu_s.shape[0]
        device = mu_s.device

        if self.mu is None:
            self.mu = torch.zeros(B, dtype=torch.float32, device=device)
            self.M2_between = torch.zeros(B, dtype=torch.float32, device=device)
            self.WithinM2 = torch.zeros(B, dtype=torch.float32, device=device)
        else:
            self.mu = self.mu.to(device)
            self.M2_between = self.M2_between.to(device)
            self.WithinM2 = self.WithinM2.to(device)

        w_s = float(n_s)

        W_new = self.W + w_s
        delta = mu_s - self.mu

        if W_new > 0:
            mu_new = self.mu + (w_s / W_new) * delta
            self.M2_between = self.M2_between + w_s * delta * (mu_s - mu_new)
            self.mu = mu_new

        self.W = W_new

        self.WithinM2 = self.WithinM2 + M2_s
        self.DF_within += max(0, n_s - 1)

    def _get_sigma_within(self):
        if self.WithinM2 is None or self.DF_within <= 0:
            return torch.ones(self.num_freq_bands) * self.cold_start_sigma

        variance = self.WithinM2 / max(1.0, self.DF_within)
        sigma = torch.sqrt(torch.clamp(variance, min=0))

        if self.DF_within < 30:
            shrinkage_factor = self.DF_within / 30.0
            sigma = shrinkage_factor * sigma + (1 - shrinkage_factor) * self.cold_start_sigma

        return sigma

    def _get_sigma_between(self):
        if self.M2_between is None or self.W <= 1:
            return torch.ones(self.num_freq_bands) * 0.5

        variance = self.M2_between / max(1.0, self.W - 1)
        return torch.sqrt(torch.clamp(variance, min=0))

    def _apply_frequency_rescaling(self, img_slice, band_masks, num_low_freq_bands, 
                                   mu_tilde, sigma_within):
        C, H, W = img_slice.shape
        device = img_slice.device

        augmented_slice = img_slice.clone()

        for c in range(C):
            fft_result = torch.fft.fft2(img_slice[c])

            fft_shifted = torch.fft.fftshift(fft_result)

            power_spectrum = torch.abs(fft_shifted) ** 2
            current_rapsd_log = torch.zeros(num_low_freq_bands, device=device)

            for b, mask in enumerate(band_masks[:num_low_freq_bands]):
                mask = mask.to(device)
                if mask.sum() > 0:
                    current_rapsd_log[b] = torch.log(power_spectrum[mask].mean() + self.epsilon)
                else:
                    current_rapsd_log[b] = torch.log(torch.tensor(self.epsilon, device=device))

            target_rapsd_log = torch.zeros(num_low_freq_bands, device=device)
            for b in range(num_low_freq_bands):
                if mu_tilde[b].abs() < 0.1:
                    base_value = current_rapsd_log[b]
                else:
                    base_value = mu_tilde[b]

                sampled_noise = torch.randn(1, device=device).item()
                sampled_noise = max(-2.0, min(2.0, sampled_noise))
                target_rapsd_log[b] = base_value + sigma_within[b] * sampled_noise

            target_rapsd = torch.exp(target_rapsd_log)
            current_rapsd = torch.exp(current_rapsd_log)

            scale_factors = torch.sqrt(target_rapsd / (current_rapsd + self.epsilon))

            fft_scaled = fft_shifted.clone()
            for b, mask in enumerate(band_masks[:num_low_freq_bands]):
                mask = mask.to(device)
                fft_scaled[mask] = fft_scaled[mask] * scale_factors[b]

            fft_unshifted = torch.fft.ifftshift(fft_scaled)

            reconstructed = torch.fft.ifft2(fft_unshifted).real

            augmented_slice[c] = torch.clamp(reconstructed, 0, 1)

        return augmented_slice


class FrequencyDeviceAugmentationHandler(FrequencyAugmentationHandler):

    def __init__(self, seed=42, aug_args=None):
        super().__init__(seed, aug_args)

    def apply_augmentation(self, imgs, args=None):
        if args is None:
            args = {}

        num_low_freq_bands = args.get('num_low_freq_bands', 3)
        augmentation_probability = args.get('augmentation_probability', 0.8)
        single_pass = args.get('single_pass', True)
        storage_device = args.get('storage_device', None)
        compute_device = args.get('compute_device', None)

        with torch.no_grad():
            if storage_device is None:
                storage_device = imgs.device
            else:
                storage_device = torch.device(storage_device)

            if compute_device is None:
                compute_device = imgs.device
            else:
                compute_device = torch.device(compute_device)

            if storage_device.type == 'cuda' and storage_device.index is None:
                storage_device = torch.device(f'cuda:{torch.cuda.current_device()}')
            if compute_device.type == 'cuda' and compute_device.index is None:
                compute_device = torch.device(f'cuda:{torch.cuda.current_device()}')

            if self.rng.random() > augmentation_probability:
                return imgs.clone().to(storage_device)

            D, C, H, W = imgs.shape

            batch_size = D
            if compute_device.type == 'cuda':
                torch.cuda.set_device(compute_device)
                free_memory = torch.cuda.mem_get_info(compute_device)[0]

                slice_memory = C * H * W * 4
                estimated_memory_per_slice = slice_memory * 6

                safe_memory = free_memory * 0.8
                max_batch_size = max(1, int(safe_memory / estimated_memory_per_slice))
                batch_size = min(D, max_batch_size)

            imgs_on_device = imgs.to(compute_device)
            augmented_imgs = torch.zeros_like(imgs, device=storage_device)

            band_masks = self._create_frequency_bands(H, W)
            B = len(band_masks)

            n_s = 0
            mu_s = torch.zeros(B, dtype=torch.float32, device=compute_device)
            M2_s = torch.zeros(B, dtype=torch.float32, device=compute_device)

            if single_pass:
                for batch_start in range(0, D, batch_size):
                    batch_end = min(batch_start + batch_size, D)
                    batch_indices = range(batch_start, batch_end)

                    imgs_batch = imgs_on_device[batch_start:batch_end]
                    augmented_batch = imgs_batch.clone()

                    for local_idx, d in enumerate(batch_indices):
                        slice_img = imgs_batch[local_idx]

                        fft_result = torch.fft.fft2(slice_img[0])
                        fft_shifted = torch.fft.fftshift(fft_result)
                        z = self._compute_rapsd_per_band(fft_shifted, band_masks)

                        alpha = n_s / (n_s + self.kappa) if n_s > 0 else 0.0

                        if self.mu is not None:
                            mu_global = self.mu.to(compute_device)
                        else:
                            mu_global = torch.zeros(B, dtype=torch.float32, device=compute_device)

                        mu_tilde = alpha * mu_s + (1 - alpha) * mu_global

                        sigma_within = self._get_sigma_within().to(compute_device)

                        augmented_batch[local_idx] = self._apply_frequency_rescaling(
                            slice_img, band_masks, num_low_freq_bands, mu_tilde, sigma_within
                        )

                        n_s, mu_s, M2_s = self._update_within_sample_stats(z, n_s, mu_s, M2_s)

                    augmented_imgs[batch_start:batch_end] = augmented_batch.to(storage_device)

                    if compute_device.type == 'cuda':
                        del imgs_batch, augmented_batch
                        torch.cuda.empty_cache()

            else:
                for batch_start in range(0, D, batch_size):
                    batch_end = min(batch_start + batch_size, D)
                    imgs_batch = imgs_on_device[batch_start:batch_end]

                    for local_idx in range(imgs_batch.shape[0]):
                        slice_img = imgs_batch[local_idx]

                        fft_result = torch.fft.fft2(slice_img[0])
                        fft_shifted = torch.fft.fftshift(fft_result)
                        z = self._compute_rapsd_per_band(fft_shifted, band_masks)

                        n_s, mu_s, M2_s = self._update_within_sample_stats(z, n_s, mu_s, M2_s)

                    if compute_device.type == 'cuda':
                        del imgs_batch
                        torch.cuda.empty_cache()

                alpha = n_s / (n_s + self.kappa)
                if self.mu is not None:
                    mu_global = self.mu.to(compute_device)
                else:
                    mu_global = mu_s.clone()

                mu_tilde = alpha * mu_s + (1 - alpha) * mu_global

                sigma_within = self._get_sigma_within().to(compute_device)

                for batch_start in range(0, D, batch_size):
                    batch_end = min(batch_start + batch_size, D)
                    imgs_batch = imgs_on_device[batch_start:batch_end]
                    augmented_batch = imgs_batch.clone()

                    for local_idx in range(imgs_batch.shape[0]):
                        slice_img = imgs_batch[local_idx]
                        augmented_batch[local_idx] = self._apply_frequency_rescaling(
                            slice_img, band_masks, num_low_freq_bands, mu_tilde, sigma_within
                        )

                    augmented_imgs[batch_start:batch_end] = augmented_batch.to(storage_device)

                    if compute_device.type == 'cuda':
                        del imgs_batch, augmented_batch
                        torch.cuda.empty_cache()

            if n_s > 0:
                self._merge_between_sample_stats(n_s, mu_s, M2_s)

            return augmented_imgs


def _normalize_k_low_position(position: str) -> str:
    if position is None:
        return "front"
    pos = str(position).strip()
    if not pos:
        return "front"
    pos_l = pos.lower().replace("-", "_").replace(" ", "")

    aliases = {
        "front": "front",
        "first": "front",
        "start": "front",
        "begin": "front",
        "head": "front",
        "low": "front",
        "lowfreq": "front",
        "low_freq": "front",
        "前": "front",
        "middle": "middle",
        "mid": "middle",
        "center": "middle",
        "centre": "middle",
        "中": "middle",
        "back": "back",
        "last": "back",
        "end": "back",
        "tail": "back",
        "high": "back",
        "highfreq": "back",
        "high_freq": "back",
        "后": "back",
    }

    if pos_l in aliases:
        return aliases[pos_l]
    if pos in aliases:
        return aliases[pos]
    raise ValueError(
        f"Unknown K_low_position: {position!r}. Expected one of: 'front', 'middle', 'back'."
    )


def _resolve_k_low_band_slice(K: int, K_low: int, position: str) -> slice:
    if K is None or int(K) <= 0:
        return slice(0, 0)
    K = int(K)
    K_low = int(K_low) if K_low is not None else 0
    if K_low <= 0:
        return slice(0, 0)
    if K_low >= K:
        return slice(0, K)

    pos = _normalize_k_low_position(position)
    if pos == "front":
        start = 0
    elif pos == "middle":
        start = (K - K_low) // 2
    elif pos == "back":
        start = K - K_low
    else:
        raise ValueError(f"Invalid normalized K_low_position: {pos!r}")
    return slice(start, start + K_low)


class Image3DDAPHandler(Image3DAugmentationHandler):

    def __init__(self, seed=42, aug_args=None):
        super().__init__(seed, aug_args)

        self.K = self.aug_args.get('K', 16)
        self.K_low = self.aug_args.get('K_low', 8)
        self.K_low_position = _normalize_k_low_position(self.aug_args.get('K_low_position', 'front'))
        self.gamma = self.aug_args.get('gamma', 1.5)
        self.eta = self.aug_args.get('eta', 0.01)
        self.beta = self.aug_args.get('beta', 0.7)
        self.lambda_inter = self.aug_args.get('lambda_inter', 0.4)
        self.lambda_intra = self.aug_args.get('lambda_intra', 0.6)
        self.rho_band = self.aug_args.get('rho_band', 0.3)
        self.exclude_dc = self.aug_args.get('exclude_dc', True)
        self.zeta_dc = self.aug_args.get('zeta_dc', 0.1)
        self.eps = self.aug_args.get('eps', 1e-12)
        self.rescale_to_original = self.aug_args.get('rescale_to_original', True)

        self.m = torch.zeros(self.K, dtype=torch.float64)
        self.v = torch.ones(self.K, dtype=torch.float64) * 1e-6
        self.initialized = False

        self.freq_bands_cache = {}

    def reset_state(self):
        self.m = torch.zeros(self.K, dtype=torch.float64)
        self.v = torch.ones(self.K, dtype=torch.float64) * 1e-6
        self.initialized = False

        self.freq_bands_cache = {}

    def apply_augmentation(self, imgs, args=None):
        if args is None:
            args = {}

        with torch.no_grad():
            orig_device = imgs.device
            orig_dtype = imgs.dtype

            storage_device = args.get('storage_device', 'cpu')
            compute_device = args.get('compute_device', None)

            storage_device = torch.device(storage_device)

            if compute_device is None:
                compute_device = orig_device
            else:
                compute_device = torch.device(compute_device)

            if storage_device.type == 'cuda' and storage_device.index is None:
                storage_device = torch.device(f'cuda:{torch.cuda.current_device()}')
            if compute_device.type == 'cuda' and compute_device.index is None:
                compute_device = torch.device(f'cuda:{torch.cuda.current_device()}')

            is_4d = (imgs.ndim == 4)
            if is_4d:
                D, C, H, W = imgs.shape
                if C == 1:
                    x_proc = imgs[:, 0, :, :]
                else:
                    x_proc = imgs.mean(dim=1)
            else:
                x_proc = imgs
                D, H, W = x_proc.shape

            batch_size = D
            if compute_device.type == 'cuda':
                torch.cuda.set_device(compute_device)
                free_memory = torch.cuda.mem_get_info(compute_device)[0]

                slice_memory = H * W * 8
                estimated_memory_per_slice = slice_memory * 10

                safe_memory = free_memory * 0.8
                max_batch_size = max(1, int(safe_memory / estimated_memory_per_slice))
                batch_size = min(D, max_batch_size)

            x_proc = x_proc.to(device=compute_device, dtype=torch.float64)

            masks = self._radial_bands(H, W, device=compute_device)

            if batch_size < D:
                x_aug_list = []
                all_mu_list = []
                all_var_intra_list = []

                for batch_start in range(0, D, batch_size):
                    batch_end = min(batch_start + batch_size, D)

                    x_batch = x_proc[batch_start:batch_end]

                    mu_batch, var_intra_batch = self._rapsd_stats(x_batch, masks)
                    all_mu_list.append(mu_batch)
                    all_var_intra_list.append(var_intra_batch)

                    alphas = self._sample_band_multipliers(mu_batch, var_intra_batch)

                    x_aug_batch = self._apply_in_frequency(x_batch, masks, alphas)

                    x_aug_list.append(x_aug_batch.to(dtype=orig_dtype, device=storage_device))

                    if compute_device.type == 'cuda':
                        del x_batch, x_aug_batch
                        torch.cuda.empty_cache()

                x_aug = torch.cat(x_aug_list, dim=0).to(orig_device)

                mu = torch.zeros(self.K, dtype=torch.float64, device=compute_device)
                var_intra = torch.zeros(self.K, dtype=torch.float64, device=compute_device)
                for i, (mu_batch, var_intra_batch) in enumerate(zip(all_mu_list, all_var_intra_list)):
                    batch_start = i * batch_size
                    batch_end = min(batch_start + batch_size, D)
                    batch_weight = (batch_end - batch_start) / D
                    mu += mu_batch * batch_weight
                    var_intra += var_intra_batch * batch_weight

            else:
                mu, var_intra = self._rapsd_stats(x_proc, masks)

                alphas = self._sample_band_multipliers(mu, var_intra)

                x_aug = self._apply_in_frequency(x_proc, masks, alphas)

                x_aug = x_aug.to(dtype=orig_dtype, device=orig_device)

                if compute_device.type == 'cuda' and compute_device != orig_device:
                    del x_proc
                    torch.cuda.empty_cache()

            self._update_online(mu)
            self.initialized = True

            if is_4d:
                x_aug = x_aug.unsqueeze(1).expand(-1, C, -1, -1)

            return x_aug

    def _radial_bands(self, H, W, device='cpu'):
        cache_key = (H, W, str(device))
        if cache_key in self.freq_bands_cache:
            return self.freq_bands_cache[cache_key]

        y = torch.arange(H, dtype=torch.float64, device=device)
        x = torch.arange(W, dtype=torch.float64, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        fy = (yy - H // 2) / float(H)
        fx = (xx - W // 2) / float(W)

        rho_max = math.sqrt(2) / 2
        rho = torch.sqrt(fx**2 + fy**2) / rho_max

        edges = [(b / self.K) ** self.gamma for b in range(self.K + 1)]

        masks = []
        for b in range(1, self.K + 1):
            mask = (rho >= edges[b - 1]) & (rho < edges[b])
            masks.append(mask)

        self.freq_bands_cache[cache_key] = masks

        return masks

    def _rapsd_stats(self, x, masks):
        S, H, W = x.shape
        device = x.device

        X = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))

        P = (torch.abs(X) ** 2).to(torch.float64)

        P_logs = torch.zeros((S, self.K), dtype=torch.float64, device=device)

        for b, mask in enumerate(masks):
            vals = P[:, mask]
            if vals.numel() > 0:
                P_logs[:, b] = torch.mean(torch.log(vals + self.eps), dim=1)
            else:
                P_logs[:, b] = 0.0

        mu = P_logs.mean(dim=0)

        if S > 1:
            var_intra = P_logs.var(dim=0)
        else:
            var_intra = torch.zeros_like(mu)

        return mu, var_intra

    def _sample_band_multipliers(self, mu, var_intra):
        device = mu.device

        if self.v.device != device:
            self.v = self.v.to(device)
        if self.m.device != device:
            self.m = self.m.to(device)

        s_inter = torch.sqrt(torch.maximum(self.v, torch.tensor(0.0, device=device)))
        s_intra = torch.sqrt(torch.maximum(var_intra, torch.tensor(0.0, device=device)))

        sigma = self.beta * torch.sqrt(
            (self.lambda_inter * s_inter) ** 2 + 
            (self.lambda_intra * s_intra) ** 2
        )

        sigma_np = sigma.cpu().numpy()

        z = np.zeros(self.K)
        z[0] = self.rng.standard_normal()
        for b in range(1, self.K):
            z[b] = self.rho_band * z[b - 1] + np.sqrt(1 - self.rho_band ** 2) * self.rng.standard_normal()

        z_tensor = torch.from_numpy(z).to(dtype=sigma.dtype, device=device)
        epsilons = sigma * z_tensor

        if self.exclude_dc:
            epsilons[0] *= self.zeta_dc

        alphas = torch.exp(epsilons - 0.5 * sigma ** 2)

        band_slice = _resolve_k_low_band_slice(self.K, self.K_low, self.K_low_position)
        keep = torch.zeros(self.K, dtype=torch.bool, device=device)
        keep[band_slice] = True
        alphas = torch.where(keep, alphas, torch.ones_like(alphas))

        return alphas

    def _apply_in_frequency(self, x, masks, alphas):
        S, H, W = x.shape
        device = x.device

        X = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))

        A = torch.abs(X)
        Theta = torch.angle(X)

        for b, mask in enumerate(masks):
            if alphas[b] != 1.0:
                A[:, mask] *= alphas[b]

        X_new = A * torch.exp(1j * Theta)

        y = torch.real(torch.fft.ifft2(torch.fft.ifftshift(X_new, dim=(-2, -1))))

        if self.rescale_to_original:
            mu0 = x.mean()
            std0 = x.std() + 1e-12
            mu1 = y.mean()
            std1 = y.std() + 1e-12
            y = mu0 + (std0 / std1) * (y - mu1)

        return y

    def _update_online(self, mu):
        if self.m.device != mu.device:
            self.m = self.m.to(mu.device)
        if self.v.device != mu.device:
            self.v = self.v.to(mu.device)

        delta = mu - self.m

        self.m = self.m + self.eta * delta

        self.v = (1 - self.eta) * (self.v + self.eta * (delta ** 2))


class Image3DDAPCHandler(Image3DAugmentationHandler):

    def __init__(self, seed=42, aug_args=None):
        super().__init__(seed, aug_args)

        self.K = self.aug_args.get('K', 16)
        self.K_low = self.aug_args.get('K_low', 8)
        self.K_low_position = _normalize_k_low_position(self.aug_args.get('K_low_position', 'front'))
        self.gamma = self.aug_args.get('gamma', 1.5)
        self.eta = self.aug_args.get('eta', 0.01)
        self.beta = self.aug_args.get('beta', 0.7)
        self.lambda_inter = self.aug_args.get('lambda_inter', 0.4)
        self.lambda_intra = self.aug_args.get('lambda_intra', 0.6)
        self.lambda_xchan = self.aug_args.get('lambda_xchan', 0.3)
        self.rho_band = self.aug_args.get('rho_band', 0.3)
        self.exclude_dc = self.aug_args.get('exclude_dc', True)
        self.zeta_dc = self.aug_args.get('zeta_dc', 0.1)
        self.eps = self.aug_args.get('eps', 1e-12)
        self.rescale_to_original = self.aug_args.get('rescale_to_original', True)

        self.channel_weights = self.aug_args.get('channel_weights', None)
        self.channel_mode = self.aug_args.get('channel_mode', 'shared')
        self.beta_xchan = self.aug_args.get('beta_xchan', 0.0)
        self.per_channel_rescale = self.aug_args.get('per_channel_rescale', True)
        self.sigma_max = float(self.aug_args.get('sigma_max', 0.0) or 0.0)
        self.log_alpha_clip = float(self.aug_args.get('log_alpha_clip', 0.0) or 0.0)
        self.alpha_min = float(self.aug_args.get('alpha_min', 0.0) or 0.0)
        self.alpha_max = float(self.aug_args.get('alpha_max', 0.0) or 0.0)
        self.alpha_smooth_sigma = float(self.aug_args.get('alpha_smooth_sigma', 0.0) or 0.0)
        self.alpha_taper_bands = int(self.aug_args.get('alpha_taper_bands', 0) or 0)
        self.max_slice_batch_size = int(self.aug_args.get('max_slice_batch_size', 0) or 0)
        self.fft_dtype = self._resolve_fft_dtype(self.aug_args.get('fft_dtype', 'float64'))

        self.m = torch.zeros(self.K, dtype=torch.float64)
        self.v = torch.ones(self.K, dtype=torch.float64) * 1e-6
        self.initialized = False

        self.freq_bands_cache = {}

    def reset_state(self):
        self.m = torch.zeros(self.K, dtype=torch.float64)
        self.v = torch.ones(self.K, dtype=torch.float64) * 1e-6
        self.initialized = False

        self.freq_bands_cache = {}

    def apply_augmentation(self, imgs, args=None):
        if args is None:
            args = {}

        with torch.no_grad():
            orig_device = imgs.device
            orig_dtype = imgs.dtype

            storage_device = args.get('storage_device', 'cpu')
            compute_device = args.get('compute_device', None)

            storage_device = torch.device(storage_device)

            if compute_device is None:
                compute_device = orig_device
            else:
                compute_device = torch.device(compute_device)

            if storage_device.type == 'cuda' and storage_device.index is None:
                storage_device = torch.device(f'cuda:{torch.cuda.current_device()}')
            if compute_device.type == 'cuda' and compute_device.index is None:
                compute_device = torch.device(f'cuda:{torch.cuda.current_device()}')

            if imgs.ndim != 4:
                raise ValueError(f"DAPC expects 4D input [D, C, H, W], got shape {imgs.shape}")

            D, C, H, W = imgs.shape

            batch_size = D
            if self.max_slice_batch_size > 0:
                batch_size = min(batch_size, self.max_slice_batch_size)

            imgs_on_device = imgs.to(device=compute_device, dtype=self.fft_dtype)

            masks = self._radial_bands(H, W, device=compute_device)

            if batch_size < D:
                x_aug_list = []
                all_mu_list = []
                all_var_intra_list = []
                all_var_xchan_list = []

                for batch_start in range(0, D, batch_size):
                    batch_end = min(batch_start + batch_size, D)

                    x_batch = imgs_on_device[batch_start:batch_end]

                    mu_batch, var_intra_batch, var_xchan_batch = self._rapsd_stats(x_batch, masks)
                    all_mu_list.append(mu_batch)
                    all_var_intra_list.append(var_intra_batch)
                    all_var_xchan_list.append(var_xchan_batch)

                    alphas = self._sample_band_multipliers(mu_batch, var_intra_batch, var_xchan_batch, C)

                    x_aug_batch = self._apply_in_frequency(x_batch, masks, alphas)

                    x_aug_list.append(x_aug_batch.to(dtype=orig_dtype, device=storage_device))

                x_aug = torch.cat(x_aug_list, dim=0).to(orig_device)
                del x_aug_list

                mu = torch.zeros(self.K, dtype=torch.float64, device=compute_device)
                var_intra = torch.zeros(self.K, dtype=torch.float64, device=compute_device)
                var_xchan = torch.zeros(self.K, dtype=torch.float64, device=compute_device)
                for i, (mu_batch, var_intra_batch, var_xchan_batch) in enumerate(zip(all_mu_list, all_var_intra_list, all_var_xchan_list)):
                    batch_start = i * batch_size
                    batch_end = min(batch_start + batch_size, D)
                    batch_weight = (batch_end - batch_start) / D
                    mu += mu_batch * batch_weight
                    var_intra += var_intra_batch * batch_weight
                    var_xchan += var_xchan_batch * batch_weight

            else:
                mu, var_intra, var_xchan = self._rapsd_stats(imgs_on_device, masks)

                alphas = self._sample_band_multipliers(mu, var_intra, var_xchan, C)

                x_aug = self._apply_in_frequency(imgs_on_device, masks, alphas)

                x_aug = x_aug.to(dtype=orig_dtype, device=orig_device)

            self._update_online(mu)
            self.initialized = True

            return x_aug

    def _resolve_fft_dtype(self, dtype_name):
        dtype_name = str(dtype_name).lower()
        if dtype_name in ('float32', 'fp32', 'single'):
            return torch.float32
        if dtype_name in ('float64', 'fp64', 'double'):
            return torch.float64
        raise ValueError(f"Unknown DAPC fft_dtype: {dtype_name}. Expected float32 or float64.")

    def _radial_bands(self, H, W, device='cpu'):
        cache_key = (H, W, str(device))
        if cache_key in self.freq_bands_cache:
            return self.freq_bands_cache[cache_key]

        y = torch.arange(H, dtype=torch.float64, device=device)
        x = torch.arange(W, dtype=torch.float64, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        fy = (yy - H // 2) / float(H)
        fx = (xx - W // 2) / float(W)

        rho_max = math.sqrt(2) / 2
        rho = torch.sqrt(fx**2 + fy**2) / rho_max

        edges = [(b / self.K) ** self.gamma for b in range(self.K + 1)]

        masks = []
        for b in range(1, self.K + 1):
            mask = (rho >= edges[b - 1]) & (rho < edges[b])
            masks.append(mask)

        self.freq_bands_cache[cache_key] = masks

        return masks

    def _rapsd_stats(self, x, masks):
        D, C, H, W = x.shape
        device = x.device

        if self.channel_weights is None:
            w = torch.ones(C, dtype=torch.float64, device=device) / C
        else:
            w = torch.tensor(self.channel_weights, dtype=torch.float64, device=device)
            w = w / (w.sum() + 1e-12)

        x_flat = x.reshape(D * C, H, W)

        X = torch.fft.fftshift(torch.fft.fft2(x_flat), dim=(-2, -1))

        P = (torch.abs(X) ** 2).to(torch.float64)

        P_logs = torch.zeros((D * C, self.K), dtype=torch.float64, device=device)

        for b, mask in enumerate(masks):
            vals = P[:, mask]
            if vals.numel() > 0:
                P_logs[:, b] = torch.mean(torch.log(vals + self.eps), dim=1)
            else:
                P_logs[:, b] = 0.0

        R = P_logs.reshape(D, C, self.K)

        mu_c = R.mean(dim=0)
        if D > 1:
            var_intra_c = R.var(dim=0)
        else:
            var_intra_c = torch.zeros_like(mu_c)

        mu = (w[:, None] * mu_c).sum(dim=0)
        var_intra = (w[:, None] * var_intra_c).sum(dim=0)

        mu_centered = mu_c - mu[None, :]
        var_xchan = (w[:, None] * (mu_centered ** 2)).sum(dim=0)

        return mu, var_intra, var_xchan

    def _sample_band_multipliers(self, mu, var_intra, var_xchan, C):
        device = mu.device

        if self.v.device != device:
            self.v = self.v.to(device)
        if self.m.device != device:
            self.m = self.m.to(device)

        s_inter = torch.sqrt(torch.maximum(self.v, torch.tensor(0.0, device=device)))
        s_intra = torch.sqrt(torch.maximum(var_intra, torch.tensor(0.0, device=device)))
        s_xchan = torch.sqrt(torch.maximum(var_xchan, torch.tensor(0.0, device=device)))

        sigma = self.beta * torch.sqrt(
            (self.lambda_inter * s_inter) ** 2 + 
            (self.lambda_intra * s_intra) ** 2 +
            (self.lambda_xchan * s_xchan) ** 2
        )

        if self.sigma_max > 0:
            sigma = torch.clamp(sigma, max=self.sigma_max)

        z = np.zeros(self.K)
        z[0] = self.rng.standard_normal()
        for b in range(1, self.K):
            z[b] = self.rho_band * z[b - 1] + np.sqrt(1 - self.rho_band ** 2) * self.rng.standard_normal()

        z_tensor = torch.from_numpy(z).to(dtype=sigma.dtype, device=device)
        epsilons = sigma * z_tensor

        if self.exclude_dc:
            epsilons[0] *= self.zeta_dc

        log_alphas = epsilons - 0.5 * sigma ** 2
        log_alphas = self._stabilize_log_alphas(log_alphas)
        window_weights = self._band_window_weights(device=device, dtype=log_alphas.dtype)
        log_alphas = log_alphas * window_weights
        base_alphas = torch.exp(log_alphas)

        if self.channel_mode == 'shared':
            alphas = base_alphas
        elif self.channel_mode == 'per_channel_jitter':
            tau = self.beta_xchan * s_xchan * window_weights
            tau_np = tau.cpu().numpy()

            noise = np.random.randn(C, self.K) * tau_np[None, :]
            noise_tensor = torch.from_numpy(noise).to(dtype=sigma.dtype, device=device)

            alphas = base_alphas[None, :] * torch.exp(noise_tensor - 0.5 * tau[None, :] ** 2)
            inactive = window_weights <= 0
            if inactive.any():
                alphas[:, inactive] = 1.0
        else:
            raise ValueError(f"Unknown channel_mode: {self.channel_mode}")

        return alphas

    def _band_window_weights(self, device, dtype):
        weights = torch.zeros(self.K, dtype=dtype, device=device)
        band_slice = _resolve_k_low_band_slice(self.K, self.K_low, self.K_low_position)
        start, stop = band_slice.start, band_slice.stop
        weights[band_slice] = 1.0

        taper = max(0, int(self.alpha_taper_bands))
        if taper > 0 and stop > start:
            for i in range(min(taper, stop - start)):
                ramp = float(i + 1) / float(taper + 1)
                weights[start + i] = min(float(weights[start + i].item()), ramp)
                weights[stop - 1 - i] = min(float(weights[stop - 1 - i].item()), ramp)

        return weights

    def _stabilize_log_alphas(self, log_alphas):
        if self.log_alpha_clip > 0:
            log_alphas = torch.clamp(log_alphas, -self.log_alpha_clip, self.log_alpha_clip)

        if self.alpha_smooth_sigma > 0:
            sigma = float(self.alpha_smooth_sigma)
            radius = max(1, int(math.ceil(3.0 * sigma)))
            coords = torch.arange(-radius, radius + 1, dtype=log_alphas.dtype, device=log_alphas.device)
            kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
            kernel = kernel / (kernel.sum() + 1e-12)
            padded = F.pad(log_alphas.view(1, 1, -1), (radius, radius), mode='replicate')
            log_alphas = F.conv1d(padded, kernel.view(1, 1, -1)).view(-1)

        if self.alpha_min > 0:
            log_alphas = torch.clamp(log_alphas, min=math.log(self.alpha_min))
        if self.alpha_max > 0:
            log_alphas = torch.clamp(log_alphas, max=math.log(self.alpha_max))

        return log_alphas

    def _apply_in_frequency(self, x, masks, alphas):
        D, C, H, W = x.shape
        device = x.device

        is_shared = (alphas.ndim == 1)

        x_flat = x.reshape(D * C, H, W)

        X = torch.fft.fftshift(torch.fft.fft2(x_flat), dim=(-2, -1))

        A = torch.abs(X)
        Theta = torch.angle(X)

        for b, mask in enumerate(masks):
            if is_shared:
                alpha_b = alphas[b]
                if alpha_b != 1.0:
                    A[:, mask] *= alpha_b
            else:
                for c in range(C):
                    alpha_bc = alphas[c, b]
                    if alpha_bc != 1.0:
                        slice_indices = torch.arange(c, D * C, C, device=device)
                        A[slice_indices[:, None], mask[None, :]] *= alpha_bc

        X_new = A * torch.exp(1j * Theta)

        y_flat = torch.real(torch.fft.ifft2(torch.fft.ifftshift(X_new, dim=(-2, -1))))

        y = y_flat.reshape(D, C, H, W)

        if self.rescale_to_original:
            if self.per_channel_rescale:
                for c in range(C):
                    mu0 = x[:, c].mean()
                    std0 = x[:, c].std() + 1e-12
                    mu1 = y[:, c].mean()
                    std1 = y[:, c].std() + 1e-12
                    y[:, c] = mu0 + (std0 / std1) * (y[:, c] - mu1)
            else:
                mu0 = x.mean()
                std0 = x.std() + 1e-12
                mu1 = y.mean()
                std1 = y.std() + 1e-12
                y = mu0 + (std0 / std1) * (y - mu1)

        return y

    def _update_online(self, mu):
        if self.m.device != mu.device:
            self.m = self.m.to(mu.device)
        if self.v.device != mu.device:
            self.v = self.v.to(mu.device)

        delta = mu - self.m

        self.m = self.m + self.eta * delta

        self.v = (1 - self.eta) * (self.v + self.eta * (delta ** 2))


class Image3DSPDAHandler(Image3DDAPCHandler):

    def __init__(self, seed=42, aug_args=None):
        super().__init__(seed, aug_args)

        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.norm_mode = self.aug_args.get('spda_norm_mode', 'minmax')
        self.percentile_low = float(self.aug_args.get('spda_percentile_low', 1.0))
        self.percentile_high = float(self.aug_args.get('spda_percentile_high', 99.0))
        self.percentile_sample_size = int(self.aug_args.get('spda_percentile_sample_size', 262144) or 0)
        self.softclip_temperature = float(self.aug_args.get('spda_softclip_temperature', 0.25))
        self.blend = float(self.aug_args.get('spda_blend', 1.0))

    def apply_augmentation(self, imgs, args=None):
        if args is None:
            args = {}

        orig_device = imgs.device
        orig_dtype = imgs.dtype

        x_aug = super().apply_augmentation(imgs, args)

        x_aug = self._restore_unit_range(x_aug, imgs)


        self.img_mean = self.img_mean.to(device=x_aug.device)
        self.img_std = self.img_std.to(device=x_aug.device)
        img_mean = self.img_mean.view(1, 3, 1, 1).to(dtype=x_aug.dtype)
        img_std = self.img_std.view(1, 3, 1, 1).to(dtype=x_aug.dtype)

        x_aug = (x_aug - img_mean) / img_std

        x_aug = x_aug.to(device=orig_device, dtype=orig_dtype)

        return x_aug

    def _restore_unit_range(self, x_aug, x_orig):
        mode = str(self.norm_mode).lower()
        eps = 1e-12

        if mode == 'minmax':
            D, C, H, W = x_aug.shape
            for c in range(C):
                channel_data = x_aug[:, c, :, :]
                channel_min = channel_data.min()
                channel_max = channel_data.max()
                x_aug[:, c, :, :] = (channel_data - channel_min) / (channel_max - channel_min + eps)
        elif mode == 'clip':
            x_aug = x_aug.clamp_(0.0, 1.0)
        elif mode == 'percentile':
            D, C, H, W = x_aug.shape
            q_low = max(0.0, min(100.0, self.percentile_low)) / 100.0
            q_high = max(0.0, min(100.0, self.percentile_high)) / 100.0
            if q_high <= q_low:
                q_low, q_high = 0.01, 0.99

            for c in range(C):
                channel_data = x_aug[:, c, :, :]
                flat = channel_data.reshape(-1)
                sample = flat
                if self.percentile_sample_size > 0 and flat.numel() > self.percentile_sample_size:
                    stride = int(math.ceil(flat.numel() / self.percentile_sample_size))
                    sample = flat[::stride][:self.percentile_sample_size]

                lo = torch.quantile(sample, q_low)
                hi = torch.quantile(sample, q_high)
                if (hi - lo).abs() <= eps:
                    x_aug[:, c, :, :] = torch.clamp(channel_data, 0.0, 1.0)
                else:
                    channel_data = torch.clamp(channel_data, lo, hi)
                    x_aug[:, c, :, :] = (channel_data - lo) / (hi - lo + eps)
        elif mode == 'softclip':
            temperature = max(self.softclip_temperature, 1e-6)
            x_aug = 0.5 + 0.5 * torch.tanh((x_aug - 0.5) / temperature)
        elif mode == 'none':
            pass
        else:
            raise ValueError(
                f"Unknown spda_norm_mode: {self.norm_mode}. "
                "Expected one of: minmax, clip, percentile, softclip, none."
            )

        if self.blend < 1.0:
            blend = max(0.0, self.blend)
            x_aug = (1.0 - blend) * x_orig.to(device=x_aug.device, dtype=x_aug.dtype) + blend * x_aug

        return x_aug.clamp_(0.0, 1.0)


def build_augmentation_handler(config):
    handler_type = config.get('type', 'weak')
    seed = config.get('seed', 42)
    aug_args = config.get('aug_args', {})

    handler_map = {
        'weak': WeakAugmentationHandler,
        'freq': FrequencyAugmentationHandler,
        'spda': Image3DSPDAHandler,
    }

    if handler_type not in handler_map:
        raise ValueError(f"Unknown augmentation handler type: {handler_type}. "
                        f"Valid types are: {list(handler_map.keys())}")

    handler_class = handler_map[handler_type]
    return handler_class(seed=seed, aug_args=aug_args)
