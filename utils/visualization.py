import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import tqdm


def show_points(coords, labels, ax, marker_size=20):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
                  s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
                  s=marker_size, edgecolor='white', linewidth=1.25)


def show_mask(mask, ax, mask_color=None, alpha=0.5):
    mask_binary = (mask > 0).astype(np.float32)

    if mask_color is not None:
        mask_color = np.clip(mask_color, 0, 1)
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])

    h, w = mask_binary.shape[-2:]
    mask_image = mask_binary.reshape(h, w, 1) * color.reshape(1, 1, -1)

    mask_image = np.clip(mask_image, 0, 1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))


def visualize_sample(raw_img, gt_mask, pred_mask, bbox, dsc_score, save_path, is_3d=False, slice_idx=None, 
                     prompt_type='box', points=None, point_labels=None):
    plt.style.use('default')

    if is_3d and slice_idx is not None:
        img_slice = raw_img[slice_idx]
        gt_slice = gt_mask[slice_idx]
        pred_slice = pred_mask[slice_idx]
        bbox_2d = bbox[:4] if bbox is not None and len(bbox) == 6 else bbox
    else:
        img_slice = raw_img
        gt_slice = gt_mask
        pred_slice = pred_mask
        bbox_2d = bbox[:4] if bbox is not None and len(bbox) == 6 else bbox

    img_slice_norm = img_slice.astype(np.float32)
    img_min, img_max = img_slice_norm.min(), img_slice_norm.max()
    if img_max > img_min:
        img_slice_norm = (img_slice_norm - img_min) / (img_max - img_min)
    else:
        img_slice_norm = np.zeros_like(img_slice_norm)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    prompt_name = 'Box' if prompt_type == 'box' else f'Points({len(points) if points is not None else 0})'
    fig.suptitle(f'DSC Score: {dsc_score:.4f} | Prompt: {prompt_name}', fontsize=16)

    axes[0, 0].imshow(img_slice_norm, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_slice_norm, cmap='gray', vmin=0, vmax=1)
    if prompt_type == 'box' and bbox_2d is not None:
        show_box(bbox_2d, axes[0, 1], edgecolor='red')
        axes[0, 1].set_title('Image + Bounding Box')
    elif prompt_type == 'point' and points is not None and len(points) > 0:
        show_points(points, point_labels, axes[0, 1], marker_size=100)
        axes[0, 1].set_title(f'Image + Points ({len(points)})')
    else:
        axes[0, 1].set_title('Image + Prompt')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(img_slice_norm, cmap='gray', vmin=0, vmax=1)
    show_mask(gt_slice, axes[1, 0], mask_color=np.array([0, 1, 0]))
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img_slice_norm, cmap='gray', vmin=0, vmax=1)
    show_mask(pred_slice, axes[1, 1], mask_color=np.array([1, 0, 0]))
    axes[1, 1].set_title('Prediction')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_best_worst_visualizations(dir_results, output_dir, data_dir, prompt_type, 
                                   num_points=1, point_strategy='random', debug=True, logger=None):
    from .bbox_utils import generate_bbox
    from .point_sampling import sample_points_from_mask

    if not debug:
        return

    visualization_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)

    logger.info("\n🎨 Generating visualizations for best and worst predictions...")

    total_tasks = 0
    for dir_path, results in dir_results.items():
        if results:
            sorted_results = sorted(results, key=lambda x: x['dsc'])
            worst_count = min(2, len(sorted_results))
            best_count = min(2, len(sorted_results))
            total_tasks += worst_count + best_count

    progress_bar = tqdm.tqdm(total=total_tasks, desc="Generating visualizations", unit="viz")

    for dir_path, results in dir_results.items():
        if not results:
            continue

        sorted_results = sorted(results, key=lambda x: x['dsc'])

        worst_results = sorted_results[:2]
        best_results = sorted_results[-2:]

        rel_dir = os.path.relpath(dir_path, data_dir)
        dir_viz_path = os.path.join(visualization_dir, rel_dir)
        os.makedirs(dir_viz_path, exist_ok=True)

        if logger:
            logger.debug(f"\n🔄 Processing visualizations for directory: {rel_dir}")

        for i, result in enumerate(worst_results):
            file_path = result['file_path']
            dsc_score = result['dsc']
            filename = os.path.basename(file_path).replace('.npz', '')

            progress_bar.set_description(f"Dir: {rel_dir} | Worst {i+1}")

            data = np.load(file_path, 'r', allow_pickle=True)
            raw_imgs = data['imgs']
            gts = data['gts']

            rel_save_path = os.path.relpath(file_path, data_dir)
            pred_path = os.path.join(output_dir, "predictions", rel_save_path)
            if os.path.exists(pred_path):
                pred_data = np.load(pred_path, 'r', allow_pickle=True)
                pred_segs = pred_data['segs']
            else:
                if logger:
                    logger.warning(f"Prediction file not found: {pred_path}")
                progress_bar.update(1)
                continue

            if gts.ndim == 3:
                slice_pixel_counts = np.sum(gts > 0, axis=(1, 2))
                key_slice_idx = np.argmax(slice_pixel_counts)
                key_slice_gt = gts[key_slice_idx]

                if prompt_type == 'box':
                    bbox = generate_bbox(key_slice_gt, max_shift=20)
                    points, point_labels = None, None
                else:
                    points, point_labels = sample_points_from_mask(
                        key_slice_gt,
                        num_points=num_points,
                        strategy=point_strategy
                    )
                    bbox = None
            else:
                if prompt_type == 'box':
                    bbox = generate_bbox(gts, max_shift=20)
                    points, point_labels = None, None
                else:
                    points, point_labels = sample_points_from_mask(
                        gts,
                        num_points=num_points,
                        strategy=point_strategy
                    )
                    bbox = None
                key_slice_idx = None

            save_path = os.path.join(dir_viz_path, f"worst_{i+1}_{filename}_dsc{dsc_score:.4f}.png")

            if gts.ndim == 3:
                visualize_sample(raw_imgs, gts, pred_segs, bbox, dsc_score, save_path,
                               is_3d=True, slice_idx=key_slice_idx,
                               prompt_type=prompt_type, points=points, point_labels=point_labels)
            else:
                visualize_sample(raw_imgs, gts, pred_segs, bbox, dsc_score, save_path, is_3d=False,
                               prompt_type=prompt_type, points=points, point_labels=point_labels)

            progress_bar.set_postfix(DSC=f"{dsc_score:.4f}")
            if logger:
                logger.debug(f"  💾 Saved worst sample {i+1}: {filename} (DSC={dsc_score:.4f})")

            progress_bar.update(1)

        for i, result in enumerate(best_results):
            file_path = result['file_path']
            dsc_score = result['dsc']
            filename = os.path.basename(file_path).replace('.npz', '')

            progress_bar.set_description(f"Dir: {rel_dir} | Best {i+1}")

            data = np.load(file_path, 'r', allow_pickle=True)
            raw_imgs = data['imgs']
            gts = data['gts']

            rel_save_path = os.path.relpath(file_path, data_dir)
            pred_path = os.path.join(output_dir, "predictions", rel_save_path)
            if os.path.exists(pred_path):
                pred_data = np.load(pred_path, 'r', allow_pickle=True)
                pred_segs = pred_data['segs']
            else:
                if logger:
                    logger.warning(f"Prediction file not found: {pred_path}")
                progress_bar.update(1)
                continue

            if gts.ndim == 3:
                slice_pixel_counts = np.sum(gts > 0, axis=(1, 2))
                key_slice_idx = np.argmax(slice_pixel_counts)
                key_slice_gt = gts[key_slice_idx]

                if prompt_type == 'box':
                    bbox = generate_bbox(key_slice_gt, max_shift=20)
                    points, point_labels = None, None
                else:
                    points, point_labels = sample_points_from_mask(
                        key_slice_gt,
                        num_points=num_points,
                        strategy=point_strategy
                    )
                    bbox = None
            else:
                if prompt_type == 'box':
                    bbox = generate_bbox(gts, max_shift=20)
                    points, point_labels = None, None
                else:
                    points, point_labels = sample_points_from_mask(
                        gts,
                        num_points=num_points,
                        strategy=point_strategy
                    )
                    bbox = None
                key_slice_idx = None

            save_path = os.path.join(dir_viz_path, f"best_{i+1}_{filename}_dsc{dsc_score:.4f}.png")

            if gts.ndim == 3:
                visualize_sample(raw_imgs, gts, pred_segs, bbox, dsc_score, save_path,
                               is_3d=True, slice_idx=key_slice_idx,
                               prompt_type=prompt_type, points=points, point_labels=point_labels)
            else:
                visualize_sample(raw_imgs, gts, pred_segs, bbox, dsc_score, save_path, is_3d=False,
                               prompt_type=prompt_type, points=points, point_labels=point_labels)

            progress_bar.set_postfix(DSC=f"{dsc_score:.4f}")
            if logger:
                logger.debug(f"  💾 Saved best sample {i+1}: {filename} (DSC={dsc_score:.4f})")

            progress_bar.update(1)

        if logger:
            logger.debug(f"  ✅ Completed visualizations for directory: {rel_dir}")

    progress_bar.close()
    if logger:
        logger.info(f"\n🎉 Generated {total_tasks} visualizations across {len(dir_results)} directories")
