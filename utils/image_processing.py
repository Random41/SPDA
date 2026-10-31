import numpy as np
import torch
from PIL import Image
import skimage


def resize_grayscale_to_rgb_and_resize(array, image_size):
    if array.ndim == 2:
        h, w = array.shape
        img_pil = Image.fromarray(array.astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)
        return img_array
    elif array.ndim == 3:
        if array.shape[2] == 3:
            img_pil = Image.fromarray(array.astype(np.uint8))
            img_resized = img_pil.resize((image_size, image_size))
            img_array = np.array(img_resized).transpose(2, 0, 1)
            return img_array
        else:
            d, h, w = array.shape
            resized_array = np.zeros((d, 3, image_size, image_size))

            for i in range(d):
                img_pil = Image.fromarray(array[i].astype(np.uint8))
                img_rgb = img_pil.convert("RGB")
                img_resized = img_rgb.resize((image_size, image_size))
                img_array = np.array(img_resized).transpose(2, 0, 1)
                resized_array[i] = img_array

            return resized_array
    else:
        raise ValueError("Input array must be 2D or 3D")


img_mean = (0.485, 0.456, 0.406)
img_std = (0.229, 0.224, 0.225)


def preprocess(img, device):
    img = resize_grayscale_to_rgb_and_resize(img, image_size=512)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).to(device)
    img = (img - torch.tensor(img_mean).view(3, 1, 1).to(device)) / torch.tensor(img_std).view(3, 1, 1).to(device)
    return img


def resize_grayscale_to_rgb_and_resize_fine(array, image_size):
    from skimage.transform import resize as skimage_resize

    array_min = array.min()
    array_max = array.max()

    if array_max - array_min > 1e-8:
        array_normalized = (array - array_min) / (array_max - array_min)
    else:
        array_normalized = np.zeros_like(array, dtype=np.float32)

    if array.ndim == 2:
        img_resized = skimage_resize(
            array_normalized, 
            (image_size, image_size), 
            mode='reflect', 
            anti_aliasing=True, 
            preserve_range=False
        )
        img_rgb = np.stack([img_resized, img_resized, img_resized], axis=0).astype(np.float32)
        return img_rgb

    elif array.ndim == 3:
        if array.shape[2] == 3:
            img_resized = skimage_resize(
                array_normalized, 
                (image_size, image_size), 
                mode='reflect', 
                anti_aliasing=True, 
                preserve_range=False
            )
            img_array = img_resized.transpose(2, 0, 1).astype(np.float32)
            return img_array
        else:
            d, h, w = array.shape
            resized_array = np.zeros((d, 3, image_size, image_size), dtype=np.float32)

            for i in range(d):
                img_resized = skimage_resize(
                    array_normalized[i], 
                    (image_size, image_size), 
                    mode='reflect', 
                    anti_aliasing=True, 
                    preserve_range=False
                )
                img_rgb = np.stack([img_resized, img_resized, img_resized], axis=0).astype(np.float32)
                resized_array[i] = img_rgb

            return resized_array
    else:
        raise ValueError("Input array must be 2D or 3D")


def preprocess_fine(img, device):

    img_resized = resize_grayscale_to_rgb_and_resize_fine(img, image_size=512)

    img_tensor = torch.from_numpy(img_resized).to(device)

    img_normalized = (img_tensor - torch.tensor(img_mean).view(3, 1, 1).to(device)) / torch.tensor(img_std).view(3, 1, 1).to(device)

    return img_normalized


def preprocess_fine_normalized(img, device):

    img_resized = resize_grayscale_to_rgb_and_resize_fine(img, image_size=512)

    img_tensor = torch.from_numpy(img_resized).to(device)


    return img_tensor


def getLargestCC(segmentation):
    labels = skimage.measure.label(segmentation)
    if labels.max() == 0:
        return segmentation.astype(bool)

    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC.astype(bool)
