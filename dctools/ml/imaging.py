import numpy as np
from PIL import Image


__all__ = ["img_to_numpy", "batch_img_to_numpy"]


def img_to_numpy(img_path: str, size: tuple = (100,100), mode: str ="RGB"):
    """Converts an image to a numpy array of pixels with given size"""
    
    return np.array(Image.open(img_path).convert("RGB").resize(size, resample=Image.Resampling.BILINEAR))

def batch_img_to_numpy(fnames: list[str], dir: str = None, size: tuple = (100,100), mode:str = "RGB"):
    """Converts a batch of images to numpy arrays of equal, given size"""
    
    imgs = np.array([])
    if dir is not None:
        for f in fnames:
            imgs = np.append(imgs, img_to_numpy(f"{dir}\f", size, mode))
    else:
        for f in fnames:
            imgs = np.append(imgs, img_to_numpy(f, size, mode))