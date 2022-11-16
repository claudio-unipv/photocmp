import numpy as np


def mse(image1, image2):
    """Mean Squared Error (MSE).

    Note: the function assumes that the depth of both images is 8
    bits.  The metric is scaled as if input values were in the [0, 1]
    range.

    """
    val = ((image1.astype(np.int32) - image2) ** 2).mean()
    return val / (255 * 255)


def psnr(image1, image2):
    """Peak Signal to Noise Ratio (PSNR)."""
    mse_val = mse(image1, image2)
    if mse_val > 0:
        return -10 * np.log10(mse_val)
    else:
        return np.inf


def stretch(image, eps=1e-6):
    """Linear scaling of the color values to the [0, 1] range."""
    num = image - image.min(axis=(0, 1), keepdims=True)
    den = np.ptp(image, axis=(0, 1), keepdims=True)
    den = np.clip(den, eps, None)
    return num / den


def normalize(image, eps=1e-6):
    """Mean-variance normalization."""
    num = image - image.mean(axis=(0, 1), keepdims=True)
    den = np.std(image, axis=(0, 1), keepdims=True)
    den = np.clip(den, eps, None)
    return num / den
    
