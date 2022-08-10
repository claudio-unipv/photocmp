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


def _ssim_window(image, window=8):
    """Extract the windows for SSIM computation."""
    # HxWxC  ==>  H
    if image.ndim != 3:
        image = image.reshape(image.shape[0], image.shape[1], -1)
    h, w, c = image.shape
    hb = h // window
    wb = w // window
    if h % window != 0 or w % window != 0:
        # Make sure that the image dimensions are multiples of the window
        image = image[:hb * window, :wb * window, :]
    windows = image.reshape(hb, window, wb, window, -1)
    windows = windows.transpose(0, 2, 4, 1, 3)
    return windows.reshape(-1, window * window)


def ssim(image1, image2, window=8):
    """Structural Simularity Index Measure (SSIM)."""
    k1 = 0.01
    k2 = 0.03
    L = 1
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    win1 = _ssim_window(image1)
    win2 = _ssim_window(image2)
    mu1 = win1.mean(1)
    mu2 = win2.mean(1)
    sigma1 = win1.var(1)
    sigma2 = win2.var(1)
    cov = (win1.astype(float) * win2).mean(1) - mu1 * mu2
    num = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2)
    return (num / den).mean()


def _test():
    image1 = (np.arange(1800) % 256).reshape(30, 20, 3).astype(np.uint8)
    image2 = image1[::-1, :, :]
    print(mse(image1, image2))
    print(psnr(image1, image2))
    print(ssim(image1, image2))


if __name__ == "__main__":
    _test()
