import numpy as np
from numba import njit, prange

# 利用最近邻滤波NNF对彩色图像进行resize
@njit("void(f4[:, :, :], f4[:, :, :])", cache=True, nogil=True, parallel=True)
def _resize_nearest_multichannel(dst, src):
    h_src, w_src, depth = src.shape
    h_dst, w_dst, depth = dst.shape
    for y_dst in prange(h_dst):
        for x_dst in range(w_dst):
            x_src = max(0, min(w_src - 1, x_dst * w_src // w_dst))
            y_src = max(0, min(h_src - 1, y_dst * h_src // h_dst))
            for c in range(depth):
                dst[y_dst, x_dst, c] = src[y_src, x_src, c]

# 利用最近邻滤波NNF对灰色图像进行resize
@njit("void(f4[:, :], f4[:, :])", cache=True, nogil=True, parallel=True)
def _resize_nearest(dst, src):
    h_src, w_src = src.shape
    h_dst, w_dst = dst.shape
    for y_dst in prange(h_dst):
        for x_dst in range(w_dst):
            x_src = max(0, min(w_src - 1, x_dst * w_src // w_dst))
            y_src = max(0, min(h_src - 1, y_dst * h_src // h_dst))
            dst[y_dst, x_dst] = src[y_src, x_src]

# TODO
# There should be an option to switch @njit(parallel=True) on or off.
# parallel=True would be faster, but might cause race conditions.
# User should have the option to turn it on or off.
@njit("Tuple((f4[:, :, :], f4[:, :, :]))(f4[:, :, :], f4[:, :], f4, i4, i4, i4, f4)", cache=True, nogil=True)
def _estimate_fb_ml(
    input_image,
    input_alpha,
    regularization,
    n_small_iterations,
    n_big_iterations,
    small_size,
    gradient_weight,
):
    h0, w0, depth = input_image.shape
    dtype = np.float32
    # 初始化
    h_prev, w_prev = 1, 1
    F_prev = np.empty((h_prev, w_prev, depth), dtype=dtype)
    B_prev = np.empty((h_prev, w_prev, depth), dtype=dtype)
    n_levels = int(np.ceil(np.log2(max(w0, h0))))
    for i_level in range(n_levels + 1):
        w = round(w0 ** (i_level / n_levels))
        h = round(h0 ** (i_level / n_levels))
        image = np.empty((h, w, depth), dtype=dtype)
        alpha = np.empty((h, w), dtype=dtype)
        # 将彩色图image改变大小为input_image的尺寸
        _resize_nearest_multichannel(image, input_image)
        # 将灰色图alpha改变大小为input_alpha的尺寸
        _resize_nearest(alpha, input_alpha)
        F = np.empty((h, w, depth), dtype=dtype)
        B = np.empty((h, w, depth), dtype=dtype)
        # 将彩色图F改变大小为F_prev的尺寸
        _resize_nearest_multichannel(F, F_prev)
        # 将彩色图B改变大小为B_prev的尺寸
        _resize_nearest_multichannel(B, B_prev)
        if w <= small_size and h <= small_size:
            n_iter = n_small_iterations
        else:
            n_iter = n_big_iterations
        b = np.zeros((2, depth), dtype=dtype)
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        # 迭代
        for i_iter in range(n_iter):
            for y in prange(h):
                for x in range(w):
                    a0 = alpha[y, x]
                    a1 = 1.0 - a0
                    a00 = a0 * a0
                    a01 = a0 * a1
                    # 由于矩阵的对称性，可以省略a10 = a01
                    a11 = a1 * a1
                    for c in range(depth):
                        b[0, c] = a0 * image[y, x, c]
                        b[1, c] = a1 * image[y, x, c]
                    for d in range(4):
                        x2 = max(0, min(w - 1, x + dx[d]))
                        y2 = max(0, min(h - 1, y + dy[d]))
                        gradient = abs(a0 - alpha[y2, x2])
                        da = regularization + gradient_weight * gradient
                        a00 += da
                        a11 += da
                        for c in range(depth):
                            b[0, c] += da * F[y2, x2, c]
                            b[1, c] += da * B[y2, x2, c]
                    determinant = a00 * a11 - a01 * a01
                    inv_det = 1.0 / determinant
                    b00 = inv_det * a11
                    b01 = inv_det * -a01
                    b11 = inv_det * a00
                    for c in range(depth):
                        F_c = b00 * b[0, c] + b01 * b[1, c]
                        B_c = b01 * b[0, c] + b11 * b[1, c]
                        F_c = max(0.0, min(1.0, F_c))
                        B_c = max(0.0, min(1.0, B_c))
                        F[y, x, c] = F_c
                        B[y, x, c] = B_c
    return F, B


def estimate_foreground_ml(
    image,
    alpha,
    regularization=1e-5,
    n_small_iterations=10,
    n_big_iterations=2,
    small_size=32,
    return_background=False,
    gradient_weight=1.0,
):
    foreground, background = _estimate_fb_ml(
        image.astype(np.float32),
        alpha.astype(np.float32),
        regularization,
        n_small_iterations,
        n_big_iterations,
        small_size,
        gradient_weight,
    )
    if return_background:
        return foreground, background
    return foreground
