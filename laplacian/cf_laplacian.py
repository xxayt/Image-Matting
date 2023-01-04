import numpy as np
import scipy.sparse
from numba import njit

def _cf_laplacian(image, epsilon, r, values, indices, indptr, is_known):
    h, w, d = image.shape
    assert d == 3
    size = 2 * r + 1
    window_area = size * size
    for yi in range(h):
        for xi in range(w):
            i = xi + yi * w
            k = i * (4 * r + 1) ** 2
            for yj in range(yi - 2 * r, yi + 2 * r + 1):
                for xj in range(xi - 2 * r, xi + 2 * r + 1):
                    j = xj + yj * w
                    if 0 <= xj < w and 0 <= yj < h:
                        indices[k] = j
                    k += 1
            indptr[i + 1] = k
    # 居中并归一化窗口颜色
    c = np.zeros((2 * r + 1, 2 * r + 1, 3))
    # 对图片的每个像素
    for y in range(r, h - r):
        for x in range(r, w - r):
            if np.all(is_known[y - r : y + r + 1, x - r : x + r + 1]):
                continue
            # 对每个颜色通道
            for dc in range(3):
                # 计算窗口中颜色通道的总和
                s = 0.0
                for dy in range(size):
                    for dx in range(size):
                        s += image[y + dy - r, x + dx - r, dc]
                # 计算居中的窗口颜色
                for dy in range(2 * r + 1):
                    for dx in range(2 * r + 1):
                        c[dy, dx, dc] = (
                            image[y + dy - r, x + dx - r, dc] - s / window_area
                        )
            # 通过正则化计算颜色通道上的协方差矩阵
            a00, a01, a02 = epsilon, 0.0, 0.0
            a11, a12, a22 = epsilon, 0.0, epsilon
            for dy in range(size):
                for dx in range(size):
                    a00 += c[dy, dx, 0] * c[dy, dx, 0]
                    a01 += c[dy, dx, 0] * c[dy, dx, 1]
                    a02 += c[dy, dx, 0] * c[dy, dx, 2]
                    a11 += c[dy, dx, 1] * c[dy, dx, 1]
                    a12 += c[dy, dx, 1] * c[dy, dx, 2]
                    a22 += c[dy, dx, 2] * c[dy, dx, 2]
            a00 /= window_area
            a01 /= window_area
            a02 /= window_area
            a11 /= window_area
            a12 /= window_area
            a22 /= window_area
            # 行列式
            det = (a00*a12*a12 + a01*a01*a22 + a02*a02*a11 - a00*a11*a22 - 2 * a01*a02*a12)
            inv_det = 1.0 / det
            # 计算协方差矩阵
            m00 = (a12 * a12 - a11 * a22) * inv_det
            m01 = (a01 * a22 - a02 * a12) * inv_det
            m02 = (a02 * a11 - a01 * a12) * inv_det
            m11 = (a02 * a02 - a00 * a22) * inv_det
            m12 = (a00 * a12 - a01 * a02) * inv_det
            m22 = (a01 * a01 - a00 * a11) * inv_det
            # 对在(2r + 1)*(2r + 1)窗口内的每个像素对((xi, yi), (xj, yj))
            for dyi in range(2 * r + 1):
                for dxi in range(2 * r + 1):
                    s = c[dyi, dxi, 0]
                    t = c[dyi, dxi, 1]
                    u = c[dyi, dxi, 2]
                    c0 = m00 * s + m01 * t + m02 * u
                    c1 = m01 * s + m11 * t + m12 * u
                    c2 = m02 * s + m12 * t + m22 * u
                    for dyj in range(2 * r + 1):
                        for dxj in range(2 * r + 1):
                            xi = x + dxi - r
                            yi = y + dyi - r
                            xj = x + dxj - r
                            yj = y + dyj - r
                            i = xi + yi * w
                            j = xj + yj * w
                            # 计算对像素对每个L_ij贡献
                            temp = (c0*c[dyj, dxj, 0] + c1*c[dyj, dxj, 1] + c2*c[dyj, dxj, 2])
                            value = (1.0 if (i == j) else 0.0) - (1 + temp) / window_area
                            dx = xj - xi + 2 * r
                            dy = yj - yi + 2 * r
                            values[i, dy, dx] += value

def cf_laplacian(image, epsilon=1e-7, radius=1, is_known=None):
    h, w, d = image.shape
    n = h * w
    if is_known is None:
        is_known = np.zeros((h, w), dtype=np.bool8)
    is_known = is_known.reshape(h, w)
    # Data for matting laplacian in csr format
    # 游标指针
    indptr = np.zeros(n + 1, dtype=np.int64)
    # 列索引
    indices = np.zeros(n * (4 * radius + 1) ** 2, dtype=np.int64)
    values = np.zeros((n, 4 * radius + 1, 4 * radius + 1), dtype=np.float64)
    # 详细计算
    _cf_laplacian(image, epsilon, radius, values, indices, indptr, is_known)
    # 列稀疏矩阵
    L = scipy.sparse.csr_matrix((values.ravel(), indices, indptr), (n, n))
    return L
