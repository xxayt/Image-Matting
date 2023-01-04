from PIL import Image
import numpy as np
import scipy.sparse
import os
import warnings
from functools import wraps

def isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def _resize_pil_image(image, size, resample="bicubic"):
    filters = {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "box": Image.BOX,
        "hamming": Image.HAMMING,
        "lanczos": Image.LANCZOS,
        "nearest": Image.NEAREST,
        "none": Image.NEAREST,
    }
    if not isiterable(size):
        size = (int(image.width * size), int(image.height * size))
    image = image.resize(size, filters[resample.lower()])
    return image

# 载入图片
def load_image(path, mode=None, size=None, resample="box"):
    image = Image.open(path)
    if mode is not None:
        mode = mode.upper()
        mode = "L" if mode == "GRAY" else mode
        image = image.convert(mode)
    if size is not None:
        image = _resize_pil_image(image, size, resample)
    image = np.array(image) / 255.0
    return image

# 加载识别trimap
def trimap_split(trimap, flatten=True, bg_threshold=0.1, fg_threshold=0.9):
    # 降维至1维
    if flatten:
        trimap = trimap.flatten()
    '''
    # 警告监测
    min_value = trimap.min()
    max_value = trimap.max()
    if min_value < 0.0:
        warnings.warn(
            "Trimap values should be in [0, 1], but trimap.min() is %s." % min_value,
            stacklevel=3,
        )
    if max_value > 1.0:
        warnings.warn(
            "Trimap values should be in [0, 1], but trimap.max() is %s." % min_value,
            stacklevel=3,
        )
    if trimap.dtype not in [np.float32, np.float64]:
        warnings.warn(
            "Unexpected trimap.dtype %s. Are you sure that you do not want to use np.float32 or np.float64 instead?"
            % trimap.dtype,
            stacklevel=3,
        )
    '''
    is_foreground = trimap >= fg_threshold
    is_background = trimap <= bg_threshold
    '''
    # 异常监测
    if is_background.sum() == 0:
        raise ValueError(
            "Trimap did not contain background values (values <= %f)" % bg_threshold
        )
    if is_foreground.sum() == 0:
        raise ValueError(
            "Trimap did not contain foreground values (values >= %f)" % fg_threshold
        )
    '''
    is_known = is_foreground | is_background
    is_unknown = ~is_known
    # 返回前景位置、背景位置、确定前后景位置、模糊位置
    return is_foreground, is_background, is_known, is_unknown

# 针对三维数组，在第三个维度进行拼接！
def stack_images(*images):
    images = [
        (image if len(image.shape) == 3 else image[:, :, np.newaxis]) for image in images
    ]
    return np.concatenate(images, axis=2)

# 保存图片
def save_image(path, image, make_directory=True):
    if image.dtype in [np.float32, np.float64]:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    if make_directory:
        directory, _ = os.path.split(path)
        if len(directory) > 0:
            os.makedirs(directory, exist_ok=True)
    Image.fromarray(image).save(path)


def blend(foreground, background, alpha):
    if len(alpha.shape) == 2:
        alpha = alpha[:, :, np.newaxis]
    return alpha * foreground + (1 - alpha) * background

# 合并多个图片到grid画布！
def make_grid(images, nx, ny, dtype=None):
    # 计算总画布大小尺寸
    shapes = [image.shape for image in images if image is not None]
    h = max(shape[0] for shape in shapes)
    w = max(shape[1] for shape in shapes)
    d = max([shape[2] for shape in shapes if len(shape) > 2], default=1)
    # 使通道数(d)相同
    for i, image in enumerate(images):
        if image is not None:
            # 若是灰度图，则添加第三维度(通道维度)
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
            # 灰度图：在多通道(d)的每个通道上复制一遍灰度图
            if image.shape[2] == 1:
                image = np.concatenate([image] * d, axis=2)
            # 彩色图：若有四通道图片(png图片有alpha透明度通道)，则三通道彩色图片需加一通道
            if image.shape[2] == 3 and d == 4:
                image = stack_images(image, np.ones(image.shape[:2], dtype=image.dtype))
            images[i] = image
    if dtype is None:
        dtype = next(image.dtype for image in images if image is not None)
    # 设置总画布
    result = np.zeros((h * ny, w * nx, d), dtype=dtype)
    # 分别各图片image加入总画布
    for y in range(ny):
        for x in range(nx):
            i = x + y * nx
            if i >= len(images):
                break
            image = images[i]
            if image is not None:
                image = image.reshape(image.shape[0], image.shape[1], -1)
                # 对应位置加入图片像素
                result[y * h : y * h + image.shape[0], x * w : x * w + image.shape[1]] = image
    return result