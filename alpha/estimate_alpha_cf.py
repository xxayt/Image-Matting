from core.util import trimap_split
from laplacian.cf_laplacian import cf_laplacian
from preconditioner.ichol import ichol
from core.cg import cg
import numpy as np


def estimate_alpha_cf(image, trimap, preconditioner=ichol, laplacian_kwargs={}, cg_kwargs={}):
    # 确定预调节器为ichol
    # 加载识别trimap图像
    is_foreground, is_background, is_known, is_unknown = trimap_split(trimap)
    # 计算拉普拉斯矩阵作为系数
    L = cf_laplacian(image, **laplacian_kwargs, is_known=is_known)
    L_U = L[is_unknown, :][:, is_unknown]
    R = L[is_unknown, :][:, is_known]
    m = is_foreground[is_known]
    # 对trimap副本降维至一维
    x = trimap.copy().flatten()
    # 共轭梯度法求解线性方程组L_U * x = -R.dot(m)
    x[is_unknown] = cg(L_U, -R.dot(m), M=preconditioner(L_U), **cg_kwargs)
    # 约束alpha矩阵数值在[0,1]之间
    alpha = np.clip(x, 0, 1).reshape(trimap.shape)
    return alpha
