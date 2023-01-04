from laplacian.laplacian import make_linear_system
from laplacian.knn_laplacian import knn_laplacian
from preconditioner.jacobi import jacobi
from core.cg import cg
import numpy as np


def estimate_alpha_knn(image, trimap, preconditioner=jacobi, laplacian_kwargs={}, cg_kwargs={}):
    # 确定预调节器为jacobi
    # 
    A, b = make_linear_system(knn_laplacian(image, **laplacian_kwargs), trimap)
    # 共轭梯度法求解线性方程组A * x = b
    x = cg(A, b, M=preconditioner(A), **cg_kwargs)
    # 约束alpha矩阵数值在[0,1]之间
    alpha = np.clip(x, 0, 1).reshape(trimap.shape)
    return alpha
