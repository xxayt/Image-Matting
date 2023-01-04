from core.util import trimap_split
import scipy.sparse

# 构造A = L + \lambda C,
def make_linear_system(L, trimap, lambda_value=100.0, return_c=False):
    # 加载识别trimap图像
    is_foreground, is_background, is_known, is_unknown = trimap_split(trimap)
    # c向量代表已知区域
    c = lambda_value * is_known
    # b向量代表前景区域
    b = lambda_value * is_foreground
    # 从对角线构造稀疏矩阵C
    C = scipy.sparse.diags(c)
    # 得到线性方程组系数A矩阵
    A = (L + C).tocsr()
    if return_c:
        return A, b, c
    return A, b
