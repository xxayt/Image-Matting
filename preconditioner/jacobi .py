# 对于矩阵A计算Jacobi预处理
def jacobi(A):
    diagonal = A.diagonal()
    inverse_diagonal = 1.0 / diagonal
    return x * inverse_diagonal
