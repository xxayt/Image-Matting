import numpy as np

# 共轭梯度法(Conjugate Gradient, CG)求解线性方程组
def Conjugate_Gradient(
    A,
    b,
    x0=None,
    atol=0.0,
    rtol=1e-7,
    maxiter=10000,
    callback=None,
    M=None,
    reorthogonalize=False,
):
    if M is None:
        def precondition(x):
            return x
    elif callable(M):
        precondition = M
    else:
        def precondition(x):
            return M.dot(x)

    x = np.zeros_like(b) if x0 is None else x0.copy()
    norm_b = np.linalg.norm(b)
    if callable(A):
        r = b - A(x)
    else:
        r = b - A.dot(x)
    norm_r = np.linalg.norm(r)
    if norm_r < atol or norm_r < rtol * norm_b:
        return x
    z = precondition(r)
    p = z.copy()
    rz = np.inner(r, z)

    for iteration in range(maxiter):
        r_old = r.copy()
        if callable(A):
            Ap = A(p)
        else:
            Ap = A.dot(p)
        alpha = rz / np.inner(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        norm_r = np.linalg.norm(r)
        if callback is not None:
            callback(A, x, b, norm_b, r, norm_r)
        if norm_r < atol or norm_r < rtol * norm_b:
            return x
        z = precondition(r)
        if reorthogonalize:
            beta = np.inner(r - r_old, z) / rz
            rz = np.inner(r, z)
        else:
            beta = 1.0 / rz
            rz = np.inner(r, z)
            beta *= rz
        p *= beta
        p += z
