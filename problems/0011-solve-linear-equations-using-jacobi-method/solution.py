import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    m = b.shape[0]
    x0 = np.zeros(m)
    for _ in range(n):
        x0 += 1/np.diag(A) * (b - A @ x0)

    return np.round(x0, 4).tolist()
        
