import numpy as np

def calculate_eigenvalues(m: list[list[float|int]]) -> list[float]:
    m = np.array(m)
    a = 1
    b = -np.trace(m)
    c = np.linalg.det(m)
    delta = np.sqrt(b**2 - 4*a*c) / (2*a)
    l1 = delta - b / (2*a)
    l2 = -delta - b / (2*a)
    return [l1, l2]