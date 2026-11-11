def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    m = len(a[0])
    if m != len(b):
        return -1
    return [[sum([a[i][k] * b[k][j] for k in range(m)])
        for j in range(len(b[0]))]
        for i in range(len(a))]
