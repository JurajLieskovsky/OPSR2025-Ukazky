import cvxpy as cp
# import numpy as np
from pprint import pprint

u = [5, 10, 8, 6, 2, 4, 5, 8, 9]

x = cp.Variable(9)
objective = cp.Maximize(x[7]+x[8])
constraints = [
    u >= -x,
    x <= u,
    x[0] + x[3] + x[5] == x[7],
    x[4] + x[6] == x[8],
    x[1] == x[3] + x[4],
    x[2] == x[5] + x[6]
]

prob = cp.Problem(objective, constraints)

result = prob.solve()
pprint(x.value.tolist())

