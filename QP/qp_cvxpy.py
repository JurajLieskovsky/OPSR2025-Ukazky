import cvxpy as cp
import numpy as np

n = 3
p = cp.Variable(n)

a = np.array([20, 25, 30])
b = np.array([5, 4, 3])
c = np.diag([0.02, 0.015, 0.01])

p_min = [200, 300, 100]
p_max = [1000, 1500, 800]
p_sum = 3000

objective = cp.Minimize(cp.quad_form(p, c) + b.T @ p + cp.sum(a)) 
constraints = [
    p >= p_min,
    p <= p_max,
    cp.sum(p) == p_sum,
]
prob = cp.Problem(objective, constraints)

result = prob.solve(solver=cp.OSQP)
print(p.value)
