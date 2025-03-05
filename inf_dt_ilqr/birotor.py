"""
1. definujeme nelin. system
2. linearizace
3. definujeme problem
4. vyresime Riccatiho rovnici
5. vyjadrime zpetnou vazbu
6. nasimulujeme
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.integrate import ode

m = 1
g = 9.81
a = 0.2
I = 1

h = 1e-2


# nelin. stavovy popis
def f(t, x, u):
    return np.array(
        [
            x[3],
            x[4],
            x[5],
            -np.sin(x[2]) * (u[0] + u[1]) / m,
            np.cos(x[2]) * (u[0] + u[1]) / m - m * g,
            a / I * (u[0] - u[1]),
        ]
    )


# lin. stavovy popis
A = np.eye(6) + h * np.array(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, -g, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

B = h * np.array(
    [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1 / m, 1 / m],
        [a / I, -a / I],
    ]
)

# LQR problem
Q = np.diag([10, 10, 10, 1, 1, 1])
R = np.eye(2)

# LQR feedback
S = solve_discrete_are(A, B, Q, R)
K = np.linalg.solve(R + B.T @ S @ B, B.T @ S @ A)

# Simulace
N = 500
x0 = np.array([1, 0, 0, 0, 0, 0])

x_eq = np.zeros(6)
u_eq = 0.5 * m * g * np.ones(2)

xs = np.zeros((6, N + 1))
us = np.zeros((2, N))

solver = ode(f).set_integrator(name="dopri5")

xs[:, 0] = x0
for k in range(N):
    us[:, k] = u_eq - K @ (xs[:, k] - x_eq)
    solver.set_initial_value(xs[:, k], 0).set_f_params(us[:, k])
    solver.integrate(h)
    xs[:, k + 1] = solver.y


print(xs[:,N])
