import numpy as np
import cvxpy as cp
from scipy.integrate import ode
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

m = 1
g = 9.81
a = 0.2
I = 1

# casovy krok
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
x_eq = [1, 1, 0, 0, 0, 0]
u_eq = 0.5 * m * g * np.ones(2)

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

# MPC problem
N = 20
Q = np.diag([10, 10, 10, 1, 1, 1])
R = np.eye(2)
Q_N = solve_discrete_are(A, B, Q, R)

# MPC CVXPY problem
x = cp.Variable((6, N + 1))
u = cp.Variable((2, N))
x_init = cp.Parameter(6)

constraints = [
    x[:, 0] == x_init,
    x[:, 1:] == A @ x[:, :-1] + B @ u,
    u >= -u_eq[:, np.newaxis],
]

objective = cp.Minimize(
    cp.sum_squares(np.sqrt(Q) @ x[:, :-1])
    + cp.sum_squares(np.sqrt(R) @ u)
    + cp.quad_form(x[:, N], Q_N)
)

problem = cp.Problem(objective, constraints)

# MPC simulation
M = 1000
x0 = [0, 4, 0, 0, 0, 0]

xs = np.zeros((6, M + 1))
us = np.zeros((2, M))

solver = ode(f).set_integrator(name="dopri5")
xs[:, 0] = x0

for k in range(M):
    # MPC
    x_init.value = xs[:, k] - x_eq
    problem.solve()
    us[:, k] = u_eq + u.value[:, 0]

    # Simulation
    solver.set_initial_value(xs[:, k], 0).set_f_params(us[:, k])
    solver.integrate(h)
    xs[:, k + 1] = solver.y

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1)

for i in range(3):
    ax1.plot(xs[i, :].T, label=f"x{i}")

for i in range(2):
    ax2.plot(us[i, :].T, label=f"u{i}")

ax1.legend()
ax2.legend()
plt.show()
