from sympy import symbols, Function, Matrix, sin, cos, diff, latex, simplify
from lagrangian2eom import massMatrix, velocityTerms, potentialTerms

t = symbols("t")

x = Function("x")(t)
y = Function("y")(t)
theta = Function("theta")(t)

# parameters
g, m, I, a = symbols("g m I a")

# generalized coordinates
q = Matrix([x, y, theta])

# kinetic and potential energy
kin = 1 / 2 * m * (diff(x, t) ** 2 + diff(y, t) ** 2) + 1 / 2 * I * diff(theta, t) ** 2
pot = m * g * y

# Quantities
M = massMatrix(kin, q, t)
c = velocityTerms(kin, q, t)
tau_p = potentialTerms(pot, q)

# Printout
print("T &=", latex(simplify(kin)), "\\\\")
print("V &=", latex(simplify(pot)))
print("M &=", latex(simplify(M)), "\\\\")
print("c &=", latex(simplify(c)), "\\\\")
print("\\tau_p &=", latex(simplify(tau_p)))
