from sympy import symbols, Function, Matrix, sin, cos, diff, latex, simplify
from lagrangian2eom import massMatrix, velocityTerms, potentialTerms, wrenchMatrix

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

# input matrix
f = Matrix([-sin(theta), cos(theta)])

r_1 = Matrix([x + a * cos(theta), y + a * sin(theta)])
r_2 = Matrix([x - a * cos(theta), y - a * sin(theta)])

B = Matrix.hstack(wrenchMatrix(r_1, q) * f, wrenchMatrix(r_2, q) * f)

# Quantities
M = massMatrix(kin, q, t)
c = velocityTerms(kin, q, t)
tau_p = potentialTerms(pot, q)

# Printout
print("T &=", latex(simplify(kin)), "\\\\")
print("V &=", latex(simplify(pot)))
print("M &=", latex(simplify(M)), "\\\\")
print("c &=", latex(simplify(c)), "\\\\")
print("\\tau_p &=", latex(simplify(tau_p)), "\\\\")
print("B &=", latex(simplify(B)))
