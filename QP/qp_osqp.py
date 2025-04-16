import osqp
import numpy as np
from scipy.sparse import csc_matrix

b = np.array([5, 4, 3])

data_C = np.array([0.02, 0.015, 0.01])
rowcol_C = np.arange(3, dtype=int)
C = 2 * csc_matrix((data_C, (rowcol_C, rowcol_C)), shape=(3,3))

data_A = np.ones(6)
row_A = np.array([0,0,0,1,2,3])
col_A = np.array([0,1,2,0,1,2])
A = csc_matrix((data_A, (row_A,col_A)), shape=(4,3))

l = np.array([3000, 200, 300, 100])
u = np.array([3000, 1000, 1500, 800])

m = osqp.OSQP()
m.setup(P=C, q=b, A=A, l=l, u=u, polish=True)
result = m.solve()

print(result.x)
