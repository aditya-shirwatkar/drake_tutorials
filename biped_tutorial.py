from pydrake.symbolic import Variable
from pydrake.systems.primitives import SymbolicVectorSystem
import pydrake.math as pymath
import numpy as np
import matplotlib.pyplot as plt


u = np.empty((4, 1), dtype=Variable)
x = np.empty((5, 1), dtype=Variable)
dx = np.empty((5, 1), dtype=Variable)

for n in range(5):
    x[n] = Variable('q' + str(n))
    dx[n] = Variable('dq' + str(n))
    if n != 5-1:
        u[n] = Variable('u' + str(n+1))


