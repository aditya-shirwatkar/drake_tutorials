import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import eq, MathematicalProgram, Solve, Variable
from pydrake.solvers.ipopt import IpoptSolver

# Discrete-time approximation of the double integrator.
dt = 0.01
A = np.eye(2) + dt*np.mat('0 1; 0 0')
print(A)
B = dt*np.mat('0; 1')
print(B)
prog = MathematicalProgram()

N = 284  # Note: I had to do a manual "line search" to find this.

# Create decision variables
u = np.empty((1, N-1), dtype=Variable)
x = np.empty((2, N), dtype=Variable)
for n in range(N-1):
  u[:,n] = prog.NewContinuousVariables(1, 'u' + str(n))
  x[:,n] = prog.NewContinuousVariables(2, 'x' + str(n))
x[:,N-1] = prog.NewContinuousVariables(2, 'x' + str(N))

# Add constraints
x0 = [-2, 0]
prog.AddBoundingBoxConstraint(x0, x0, x[:,0])
for n in range(N-1):
  # Will eventually be prog.AddConstraint(x[:,n+1] == A@x[:,n] + B@u[:,n])
  # See drake issues 12841 and 8315
  prog.AddConstraint(eq(x[:,n+1],A.dot(x[:,n]) + B.dot(u[:,n])))
  prog.AddBoundingBoxConstraint(-1, 1, u[:,n])
xf = [0, 0]
prog.AddBoundingBoxConstraint(xf, xf, x[:, N-1])

solver = IpoptSolver()
result = solver.Solve(prog)

# result = Solve(prog)

x_sol = result.GetSolution(x)
assert(result.is_success()), "Optimization failed"

plt.figure()
plt.plot(x_sol[0,:], x_sol[1,:])
plt.xlabel('q')
plt.ylabel('qdot')
plt.show()

#############################################################

print("\n==========================\n")

import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import LinearSystem, DirectTranscription

# Discrete-time approximation of the double integrator.
dt = 0.01
A = np.eye(2) + dt*np.mat('0 1; 0 0')
B = dt*np.mat('0; 1')
C = np.eye(2)
D = np.zeros((2,1))
sys = LinearSystem(A, B, C, D, dt)

print(A)
print(B)
print(C)
print(D)

prog = DirectTranscription(sys, sys.CreateDefaultContext(), N)
prog.AddBoundingBoxConstraint(x0, x0, prog.initial_state())
prog.AddBoundingBoxConstraint(xf, xf, prog.final_state())
prog.AddConstraintToAllKnotPoints(prog.input()[0] <= 1)
prog.AddConstraintToAllKnotPoints(prog.input()[0] >= -1)

result = Solve(prog)
x_sol = prog.ReconstructStateTrajectory(result)
assert(result.is_success()), "Optimization failed"
plt.figure()
x_values = x_sol.vector_values(x_sol.get_segment_times())

plt.plot(x_values[0,:], x_values[1,:])
plt.xlabel('q')
plt.ylabel('qdot')
plt.show()