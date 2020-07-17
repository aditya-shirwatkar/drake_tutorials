
"Solve the optimization problem :"

# Once all the decision variables/constraints/costs are added to prog, 
# we are ready to solve the optimization problem.

# Automatically choosing a solver
# The simplest way to solve the optimization problem is to call Solve() function. 
# Drake's MathematicalProgram analyzes the type of the constraints/costs, 
# and then calls an appropriate solver for your problem. 
# The result of calling Solve() is stored inside the return argument. Here is a code snippet

"""
Solves a simple optimization problem
       min x(0)^2 + x(1)^2
subject to x(0) + x(1) = 1
           x(0) <= x(1)
"""

from pydrake.all import eq, MathematicalProgram, Solve, Variable
from pydrake.common.containers import namedview
import numpy as np

State = namedview("q",
		["l_tip","q1", "q2", "q3", "q4", "q5", "r_tip", ])
Statedot = namedview("dq", 
		["l_tipdot","q1dot", "q2dot", "q3dot", "q4dot", "q5dot", "r_tipdot"])

class Model(MathematicalProgram):
	def __init__(self):
		super().__init__()

		self.N = 50; self.T = 1.
		self.num_steps = 2
		self.step_max = 0.25; self.tauMax = 1.5
		self.pi = np.pi; 
		self.l = np.array([0.25,0.25,0.25,0.25,0.25])
		self.m = np.array([0.25,0.25,0.25,0.25,0.25])
		self.i = self.m * (self.l**2) /12
		self.g = 9.81
		self.h = self.T/self.N

		# Create decision variables
		self.u = np.empty((4, self.N), dtype=Variable)
		self.x = np.empty((7, self.N), dtype=Variable)
		self.dx = np.empty((7, self.N), dtype=Variable)
		self.prog = MathematicalProgram()

		self.createVariables(self.u, self.x, self.dx)

		# # Set up the optimization problem.
		# self.state = self.prog.NewContinuousVariables(14, 'state')
		# self.u = self.prog.NewContinuousVariables(4, 'u')
		# self.q = BipedState(self.state)[0:7]
		# self.dq = BipedState(self.state)[7:14]

	def createVariables(self, u, x, dx):
		for n in range(self.N):
			u[:,n] = self.prog.NewContinuousVariables(4, 'u' + str(n))
			dx[:,n] = self.prog.NewContinuousVariables(7, 'dx' + str(n))
			x[:,n] = self.prog.NewContinuousVariables(7, 'x' + str(n))
 

f = Model()
print('u = ', f.u.shape, 'x = ', f.x.shape, 'dx = ', f.dx.shape)
# def constraint_eval(z):
#        p = pymath.cos(z)
#        return (1 - forwarddiff.derivative(p, z))

# prog.AddConstraint(constraint_eval, lb=[0], ub=[0], vars=x)
# prog.AddCost(x[0]**2 + dx[0]** 2)

# # Now solve the optimization problem.
# result = Solve(prog)

# # print out the result.
# print("Success? ", result.is_success())
# # Print the solution to the decision variables.
# print('x* = ', result.GetSolution(x))
# # Print the optimal cost.
# print('optimal cost = ', result.get_optimal_cost())
# # Print the name of the solver that was called.
# print('solver is: ', result.get_solver_id().name())

# print("\n==========================\n")

# # Some optimization solution is infeasible (doesn't have a solution). 
# # For example in the following code example, result.get_solution_result() will not report kSolutionFound.

# """
# An infeasible optimization problem.
# """
# prog = MathematicalProgram()
# x = prog.NewContinuousVariables(1)[0]
# y = prog.NewContinuousVariables(1)[0]
# prog.AddConstraint(x + y >= 1)
# prog.AddConstraint(x + y <= 0)
# prog.AddCost(x)

# result = Solve(prog)
# print("Success? ", result.is_success())
# print(result.get_solution_result())

# print("\n==========================\n")

