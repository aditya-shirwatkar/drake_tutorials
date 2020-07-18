
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
from numpy import sin 
from numpy import cos 

import pydrake.math as pymath

State = namedview("q",
		["q1", "q2", "q3", "q4", "q5"])
Statedot = namedview("dq", 
		["q1dot", "q2dot", "q3dot", "q4dot", "q5dot"])

class Model(MathematicalProgram):
	def __init__(self):
		super().__init__()

		self.N = 3; self.T = 1.
		self.num_steps = 2
		self.step_max = 0.25; self.tauMax = 1.5
		self.pi = np.pi; 
		self.l = np.array([0.25,0.25,0.25,0.25,0.25])
		self.m = np.array([0.25,0.25,0.25,0.25,0.25])
		self.i = self.m * (self.l**2) /12
		self.g = 9.81
		self.h = self.T/self.N

		# Create decision variables
		# self.u = np.empty((4, self.N), dtype=Variable)
		# self.x = np.empty((5, self.N), dtype=Variable)
		# self.dx = np.empty((5, self.N), dtype=Variable)
		# self.left_foot = np.empty((1, self.N), dtype=Variable)
		# self.right_foot = np.empty((1, self.N), dtype=Variable)

		self.prog = MathematicalProgram()

		self.CreateVariables()

		# self.ComputeTransformations()

		self.AddConstraints()

		# # Set up the optimization problem.
		# self.state = self.prog.NewContinuousVariables(14, 'state')
		# self.u = self.prog.NewContinuousVariables(4, 'u')
		# self.q = BipedState(self.state)[0:7]
		# self.dq = BipedState(self.state)[7:14]

	def CreateVariables(self):
		# for n in range(self.N):
		self.u = self.prog.NewContinuousVariables(4, self.N,'u')
		self.dx = self.prog.NewContinuousVariables(5, self.N, 'dx')
		self.x = self.prog.NewContinuousVariables(5, self.N, 'x')
		self.left_foot= self.prog.NewContinuousVariables(2, self.N, 'left_tip')
		self.foot_contacts = self.prog.NewBinaryVariables(2, self.N, 'foot_contacts')
		# self.right_foot_contact = self.prog.NewBinaryVariables(1, self.N, 'right_contact')

	def ComputeTransformations(self):
		self.world_frame = np.array([0, 0]).reshape(2, 1)
		self.T01 = []; self.T12 = []; self.T23 = []; self.T24 = []; self.T45 = []
		for n in range(self.N):
			s1, c1 = pymath.sin(self.x[0, n]), pymath.cos(self.x[0, n]) 
			s2, c2 = pymath.sin(self.x[1, n]), pymath.cos(self.x[1, n]) 
			s3, c3 = pymath.sin(self.x[2, n]), pymath.cos(self.x[2, n]) 
			s4, c4 = pymath.sin(self.x[3, n]), pymath.cos(self.x[3, n]) 
			s5, c5 = pymath.sin(self.x[4, n]), pymath.cos(self.x[4, n]) 
			
			self.T01.append(np.array([
				[c1, s1, self.left_foot[0, n]],
				[-s1, c1, self.left_foot[1, n]],
				[0, 0, 1]
			]))
			self.T12.append( self.T01[-1] @ (np.array(
											[[c2, s2, self.l[0]],
											[-s2, c2, self.l[0]],
											[0, 0, 1]]
											))
							)
			self.T23.append( self.T12[-1] @ (np.array(
											[[c3, s3, self.l[1]],
											[-s3, c3, self.l[1]],
											[0, 0, 1]]
											))
							)
			self.T24.append( self.T12[-1] @ (np.array(
											[[c4, s4, self.l[0]],
											[-s4, c4, self.l[0]],
											[0, 0, 1]]
											))
							)
			# print(self.T01)

	def AddConstraints(self):

		def collocation(z):
			q = z[0]
			dq = z[1]
			ddq = self.ComputeManipulators(q, dq)

		# for n in range(self.N):
		self.prog.AddConstraint(self.collocation, lb=[0, 0], ub=[0, 0], vars=[x, dx])

	def ComputeManipulators(self, q, dq):
		n = np.array([
			self.l[0]*np.sum(self.m[1:]),
			self.l[1]*np.sum(self.m[2:]),
			0,
			self.l[3]*np.sum(self.m[4:]),
			0.
		])
		p = np.array([
			[			 self.i[0] + n[0]*self.l[0],                                      0,                                     0,                                       0,                          0],
			[(self.m[0]*self.l[0] + n[0])*self.l[1],             self.i[1] + n[1]*self.l[1],                                     0,                                       0,                          0],
			[(self.m[0]*self.l[0] + n[0])*self.l[2], (self.m[1]*self.l[1] + n[1])*self.l[2],             self.i[2] + n[2]*self.l[2],                                      0,                          0],
			[(self.m[0]*self.l[0] + n[0])*self.l[3], (self.m[1]*self.l[1] + n[1])*self.l[3], (self.m[2]*self.l[2] + n[2])*self.l[3],             self.i[3] + n[3]*self.l[3],                          0],
			[(self.m[0]*self.l[0] + n[0])*self.l[4], (self.m[1]*self.l[1] + n[1])*self.l[4], (self.m[2]*self.l[2] + n[2])*self.l[4], (self.m[3]*self.l[3] + n[3])*self.l[4], self.i[4] + n[4]*self.l[4]],
		])
		q = np.array([
			[],
			[],
			[],
			[],
			[]
		])

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

