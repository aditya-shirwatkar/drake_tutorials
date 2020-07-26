
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
# from pydrake.solvers.snopt import SnoptSolver
from pydrake.solvers.ipopt import IpoptSolver
from matplotlib import pyplot as plt

from pydrake.common.containers import namedview
from pydrake.autodiffutils import AutoDiffXd
import numpy as np
# from numpy import sin 
# from numpy import cos 

import pydrake.math as pymath


State = namedview("q",
		["q1", "q2", "q3", "q4", "q5"])
Statedot = namedview("dq", 
		["q1dot", "q2dot", "q3dot", "q4dot", "q5dot"])

class Model(MathematicalProgram):
	def __init__(self):
		super().__init__()

		self.N = 10; self.T = 1.
		self.num_steps = 2
		self.step_max = 0.25; self.tauMax = 1.5
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

		# self.getKinematics()

		# self.ComputeTransformations()

		self.SetConstraints()

		# self.Visualize()

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
		self.l_force= self.prog.NewContinuousVariables(2, self.N, 'left_force')
		self.r_force= self.prog.NewContinuousVariables(2, self.N, 'right_force')
		# self.foot_contacts = self.prog.NewBinaryVariables(2, self.N, 'foot_contacts')
		# self.right_foot_contact = self.prog.NewBinaryVariables(1, self.N, 'right_contact')

	def getKinematics(self):
		self.p1 = [];self.p2 = [];self.p3 = [];self.p4 = [];self.p5 = []
		for n in range(self.N):
			s1, c1 = pymath.sin(self.x[0, n]), pymath.cos(self.x[0, n]) 
			s2, c2 = pymath.sin(self.x[1, n]), pymath.cos(self.x[1, n]) 
			s3, c3 = pymath.sin(self.x[2, n]), pymath.cos(self.x[2, n]) 
			s4, c4 = pymath.sin(self.x[3, n]), pymath.cos(self.x[3, n]) 
			s5, c5 = pymath.sin(self.x[4, n]), pymath.cos(self.x[4, n]) 

			self.p1.append([self.l[0]*s1, self.l[0]*c1])
			self.p2.append([self.l[1]*s2+self.p1[-1][0], self.l[1]*c2+self.p1[-1][1]])
			self.p3.append([self.l[2]*s3+self.p2[-1][0], self.l[2]*c3+self.p2[-1][1]])
			self.p4.append([self.l[3]*s4+self.p3[-1][0], self.l[3]*c4+self.p3[-1][1]])
			self.p5.append([self.l[4]*s5+self.p4[-1][0], self.l[4]*c5+self.p4[-1][1]])

	def SetConstraints(self):

		h = self.h

		def collocation_dot(z):
			ceq = []
			q1 = [z[0], z[1], z[2], z[3], z[4]]
			dq1 = [z[5], z[6], z[7], z[8], z[9]]
			u1 = [z[10], z[11], z[12], z[13]]
			ddq1 = self.ComputeManipulators(q1, dq1, u1)

			q2 = [z[0+14], z[1+14], z[2+14], z[3+14], z[4+14]]
			dq2 = [z[5+14], z[6+14], z[7+14], z[8+14], z[9+14]]
			u2 = [z[10+14], z[11+14], z[12+14], z[13+14]]
			ddq2 = self.ComputeManipulators(q2, dq2, u2)
			
			ceq.extend([((ddq2[i] - ddq1[i]) - (h*(dq2[i] - dq1[i]))) for i in range(len(dq1))])
			# ceq.extend([((dq2[i] - dq1[i]) - (h*(q2[i] - q1[i]))) for i in range(len(dq1))])
			
			# ceq.extend([(u[i] - q[i]) for i in range(len(u))])
			# print(len(ceq))
			
			return np.array(ceq)

		def collocation(z):
			ceq = []
			q1 = [z[0], z[1], z[2], z[3], z[4]]
			dq1 = [z[5], z[6], z[7], z[8], z[9]]
			u1 = [z[10], z[11], z[12], z[13]]
			ddq1 = self.ComputeManipulators(q1, dq1, u1)

			q2 = [z[0+14], z[1+14], z[2+14], z[3+14], z[4+14]]
			dq2 = [z[5+14], z[6+14], z[7+14], z[8+14], z[9+14]]
			u2 = [z[10+14], z[11+14], z[12+14], z[13+14]]
			ddq2 = self.ComputeManipulators(q2, dq2, u2)
			
			# ceq.extend([((ddq2[i] - ddq1[i]) - (h*(dq2[i] - dq1[i]))) for i in range(len(dq1))])
			ceq.extend([((dq2[i] - dq1[i]) - (h*(q2[i] - q1[i]))) for i in range(len(dq1))])
			
			# ceq.extend([(u[i] - q[i]) for i in range(len(u))])
			# print(len(ceq))
			
			return np.array(ceq)

		def positional(z):
			p1, p2, p3, p4, p5 = self.ConstraintKinematics(z)
			return np.array([p5[0], p5[1]])

		x0 = [-0.6,0.7,0.0,-0.5,-0.3]
		self.prog.AddBoundingBoxConstraint(x0, x0, self.x[:,0])
		# self.prog.AddConstraint(positional, lb=[-3*self.step_max, 0], ub=[0, 0],
		# 					vars=[self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0], self.x[4,0]])

		for n in range(self.N - 1):
			self.prog.AddConstraint(collocation_dot, lb=np.array([0]*5), ub=np.array([0]*5), 
							vars=[self.x[0,n], self.x[1,n], self.x[2,n], self.x[3,n], self.x[4,n], 
								self.dx[0,n], self.dx[1,n],self.dx[2,n],self.dx[3,n],self.dx[4,n],  
								self.u[0,n], self.u[1,n], self.u[2,n], self.u[3,n],
								self.x[0,n+1], self.x[1,n+1], self.x[2,n+1], self.x[3,n+1], self.x[4,n+1], 
								self.dx[0,n+1], self.dx[1,n+1],self.dx[2,n+1],self.dx[3,n+1],self.dx[4,n+1],  
								self.u[0,n+1], self.u[1,n+1], self.u[2,n+1], self.u[3,n+1]])

			self.prog.AddConstraint(collocation, lb=np.array([0]*5), ub=np.array([0]*5), 
							vars=[self.x[0,n], self.x[1,n], self.x[2,n], self.x[3,n], self.x[4,n], 
								self.dx[0,n], self.dx[1,n],self.dx[2,n],self.dx[3,n],self.dx[4,n],  
								self.u[0,n], self.u[1,n], self.u[2,n], self.u[3,n],
								self.x[0,n+1], self.x[1,n+1], self.x[2,n+1], self.x[3,n+1], self.x[4,n+1], 
								self.dx[0,n+1], self.dx[1,n+1],self.dx[2,n+1],self.dx[3,n+1],self.dx[4,n+1],  
								self.u[0,n+1], self.u[1,n+1], self.u[2,n+1], self.u[3,n+1]])

			self.prog.AddCost(sum(self.x[:,n]**2) + sum(self.dx[:,n]**2) + sum(self.u[:,n]**2))

			self.prog.AddBoundingBoxConstraint([-self.tauMax]*4, [self.tauMax]*4, self.u[:,n])
			self.prog.AddBoundingBoxConstraint([-np.pi]*5, [np.pi]*5, self.x[:,n])
			self.prog.AddBoundingBoxConstraint([-np.pi]*5, [np.pi]*5, self.dx[:,n])
		
		self.prog.AddConstraint(positional, lb=[self.step_max, 0], ub=[self.step_max, 0],
							vars=[self.x[0,-1], self.x[1,-1], self.x[2,-1], self.x[3,-1], self.x[4,-1]])

		# self.prog.AddBoundingBoxConstraint(x0[::-1], x0[::-1], self.x[:,-1])

	def ComputeManipulators(self, q, dq, u):
		n = np.array([
			self.l[0]*np.sum(self.m[1:]),
			self.l[1]*np.sum(self.m[2:]),
			0,
			self.l[3]*np.sum(self.m[4:]),
			0.
		])
		p = np.array([
			[			 self.i[0] + n[0]*self.l[0],                                      0,                                     0,                                       0,                          0],
			[(self.m[0]*self.l[0]/2 + n[0])*self.l[1],             self.i[1] + n[1]*self.l[1],                                     0,                                       0,                          0],
			[(self.m[0]*self.l[0]/2 + n[0])*self.l[2], (self.m[1]*self.l[1]/2 + n[1])*self.l[2],             self.i[2] + n[2]*self.l[2],                                      0,                          0],
			[(self.m[0]*self.l[0]/2 + n[0])*self.l[3], (self.m[1]*self.l[1]/2 + n[1])*self.l[3], (self.m[2]*self.l[2]/2 + n[2])*self.l[3],             self.i[3] + n[3]*self.l[3],                          0],
			[(self.m[0]*self.l[0]/2 + n[0])*self.l[4], (self.m[1]*self.l[1]/2 + n[1])*self.l[4], (self.m[2]*self.l[2]/2 + n[2])*self.l[4], (self.m[3]*self.l[3]/2 + n[3])*self.l[4], self.i[4] + n[4]*self.l[4]],
		])
		asin = (np.array([
			[np.sin(        0), np.sin(q[0]-q[1]), np.sin(q[0]-q[2]), np.sin(q[0]+q[3]), np.sin(q[0]+q[4])],
			[np.sin(q[1]-q[0]), np.sin(        0), np.sin(q[1]-q[2]), np.sin(q[1]+q[3]), np.sin(q[1]+q[4])],
			[np.sin(q[2]-q[0]), np.sin(q[2]-q[1]), np.sin(        0), np.sin(q[2]+q[3]), np.sin(q[2]+q[4])],
			[np.sin(q[3]+q[0]), np.sin(q[3]+q[1]), np.sin(q[3]+q[2]), np.sin(        0), np.sin(q[3]-q[4])],
			[np.sin(q[4]+q[0]), np.sin(q[4]+q[1]), np.sin(q[4]+q[2]), np.sin(q[4]-q[3]), np.sin(        0)]
		]))
		acos = (np.array([
			[np.cos(        0), np.cos(q[0]-q[1]), np.cos(q[0]-q[2]), np.cos(q[0]+q[3]), np.cos(q[0]+q[4])],
			[np.cos(q[1]-q[0]), np.cos(        0), np.cos(q[1]-q[2]), np.cos(q[1]+q[3]), np.cos(q[1]+q[4])],
			[np.cos(q[2]-q[0]), np.cos(q[2]-q[1]), np.cos(        0), np.cos(q[2]+q[3]), np.cos(q[2]+q[4])],
			[np.cos(q[3]+q[0]), np.cos(q[3]+q[1]), np.cos(q[3]+q[2]), np.cos(        0), np.cos(q[3]-q[4])],
			[np.cos(q[4]+q[0]), np.cos(q[4]+q[1]), np.cos(q[4]+q[2]), np.cos(q[4]-q[3]), np.cos(        0)]
		]))
		g = np.array([
			-(self.m[0]*self.l[0]/2 + n[0])*self.g,
			-(self.m[1]*self.l[1]/2 + n[1])*self.g,
			-(self.m[2]*self.l[2]/2 + n[2])*self.g,
			(self.m[3]*self.l[3]/2 + n[3])*self.g,
			(self.m[4]*self.l[4]/2 + n[4])*self.g
		]).reshape(5, 1)
		M = p*acos
		C = p*asin
		G = g*np.sin(q).reshape(5, 1)
		T = np.array([0, u[0], u[1], u[2], u[3]]).reshape(5, 1)
		ddq = pymath.inv(M) @ (T - G - (C @ (np.array(dq).reshape(5, 1)**2)))
		return ddq

	def ConstraintKinematics(self, x):
		p1 = [];p2 = [];p3 = [];p4 = [];p5 = []
		s1, c1 = pymath.sin(x[0]), pymath.cos(x[0]) 
		s2, c2 = pymath.sin(x[1]), pymath.cos(x[1]) 
		s3, c3 = pymath.sin(x[2]), pymath.cos(x[2]) 
		s4, c4 = pymath.sin(x[3]), pymath.cos(x[3]) 
		s5, c5 = pymath.sin(x[4]), pymath.cos(x[4]) 

		p1 = [self.l[0]*s1, self.l[0]*c1]
		p2 = [self.l[1]*s2+p1[0], self.l[1]*c2+p1[1]]
		p3 = [self.l[2]*s3+p2[0], self.l[2]*c3+p2[1]]
		p4 = [self.l[3]*s4+p3[0], self.l[3]*c4+p3[1]]
		p5 = [self.l[4]*s5+p4[0], self.l[4]*c5+p4[1]]

		return p1, p2, p3, p4, p5

f = Model()

result = Solve(f.prog)

assert(result.is_success()), "Optimization failed"

print("Success? ", result.is_success())
# Print the solution to the decision variables.
# print('x* = ', result.GetSolution(x))
# Print the optimal cost.
print('optimal cost = ', result.get_optimal_cost())
# Print the name of the solver that was called.
print('solver is: ', result.get_solver_id().name())

time = np.linspace(0, f.T, f.N)
x_sol = result.GetSolution(f.x)
dx_sol = result.GetSolution(f.dx)
u_sol = result.GetSolution(f.u)
plt.figure()
plt.subplot(311)
plt.plot(time, x_sol[0,:], label='q1')
plt.plot(time, x_sol[1,:], label='q2')
plt.plot(time, x_sol[2,:], label='q3')
plt.plot(time, x_sol[3,:], label='q4')
plt.plot(time, x_sol[4,:], label='q5')

plt.subplot(312)
plt.plot(time, dx_sol[0,:], label='dq1')
plt.plot(time, dx_sol[1,:], label='dq2')
plt.plot(time, dx_sol[2,:], label='dq3')
plt.plot(time, dx_sol[3,:], label='dq4')
plt.plot(time, dx_sol[4,:], label='dq5')

plt.subplot(313)
plt.plot(time, u_sol[0,:], label='u1')
plt.plot(time, u_sol[1,:], label='u2')
plt.plot(time, u_sol[2,:], label='u3')
plt.plot(time, u_sol[3,:], label='u4')

# plt.xlabel('q')
# plt.ylabel('qdot')
plt.show()
# solver = IpoptSolver()
# result = solver.Solve(f.prog, None, None)

# assert(result.is_success()), "Optimization failed"

# print("Success? ", result.is_success())
# # Print the solution to the decision variables.
# # print('x* = ', result.GetSolution(x))
# # Print the optimal cost.
# print('optimal cost = ', result.get_optimal_cost())
# # Print the name of the solver that was called.
# print('solver is: ', result.get_solver_id().name())





	# def ComputeTransformations(self):
	# 	self.world_frame = np.array([0, 0]).reshape(2, 1)
	# 	self.T01 = []; self.T12 = []; self.T23 = []; self.T24 = []; self.T45 = []
	# 	for n in range(self.N):
	# 		s1, c1 = np.sin(self.x[0, n]), np.cos(self.x[0, n]) 
	# 		s2, c2 = np.sin(self.x[1, n]), np.cos(self.x[1, n]) 
	# 		s3, c3 = np.sin(self.x[2, n]), np.cos(self.x[2, n]) 
	# 		s4, c4 = np.sin(self.x[3, n]), np.cos(self.x[3, n]) 
	# 		s5, c5 = np.sin(self.x[4, n]), np.cos(self.x[4, n]) 
			
	# 		self.T01.append(np.array([
	# 			[c1, s1, self.left_foot[0, n]],
	# 			[-s1, c1, self.left_foot[1, n]],
	# 			[0, 0, 1]
	# 		]))
	# 		self.T12.append( self.T01[-1] @ (np.array(
	# 										[[c2, s2, self.l[0]],
	# 										[-s2, c2, self.l[0]],
	# 										[0, 0, 1]]
	# 										))
	# 						)
	# 		self.T23.append( self.T12[-1] @ (np.array(
	# 										[[c3, s3, self.l[1]],
	# 										[-s3, c3, self.l[1]],
	# 										[0, 0, 1]]
	# 										))
	# 						)
	# 		self.T24.append( self.T12[-1] @ (np.array(
	# 										[[c4, s4, self.l[0]],
	# 										[-s4, c4, self.l[0]],
	# 										[0, 0, 1]]
	# 										))
	# 						)
	# 		# print(self.T01)

	# def Visualize(self):
	# 	fig = plt.figure()
	# 	time = np.linspace(0, self.T, self.N)
	# 	ax = plt.gca()
	# 	# ax.plot(curve_x, 9./curve_x)
	# 	# ax.plot(-curve_x, -9./curve_x)
	# 	# ax.plot(0, 0, 'o')
	# 	x_init = [-0.6,0.7,0.0,-0.5,-0.3]
	# 	point_x1, = ax.plot([], [], 'x')
	# 	ax.axis('equal')

	# 	def update(x):
	# 		global iter_count
	# 		point_x.set_xdata()
	# 		point_x.set_ydata(x)
	# 		ax.set_title(f"iteration {iter_count}")
	# 		fig.canvas.draw()
	# 		fig.canvas.flush_events()
	# 		# Also update the iter_count variable in the callback.
	# 		# This shows we can do more than just visualization in
	# 		# callback.
	# 		iter_count += 1
	# 		plt.pause(0.1)
	# 	iter_count = 0

	# 	for n in range(self.N):
	# 		self.prog.AddVisualizationCallback(update, self.x[:,n])
















# def constraint_eval(z):
#        p = np.cos(z)
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

