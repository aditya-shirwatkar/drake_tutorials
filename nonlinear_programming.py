" Nonlinear Program "
# A Nonlinear Programming (NLP) problem is a special type of optimization problem. 
# The cost and/or constraints in an NLP are nonlinear functions of decision variables. 
# The mathematical formulation of a general NLP is
# find x, min f(x) , subject to gi(x) ≤ 0
# where f(x) is the cost function, and gi(x) is the i'th constraint.

# An NLP is typically solved through gradient-based optimization (like gradient descent, SQP, interior point methods, etc). 
# These methods rely on the gradient of the cost/constraints ∂f/∂x, ∂gi/∂x. 
# pydrake can compute the gradient of many functions through automatic differentiation, 
# so very often the user doesn't need to manually provide the gradient.

###############################################################################

print("\n==========================\n")

" Setting the objective "
# The user can call AddCost function to add a nonlinear cost into the program. 
# Note that the user can call AddCost repeatedly, and the program will evaluatate the summation 
# of each individual cost as the total cost.

# Adding a cost through a python function
# We can define a cost through a python function, and then add this python function to the
# objective through AddCost function. When adding a cost, 
# we should provide the variable associated with that cost, 
# the syntax is AddCost(cost_evaluator, vars=associated_variables), 
# which means the cost is evaluated on the associated_variables. 
# In the code example below, We first demonstrate how to construct 
# an optimization program with 3 decision variables, 
# then we show how to add a cost through a python function.

from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
import numpy as np

# Create an empty MathematicalProgram named prog (with no decision variables,
# constraints or costs)
prog = MathematicalProgram()
# Add three decision variables x[0], x[1], x[2]
x = prog.NewContinuousVariables(3, "x")

def cost_fun(z):
    cos_z = np.cos(z[0] + z[1])
    sin_z = np.sin(z[0] + z[1])
    return cos_z**2 + cos_z + sin_z
# Add the cost evaluated with x[0] and x[1], think of vars= z (for the function)
cost1 = prog.AddCost(cost_fun, vars=[x[0], x[1]])
print(cost1)

# Notice that by changing the argument vars in AddCost function, 
# we can add the cost to a different set of variables. 
# In the code example below, we use the same python function cost_fun, 
# but impose this cost on the variable x[0], x[2].

cost2 = prog.AddCost(cost_fun, vars=[x[0], x[2]])
print(cost2)

# Adding cost through a lambda function
# A more compact approach to add a cost is through a lambda function. 
# For example, the code below adds a cost x[1]^2 + x[0] to the optimization program.

# Add a cost x[1]**2 + x[0] using a lambda function.
cost3 = prog.AddCost(lambda z: z[0]**2 + z[1], vars = [x[0], x[1]])
print(cost3)

# If we change the associated variables, then it represents a different cost. 
# For example, we can use the same lambda function,
# but add the cost x[1]^2 + x[2] to the program by changing the argument to vars, uncomment below 

# cost4 = prog.AddCost(lambda z: z[0]**2 + z[1], vars = x[1:])
# print(cost4)

" Adding quadratic cost "
# In NLP, adding a quadratic cost 0.5x.T*Q*x + b.T*x + c is very common. 
# pydrake provides multiple functions to add quadratic cost, including
## AddQuadraticCost
## AddQuadraticErrorCost
## AddL2NormCost

# AddQuadraticCost
# We can add a simple quadratic expression as a cost.
cost4 = prog.AddQuadraticCost(x[0]**2 + 3 * x[1]**2 + 2*x[0]*x[1] + 2*x[1] * x[0] + 1)
print(cost4)
# If the user knows the matrix form of Q and b, 
# then it is faster to pass in these matrices to AddQuadraticCost, 
# instead of using the symbolic quadratic expression as above.
# Add a cost x[0]**2 + 2*x[1]**2 + x[0]*x[1] + 3*x[1] + 1.
cost5 = prog.AddQuadraticCost(
    Q=np.array([[2., 1], [1., 4.]]),
    b=np.array([0., 3.]),
    c=1.,
    vars=x[:2])
print(cost5)

# AddQuadraticErrorCost
# This function adds a cost of the form (x−xdes).T*Q*(x−xdes).
cost6 = prog.AddQuadraticErrorCost(
    Q=np.array([[1, 0.5], [0.5, 1]]),
    x_desired=np.array([1., 2.]),
    vars=x[1:])
print(cost6)

# AddL2NormCost
# This function adds a quadratic cost of the form (Ax−b).T(Ax−b)
# Add the L2 norm cost on (A*x[:2] - b).dot(A*x[:2]-b)
cost7 = prog.AddL2NormCost(
    A=np.array([[1., 2.], [2., 3], [3., 4]]),
    b=np.array([2, 3, 1.]),
    vars=x[:2])
print(cost7)

###############################################################################

print("\n==========================\n")

"Adding constraints"
# Drake supports adding constraints in the following form
# lower ≤ g(x) ≤ upper
# where g(x) returns a numpy vector.

# The user can call AddConstraint(g, lower, upper, vars=x) to add the constraint. Here g must be a python function (or a lambda function).

## Define a python function to add the constraint x[0]**2 + 2x[1]<=1, -0.5<=sin(x[1])<=0.5
def constraint_evaluator1(z):
    return np.array([z[0]**2+2*z[1], np.sin(z[1])])

constraint1 = prog.AddConstraint(
    constraint_evaluator1,
    lb=np.array([-np.inf, -0.5]),
    ub=np.array([1., 0.5]),
    vars=x[:2])
print(constraint1)

# Add another constraint using lambda function.
constraint2 = prog.AddConstraint(
    lambda z: np.array([z[0]*z[1]]),
    lb=[0.],
    ub=[1.],
    vars=[x[2]])
print(constraint2)


###############################################################################

print("\n==========================\n")

" Solving the nonlinear program "
# Once all the constraints and costs are added to the program, 
# we can call the Solve function to solve the program and call GetSolution to obtain the results. 
# Solving an NLP requires an initial guess on all the decision variables. 
# If the user doesn't specify an initial guess, we will use a zero vector by default.

# Setting the initial guess
# Some NLPs might have many decision variables. 
# In order to set the initial guess for these decision variables, 
# we provide a function SetDecisionVariableValueInVector to set 
# the initial guess of a subset of decision variables. 
# For example, in the problem below, we want to find the two closest points p1 and p2, 
# where p1 is on the unit circle, and p2 is on the curve y=x^2, 
# we can set the initial guess for these two variables separately by 
# calling SetDecisionVariableValueInVector.

import matplotlib.pyplot as plt
prog = MathematicalProgram()
p1 = prog.NewContinuousVariables(2, "p1")
p2 = prog.NewContinuousVariables(2, "p2")

# Add the constraint that p1 is on the unit circle centered at (0, 2)
prog.AddConstraint(
    lambda z: [z[0]**2 + (z[1]-2)**2],
    lb=np.array([1.]),
    ub=np.array([1.]),
    vars=p1)

# Add the constraint that p2 is on the curve y=x*x
prog.AddConstraint(
    lambda z: [z[1] - z[0]**2],
    lb=[0.],
    ub=[0.],
    vars=p2)

# Add the cost on the distance between p1 and p2
prog.AddQuadraticCost((p1-p2).dot(p1-p2))

# Now create an initial guess of the problem
initial_guess = np.empty(prog.num_vars())
print(initial_guess)
# Set the value of p1 in initial guess to be [0, 1]
prog.SetDecisionVariableValueInVector(p1, [0., 1.], initial_guess)
# Set the value of p2 in initial guess to be [1, 1]
prog.SetDecisionVariableValueInVector(p2, [1., 1.], initial_guess)
print(initial_guess)

fig = plt.figure()
ax = plt.gca()
p1_init = [0., 1.]
p2_init = [1., 1.]
point_p1, = ax.plot(p1_init[0], p1_init[1], 'x')
point_p2, = ax.plot(p2_init[0], p2_init[1], 'x')

ax.axis('equal')
ax.plot(np.cos(np.linspace(0, 2*np.pi, 100)), 2+np.sin(np.linspace(0, 2*np.pi, 100)))
ax.plot(np.linspace(-2, 2, 100), np.power(np.linspace(-2, 2, 100), 2))

def update1(x):
    global iter_count
    point_p1.set_xdata(x[0])
    point_p1.set_ydata(x[1])
    ax.set_title(f"iteration {iter_count}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    # Also update the iter_count variable in the callback.
    # This shows we can do more than just visualization in
    # callback.
    iter_count += 1
    plt.pause(0.1)
def update2(x):
    global iter_count
    point_p2.set_xdata(x[0])
    point_p2.set_ydata(x[1])
    ax.set_title(f"iteration {iter_count}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    # Also update the iter_count variable in the callback.
    # This shows we can do more than just visualization in
    # callback.
    iter_count += 1
    plt.pause(0.1)

iter_count = 0
prog.AddVisualizationCallback(update1, p1)
prog.AddVisualizationCallback(update2, p2)

# Now solve the program
result = Solve(prog, initial_guess)
print(f"Is optimization successful? {result.is_success()}")
p1_sol = result.GetSolution(p1)
p2_sol = result.GetSolution(p2)
print(f"solution to p1 {p1_sol}")
print(f"solution to p2 {p2_sol}")
print(f"optimal cost {result.get_optimal_cost()}")

# Plot the solution.
plt.figure()
plt.plot(np.cos(np.linspace(0, 2*np.pi, 100)), 2+np.sin(np.linspace(0, 2*np.pi, 100)))
plt.plot(np.linspace(-2, 2, 100), np.power(np.linspace(-2, 2, 100), 2))
plt.plot(p1_sol[0], p1_sol[1], '*')
plt.plot(p2_sol[0], p2_sol[1], '*')
plt.axis('equal')
plt.show()

###############################################################################

print("\n==========================\n")





