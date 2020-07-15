" Initialize a MathematicalProgram object : "
# To initialize this class, first create an empty MathematicalProgram as
print("\n==========================\n")
from pydrake.solvers.mathematicalprogram import MathematicalProgram
import numpy as np
import matplotlib.pyplot as plt

## Create an empty `MathematicalProgram` named prog (with no decision variables, 
## constraints or cost function)
prog = MathematicalProgram()

###############################################################################

" Adding decision variables : "
# Shown below, the function NewContinuousVariables adds two new continuous decision variables to prog. 
# The newly added variables are returned as x in a numpy array.
# Note the range of the variable is a continuous set, as opposed to binary variables which only take discrete value 0 or 1.
x = prog.NewContinuousVariables(2)

# The default names of the variable in x are "x(0)" and "x(1)". 
# The next line prints the default names and types in x, 
# whereas the second line prints the symbolic expression "1 + 2x[0] + 3x[1] + 4x[1]".
print(x)
print(1 + 2*x[0] + 3*x[1] + 4*x[1])
print("\n==========================\n")
# To create an array y of two variables named "dog(0)"" and "dog(1)", 
# pass the name "dog" as a second argument to `NewContinuousVariables()`. 
# Also shown below is the printout of the two variables in y and a symbolic expression involving y
y = prog.NewContinuousVariables(2, "dog")
print(y)
print(y[0] + y[0] + y[1] * y[1] * y[1])
print("\n==========================\n")
# To create a 3×2 matrix of variables named "A", type
var_matrix = prog.NewContinuousVariables(3, 2, "A")
print(var_matrix)
print("\n==========================\n")

###############################################################################

" Adding constraints : "
# There are many ways to impose constraints on the decision variables. 
# This tutorial shows a few simple examples. 
# Refer to the links at the bottom of this document for other types of constraints.

# AddConstraint Function
# The simplest way to add a constraint is with `MathematicalProgram.AddConstraint()`.

# Add the constraint x(0) * x(1) = 1 to prog
prog.AddConstraint(x[0] * x[1] == 1)
# You can also add inequality constraints to prog such as
prog.AddConstraint(x[0] >= 0)
prog.AddConstraint(x[0] - x[1] <= 0)
# prog automatically analyzes these symbolic inequality constraint expressions and 
# determines they are all linear constraints on x.

################################################################################

" Adding Cost functions : "
# In a complicated optimization problem, it is often convenient to write the total cost function f(x) 
# as a sum of individual cost functions f(x)= ∑ g(x)

# AddCost method
# The simplest way to add an individual cost function g(x) to the total cost function f(x) 
# is with the MathematicalProgram.AddCost() method (as shown below).

# Add a cost x(0)**2 + 3 to the total cost. 
# Since prog doesn't have a cost before, now the total cost is x(0)**2 + 3
prog.AddCost(x[0] ** 2 + 3)
# To add another individual cost function x(0)+x(1) to the total cost function f(x), 
# simply call AddCost() again as follows
prog.AddCost(x[0] + x[1])

# now the total cost function becomes x(0)^2 + x(0) + x(1) + 3.

# `prog` can analyze each of these individual cost functions and determine that 
# x(0)^2 + 3 is a convex quadratic function, and x(0) + x(1) is a linear function of x.

################################################################################

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

from pydrake.solvers.mathematicalprogram import Solve
# Set up the optimization problem.
prog = MathematicalProgram()
x = prog.NewContinuousVariables(2)
prog.AddConstraint(x[0] + x[1] == 1)
prog.AddConstraint(x[0] <= x[1])
prog.AddCost(x[0] **2 + x[1] ** 2)

# Now solve the optimization problem.
result = Solve(prog)

# print out the result.
print("Success? ", result.is_success())
# Print the solution to the decision variables.
print('x* = ', result.GetSolution(x))
# Print the optimal cost.
print('optimal cost = ', result.get_optimal_cost())
# Print the name of the solver that was called.
print('solver is: ', result.get_solver_id().name())

print("\n==========================\n")

# Some optimization solution is infeasible (doesn't have a solution). 
# For example in the following code example, result.get_solution_result() will not report kSolutionFound.

"""
An infeasible optimization problem.
"""
prog = MathematicalProgram()
x = prog.NewContinuousVariables(1)[0]
y = prog.NewContinuousVariables(1)[0]
prog.AddConstraint(x + y >= 1)
prog.AddConstraint(x + y <= 0)
prog.AddCost(x)

result = Solve(prog)
print("Success? ", result.is_success())
print(result.get_solution_result())

print("\n==========================\n")

################################################################################

" Add callback : "
# Some solvers support adding a callback function in each iteration. 
# One usage of the callback is to visualize the solver progress in the current iteration. 
# MathematicalProgram supports this usage through the function AddVisualizationCallback, 
# although the usage is not limited to just visualization, the callback function can do anything. 
# Here is an example,

# Visualize the solver progress in each iteration through a callback
# Find the closest point on a curve to a desired point.

fig = plt.figure()
curve_x = np.linspace(1, 20, 200)
ax = plt.gca()
ax.plot(curve_x, 9./curve_x)
ax.plot(-curve_x, -9./curve_x)
ax.plot(0, 0, 'o')
x_init = [0., 1.]
point_x, = ax.plot(x_init[0], x_init[1], 'x')
ax.axis('equal')

def update(x):
    global iter_count
    point_x.set_xdata(x[0])
    point_x.set_ydata(x[1])
    ax.set_title(f"iteration {iter_count}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    # Also update the iter_count variable in the callback.
    # This shows we can do more than just visualization in
    # callback.
    iter_count += 1
    plt.pause(0.1)
    
iter_count = 0
prog = MathematicalProgram()
x = prog.NewContinuousVariables(2)
prog.AddConstraint(x[0] * x[1] == 9)
prog.AddCost(x[0]**2 + x[1]**2)
prog.AddVisualizationCallback(update, x)
result = Solve(prog, x_init)

