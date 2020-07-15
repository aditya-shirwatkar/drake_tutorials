" Linear Program (LP) Tutorial "
# A linear program (LP) is a special type of optimization problem. 
# The cost and constraints in an LP is a linear (affine) function of decision variables. 
# The mathematical formulation of a general LP is
# min (x*c.T*x+d) , subject to A*x ≤ b 
# A linear program can be solved by many open source or commercial solvers. 
# Drake supports some solvers including SCS, Gurobi, Mosek, etc. 
# Please see the Doxygen page for a complete list of supported solvers. 
# Note that some commercial solvers (such as Gurobi and Mosek) are not included in the pre-compiled Drake binaries, and therefore not on Binder/Colab.
# Drake's API supports multiple functions to add linear cost and constraints. 
# We briefly go through some of the functions in this tutorial. 
# For a complete list of functions, please check the Doxygen.

###############################################################################

print("\n==========================\n")

" Add linear cost "
# The easiest way to add linear cost is to call AddLinearCost function. 
# We first demonstrate how to construct an optimization program with 2 decision variables, 
# then we will call AddLinearCost to add the cost.

from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
import numpy as np

# Create an empty MathematicalProgram named prog (with no decision variables,
# constraints or costs)
prog = MathematicalProgram()
# Add two decision variables x[0], x[1].
x = prog.NewContinuousVariables(2, "x")

# We can call AddLinearCost(expression) to add a new linear cost. 
# expression is a symbolic linear expression of the decision variables.

# Add a symbolic linear expression as the cost.
cost1 = prog.AddLinearCost(x[0] + 3 * x[1] + 2)
# Print the newly added cost
print(cost1)
# The newly added cost is stored in prog.linear_costs().
print(prog.linear_costs()[0])

# If we call AddLinearCost again, the total cost stored in prog is the summation of all the costs. 
# You can see that prog.linear_costs() will have two entries.

cost2 = prog.AddLinearCost(2 * x[1] + 3)
print(f"number of linear cost objects: {len(prog.linear_costs())}")

# If you know the coefficient of the linear cost as a vector, 
# you could also add the cost by calling AddLinearCost(e, f, x) 
# which will add a linear cost eTx+f to the optimization program

# We add a linear cost 3 * x[0] + 4 * x[1] + 5 to prog by specifying the coefficients
# [3., 4] and the constant 5 in AddLinearCost
cost3 = prog.AddLinearCost([3., 4.], 5., x)
print(cost3)

# Lastly, the user can call AddCost to add a linear expression to the linear cost. 
# Drake will analyze the structure of the expression, 
# if Drake determines the expression is linear, then the added cost is linear.

print(f"number of linear cost objects before calling AddCost: {len(prog.linear_costs())}")
# Call AddCost to add a linear expression as linear cost. After calling this function,
# len(prog.linear_costs()) will increase by 1.
cost4 = prog.AddCost(x[0] + 3 * x[1] + 5)
print(f"number of linear cost objects after calling AddCost: {len(prog.linear_costs())}")


###############################################################################

print("\n==========================\n")

" Add linear constraints "
# We have three types of linear constraints
# Bounding box constraint. A lower/upper bound on the decision variable: lower≤x≤upper.
# Linear equality constraint: Ax=b.
# Linear inequality constraint: lower<=Ax<=upper.

# AddLinearConstraint and AddConstraint function
# The easiest way to add linear constraints is to call AddConstraint or AddLinearConstraint function,
# which can handle all three types of linear constraint. 
# Compared to the generic AddConstraint function, 
# AddLinearConstraint does more sanity will refuse to add the 
# constraint if it is not linear.

prog = MathematicalProgram()
x = prog.NewContinuousVariables(2, "x")
y = prog.NewContinuousVariables(3, "y")

# Call AddConstraint to add a bounding box constraint x[0] >= 1
bounding_box1 = prog.AddConstraint(x[0] >= 1)
print(f"number of bounding box constraint objects: {len(prog.bounding_box_constraints())}")

# Call AddLinearConstraint to add a bounding box constraint x[1] <= 2
bounding_box2 = prog.AddLinearConstraint(x[1] <= 2)
print(f"number of bounding box constraint objects: {len(prog.bounding_box_constraints())}")

# Call AddConstraint to add a linear equality constraint x[0] + y[1] == 3
linear_eq1 = prog.AddConstraint(x[0] + y[1] == 3.)
print(f"number of linear equality constraint objects: {len(prog.linear_equality_constraints())}")

# Call AddLinearConstraint to add a linear equality constraint x[1] + 2 * y[2] == 1
linear_eq2 = prog.AddLinearConstraint(x[1] + 2 * y[2] == 1)
print(f"number of linear equality constraint objects: {len(prog.linear_equality_constraints())}")

# Call AddConstraint to add a linear inequality constraint x[0] + 3*x[1] + 2*y[2] <= 4
linear_ineq1 = prog.AddConstraint(x[0] + 3*x[1] + 2*y[2] <= 4)
print(f"number of linear inequality constraint objects: {len(prog.linear_constraints())}")

# Call AddLinearConstraint to add a linear inequality constraint x[1] + 4 * y[1] >= 2
linear_ineq2 = prog.AddLinearConstraint(x[1] + 4 * y[1] >= 2)
print(f"number of linear inequality constraint objects: {len(prog.linear_constraints())}")

# AddLinearConstraint will check if the constraint is actually linear, 
# and throw an exception if the constraint is not linear.

# Add a nonlinear constraint square(x[0]) == 2 by calling AddLinearConstraint. This should
# throw an exception
try:
    prog.AddLinearConstraint(x[0] ** 2 == 2)
except RuntimeError as err:
    print(err.args)

# If the users know the coefficients of the constraint as a matrix, 
# they could also call AddLinearConstraint(A, lower, upper, x) 
# to add a constraint lower ≤ Ax ≤ upper. 
# This version of the method does not construct any symbolic representations, 
# and will be more efficient especially when A is very large.
# Add a linear constraint 2x[0] + 3x[1] <= 2, 1 <= 4x[1] + 5y[2] <= 3.
# This is equivalent to lower <= A * [x;y[2]] <= upper with
# lower = [-inf, 1], upper = [2, 3], A = [[2, 3, 0], [0, 4, 5]].
linear_constraint = prog.AddLinearConstraint(
    A=[[2., 3., 0], [0., 4., 5.]],
    lb=[-np.inf, 1],
    ub=[2., 3.],
    vars=np.hstack((x, y[2])))
print(linear_constraint)

###############################################################################

print("\n==========================\n")

" AddBoundingBoxConstraint "
# If your constraint is a bounding box constraint (i.e. lower≤x≤upper), 
# apart from calling AddConstraint or AddLinearConstraint, 
# you could also call AddBoundingBoxConstraint(lower, upper, x), 
# which will be slightly faster than AddConstraint and AddLinearConstraint.

# Add a bounding box constraint -1 <= x[0] <= 2, 3 <= x[1] <= 5
bounding_box3 = prog.AddBoundingBoxConstraint([-1, 3], [2, 5], x)
print(bounding_box3)

# Add a bounding box constraint 3 <= y[i] <= 5 for all i.
bounding_box4 = prog.AddBoundingBoxConstraint(3, 5, y)
print(bounding_box4)

" AddLinearEqualityConstraint "
# If your constraint is a linear equality constraint (i.e. Ax=b), 
# apart from calling AddConstraint or AddLinearConstraint, 
# you could also call AddLinearEqualityConstraint to be more specific 
# (and slightly faster than AddConstraint and AddLinearConstraint).

# Add a linear equality constraint 4 * x[0] + 5 * x[1] == 1
linear_eq3 = prog.AddLinearEqualityConstraint(np.array([[4, 5]]), np.array([1]), x)
print(linear_eq3)


###############################################################################

print("\n==========================\n")

" Solving Linear Program "
# Once all the constraints and costs are added to the program, 
# we can call Solve function to solve the program and call GetSolution to obtain the results.

# Solve an optimization program
# min -3x[0] - x[1] - 5x[2] -x[3] + 2
# s.t 3x[0] + x[1] + 2x[2] = 30
#     2x[0] + x[1] + 3x[2] + x[3] >= 15
#     2x[1] + 3x[3] <= 25
#     -100 <= x[0] + 2x[2] <= 40
#   x[0], x[1], x[2], x[3] >= 0, x[1] <= 10

prog = MathematicalProgram()
# Declare x as decision variables.
x = prog.NewContinuousVariables(4)
# Add linear costs. To show that calling AddLinearCosts results in the sum of each individual
# cost, we add two costs -3x[0] - x[1] and -5x[2]-x[3]+2
prog.AddLinearCost(-3*x[0] -x[1])
prog.AddLinearCost(-5*x[2] - x[3] + 2)
# Add linear equality constraint 3x[0] + x[1] + 2x[2] == 30
prog.AddLinearConstraint(3*x[0] + x[1] + 2*x[2] == 30)
# Add Linear inequality constraints
prog.AddLinearConstraint(2*x[0] + x[1] + 3*x[2] + x[3] >= 15)
prog.AddLinearConstraint(2*x[1] + 3*x[3] <= 25)
# Add linear inequality constraint -100 <= x[0] + 2x[2] <= 40
prog.AddLinearConstraint(A=[[1., 2.]], lb=[-100], ub=[40], vars=[x[0], x[2]])
prog.AddBoundingBoxConstraint(0, np.inf, x)
prog.AddLinearConstraint(x[1] <= 10)

# Now solve the program.
result = Solve(prog)
print(f"Is solved successfully: {result.is_success()}")
print(f"x optimal value: {result.GetSolution(x)}")
print(f"optimal cost: {result.get_optimal_cost()}")

###############################################################################

print("\n==========================\n")

