
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

