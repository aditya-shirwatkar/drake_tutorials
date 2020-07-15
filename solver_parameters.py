print(" Solver Parameters Tutorial ")

" Setting solver parameters : "
# Many solvers allow the users to adjust the parameters. 
# When calling Solve() function, Drake will use the default parameters 
# for the solver (iterations, optimality tolerance, etc). 
# You could modify these parameters in two ways, by either 
# calling MathematicalProgram::SetSolverOption, or 
# pass a SolverOptions argument to the Solve() function.
print("\n==========================\n")

# Calling MathematicalProgram::SetSolverOption
# By calling MathematicalProgram::SetSolverOption(solver_id, option_name, option_value), 
# you can set a parameter for a specific solver (with the matching solver_id). 
# The option_name is specific to that solver (for example, here is a list of IPOPT parameters). 
# Note that MathematicalProgram object will store this solver parameter, 
# and this parameter will be applied in the Solve() call, 
# if that specific solver (with the matching solver_id) is invoked.
# In the following code snippet, we show an example of setting the options of IPOPT.

from pydrake.solvers.mathematicalprogram import MathematicalProgram, SolverOptions, Solve
from pydrake.solvers.ipopt import IpoptSolver
import numpy as np
prog = MathematicalProgram()
x = prog.NewContinuousVariables(2)
prog.AddCost(x[0]**2 + x[1] ** 2)
prog.AddConstraint(x[0] + x[1] == 1)

# Set the maximum iteration for IPOPT to be 1.
# max_iter is a parameter of IPOPT solver, explained in
# https://www.coin-or.org/Ipopt/documentation/node42.html
prog.SetSolverOption(IpoptSolver().solver_id(), "max_iter", 1)
solver = IpoptSolver()
result = solver.Solve(prog, np.array([10, 1]), None)
print("With fewer maximum iteration, IPOPT hasn't converged to optimality yet (The true optimal is [0.5, 0.5])")
print("Success? ", result.is_success())
print(result.get_solution_result())
print("IPOPT x*= ", result.GetSolution(x))

print("\n==========================\n")

###############################################################################


# Also note that setting the parameter of a solver doesn't mean that 
# result = Solve(prog) will invoke that solver. 
# The invoked solver is determined by Drake, 
# to choose whichever solver it thinks most appropriate.

# In the following snippet, although we set the solver options for IPOPT,
# Drake chooses another solver (which can solve this particular problem in the closed form.)

prog.SetSolverOption(IpoptSolver().solver_id(), "max_iter", 1)
result = Solve(prog)
print(result.get_solver_id().name())

print("\n==========================\n")

###############################################################################

" Passing a SolverOptions to Solve function : "
# Another way of setting the solver options is to pass in a SolverOptions 
# object as an argument to Solve function. 
# MathematicalProgram will not store this SolverOptions object.

# In the following example, in the first Solve call, 
# it uses the SolverOptions object to set the parameter for IPOPT; 
# in the second Solve call, it uses the default IPOPT parameters, 
# hence we get different results from two Solve calls.

prog = MathematicalProgram()
x = prog.NewContinuousVariables(2)
prog.AddCost(x[0]**2 + x[1] ** 2)
prog.AddConstraint(x[0] + x[1] == 1)

solver_options = SolverOptions()
solver_options.SetOption(IpoptSolver().solver_id(), "max_iter", 1)
solver = IpoptSolver()

# Call Solve with solver_options, IPOPT will use `max_iter` = 1
result = solver.Solve(prog, np.array([10, 1]), solver_options)
print("Success? ", result.is_success())
print(result.get_solution_result())
# Call Solve without solver_options, IPOPT will use the default options.
result = solver.Solve(prog, np.array([10, 1]), None)
print("Success? ", result.is_success())
print(result.get_solution_result())

print("\n==========================\n")

