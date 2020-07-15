" Updating costs and constraints in MathematicalProgram "
# Often cases after we solve an optimization problem, 
# we want to tweak its constraints and costs slightly, 
# and then resolve the updated problem. 
# One use case is in model predictive control, 
# where in each time instance we solve a new optimization problem, 
# whose constraints/costs are just slightly different 
# from the one in the previous time instance.

# Instead of constructing a new MathematicalProgram object, 
# we could update the constraints/costs in the old MathematicalProgram object, 
# and then solve the updated problem. 
# To do so, many constraints/costs object provide an "update" function. 
# In this tutorial we show how to update certain types of constraints/costs

###############################################################################

print("\n==========================\n")

" Updating LinearCost "
# For a linear cost a.T*x + b, we could call LinearCost.UpdateCoefficients() 
# function to update the linear coefficient a vector or the constant term b

from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
import numpy as np

prog = MathematicalProgram()
x = prog.NewContinuousVariables(2)
cost1 = prog.AddLinearCost(2*x[0] + 3 * x[1] + 2)
print(f"original cost: {cost1}")
prog.AddBoundingBoxConstraint(-1, 1, x)
result = Solve(prog)
print(f"optimal solution: {result.GetSolution(x)}")
print(f"original optimal cost: {result.get_optimal_cost()}")

# Now update the cost to 3x[0] - 4x[1] + 5
cost1.evaluator().UpdateCoefficients(new_a=[3, -4], new_b=5)
print(f"updated cost: {cost1}")
# Solve the optimization problem again with the updated cost.
result = Solve(prog)
print(f"updated optimal solution: {result.GetSolution(x)}")
print(f"updated optimal cost: {result.get_optimal_cost()}")

###############################################################################

print("\n==========================\n")

" Updating QuadraticCost : "
# For a quadratic cost in the form 0.5*x.T*Q*x + b*x + c, 
# we can also call QuadraticCost.UpdateCoefficients to update its Q,b,c terms

prog = MathematicalProgram()
x = prog.NewContinuousVariables(2)
cost1 = prog.AddQuadraticCost(x[0]**2 + 2 * x[1]**2 + x[0]*x[1] + 3*x[0] + 5)
print(f" original cost: {cost1}")
cost1.evaluator().UpdateCoefficients(new_Q=[[1., 2], [2., 4]], new_b=[1, 2.], new_c= 4)
print(f" updated cost: {cost1}")

###############################################################################

print("\n==========================\n")

" Updating the bounds for any constraint "
# For any constraint lower ≤ f(x) ≤ upper, we can update its bounds by
## Constraint.UpdateLowerBound(new_lb) to change its lower bound to new_lb.
## Constraint.UpdateUpperBound(new_ub) to change its upper bound to new_ub.
## Constraint.set_bounds(new_lb, new_ub) up change both its lower and upper bounds

prog = MathematicalProgram()
x = prog.NewContinuousVariables(2)
constraint1 = prog.AddLinearConstraint(x[0] + 3 * x[1] <= 2)
print(f"original constraint: {constraint1}")
constraint1.evaluator().UpdateLowerBound([-1])
print(f"updated constraint: {constraint1}")
constraint1.evaluator().UpdateUpperBound([3])
print(f"updated constraint: {constraint1}")
constraint1.evaluator().set_bounds(new_lb=[-5], new_ub=[10])
print(f"updated constraint: {constraint1}")

###############################################################################

print("\n==========================\n")

" Update linear constraint coefficients and bounds : "
# For a linear constraint lower≤Ax≤upper, 
# we can call LinearConstraint.UpdateCoefficients(new_A, new_lb, new_ub) to update the 
# constraint as new_lb≤new_A∗x≤new_ub.

# For a linear equality constraint Ax=b, 
# we can call LinearEqualityConstraint.UpdateCoefficients(Aeq, beq) to update the
# constraint to Aeq∗x=beq.

prog = MathematicalProgram()
x = prog.NewContinuousVariables(2)
linear_constraint = prog.AddLinearConstraint(3 * x[0] + 4 * x[1] <= 5)
linear_eq_constraint = prog.AddLinearConstraint(5 * x[0] + 2 * x[1] == 3)
print(f"original linear constraint： {linear_constraint}")
linear_constraint.evaluator().UpdateCoefficients(new_A = [[1, 3]], new_lb=[-2], new_ub=[3])
print(f"updated linear constraint： {linear_constraint}")

print(f"original linear equality constraint： {linear_eq_constraint}")
linear_eq_constraint.evaluator().UpdateCoefficients(Aeq=[[3, 4]], beq=[2])
print(f"updated linear equality constraint： {linear_eq_constraint}")

###############################################################################

print("\n==========================\n")
