" Modeling Dynamical Systems"
# This Tutorial provides a short tutorial for modeling input-output dynamical systems in Drake. It covers
## Writing your own simple dynamical systems,
## Simulating a dynamical system and plotting the results,
## Constructing a block diagram out of simple systems.

###############################################################################
###############################################################################

" Writing your own dynamics "
# In this section, we will describe how you can write your own dynamics 
# class in pydrake. Every user system should derive from the 
# pydrake.systems.framework.LeafSystem class.

# However, many dynamical systems we are interested in are 
# represented by a simple vector field in state-space form, 
# where we often use x to denote the state vector, u for the input vector, 
# and y for the output vector. 
# To make it even easier to implement systems in this form, 
# we have another subclass pydrake.systems.primitives.SymbolicVectorSystem 
# that makes it very easy to author these simple systems.
# u → System → y

###############################################################################
print("\n==========================\n")
print("\n------Writing Your Own Dynamics----------\n")

"Using SymbolicVectorSystem"
# Consider a basic continuous-time, nonlinear, input-output dynamical system 
# described by the following state-space equations:
# xdot = f(t,x,u), y= g(t,x,u).
# In pydrake, you can instantiate a system of this form 
# where f() and g() are anything that you can write in Python 
# using operations supported by the Drake symbolic engine, 
# as illustrated by the following example.

# Consider the system
# xdot = −x + x^3, y = x
# This system has zero inputs, one (continuous) state variable, 
# and one output. It can be implemented in Drake using the following code:
from pydrake.symbolic import Variable
from pydrake.systems.primitives import SymbolicVectorSystem

# Define a new symbolic Variable
x = Variable("x")

# Define the System.  
continuous_vector_system = SymbolicVectorSystem(state=[x], dynamics=[-x + x**3], output=[x])
print(f'Continous System using SymnolicVectorSystem : {continuous_vector_system}')
# That's it! The continuous_vector_system variable is now an instantiation of 
# a Drake System class, that can be used in a number of ways that we will 
# illustrate below. Note that the state argument expects a vector of 
# symbolic::Variable (python lists get automatically converted), and 
# the dynamics and output arguments expect a vector of symbolic::Expressions.

#########################

# Implementing a basic discrete-time system in Drake is very analogous to 
# implementing a continuous-time system. The discrete-time system given by:
# x[n+1]=f(n,x,u), y[n]=g(n,x,u),
# can be implemented as seen in the following example.

# Consider the system
# x[n+1]=x^3[n], y[n]=x[n].
# This system has zero inputs, one (discrete) state variable, and one output. 
# It can be implemented in Drake using the following code:
from pydrake.symbolic import Variable
from pydrake.systems.primitives import SymbolicVectorSystem

# Define a new symbolic Variable
x = Variable("x")

# Define the System.  Note the additional argument specifying the time period.
discrete_vector_system = SymbolicVectorSystem(state=[x], dynamics=[x**3], output=[x], time_period=1.0)
print(f'Discrete System using SymnolicVectorSystem : {discrete_vector_system}')

###############################################################################
print("\n==========================\n")

" Deriving from LeafSystem "
# Although using SymbolicVectorSystems are a nice way to get started, 
# in fact Drake supports authoring a wide variety of systems with multiple inputs and outputs, 
# mixed discrete- and continuous- dynamics, hybrid dynamics with guards and resets, 
# systems with constraints, and even stochastic systems. 
# To expose more of the underlying system framework, 
# you can derive your system from pydrake.systems.framework.LeafSystem directly, 
# instead of using the simplified SymbolicVectorSystem interface.

from pydrake.systems.framework import BasicVector, LeafSystem

# Define the system.
class SimpleContinuousTimeSystem(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        
        self.DeclareContinuousState(1)                                        # One state variable.
        self.DeclareVectorOutputPort("y", BasicVector(1), self.CopyStateOut)  # One output.

    # xdot(t) = -x(t) + x^3(t)
    def DoCalcTimeDerivatives(self, context, derivatives):
        x = context.get_continuous_state_vector().GetAtIndex(0)
        xdot = -x + x**3
        derivatives.get_mutable_vector().SetAtIndex(0, xdot)

    # y = x
    def CopyStateOut(self, context, output):
        x = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(x)

# Instantiate the System        
continuous_system = SimpleContinuousTimeSystem()
print(f'Continuous System using LeafSystem : {continuous_system}')

# Define the system.
class SimpleDiscreteTimeSystem(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        
        self.DeclareDiscreteState(1)                                          # One state variable.
        self.DeclareVectorOutputPort("y", BasicVector(1), self.CopyStateOut)  # One output.
        self.DeclarePeriodicDiscreteUpdate(1.0)                               # One second timestep.

    # x[n+1] = x^3[n]
    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        x = context.get_discrete_state_vector().GetAtIndex(0)
        xnext = x**3
        discrete_state.get_mutable_vector().SetAtIndex(0, xnext)

    # y = x
    def CopyStateOut(self, context, output):
        x = context.get_discrete_state_vector().CopyToVector()
        output.SetFromVector(x)

# Instantiate the System        
discrete_system = SimpleDiscreteTimeSystem()
print(f'Discrete System using LeafSystem : {discrete_system}')

# This code implements the same systems we implemented above. 
# Unlike the SymbolicVectorSystem version, you can type any valid python/numpy code 
# into the methods here -- it need not be supported by pydrake.symbolic. 
# Also, we now have the opportunity to overload many more pieces of functionality in LeafSystem, 
# to produce much more complex systems 
# (with multiple input and output ports, mixed discrete- and continuous- state, more structured state, hybrid systems with witness functions and reset maps, etc). 
# But declaring a LeafSystem this way does not provide support for Drake's autodiff and symbolic tools out of the box; 
# to do that we need to add a few more lines to support templates.
# Drake also supplies a number of other helper classes and methods that derive 
# from or construct a LeafSystem, such as the pydrake.systems.primitives.LinearSystem class 
# or the pydrake.systems.primitives.Linearize() method. And in many cases, 
# like simulating the dynamics and actuators/sensors of robots, 
# most of the classes that you need have already been implemented.

###############################################################################
###############################################################################

print("\n==========================\n")
print("\n------Simulation----------\n")

"Simulation"
# Once you have acquired a System object describing the dynamics of interest, 
# the most basic thing that you can do is to simulate it. 
# This is accomplished with the pydrake.framework.analysis.Simulator class. 
# This class provides access to a rich suite of numerical integration routines, 
# with support for variable-step integration, stiff solvers, and event detection.
# In order to view the data from a simulation after the simulation has run, 
# you should add a pydrake.framework.primitives.SignalLogger system to your diagram.

# Use the following code to simulate the continuous time system we defined above, and plot the results:"
import matplotlib.pyplot as plt
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogOutput

# Create a simple block diagram containing our system.
builder = DiagramBuilder()
system = builder.AddSystem(SimpleContinuousTimeSystem())
logger = LogOutput(system.get_output_port(0), builder)
diagram = builder.Build()

# Set the initial conditions, x(0).
context = diagram.CreateDefaultContext()
context.SetContinuousState([0.9])

# Create the simulator, and simulate for 10 seconds.
simulator = Simulator(diagram, context)
simulator.AdvanceTo(10)

# Plot the results.
plt.figure()
plt.plot(logger.sample_times(), logger.data().transpose())
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()

# Create a simple block diagram containing our system.
builder = DiagramBuilder()
system = builder.AddSystem(SimpleDiscreteTimeSystem())
logger = LogOutput(system.get_output_port(0), builder)
diagram = builder.Build()

# Create the simulator.
simulator = Simulator(diagram)

# Set the initial conditions, x(0).
state = simulator.get_mutable_context().get_mutable_discrete_state_vector()
state.SetFromVector([0.9])

# Simulate for 10 seconds.
simulator.AdvanceTo(10)

# Plot the results
plt.figure()
plt.stem(logger.sample_times(), logger.data().transpose(), use_line_collection=True)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.show()

# Go ahead and try using the SymbolicVectorSystem versions instead. They work, too.
# For many systems, the simulation will run much faster than real time. 
# If you would like to tell the simulator to slow down (if possible) to 
# some multiple of the real time clock, then consider using the set_target_realtime_rate() method 
# of the Simulator. This is useful, for example, if you are trying to 
# animate your robot as it simulates, and would like the benefit of physical intuition, 
# or if you are trying to use the simulation as a part of a multi-process real-time control system.

###############################################################################

print("\n==========================\n")
print("\n------Pendulum Example----------\n")


" The System `Context` "
# If you were looking carefully, you might have noticed a few instances of the word "context" 
# in the code snippets above. The Context is a core concept in the Drake systems framework: 
# the Context captures all of the (potentially) dynamic information that a System 
# requires to implement its core methods: this includes the time, the state, any inputs, and 
# any system parameters. The Context of a System is everything you need to know for simulation 
# (or control design, ...), and given a Context all methods called on a System should be 
# completely deterministic/repeatable.
# Systems know how to create an instance of a Context (see CreateDefaultContext).
# In the simulation example above, the Simulator created a Context for us. 
# We retrieved the Context from the Simulator in order to set the initial conditions (state) 
# of the system before running the simulation.

# Note that a Context is not completely defined unless all of the input ports are connected 
# (simulation and other method calls will fail if they are not). For input ports that are 
# not directly tied to the output of another system, consider using the FixInputPort method 
# of the Context.

" Combinations of Systems: Diagram and DiagramBuilder "
# The real modeling power of Drake comes from combining many smaller systems 
# together into more complex systems. The concept is very simple: we use the 
# DiagramBuilder class to AddSystem()s and to Connect() input ports to output ports or 
# to expose them as inputs/output of the diagram. 
# Then we call Build() to generate the new Diagram instance, which is just another
# System in the framework, and can be simulated or analyzed using the entire suite of tools.

# In the example below, we connect three subsystems (a plant, a controller, and a logger), 
# and expose the input of the controller as an input to the Diagram being constructed:

import numpy as np

from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.examples.pendulum import PendulumPlant
from pydrake.systems.controllers import PidController
from pydrake.systems.framework import GenerateHtml
from pydrake.systems.primitives import LogOutput

builder = DiagramBuilder()

# First add the pendulum.
pendulum = builder.AddSystem(PendulumPlant())
pendulum.set_name("pendulum")

controller = builder.AddSystem(PidController(kp=[10.], ki=[10.], kd=[10.]))
controller.set_name("controller")

# Now "wire up" the controller to the plant.
builder.Connect(pendulum.get_state_output_port(), controller.get_input_port_estimated_state())
builder.Connect(controller.get_output_port_control(), pendulum.get_input_port())

# Make the desired_state input of the controller an input to the diagram.
builder.ExportInput(controller.get_input_port_desired_state())

# Log the state of the pendulum.
logger = LogOutput(pendulum.get_state_output_port(), builder)
logger.set_name("logger")

diagram = builder.Build()
diagram.set_name("diagram")

html_str = GenerateHtml(diagram)
f = open("Diagram.html","w")
f.write(html_str)
f.close()

import webbrowser
new = 2 # open in a new tab, if possible
webbrowser.open_new_tab('Diagram.html')

import matplotlib.pyplot as plt

# Set up a simulator to run this diagram.
simulator = Simulator(diagram)
context = simulator.get_mutable_context()

# We'll try to regulate the pendulum to a particular angle.
desired_angle = np.pi/2.

# First we extract the subsystem context for the pendulum.
pendulum_context = diagram.GetMutableSubsystemContext(pendulum, context)
# Then we can set the pendulum state, which is (theta, thetadot).
pendulum_context.get_mutable_continuous_state_vector().SetFromVector([0., 0.2])

# The diagram has a single input port (port index 0), which is the desired_state.
context.FixInputPort(0, [desired_angle, 0.])

# Reset the logger only because we've written this notebook with the opportunity to 
# simulate multiple times (in this cell) using the same logger object.  This is 
# often not needed.
logger.reset()

# Simulate for 10 seconds.
simulator.AdvanceTo(20)

# Plot the results.
t = logger.sample_times()
plt.figure()
plt.title('PID Control of the Pendulum')
# Plot theta.
plt.subplot(211)
plt.plot([t[0], t[-1]], [desired_angle, desired_angle], 'g' )
plt.xlabel('time (seconds)')
plt.plot(t, logger.data()[0,:],'r.-')
plt.ylabel('theta (rad)')
# Plot theta_dot
plt.subplot(212)
plt.plot(t, logger.data()[1,:],'b.-')
plt.plot([t[0], t[-1]], [0, 0], 'g' )
plt.xlabel('time (seconds)')
plt.ylabel('theta_dot (rad)')
# Draw a line for the desired angle.
plt.show()
###############################################################################

print("\n==========================\n")

import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [0, np.cos(logger.data()[0, i])]
    thisy = [0, np.sin(logger.data()[0, i])]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*20/len(t)))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(logger.data()[0, :])),
                              interval=25, blit=True, init_func=init)

ani.save('double_pendulum.mp4', fps=60)
plt.show()












