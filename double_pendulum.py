import sympy as sym
import math
from sympy import Function, dsolve, Derivative, checkodesol
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# defining using sympy for symbiolic use
m1 = sym.symbols("m1")
m2 = sym.symbols("m2")
l1 = sym.symbols("l1")
l2 = sym.symbols("l2")
time = sym.symbols('t')
g = sym.symbols('g')
theta1 = sym.symbols(r'\theta_1', cls=sym.Function)
theta2 = sym.symbols(r'\theta_2', cls=sym.Function)
theta1 = theta1(time)
theta2 = theta2(time)
theta1_d = sym.diff(theta1, time)
theta2_d = sym.diff(theta2, time)
theta1_dd = sym.diff(theta1_d, time)
theta2_dd = sym.diff(theta2_d, time)

# Coordinates
x1 = l1 * sym.sin(theta1)
x2 = l1 * sym.sin(theta1) + l2 * sym.sin(theta2)
y1 = -l1 * sym.cos(theta1)
y2 = -l1 * sym.cos(theta1) - l2 * sym.cos(theta2)

T1 = (m1 * ((x1.diff(time))**2) + m1 * ((y1.diff(time))**2) + m2 * ((x2.diff(time))**2) + m2 * ((y2.diff(time))**2)) / 2  # Kinetic Energy due to translation
T2 = m1 * l1**2 * theta1_d**2 / 24 + m2 * l2**2 * theta2_d**2 / 24  # Rotational kinetic energy where I = ML^2/12
U = -g * (m1 * l1 * sym.cos(theta1) + m2 * (l1 * sym.cos(theta1) + l2 * sym.cos(theta2)))  # Net Potential Energy

L = (T1 + T2) - U  # Lagrange of the system

# Euler-Lagrange equations
eqn1 = sym.diff(L, theta1) - sym.diff(L, theta1_d, time).simplify()
eqn2 = sym.diff(L, theta2) - sym.diff(L, theta2_d, time).simplify()

soln = sym.solve([eqn1, eqn2], (theta1_dd, theta2_dd), simplify = False, rational = False)

# Solve for angular accelerations
theta1_dd_sol = sym.simplify(soln[theta1_dd])
theta2_dd_sol = sym.simplify(soln[theta2_dd])

# Convert symbolic solutions to numerical functions
theta1_dd_func = sym.lambdify((theta1, theta1_d, theta2, theta2_d, m1, m2, l1, l2, g), theta1_dd_sol, modules="sympy")
theta2_dd_func = sym.lambdify((theta1, theta1_d, theta2, theta2_d, m1, m2, l1, l2, g), theta2_dd_sol, modules="sympy")

# Numerical Simulation
t = np.linspace(0, 40, 1001)
m1 = float(input("Enter mass of ball 1 in Kgs: "))
m2 = float(input("Enter mass of ball 2 in Kgs: "))
l1 = float(input("Enter length of string 1 in m: "))
l2 = float(input("Enter length of string 2 in m: "))
g = 9.81

def double_pendulum_derivatives(y, t, m1, m2, l1, l2, g):
    theta1, z1, theta2, z2 = y

    theta1_dd = theta1_dd_func(theta1, z1, theta2, z2, m1, m2, l1, l2, g)
    theta2_dd = theta2_dd_func(theta1, z1, theta2, z2, m1, m2, l1, l2, g)

    return [z1, theta1_dd, z2, theta2_dd]

# Initial conditions
theta1_0 = float(input("Enter initial angle of pendulum 1 in degrees: ")) * np.pi / 180
theta2_0 = float(input("Enter initial angle of pendulum 2 in degrees: ")) * np.pi / 180
z1_0 = 0.0
z2_0 = 0.0

initial_conditions = [theta1_0, z1_0, theta2_0, z2_0]
solution = odeint(double_pendulum_derivatives, initial_conditions, t, args=(m1, m2, l1, l2, g))

# Extract angles
theta1, theta2 = solution[:, 0], solution[:, 2]

# defining once again with numpy for numerical use
x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)

# Plot animation
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-l1 - l2 - 1, l1 + l2 + 1)
ax.set_ylim(-l1 - l2 - 1, l1 + l2 + 1)
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
line1, = ax.plot([], [], 'o-', lw=2, color="red")  # For ball 1 trail (pendulum 1)
trail, = ax.plot([], [], '-', lw=1, alpha=0.5)
trail1, = ax.plot([], [], '-', lw=1, alpha=0.5, color="red")  # For ball 1 trail (pendulum 1)
trail_points = 200
trail_data = np.zeros((trail_points, 2))
trail_data1 = np.zeros((trail_points, 2))  # For ball 1 trail data


def init():
    line.set_data([], [])
    line1.set_data([], [])
    trail.set_data([], [])
    trail1.set_data([], [])
    return line, line1, trail, trail1

def update(frame):
    global trail_data, trail_data1

    # For ball 2
    x = [0, x1[frame], x2[frame]]
    y = [0, y1[frame], y2[frame]]

    # For ball 1 trail
    x1_line = [0, x1[frame]]
    y1_line = [0, y1[frame]]

    line.set_data(x, y)
    line1.set_data(x1_line, y1_line)

    # Update the trails
    trail_data = np.vstack((trail_data[1:], [[x2[frame], y2[frame]]]))
    trail_data1 = np.vstack((trail_data1[1:], [[x1[frame], y1[frame]]]))

    trail.set_data(trail_data[:, 0], trail_data[:, 1])
    trail1.set_data(trail_data1[:, 0], trail_data1[:, 1])

    return line, line1, trail, trail1

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=20)
plt.show()

# Ploting x vs t and y vs t with theta1 and theta2 curves
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# x vs t graph
ax1.plot(t, x1, label="x1 (Pendulum 1)")
ax1.plot(t, x2, label="x2 (Pendulum 2)")
ax1.set_xlabel("Time in (s)")
ax1.set_ylabel("X Position (m)")
ax1.legend()
ax1.grid()

# y vs t graph
ax2.plot(t, y1, label="y1 (Pendulum 1)")
ax2.plot(t, y2, label="y2 (Pendulum 2)")
ax2.set_xlabel("Time in (s)")
ax2.set_ylabel("Y Position (m)")
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()