# Simulate a simple SHO with damping and a driving force in the time domain using scipy.integrate.odeint
#
# The equation of motion is:
# m*x'' + gamma*x' + m*omega_0^2*x = f(t)
# where x is the position of the oscillator, gamma is the damping coefficient, omega_0 is the natural frequency of the oscillator without damping, and f(t) is the driving force.   
#
# The equation of motion can be rewritten as a system of first order ODEs:
# x' = v
# v' = (1/m)*(f(t) - gamma*v - m*omega_0^2*x)
# where v is the velocity of the oscillator.
#
# The system of ODEs can be solved using scipy.integrate.odeint.
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the system of ODEs
def sho(y, t, m, gamma, omega_0, f):
    x, v = y
    dydt = [v, (1/m)*(f(t) - 2*m*gamma*v - m*omega_0**2*x)]
    return dydt

# Define the driving force
def f(t):
    return 2 * np.sin(omega_0 * t)

# Define parameters
omega_0 = 0.15    # Natural frequency of the oscillator without damping
gamma = 0.005 # Damping coefficient
x_0 = 0   # Initial position of the oscillator
v_0 = 0 # Initial velocity of the oscillator
m = 1   # Mass of the oscillator

# Define time
T = 1000
N = 1000 # Number of discretization points
t = np.linspace(0,T,N)

# Solve the system of ODEs
sol = odeint(sho, [x_0, v_0], t, args=(m, gamma, omega_0, f))

# Plot the position of the oscillator
plt.figure()
plt.plot(t,sol[:,0])
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Driven SHO with Damping')
plt.show()