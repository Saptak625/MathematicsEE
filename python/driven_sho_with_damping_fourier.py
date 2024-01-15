# Simulate a simple SHO with damping and a driving force in the time domain using fourier transforms
#
# The equation of motion is:
# x'' + 2*gamma*x' + omega_0^2*x = f(t)/m
# where x is the position of the oscillator, gamma is the damping coefficient, omega_0 is the natural frequency of the oscillator without damping, and f(t) is the driving force.
#
# Taking the Fourier transform of both sides of the equation of motion gives:
# -omega^2*X + 2*gamma*i*omega*X + omega_0^2*X = F/m
# where X is the Fourier transform of x, omega is the frequency, and F is the Fourier transform of f(t).
#
# Solving for X gives:
# X = F/(m*(omega_0^2 - omega^2 + 2*i*gamma*omega))
#
# Taking the inverse Fourier transform of X gives the position of the oscillator in the time domain.

import numpy as np
import matplotlib.pyplot as plt

# Define the driving force
def f(t):
    return np.exp(-t/500)

# Define parameters
omega_0 = 0.15 / (2 * np.pi)    # Natural frequency of the oscillator without damping
gamma = 0.0001 # Damping coefficient.
m = 1000   # Mass of the oscillator

# Define time
T = 2500.0
N = 100000 # Number of discretization points
dt = T/N
t = np.linspace(0,T,N)

# Make sure to satify the x_0 and v_0 initial conditions
# Take the Fourier transform of f(t)
F = np.fft.fft(f(t))

# Define omega
omega = np.fft.fftfreq(N, d=dt)

# Solve for X
x_0 = 0.5 # Initial position of the oscillator
v_0 = 0 # Initial velocity of the oscillator
C_1 = (-82.385 * x_0 * omega_0**2 - 31.91 * v_0 * omega_0)/dt
C_2 = (4.78 * x_0 * omega_0**2 - 0.3137 * v_0 * omega_0)/dt
X = (F - m*C_1*1j*omega - m*(2*gamma*C_1 + C_2))/(m*(omega_0**2 - np.power(omega, 2) + 2*1j*gamma*omega))

# Plot the Fourier transform of f(t) and X
plt.figure("Driven SHO with Damping")
plt.plot(omega,F.real, label='Real f(t)')
plt.plot(omega,F.imag, label='Imaginary f(t)')
plt.plot(omega,X.real, label='Real X')
plt.plot(omega,X.imag, label='Imaginary X')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Driven SHO with Damping')
plt.legend(loc=1)
plt.grid()

# Take the inverse Fourier transform of X
x = np.fft.ifft(X)

# Plot the position of the oscillator
plt.figure('Driven SHO with Damping Position')
plt.plot(t,x.real, label='Fourier Solution')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Driven SHO with Damping')
plt.grid()
plt.show()