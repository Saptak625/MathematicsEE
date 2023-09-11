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
    return 2 * np.sin(omega_0 * t)

# Define parameters
omega_0 = 0.15    # Natural frequency of the oscillator without damping
gamma = 0.005 # Damping coefficient
x_0 = 0   # Initial position of the oscillator
m = 1   # Mass of the oscillator

# Define time
T = 1000
N = 1000 # Number of discretization points
dt = T/N
t = np.linspace(0,T,N)

# Make sure to satify the x_0 and v_0 initial conditions
# Take the Fourier transform of f(t)
F = np.fft.fft(f(t))

# Define omega
omega = 2*np.pi * np.fft.fftfreq(N, d=dt)
omega = np.fft.fftshift(omega)

# Solve for X
X = F/(m*(omega_0**2 - np.power(omega, 2) + 2*1j*gamma*omega))

# Set the initial conditions where omega = 0
# X[omega == 0] = x_0

# Plot the Fourier transform of f(t) and X
plt.figure()
plt.plot(omega,abs(F), 'rx', label='Fourier Transform of f(t)')
plt.plot(omega,abs(X), 'bx', label='Fourier Transform of x(t)')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Driven SHO with Damping')
plt.legend(loc=1)
plt.grid()
plt.show()

# Take the inverse Fourier transform of X
x = np.fft.ifft(X)

# Plot the position of the oscillator
plt.figure()
plt.plot(t,x.real)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Driven SHO with Damping')
plt.grid()
plt.show()