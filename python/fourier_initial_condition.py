# Use Fourier Transforms to have some initial conditions and then evolve the system in time.
import numpy as np
import matplotlib.pyplot as plt

# Parameters
omega_0 = 15    # Natural frequency of the oscillator without damping
gamma = 0.5 # Damping coefficient
x_0 = 0.5   # Initial position of the oscillator
m = 1   # Mass of the oscillator

# Define time
T = 10
N = 1000 # Number of discretization points
dt = T/N
t = np.linspace(0,T,N)

# Define the driving force
def f(t):
    return np.zeros(len(t))

# Make sure to satify the x_0 and v_0 initial conditions
# Take the Fourier transform of f(t)
F = np.fft.fft(f(t))

# Define omega
omega = 2*np.pi * np.fft.fftfreq(N, d=dt)
omega = np.fft.fftshift(omega)

# Solve for X
X = F/(m*(omega_0**2 - np.power(omega, 2) + 2*1j*gamma*omega))

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
x = np.fft.ifft(X) * len(X)

# Set the initial conditions where t = 0
x[0] = x_0

# Take the Fourier transform of x(t)
X = np.fft.fft(x)
x = np.fft.ifft(X) * len(X)

# Plot the position of the oscillator
plt.figure()
plt.plot(t,x.real)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Driven SHO with Damping')
plt.grid()
plt.show()