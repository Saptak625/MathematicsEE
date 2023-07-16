# Implementation of the driven sho with damping equation in 1D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Simulate in Fourier frequency domain
T = 1000
N = 1000 # Number of discretization points
dt = T/N
t = np.arange(0,T,dt)

# Define parameters
omega_0 = 0.2    # Natural frequency of the oscillator without damping
gamma = 0.1 # Damping coefficient
f = 0.2 * np.sin(2 * omega_0 * t)    # Driving force function
x_0 = 0.5   # Initial position of the oscillator
m = 200   # Mass of the oscillator

# Define discrete wavenumbers
omega = 2*np.pi*np.fft.fftfreq(N, d=dt)

# Take the Fourier transform of the driving force
fhat = np.fft.fft(f)

# Define the Fourier transform of the position of the oscillator
xhat = fhat / (m*(omega_0**2 - omega**2 + 2j*gamma*omega))

# Plot the Fourier transform of the position of the oscillator
plt.figure()
plt.plot(omega,xhat.real)
plt.plot(omega,xhat.imag)
plt.xlabel('Frequency')
plt.ylabel('Position')
plt.title('Driven SHO with Damping')
plt.legend(['Real','Imaginary'])

# Take the inverse Fourier transform of the position of the oscillator
x = np.fft.ifft(xhat)

# Plot the forces
plt.figure()
plt.plot(t, f, label='Driving Force')
plt.plot(t, -m*(omega_0**2)*x, label='Spring Force')
plt.plot(t, -2*m*gamma*np.gradient(x, dt), label='Damping Force')
plt.plot(t, f - m*(omega_0**2)*x - 2*m*gamma*np.gradient(x, dt), '--', label='Net Force')
plt.xlabel('Time')
plt.ylabel('Force')
plt.title('All Forces')
plt.grid(True)
plt.legend()

# Plot the position of the oscillator
plt.figure()
plt.plot(t,x.real)
plt.plot(t,x.imag)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Driven SHO with Damping')
plt.legend(['Real','Imaginary'])
plt.show()