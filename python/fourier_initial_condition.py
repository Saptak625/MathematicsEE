# Use Fourier Transforms to have some initial conditions and then evolve the system in time.
import numpy as np
import matplotlib.pyplot as plt

# Parameters
omega_0 = 0.15    # Natural frequency of the oscillator without damping
gamma = 0.005 # Damping coefficient
m = 1   # Mass of the oscillator

# Define time
T = 1000
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
# X(\omega) = \frac{F(\omega) - m C_1 \left( i \omega + 2 \gamma \right) - m C_2}{m \left( \omega_0^2 - \omega^2 + 2 \gamma i \omega \right)}
C_1 = 0.5 # Velocity initial condition
C_2 = 2*gamma*C_1 # Position initial condition
X = (F - m*C_1*1j*omega - m*(2*gamma*C_1 - C_2))/(m*(omega_0**2 - np.power(omega, 2) + 2*1j*gamma*omega))

# Plot the Fourier transform of f(t) and X
plt.figure()
plt.plot(omega,abs(F), label='Fourier Transform of f(t)')
plt.plot(omega,abs(X), label='Fourier Transform of x(t)')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Driven SHO with Damping')
plt.legend(loc=1)
plt.grid()

# Take the inverse Fourier transform of X
x = np.fft.ifft(X) * len(X)

# Plot the position of the oscillator
plt.figure()
plt.plot(t,x.real)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Driven SHO with Damping')
plt.grid()
plt.show()