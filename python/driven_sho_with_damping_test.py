import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

def simulate_driven_oscillator(m, c, k, F, dt):
    # Parameters
    num_samples = len(F)
    t = np.arange(0, num_samples * dt, dt)
    frequencies = np.fft.fftfreq(num_samples, dt)
    
    # Fourier transform of the driving force
    F_hat = fft(F)
    
    # Calculate the Fourier transform of the displacement
    X_hat = F_hat / (-m * (2 * np.pi * frequencies)**2 + 1j * 2 * np.pi * frequencies * c + k)
    
    # Inverse Fourier transform to obtain the displacement in time domain
    x = ifft(X_hat)
    
    return t, x.real

# Parameters
m = 0.1  # mass
c = 0  # damping coefficient
k = 8.0  # spring constant
dt = 0.01  # time step

# Time domain parameters
t_total = 10.0  # total simulation time
t = np.arange(0, t_total, dt)

# Driving force (example: sinusoidal)
omega = 15  # angular frequency
F = np.sin(np.sqrt(2) * omega * t)

# Simulate the driven oscillator
t_sim, x_sim = simulate_driven_oscillator(m, c, k, F, dt)

# Plot the forces
plt.figure()
plt.plot(t, F, label='Driving Force')
plt.plot(t, -k*x_sim, label='Spring Force')
plt.plot(t, -c*np.gradient(x_sim, dt), label='Damping Force')
plt.plot(t, F - k*x_sim - c*np.gradient(x_sim, dt), '--', label='Net Force')
plt.xlabel('Time')
plt.ylabel('Force')
plt.title('All Forces')
plt.grid(True)
plt.legend()

# Plot the results
plt.figure()
plt.plot(t_sim, x_sim)
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Driven Harmonic Oscillator with Damping')
plt.grid(True)
plt.show()
