# Take the Fourier Transform of the unit function.

import numpy as np
import matplotlib.pyplot as plt

# Define the unit function
def f(t):
    return np.ones_like(t)

t = np.linspace(-10,10,1000)

# Take the Fourier transform of f(t)
F = np.fft.fft(f(t))
F_sine = np.fft.fft(np.sin(5*t))

# Define omega
omega = 2 * np.pi * np.fft.fftfreq(len(t), d=t[1]-t[0])

# Plot the Fourier transform of f(t)
plt.figure()
plt.plot(omega, abs(F) / len(t))
plt.plot(omega, abs(F_sine) / len(t))
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Fourier Transform of the Unit Function')
plt.grid()
plt.show()