# Implementation of the Black-Scholes equation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d

sigma = 10    # Volatility of underlying asset
r = 0.05     # Risk-free interest rate
L = 50  # Length of domain
N = 1000 # Number of discretization points
ds = L/N
s = np.arange(0,L,ds) # Define x domain

# Take the Fourier transform of S(t)
# Perform one-sided FFT
fourier_unit = np.fft.fft(np.ones_like(s)) / len(s)
fourier_S = -1j * fourier_unit
fourier_S_squared = -1 * fourier_unit

# Define discrete wavenumbers
kappa = 2*np.pi*np.fft.fftfreq(N, d=ds)

# Initial condition
v0 = s/4 + L/4
v0hat = np.fft.fft(v0)

# SciPy's odeint function doesn't play well with complex numbers, so we recast 
# the state v0hat from an N-element complex vector to a 2N-element real vector
v0hat_ri = np.concatenate((v0hat.real,v0hat.imag))

# Simulate in Fourier frequency domain
def black_scholes(vhat_ri,t,kappa,sigma,r, fourier_S, fourier_S_squared):
    vhat = vhat_ri[:N] + (1j) * vhat_ri[N:]
    d_vhat = (sigma**2 / (4 * np.pi)) * (np.power(kappa,2)) * fourier_S_squared * vhat - (r / (2 * np.pi)) * (1j) * kappa * fourier_S * vhat + r * vhat
    d_vhat_ri = np.concatenate((d_vhat.real,d_vhat.imag)).astype('float64')
    return d_vhat_ri

vhat_ri = odeint(black_scholes, v0hat_ri, s, args=(kappa, sigma, r, fourier_S, fourier_S_squared))

vhat = vhat_ri[:,:N] + (1j) * vhat_ri[:,N:]

v = np.zeros_like(vhat)

for k in range(len(s)):
    v[k,:] = np.fft.ifft(vhat[k,:])

v = v.real    

# Mesh plot
v_plot = v[0:-1:10,:]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=22.5, azim=-135, roll=0)
x = np.arange(v_plot.shape[1])
y = np.arange(v_plot.shape[0])
X, Y = np.meshgrid(x, y)
cmap = 'plasma'
ax.plot_surface(X, Y, v_plot, cmap=cmap)
cbar = plt.colorbar(ax.plot_surface(X, Y, v_plot, cmap=cmap), ax=ax)
cbar.set_label('Option Price')

ax.set_xlabel('Expiry Time')
ax.set_ylabel('Stock Price')
ax.set_zlabel('Option Price')
ax.set_title('Black-Scholes Equation Numerical Solution')

# Image plot
# plt.figure()
# plt.imshow(np.flipud(v_plot), aspect=8)
# plt.axis('off')
plt.show()