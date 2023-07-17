# Implementation of the Black-Scholes equation
# \frac{d \hat{V}}{dt} = \frac{1}{4 \pi}\sigma^2 \kappa^2 ( \mathcal{F} \left\{ S^2 \right\} * \hat{V} ) - \frac{1}{2 \pi} r i \kappa ( \mathcal{F} \left\{ S \right\} * \hat{V} ) + r \hat{V}
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d

sigma = 1    # Volatility of underlying asset
r = 0.05     # Risk-free interest rate
L = 100  # Length of domain
N = 1000 # Number of discretization points
ds = L/N
s = np.arange(0,L,ds) # Define x domain

# Define discrete wavenumbers
kappa = 2*np.pi*np.fft.fftfreq(N, d=ds)

# Initial condition
v0 = np.zeros_like(s)
v0[int((L/2 - L/10)/ds):int((L/2 + L/10)/ds)] = 1
v0hat = np.fft.fft(v0)

# SciPy's odeint function doesn't play well with complex numbers, so we recast 
# the state v0hat from an N-element complex vector to a 2N-element real vector
v0hat_ri = np.concatenate((v0hat.real,v0hat.imag))

# Simulate in Fourier frequency domain
dt = 0.1
t = np.arange(0,1000,dt)

def rhsHeat(vhat_ri,t,kappa,sigma,r):
    vhat = vhat_ri[:N] + (1j) * vhat_ri[N:]
    d_vhat = (sigma**2 / (4 * np.pi)) * (np.power(kappa,2)) * vhat - (r / (2 * np.pi)) * (1j) * kappa * vhat + r * vhat
    d_vhat_ri = np.concatenate((d_vhat.real,d_vhat.imag)).astype('float64')
    return d_vhat_ri

vhat_ri = odeint(rhsHeat, v0hat_ri, t, args=(kappa, sigma, r))

vhat = vhat_ri[:,:N] + (1j) * vhat_ri[:,N:]

v = np.zeros_like(vhat)

for k in range(len(t)):
    v[k,:] = np.fft.ifft(vhat[k,:])

v = v.real    

# Mesh plot
v_plot = v[0:-1:10,:]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=22.5, azim=45, roll=0)

x = np.arange(v_plot.shape[1])
y = np.arange(v_plot.shape[0])
X, Y = np.meshgrid(x, y)

cmap = 'plasma'
ax.plot_surface(X, Y, v_plot, cmap=cmap)
cbar = plt.colorbar(ax.plot_surface(X, Y, v_plot, cmap=cmap), ax=ax)
cbar.set_label('Temperature')

ax.set_xlabel('Position')
ax.set_ylabel('Time')
ax.set_zlabel('Temperature')
ax.set_title('Heat Equation Numerical Solution')

# Image plot
# plt.figure()
# plt.imshow(np.flipud(u), aspect=8)
# plt.axis('off')
plt.show()