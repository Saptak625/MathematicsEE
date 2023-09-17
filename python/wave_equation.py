import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d

c = 3    # Wave speed
L = 20   # Length of domain
N = 1000 # Number of discretization points
dx = L/N
x = np.arange(-L/2,L/2,dx) # Define x domain

# Define discrete wavenumbers
kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)

# Initial condition
u0 = 1/np.cosh(x)
u0hat = np.fft.fft(u0)

# SciPy's odeint function doesn't play well with complex numbers, so we recast 
# the state u0hat from an N-element complex vector to a 2N-element real vector
u0hat_ri = np.concatenate((u0hat.real,u0hat.imag))

# Simulate in Fourier frequency domain
dt = 0.025
t = np.arange(0,500*dt,dt)

def rhsWave(uhat_ri,t,kappa,c):
    uhat = uhat_ri[:N] + (1j) * uhat_ri[N:]
    d_uhat = -c*(1j)*kappa*uhat
    d_uhat_ri = np.concatenate((d_uhat.real,d_uhat.imag)).astype('float64')
    return d_uhat_ri

uhat_ri = odeint(rhsWave, u0hat_ri, t, args=(kappa,c))
uhat = uhat_ri[:,:N] + (1j) * uhat_ri[:,N:]

# Inverse FFT to bring back to spatial domain
u = np.zeros_like(uhat)

for k in range(len(t)):
    u[k,:] = np.fft.ifft(uhat[k,:])

u = u.real   


# Waterfall plot
u_plot = u[0:-1:10,:]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=22.5, azim=45, roll=0)

x = np.arange(u_plot.shape[1])
y = np.arange(u_plot.shape[0])
X, Y = np.meshgrid(x, y)

cmap = 'viridis'
ax.plot_surface(X, Y, u_plot, cmap=cmap)
cbar = plt.colorbar(ax.plot_surface(X, Y, u_plot, cmap=cmap), ax=ax)
cbar.set_label('Displacement')

ax.set_xlabel('Position')
ax.set_ylabel('Time')
ax.set_zlabel('Displacement')
ax.set_title('Wave Equation Numerical Solution')
    
# Image plot
# plt.figure()
# plt.imshow(np.flipud(u), aspect=8)
# plt.axis('off')
plt.show()