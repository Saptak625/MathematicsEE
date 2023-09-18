from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# Allee effect
# Define the system of ODEs
def rhsAlleeEffect(n, t, r, K, a):
    return -r * n * (1 - (n / K)) * (1 - (n / a))

# Define parameters
r = 0.2
K = 300
a = 120
n_0 = 100

# Define time
T = 100
N = 1000 # Number of discretization points
t = np.linspace(0,T,N)

# Solve the system of ODEs
sol = odeint(rhsAlleeEffect, n_0, t, args=(r, K, a))

# Plot the position of the oscillator
plt.figure()
plt.plot(t,sol[:,0], label='Population')
plt.plot(t, K * np.ones(len(t)), '--', label='Carrying Capacity')
plt.plot(t, a * np.ones(len(t)), '--', label='Allee Threshold')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Allee Effect')
plt.legend(loc=1)
plt.grid()
plt.show()
