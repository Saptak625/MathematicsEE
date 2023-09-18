# Simulate the competitive Lotka-Volterra equations with a changing carrying capacity
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the system of ODEs
def K(t):
    return 25 * np.sin((2 * np.pi / 365) * t) + 50 * np.sin((2 * np.pi / 3 / 365) * t) + 200

def lotka_volterra(s, t, r_x, r_y, alpha_xy, alpha_yx):
    # \frac{dx}{dt} &= r_x x \left(1 - \left(\frac{x + \alpha_{xy} y}{K(t)}\right)\right) \\ 
    # \frac{dy}{dt} &= r_y y \left(1 - \left(\frac{y + \alpha_{yx} x}{K(t)}\right)\right)
    x, y = s

    dxdt = r_x * x * (1 - ((x + alpha_xy * y) / K(t)))
    dydt = r_y * y * (1 - ((y + alpha_yx * x) / K(t)))

    return [dxdt, dydt]

# Define parameters
r_x = 0.05
r_y = 0.2
alpha_xy = 0.2
alpha_yx = 0.15
x_0 = 180
y_0 = 130

# Define time
T = 365 * 10
N = 4000 # Number of discretization points
t = np.linspace(0,T,N)

# Solve the system of ODEs
sol = odeint(lotka_volterra, [x_0, y_0], t, args=(r_x, r_y, alpha_xy, alpha_yx))

# Plot the position of the oscillator
plt.figure()
plt.plot(t,sol[:,0], label='Population X')
plt.plot(t,sol[:,1], label='Population Y')
plt.plot(t,sol[:,0]+sol[:,1], label='Total Population', linestyle='--')
plt.plot(t,2*K(t), label='Carrying Capacity', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Competitive Lotka-Volterra')
plt.legend(loc=1)
plt.grid()
plt.show()


