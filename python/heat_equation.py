# Implementation of the heat equation in 1D
import numpy as np
import matplotlib.pyplot as plt


def heat_equation(x, t, u0, D, L):
    """
    Solves the heat equation in 1D
    :param x: 1D array of x values
    :param t: 1D array of t values
    :param u0: Initial condition
    :param D: Diffusion coefficient
    :param L: Length of the domain
    :return: 2D array of u(x,t)
    """
    # Create a 2D array to store the solution
    u = np.zeros((len(x), len(t)))
    # Compute the solution including the boundary conditions
    for i in range(len(x)):
        for j in range(len(t)):
            u[i, j] = u0 * np.exp(-D * (np.pi / L) ** 2 * t[j]) * np.sin(np.pi * x[i] / L)
    return u

y = heat_equation(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 1, 1, 1)

# Plot the solution using a waterfall plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
ax.plot_wireframe(X, Y, y, rstride=10, cstride=10)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
plt.show()