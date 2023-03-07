'''
Calculate various quantum particle cases in a 1D space,
using grid method.
'''

import numpy as np
from matplotlib import pyplot as plt

pi = np.pi
# These constants are in atomic units
m = 1000.0
# The omega of harmonic oscillator corresponding to 600cm-1
omega = 0.0027338
h_bar = 1.0
# Boundaries
x_left = -3.0
x_right = 3.0
# n is the number of grid points and is frequently used.
# The number of n determines the accuracy of calculation.
n = 500
L = (x_right - x_left)
# terminal points not counted.
step = L / (n + 1)
grid_points = [(x_left + (i + 1) * step) for i in range(n)]


def harmonic_oscillator(i):
    # potential as a function of position
    v = 0.5 * m * (omega * grid_points[i]) ** 2
    return v


def infinite_wall(i):
    if (grid_points[i] < -2) or (grid_points[i] > 2):
        return 65535
    else:
        return 0


def finite_wall(i):
    if grid_points[i] < -2:
        return 65535
    elif grid_points[i] > 1:
        return 0.01
    else:
        return 0


def penetration(i):
    if grid_points[i] < -2:
        return 65535
    elif 0 < grid_points[i] < 0.2:
        return 0.1
    else:
        return 0


def calculate(potential_function, state):
    # H is the Hamiltonian
    H = np.zeros((n, n), dtype=float)
    # U is the transformation matrix
    U = np.zeros((n, n), dtype=float)
    # T is the kinetic energy given by (h_bar^2/2m)(d^2 psi/dx^2)
    T = np.zeros((n, n), dtype=float)
    # V is the potential energy given by the potential function
    V = np.zeros((n, n), dtype=float)
    potential = []
    for i in range(n):
        for j in range(n):
            U[i][j] = np.sqrt(2.0 / (n + 1)) * np.sin((i + 1) * (j + 1) * pi / (n + 1))
            if i == j:
                # This basis begins from i = 0
                T[i][j] = 0.5 * ((h_bar * (i+1) * pi) / L) ** 2 / m
                V[i][j] = potential_function(i)
                potential.append(V[i][j])
            else:
                T[i][j] = 0
                V[i][j] = 0
    temp = np.dot(U, np.dot(T, U))
    H = np.add(temp, V)
    # print(H)
    w, v = np.linalg.eigh(H)
    # print(w)
    # draw the potential line
    plt.plot(grid_points, potential)
    # draw the wavefunction
    E = w[state] * 27.2114 * 8065.51
    print("The energy is", E, "cm^(-1)")
    plt.plot(grid_points, v[:, state])
    plt.ylim((-0.2, 0.2))
    plt.show()


if __name__ == '__main__':
    # 1st excited state of a harmonic oscillator
    calculate(harmonic_oscillator, 1)
    # 2rd excited state of an infinite potential wall
    calculate(infinite_wall, 2)
    # Ground states of a finite wall
    calculate(finite_wall, 0)
    # fourth excited state of penetration wall
    calculate(penetration, 4)
