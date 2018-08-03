# Modelling and Visualisation
# Checkpoint 3
# Cahn-Hilliard Initial value problem
# Simon McLaren

import numpy as np
import random
import matplotlib.pyplot as plt

class cahnhilliard():

    # Initialisaton function
    # Size is the dimensions of the order parameter lattice
    # Noise is a number above/below which the order parameter will vary at each lattice site from the reference value
    # A good amount of noise is <0.1
    def __init__(self, noise, size, ref_value):
        self.dim = int(size)
        self.phi_lattice = np.empty((size, size))
        self.phi_lattice.fill(float(ref_value))
        for i in range(size):
            for j in range(size):
                # Add noise to each point
                random_num = (2 * (random.random() - 0.5)) * noise
                self.phi_lattice[i][j] += random_num

    # Single timestep update of the system
    def update(self, a, k, M, dx, dt):
        mu_lattice = np.empty((self.dim, self.dim))

        # Find current chemical potential lattice
        for i in range(self.dim):
            for j in range(self.dim):
                # Find nearest neighbour values for the current phi lattice point
                i_up = i + 1
                if i == self.dim - 1: i_up = 0
                j_up = j + 1
                if j == self.dim - 1: j_up = 0
                i_down = i - 1
                if i == 0: i_down = self.dim - 1
                j_down = j - 1
                if j == 0: j_down = self.dim - 1
                mu_lattice[i][j] = -a * self.phi_lattice[i][j] + a * self.phi_lattice[i][j]**3 - \
                                    k * (self.phi_lattice[i_up][j] + self.phi_lattice[i_down][j]
                                        + self.phi_lattice[i][j_down] + self.phi_lattice[i][j_up]
                                        - 4 * self.phi_lattice[i][j]) / dx**2

        # Update the order parameter lattice
        phi_old = self.phi_lattice
        for i in range(self.dim):
            for j in range(self.dim):
                # Find nearest neighbour values for the current mu lattice point
                i_up = i + 1
                if i == self.dim - 1: i_up = 0
                j_up = j + 1
                if j == self.dim - 1: j_up = 0
                i_down = i - 1
                if i == 0: i_down = self.dim - 1
                j_down = j - 1
                if j == 0: j_down = self.dim - 1
                self.phi_lattice[i][j] = phi_old[i][j] + ((dt * M) / dx**2) * (mu_lattice[i_up][j] + mu_lattice[i_down][j]
                                                                      + mu_lattice[i][j_down] + mu_lattice[i][j_up]
                                                                      - 4 * mu_lattice[i][j])

    # Calculate free energy for the order parameter lattice
    def calculate_f_density(self, a, k, dx):
        total_free_energy = 0.
        for i in range(self.dim):
            for j in range(self.dim):
                # Find nearest neighbour values for the current phi lattice point
                i_up = i + 1
                if i == self.dim - 1: i_up = 0
                j_up = j + 1
                if j == self.dim - 1: j_up = 0
                i_down = i - 1
                if i == 0: i_down = self.dim - 1
                j_down = j - 1
                if j == 0: j_down = self.dim - 1
                # Calculated grad phi squared for the current point in the lattice
                grad_phi_squared = ((self.phi_lattice[i_up][j] - self.phi_lattice[i_down][j]) / (2 * dx))**2 \
                                   + ((self.phi_lattice[i][j_up] - self.phi_lattice[i][j_down]) / (2 * dx))**2
                total_free_energy += - (a / 2) * self.phi_lattice[i][j]**2 + (a / 4) \
                * self.phi_lattice[i][j]**4 + (k / 2) * grad_phi_squared
        return total_free_energy

    # Simulate the a particular system for given parameters
    # Returns the free energy and time data in lists
    @staticmethod
    def simulate(lattice, N_steps, a, k, M, dx, dt, animate_freq, animate=False):

        # Data lists
        t_list = []
        total_free_energy_list = []

        # Run simulation for N steps
        for t in range(N_steps):
            t_list.append(t)

            # Animation
            if t % animate_freq == 0 and animate == True:
                print t
                plt.cla()
                im = plt.imshow(lattice.phi_lattice, vmin=-1., vmax=1., cmap='PiYG', animated=True)
                plt.pause(0.0001)

            f_e = lattice.calculate_f_density(a, k, dx)
            total_free_energy_list.append(f_e)
            lattice.update(a, k, M, dx, dt)
        return t_list, total_free_energy_list

def main():
    # Parameters
    noise = 0.1
    dimension = 30
    ref_value = 0.
    N_steps = 10000
    a = 0.1
    k = 0.1
    M = 0.1
    dx = 1.
    dt = 2.5
    animate_freq = 100

    ch_lattice = cahnhilliard(noise, dimension, ref_value)

    # Animation
    fig = plt.figure()
    im = plt.imshow(ch_lattice.phi_lattice, vmin=-1., vmax=1., cmap='PiYG', animated=True)
    plt.ion()

    t_list, total_free_energy_list = cahnhilliard.simulate(ch_lattice, N_steps, a, k, M, dx, dt, animate_freq, animate=True)

    plt.ioff()
    plt.clf()
    plt.plot(t_list, total_free_energy_list)
    plt.savefig('free_energy_against_time.png')
    plt.show()

main()




