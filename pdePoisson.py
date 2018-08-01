# Modelling and Visualisation
# Checkpoint 3
# Poisson Boundary value problem
# Simon McLaren

############################################
# NOTES ON THING THAT ARE WRONG
############################################

# Boundary conditions for A, the vector potential, are that it has PBC's in the z direction but nowhere else
# Not sure if it has to go to 0 at the boundaries, check this

import numpy as np
import random
import matplotlib.pyplot as plt
import math

# Class to solve the Poisson equation for a charge distribution
class poisson():

    # Initialisaton function
    # Size is the dimensions of the lattice
    # Noise is a number above/below which the values will vary at each lattice site from the reference value
    # A good amount of noise is <0.1
    # The boundary conditions are phi = 0 on the boundary for electric field
    # The boundary conditions for the vector potential are ... (FIND OUT!)
    def __init__(self, noise, size, ref_value, charge_dist, boundary_conditions):
        self.dim = int(size)
        self.phi_lattice = np.zeros((self.dim, self.dim, self.dim))
        if boundary_conditions == 'Electric field':
            for i in range(self.dim):
                for j in range(self.dim):
                    for k in range(self.dim):
                        # Add noise to each point
                        random_num = (2 * noise) * (random.random() - 0.5)
                        self.phi_lattice[i][j][k] = ref_value + random_num
                        # Set boundary conditions
                        if i == self.dim - 1 or j == self.dim - 1 or k == self.dim - 1 or i == 0 or j == 0 or k == 0:
                            self.phi_lattice[i][j][k] = 0.
        elif boundary_conditions == 'Vector potential':
            for i in range(self.dim):
                for j in range(self.dim):
                    random_num = (2 * noise) * (random.random() - 0.5)
                    for k in range(self.dim):
                        # Add noise to each point
                        self.phi_lattice[i][j][k] = ref_value + random_num
                        # Set boundary conditions
                        #if i == self.dim - 1 or j == self.dim - 1 or k == self.dim - 1 or i == 0 or j == 0 or k == 0:
                        #    self.phi_lattice[i][j][k] = 0.
        if charge_dist == "Centre charge":
            self.charge_dist = np.zeros((self.dim, self.dim, self.dim))
            self.charge_dist[self.dim / 2][self.dim / 2][self.dim / 2] = 1.
        elif charge_dist == "Line of charges":
            self.charge_dist = np.zeros((self.dim, self.dim, self.dim))
            self.charge_dist[self.dim / 2][self.dim /2][:] = 1.
    # Jacobi algorithm
    # e_0 and dx set to 1 (without loss of generality)
    def Jacobi_update(self):
        phi_new = np.copy(self.phi_lattice)
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    # Find nearest neighbour values for the current lattice point
                    i_up = i + 1
                    if i == self.dim - 1: i_up = 0
                    j_up = j + 1
                    if j == self.dim - 1: j_up = 0
                    k_up = k + 1
                    if k == self.dim - 1: k_up = 0
                    i_down = i - 1
                    if i == 0: i_down = self.dim - 1
                    j_down = j - 1
                    if j == 0: j_down = self.dim - 1
                    k_down = k - 1
                    if k == 0: k_down = self.dim - 1
                    phi_new[i][j][k] = (1. / 6.) * (self.phi_lattice[i_up][j][k]
                                                             + self.phi_lattice[i_down][j][k]
                                                             + self.phi_lattice[i][j_up][k]
                                                             + self.phi_lattice[i][j_down][k]
                                                             + self.phi_lattice[i][j][k_up]
                                                             + self.phi_lattice[i][j][k_down]
                                                             + self.charge_dist[i][j][k])
        return phi_new

    # Gauss Seidel algorithm
    # Phi array is updated in place instead of having 2 copies (an old and a new)
    # e_0 and dx set to 1 (without loss of generality)
    def Gauss_Seidel_update(self):
        phi_new = np.copy(self.phi_lattice)
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    # Find nearest neighbour values for the current lattice point
                    i_up = i + 1
                    if i == self.dim - 1: i_up = 0
                    j_up = j + 1
                    if j == self.dim - 1: j_up = 0
                    k_up = k + 1
                    if k == self.dim - 1: k_up = 0
                    i_down = i - 1
                    if i == 0: i_down = self.dim - 1
                    j_down = j - 1
                    if j == 0: j_down = self.dim - 1
                    k_down = k - 1
                    if k == 0: k_down = self.dim - 1
                    phi_new[i][j][k] = (1. / 6.) * (phi_new[i_up][j][k]
                                                             + phi_new[i_down][j][k]
                                                             + phi_new[i][j_up][k]
                                                             + phi_new[i][j_down][k]
                                                             + phi_new[i][j][k_up]
                                                             + phi_new[i][j][k_down]
                                                             + self.charge_dist[i][j][k])
        return phi_new

    # SOR update algorithm
    # using the Gauss Seidel algorithm
    def SOR_update(self, omega):
        phi_GS = np.copy(self.phi_lattice)
        phi_SOR = np.copy(self.phi_lattice)
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    # Find nearest neighbour values for the current lattice point
                    i_up = i + 1
                    if i == self.dim - 1: i_up = 0
                    j_up = j + 1
                    if j == self.dim - 1: j_up = 0
                    k_up = k + 1
                    if k == self.dim - 1: k_up = 0
                    i_down = i - 1
                    if i == 0: i_down = self.dim - 1
                    j_down = j - 1
                    if j == 0: j_down = self.dim - 1
                    k_down = k - 1
                    if k == 0: k_down = self.dim - 1
                    phi_GS[i][j][k] = (1. / 6.) * (phi_GS[i_up][j][k] # Gauss-Seidel approx at n+1 timestep
                                                             + phi_GS[i_down][j][k]
                                                             + phi_GS[i][j_up][k]
                                                             + phi_GS[i][j_down][k]
                                                             + phi_GS[i][j][k_up]
                                                             + phi_GS[i][j][k_down]
                                                             + self.charge_dist[i][j][k])
                    phi_SOR[i][j][k] = (1 - omega) * self.phi_lattice[i][j][k] + omega * phi_GS[i][j][k]
        return phi_SOR

    # Update lattice
    def set_lattice(self, new_lattice):
        self.phi_lattice = new_lattice

    # Returns the solution of Poissons equation (using Jacobi or Gauss-Seidel (with SOR) method) and the
    # timesteps taken to achieve steady state
    # The system is put into a time dependent problem, and once it reaches a threshold for steady state
    # we stop the loop and take the last value of the field as the solution
    @staticmethod
    def solve_Poissons_equation(lattice, threshold_tolerance, algorithm, omega):
        tolerance = 10000.
        counter = 1
        if algorithm == 'Jacobi':
            while tolerance > threshold_tolerance:
                #print counter, tolerance
                counter += 1
                phi_old_SOR = lattice.phi_lattice
                phi_new_SOR = lattice.Jacobi_update()
                lattice.set_lattice(phi_new_SOR)
                tolerance = 0.
                for i in range(lattice.dim):
                    for j in range(lattice.dim):
                        for k in range(lattice.dim):
                            tolerance += abs(phi_old_SOR[i][j][k] - phi_new_SOR[i][j][k])
        elif algorithm == 'G-S':
            while tolerance > threshold_tolerance:
                #print counter, tolerance
                counter += 1
                phi_old_SOR = lattice.phi_lattice
                phi_new_SOR = lattice.SOR_update(omega)
                lattice.set_lattice(phi_new_SOR)
                tolerance = 0.
                for i in range(lattice.dim):
                    for j in range(lattice.dim):
                        for k in range(lattice.dim):
                            tolerance += abs(phi_old_SOR[i][j][k] - phi_new_SOR[i][j][k])
        
        return lattice.phi_lattice, counter

    # Calculate electric field from field and write to an output file
    @staticmethod
    def calculate_E_field(phi_lattice):

        dimension = phi_lattice.shape[0]
        E = np.empty((dimension, dimension, dimension, 3))

        x_range = []
        y_range = []
        E_x_list = []
        E_y_list = []

        for i in range(dimension):
            for j in range(dimension):
                for k in range(dimension):
                    # Find nearest neighbour values for the current lattice point
                    i_up = i + 1
                    if i == dimension - 1: i_up = 0
                    j_up = j + 1
                    if j == dimension - 1: j_up = 0
                    k_up = k + 1
                    if k == dimension - 1: k_up = 0
                    i_down = i - 1
                    if i == 0: i_down = dimension - 1
                    j_down = j - 1
                    if j == 0: j_down = dimension - 1
                    k_down = k - 1
                    if k == 0: k_down = dimension - 1
                    
                    # Find E field components, setting dx = 1 without loss of generality
                    E[i][j][k][0] = - ((phi_lattice[i_up][j][k] - phi_lattice[i_down][j][k]) / 2.)
                    E[i][j][k][1] = - ((phi_lattice[i][j_up][k] - phi_lattice[i][j_down][k]) / 2.)
                    E[i][j][k][2] = - ((phi_lattice[i][j][k_up] - phi_lattice[i][j][k_down]) / 2.)

                    # Magnitude of E field at the current point
                    mag_E = math.sqrt(E[i][j][k][0] ** 2 + E[i][j][k][1] ** 2 + E[i][j][k][2] ** 2)

                    # Take a slice half way along the z axis
                    if k == int(dimension / 2):
                        x_range.append(i)
                        y_range.append(j)
                        # Add normalised E field values to the lists
                        E_x_list.append(E[i][j][k][0] / mag_E)
                        E_y_list.append(E[i][j][k][1] / mag_E)
                    
        return E, np.array(E_x_list), np.array(E_y_list), np.array(x_range), np.array(y_range)

    @staticmethod
    def calculate_B_field(phi_lattice):

        dimension = phi_lattice.shape[0]
        B = np.empty((dimension, dimension, dimension, 3))

        x_range = []
        y_range = []
        B_x_list = []
        B_y_list = []

        for i in range(dimension):
            for j in range(dimension):
                for k in range(dimension):
                    # Find nearest neighbour values for the current lattice point
                    i_up = i + 1
                    if i == dimension - 1: i_up = 0
                    j_up = j + 1
                    if j == dimension - 1: j_up = 0
                    i_down = i - 1
                    if i == 0: i_down = dimension - 1
                    j_down = j - 1
                    if j == 0: j_down = dimension - 1

                    # Find B field components, setting dx = 1 without loss of generality
                    B[i][j][k][0] = ((phi_lattice[i][j_up][k] - phi_lattice[i][j_down][k]) / 2.)
                    B[i][j][k][1] = - ((phi_lattice[i_up][j][k] - phi_lattice[i_down][j][k]) / 2.)
                    B[i][j][k][2] = 0.

                    mag_B = math.sqrt(B[i][j][k][0] ** 2 + B[i][j][k][1] ** 2 + B[i][j][k][2] ** 2)

                    # Take a slice half way along the z axis
                    if k == int(dimension / 2):
                        x_range.append(i)
                        y_range.append(j)
                        # Add normalised B field values to the list
                        B_x_list.append(B[i][j][k][0] / mag_B)
                        B_y_list.append(B[i][j][k][1] / mag_B)

        return B, np.array(B_x_list), np.array(B_y_list), np.array(x_range), np.array(y_range)

# Omega parameter space search
def main():
    # Parameters
    noise = 0.1
    size = 50
    ref_value = 0.
    threshold_tolerance = 0.5
    charge_dist = "Centre charge"
    algorithm = 'G-S'
    boundary_conditions = 'Electric field'
    omega_list = np.linspace(1., 1.9, 10)
    timesteps_list = []

    for omega in omega_list:
        lattice = poisson(noise, size, ref_value, charge_dist, boundary_conditions)

        phi_solution, timesteps = poisson.solve_Poissons_equation(lattice, threshold_tolerance, algorithm, omega)

        timesteps_list.append(timesteps)

        print omega, timesteps

    plt.plot(omega_list, timesteps_list)
    plt.savefig('omega_against_timesteps.png')
    plt.show()

# Solving for Electric field
def main2():
    noise = 0.1
    size = 50
    ref_value = 0.
    threshold_tolerance = 0.5
    charge_dist = "Centre charge"
    algorithm = 'G-S'
    boundary_conditions = 'Electric field'
    omega = 1.

    E_outfile = open('E_datafile.dat', 'w')
    phi_outfile = open('phi_datafile.dat', 'w')

    lattice = poisson(noise, size, ref_value, charge_dist, boundary_conditions)

    phi_solution, timesteps = poisson.solve_Poissons_equation(lattice, threshold_tolerance, algorithm, omega)

    E_solution, E_x_list, E_y_list, x_range, y_range = poisson.calculate_E_field(phi_solution)

    for i in range(lattice.dim):
        for j in range(lattice.dim):
            for k in range(lattice.dim):
                phi_outfile.write("{0:f} {1:f} {2:f} {3:f}\n".format(i, j, k, phi_solution[i][j][k]))

    for i in range(lattice.dim):
        for j in range(lattice.dim):
            for k in range(lattice.dim):
                E_outfile.write("{0:f} {1:f} {2:f} {3:f} {4:f} {5:f}\n".format(i, j, k,
                                                                               E_solution[i][j][k][0],
                                                                               E_solution[i][j][k][1],
                                                                               E_solution[i][j][k][2]))

    plt.imshow(phi_solution[:][:][lattice.dim / 2])
    plt.savefig('phi_solution.png')
    plt.show()
    plt.clf()

    plt.quiver(x_range, y_range, E_x_list, E_y_list)
    plt.savefig('E_solution.png')
    plt.show()

    E_outfile.close()
    phi_outfile.close()

# Solving for Vector potential
def main3():
    noise = 0.1
    size = 50
    ref_value = 0.
    threshold_tolerance = 16.8
    charge_dist = "Line of charges"
    algorithm = 'G-S'
    boundary_conditions = 'Vector potential'
    omega = 1.

    B_outfile = open('B_datafile.dat', 'w')
    A_outfile = open('A_datafile.dat', 'w')

    lattice = poisson(noise, size, ref_value, charge_dist, boundary_conditions)

    A_solution, timesteps = poisson.solve_Poissons_equation(lattice, threshold_tolerance, algorithm, omega)

    B_solution, B_x_list, B_y_list, x_range, y_range = poisson.calculate_B_field(A_solution)

    for i in range(lattice.dim):
        for j in range(lattice.dim):
            for k in range(lattice.dim):
                A_outfile.write("{0:f} {1:f} {2:f} {3:f}\n".format(i, j, k, A_solution[i][j][k]))

    for i in range(lattice.dim):
        for j in range(lattice.dim):
            for k in range(lattice.dim):
                B_outfile.write("{0:f} {1:f} {2:f} {3:f} {4:f} {5:f}\n".format(i, j, k,
                                                                               B_solution[i][j][k][0],
                                                                               B_solution[i][j][k][1],
                                                                               B_solution[i][j][k][2]))

    plt.imshow(A_solution[:][:][lattice.dim / 2])
    plt.savefig('A_solution.png')
    plt.show()
    plt.clf()

    plt.quiver(x_range, y_range, B_x_list, B_y_list)
    plt.savefig('B_solution.png')
    plt.show()

    B_outfile.close()
    A_outfile.close()

main()
