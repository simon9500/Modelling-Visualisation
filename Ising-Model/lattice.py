# Modelling and Visualisation in Physics
# Checkpoint 1
# Simon McLaren

import numpy as np
import random
import math
import sys
import matplotlib.pyplot as plt

# Class to simulate lattice of spins using the Ising model
class spin_lattice():

    def __init__(self, init, lx, ly):

        # Define initial variables
        self.lx = lx
        self.ly = ly
        if init == 'Random':
            self.lattice = np.random.choice([1, -1], size=(self.lx, self.ly))
        elif init == 'Allup':
            self.lattice = np.ones((self.lx, self.ly), dtype=np.int)
        elif init == 'Alldown':
            self.lattice = -1 * np.ones((self.lx, self.ly), dtype=np.int)

    # Reset lattice according to conditions defined by 'init'
    def reset_lattice(self, init):
        if init == 'Random':
            self.lattice = np.random.choice([1, -1], size=(self.lx, self.ly))
        elif init == 'Allup':
            self.lattice = np.ones((self.lx, self.ly), dtype=np.int)
        elif init == 'Alldown':
            self.lattice = -1 * np.ones((self.lx, self.ly), dtype=np.int)

    # TOTAL ENERGY of an array of spins in the Ising model
    def total_energy(self):  # array of spins, float parameter
        E = 0.  # Energy
        for i in range(self.lx):
            for j in range(self.ly):
                i_up = i + 1
                if i == self.lx - 1: i_up = 0
                j_up = j + 1
                if j == self.ly - 1: j_up = 0
                E += -1. * self.lattice[i][j] * (self.lattice[i_up][j] + self.lattice[i][j_up])
        return E

    # Return energy contribution from a single spin and all it's nearest neighbours
    def local_energy(self, spin):
        i = spin[0]
        j = spin[1]
        i_up = i + 1
        if i == self.lx - 1: i_up = 0
        j_up = j + 1
        if j == self.ly - 1: j_up = 0
        i_down = i - 1
        if i == 0: i_down = self.lx - 1
        j_down = j - 1
        if j == 0: j_down = self.ly - 1
        return float(-1. * self.lattice[i][j] \
               * (self.lattice[i][j_up] + self.lattice[i_up][j] +
                  self.lattice[i][j_down] + self.lattice[i_down][j]))

    # Simple binary check to see if spins are nearest neighbours
    def check_if_nearest_neighbours(self, spin1, spin2):
        i_1 = spin1[0]
        j_1 = spin1[1]
        i_2 = spin2[0]
        j_2 = spin2[1]
        # Find nearest neighbours of spin 1
        i_up_1 = i_1 + 1
        if i_1 == self.lx - 1: i_up_1 = 0
        j_up_1 = j_1 + 1
        if j_1 == self.ly - 1: j_up_1 = 0
        i_down_1 = i_1 - 1
        if i_1 == 0: i_down_1 = self.lx - 1
        j_down_1 = j_1 - 1
        if j_1 == 0: j_down_1 = self.ly - 1
        if i_up_1 == i_2 or i_down_1 == i_2 or j_up_1 == j_2 or j_down_1 == j_2:
            return True
        else:
            return False

    # Flip a single spin
    def flip_spin(self, flip_choice):
        self.lattice[flip_choice[0]][flip_choice[1]] *= -1

    # Switch 2 spins
    def switch_spins(self, spin1, spin2):
        spin1_value = self.lattice[spin1[0]][spin1[1]]
        spin2_value = self.lattice[spin2[0]][spin2[1]]
        self.lattice[spin1[0]][spin1[1]] = spin2_value
        self.lattice[spin2[0]][spin2[1]] = spin1_value

    # Calculate absolute value of the intensive magnetisation of a spin array
    def total_magnetisation(self):
        M = 0
        for i in range(self.lx):
            for j in range(self.ly):
                M += self.lattice[i][j]
        return abs(M)  # / (lx * ly)

    def energy_change_kawasaki(self, spin1, spin2):
        if self.lattice[spin1[0]][spin1[1]] == self.lattice[spin2[0]][spin2[1]]:
            return 0.
        else:
            # Check if spin 1 is adjacent to spin 2
            if self.check_if_nearest_neighbours(spin1, spin2):
                return -2 * self.local_energy(spin1) - 2 * self.local_energy(spin2) + 4.
            else:
                return -2 * self.local_energy(spin1) - 2 * self.local_energy(spin2)

    def metropolis_test_glauber(self, T):
        proposed_flip = [random.randint(0, self.ly - 1), random.randint(0, self.lx - 1)]
        E_change = -2 * self.local_energy(proposed_flip)
        if E_change <= 0.:
            self.flip_spin(proposed_flip)
        elif E_change > 0.:
            # Flip spin at proposed position with probabilty min{1, exp(-E_change)}
            if random.random() <= math.exp(- E_change / T):
                self.flip_spin(proposed_flip)

    def metropolis_test_kawasaki(self, T):
        proposed_flip_1 = [random.randint(0, self.ly - 1), random.randint(0, self.lx - 1)]
        proposed_flip_2 = [random.randint(0, self.ly - 1), random.randint(0, self.lx - 1)]
        E_change = self.energy_change_kawasaki(proposed_flip_1, proposed_flip_2)
        if E_change <= 0.:
            self.switch_spins(proposed_flip_1, proposed_flip_2)
        elif E_change > 0.:
            # Switch spins with probabilty min{1, exp(-E_change)}
            if random.random() <= math.exp(- E_change / T):
                self.switch_spins(proposed_flip_1, proposed_flip_2)

    # Bootstrap method for calculating errors on a list of measurements
    @staticmethod
    def bootstrap(lattice, M_list, T, measurement, k=None):
        n = len(M_list)
        if k == None: # Default k value is n
            k = n
        avmeas = 0.
        avmeas_2 = 0.
        for i in range(k):
            avM = 0.
            avM_2 = 0.
            for j in range(n):
                M_rdm = M_list[random.randint(0, n - 1)]
                avM += M_rdm
                avM_2 += M_rdm * M_rdm
            # Choose measurement to find error on
            if measurement == 'C_v':
                C_v = (avM_2 / n - (avM / n) ** 2) / (lattice.lx * lattice.ly * T ** 2)
                avmeas += C_v
                avmeas_2 += C_v * C_v
            if measurement == 'X':
                X = (avM_2 / n - (avM / n) ** 2) / (lattice.lx * lattice.ly * T)
                avmeas += X
                avmeas_2 += X * X
        return math.sqrt(abs(avmeas_2 / k - (avmeas / k) ** 2))

    # Jackknife method for calculating errors on a list of measurements (C_v or X)
    @staticmethod
    def jackknife(lattice, meas_list, T, measurement):
        n = len(meas_list)
        avmeas = 0.
        avmeas_2 = 0.
        for i in range(n):
            avM = 0.
            avM_2 = 0.
            for j in range(n):
                if i != j:
                    M = meas_list[j]
                    avM += M
                    avM_2 += M * M
            if measurement == 'C_v':
                C_v = (avM_2 / n - (avM / n) ** 2) / (lattice.lx * lattice.ly * T ** 2)
                avmeas += C_v
                avmeas_2 += C_v * C_v
            if measurement == 'X':
                X = (avM_2 / n - (avM / n) ** 2) / (lattice.lx * lattice.ly * T)
                avmeas += X
                avmeas_2 += X * X
        return math.sqrt(abs(avmeas_2 - (avmeas ** 2) / n))

    @staticmethod
    def simulation(lattice, output_frequency, num_sweeps, T, dynamics, start_meas, animate=False):
        sweep_length = lattice.ly * lattice.lx  # Number of points on the lattice

        E_list = []
        M_list = []
        sweep_list = []
        avE = 0.
        avE_2 = 0.
        avM = 0.
        avM_2 = 0.
        ndata = 0

        # Simulation of n sweeps for Glauber dynamics
        if dynamics == 'Glauber':
            for i in range(num_sweeps):
                for j in range(sweep_length):
                    lattice.metropolis_test_glauber(T)
                # Record measurements every 10 sweeps
                if i % output_frequency == 0:
                    if i > start_meas:
                        E = lattice.total_energy()
                        E_list.append(E)
                        M = lattice.total_magnetisation()
                        M_list.append(M)
                        sweep_list.append(i)
                        avE += E
                        avE_2 += E * E
                        avM += M
                        avM_2 += M * M
                        ndata += 1
                    if animate == True:
                        plt.cla()
                        im = plt.imshow(lattice.lattice, vmin=-1., vmax=1., cmap='jet', animated=True)
                        plt.draw()
                        plt.pause(0.0001)

        # Simulation of n sweeps for Kawasaki dynamics
        if dynamics == 'Kawasaki':
            for i in range(num_sweeps):
                for j in range(sweep_length):
                    lattice.metropolis_test_kawasaki(T)
                # Record measurements every 10 sweeps
                if i % output_frequency == 0:
                    if i > start_meas:
                        E = lattice.total_energy()
                        E_list.append(E)
                        M = lattice.total_magnetisation()
                        M_list.append(M)
                        sweep_list.append(i)
                        avE += E
                        avE_2 += E * E
                        avM += M
                        avM_2 += M * M
                        ndata += 1
                    if animate == True:
                        plt.cla()
                        im = plt.imshow(lattice.lattice, vmin=-1., vmax=1., cmap='jet', animated=True)
                        plt.draw()
                        plt.pause(0.0001)

        X = (avM_2 / ndata - (avM / ndata) ** 2) / (lattice.lx * lattice.ly * T)
        C_v = (avE_2 / ndata - (avE / ndata) ** 2) / (lattice.lx * lattice.ly * T ** 2)
        M = float(avM / ndata)
        E = float(avE / ndata)

        return X, C_v, M, E, E_list, M_list, sweep_list

    # Standard error of the mean computation for uncorrelated measurements
    @staticmethod
    def standard_error(list):
        list = np.array(list)
        return math.sqrt((np.mean(np.square(list)) - np.mean(list) ** 2)/(len(list) - 1))

def main():
    # Read names of output file and input file from command line
    if len(sys.argv) != 3:
        print "Wrong number of arguments."
        print "Usage: " + sys.argv[0] + " <input file> + <output file>"
        quit()
    else:
        infileName = sys.argv[1]
        outfileName = sys.argv[2]

    # Open input parameters file for reading
    infile = open(infileName, "r")

    # Open output file for appending
    outfile = open(outfileName, "a")

    # Read particle data from input file
    line = infile.readline()
    tokens = line.split()

    # Define initial variables
    init = str(tokens[0]) # Initial state of spin lattice
    num_sweeps = int(tokens[1]) # Number of sweeps in simulation
    lx = int(tokens[2])
    ly = int(tokens[3])
    T = float(tokens[4]) # Temperature
    Dynamics = str(tokens[5]) # Dynamics (glauber/kawasaki)
    output_frequency = int(tokens[6]) # Frequency of measurements
    start_meas = int(tokens[7]) # Start of measurement taking (sweep number)
    error_method = str(tokens[8]) # Error calculation method for X/C_v

    lattice = spin_lattice(init, lx, ly)

    # Simulate lattice according to user-given variables
    X, C_v, avM, avE, E_list, M_list, sweep_list = \
        spin_lattice.simulation(lattice, output_frequency, num_sweeps, T, Dynamics, start_meas, animate=True)

    # Find C_v and X errors
    if error_method == 'Jackknife':
        X_error = spin_lattice.jackknife(lattice, M_list, T, 'X')
        C_v_error = spin_lattice.jackknife(lattice, E_list, T, 'C_v')
    if error_method == 'Bootstrap':
        X_error = spin_lattice.bootstrap(lattice, M_list, T, 'X')
        C_v_error = spin_lattice.bootstrap(lattice, E_list, T, 'C_v')
    # Find errors on the average M and E
    avM_error = spin_lattice.standard_error(M_list)
    avE_error = spin_lattice.standard_error(E_list)
    outfile.write("{0:f} {1:f} {2:f} {3:f} {4:f} {5:f} {6:f} {7:f} {8:f}\n".
                format(T, X, X_error, C_v, C_v_error, avE, avE_error, avM, avM_error))

    # plt.ioff()
    # plt.clf()
    # plt.plot(sweep_list, E_list, 'r+')
    # plt.savefig("E_vs_sweep.png")
    # plt.clf()
    # plt.plot(sweep_list, M_list, 'r+')
    # plt.savefig("M_vs_sweep.png")
    # plt.clf()

    outfile.close()
    infile.close()

main()




