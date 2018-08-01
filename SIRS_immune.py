# Modelling and Visualisation in Physics
# Checkpoint 2
# SIRS Model simulation with immune cells
# Simon McLaren

import numpy as np
# import matplotlib.animation as animation
import matplotlib.pyplot as plt
import random
# import math
import sys

# Class to simulate the SIRS model
class SIRS_immune():

    # Initialise lattice
    # 0: Susceptible, 1: Infected, 2: Recovered, 3: Immune
    def __init__(self, init, lx, ly, p1, p2, p3, immune_frac):

        # Define initial variables
        self.lx = int(lx)
        self.ly = int(ly)
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        if init == 'Random':
            self.lattice = np.random.choice([3, 2, 1, 0], size=(self.lx, self.ly))
        elif init == 'Immune fraction':
            self.lattice = np.random.choice([2, 1, 0], size=(self.lx, self.ly))  # Set lattice to be S, I or R only
            Num_R = int(immune_frac * self.lx * self.ly)  # Number of immune cells
            self.lattice = self.lattice.flatten()  # Flatten lattice to 1D
            i = 0
            # Fill 1D lattice up sequentially with immune cells
            while i < Num_R:
                self.lattice[i] = 3
                i += 1
            np.random.shuffle(self.lattice)  # Randomly shuffle the lattice
            self.lattice = np.reshape(self.lattice, (self.lx, self.ly))  # Put lattice back to original shape

    # Reset lattice according to conditions defined by 'init'
    def reset_lattice(self, init, immune_frac):
        if init == 'Random':
            self.lattice = np.random.choice([3, 2, 1, 0], size=(self.lx, self.ly))
        elif init == 'Immune fraction':
            self.lattice = np.random.choice([2, 1, 0], size=(self.lx, self.ly))  # Set lattice to be S, I or R only
            Num_R = int(immune_frac * self.lx * self.ly)  # Number of immune cells
            self.lattice = self.lattice.flatten()  # Flatten lattice to 1D
            i = 0
            # Fill 1D lattice up sequentially with immune cells
            while i < Num_R:
                self.lattice[i] = 3
                i += 1
            np.random.shuffle(self.lattice)  # Randomly shuffle the lattice
            self.lattice = np.reshape(self.lattice, (self.lx, self.ly)) # Put lattice back to original shape

    # Set new probabilities for system update
    def set_probabilities(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    # Returns true if at least one nearest neighbour if infected, otherwise false
    def nearest_neighbours_infected(self, cell):
        i = cell[0]
        j = cell[1]
        i_up = i + 1
        if i == self.lx - 1: i_up = 0
        j_up = j + 1
        if j == self.ly - 1: j_up = 0
        i_down = i - 1
        if i == 0: i_down = self.lx - 1
        j_down = j - 1
        if j == 0: j_down = self.ly - 1
        nearest_neighbours = [self.lattice[i][j_up], self.lattice[i][j_down],
                              self.lattice[i_up][j], self.lattice[i_down][j]]
        for nn in nearest_neighbours:
            if nn == 1:
                return True
        return False

    # Sequential stochastic update for 1 sweep
    def update(self):
        N_steps = self.lx * self.ly
        for i in range(N_steps):
            # Pick a random cell as x, y coords
            x = random.randint(0, self.ly - 1)
            y = random.randint(0, self.lx - 1)
            cell_state = self.lattice[x, y] # Find the cell state
            random_num = random.random() # Find a random number between 0 and 1
            # Conditionals to determine behaviour of cell
            if cell_state == 0 and random_num <= self.p1 and self.nearest_neighbours_infected([x, y]):
                self.lattice[x][y] = 1
            elif cell_state == 1 and random_num <= self.p2:
                self.lattice[x][y] = 2
            elif cell_state == 2 and random_num <= self.p3:
                self.lattice[x][y] = 0

    # Returns the fraction of infected cells
    def infected_fraction(self):
        infected = 0.
        for i in range(self.lx):
            for j in range(self.ly):
                if self.lattice[i][j] == 1:
                    infected += 1.
        return infected / (self.lx * self.ly)

    # Simulate a single lattice for N steps
    # Loops through all sweeps even if system goes to an absorbing state
    # Returns array of all infected fractions and corresponding sweep numbers
    @staticmethod
    def simulate(lattice, N_steps, animate_freq, animate=False):
        infected_fraction = []
        # Loop through all sweeps
        for step in range(N_steps):
            lattice.update() # Update lattice according to rules
            current_inf_frac = lattice.infected_fraction()
            infected_fraction.append(current_inf_frac)
            if step % animate_freq == 0 and animate == True:
                plt.cla()
                im = plt.imshow(lattice.lattice, vmin=0., vmax=3., cmap='jet', animated=True)
                plt.draw()
                plt.pause(0.0001)
        infected_fraction = np.array(infected_fraction)
        return infected_fraction, np.arange(len(infected_fraction))

def main():

    # Read probability values and system size from user
    if len(sys.argv) != 3:
        print "Wrong number of arguments."
        print "Usage: " + sys.argv[0] + " <input file> + <output file>"
        quit()
    else:
        infileName = sys.argv[1]
        outfileName = sys.argv[2]

    infile = open(infileName, 'r')
    outfile = open(outfileName, "w")

    line = infile.readline()
    tokens = line.split()

    size = int(tokens[0])
    p1 = float(tokens[1])
    p2 = float(tokens[2])
    p3 = float(tokens[3])
    animate_freq = int(tokens[4])
    N_steps = int(tokens[5])

    init = 'Immune fraction'
    SIRS_lattice = SIRS_immune(init, size, size, p1, p2, p3, 0.)

    # Animation
    fig = plt.figure()
    im = plt.imshow(SIRS_lattice.lattice, vmin=0., vmax=3., cmap='jet', animated=True)
    plt.ion()

    immune_fractions = np.linspace(0.18, 0.3, 10) # List of immune fractions to simulate

    # Data lists
    mean_inf_frac = []
    var_inf_frac = []

    for im in immune_fractions:
        SIRS_lattice.reset_lattice(init, im)
        infected_fraction, sweep_list = SIRS_immune.simulate(SIRS_lattice, N_steps, animate_freq, animate=True)

        current_variance = np.var(infected_fraction)
        # If system is in absorbing state at end of simulation set average infected cells to 0
        # otherwise calculate mean as normal
        if infected_fraction[len(infected_fraction) - 1] == 0.:
            current_mean = 0.
        else:
            current_mean = np.mean(infected_fraction)

        mean_inf_frac.append(current_mean)
        var_inf_frac.append(current_variance)

        print im, current_mean, current_variance
        outfile.write("{0:f} {1:f} {2:f}\n".format(im, current_mean, current_variance))

    outfile.close()

main()
