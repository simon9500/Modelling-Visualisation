# Modelling and Visualisation in Physics
# Checkpoint 2
# Game of Life simulation
# Simon McLaren

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# Class to simulate the 'Game of Life'
class gameoflife():

    def __init__(self, init, lx, ly):

        # Define initial variables
        self.lx = int(lx)
        self.ly = int(ly)
        if init == 'Random':
            self.lattice = np.random.choice([1, 0], size=(self.lx, self.ly))
        elif init == 'Allalive':
            self.lattice = np.ones((self.lx, self.ly), dtype=np.int)
        elif init == 'Alldead':
            self.lattice = np.zeros((self.lx, self.ly), dtype=np.int)
        elif init == 'Glider':
            # Set the glider to the top left corner
            self.lattice = np.zeros((self.lx, self.ly), dtype=np.int)
            self.lattice[0][1] = 1
            self.lattice[1][2] = 1
            self.lattice[2][0] = 1
            self.lattice[2][1] = 1
            self.lattice[2][2] = 1
        elif init == 'Oscillator':
            # Set the oscillator to the centre of the lattice
            self.lattice = np.zeros((self.lx, self.ly), dtype=np.int)
            self.lattice[self.lx / 2][self.ly / 2] = 1
            self.lattice[self.lx / 2][self.ly / 2 + 1] = 1
            self.lattice[self.lx / 2][self.ly / 2 - 1] = 1

    # Reset lattice according to conditions defined by 'init'
    def reset_lattice(self, init):
        if init == 'Random':
            self.lattice = np.random.choice([1, 0], size=(self.lx, self.ly))
        elif init == 'Allalive':
            self.lattice = np.ones((self.lx, self.ly), dtype=np.int)
        elif init == 'Alldead':
            self.lattice = np.zeros((self.lx, self.ly), dtype=np.int)
        elif init == 'Glider':
            # Set the glider to the top left corner
            self.lattice = np.zeros((self.lx, self.ly), dtype=np.int)
            self.lattice[0][1] = 1
            self.lattice[1][2] = 1
            self.lattice[2][0] = 1
            self.lattice[2][1] = 1
            self.lattice[2][2] = 1
        elif init == 'Oscillator':
            # Set the oscillator to the centre of the lattice
            self.lattice = np.zeros((self.lx, self.ly), dtype=np.int)
            self.lattice[self.lx / 2][self.ly / 2] = 1
            self.lattice[self.lx / 2][self.ly / 2 + 1] = 1
            self.lattice[self.lx / 2][self.ly / 2 - 1] = 1

    # Returns the number of nearest neighbour cells that are dead/alive
    def nearest_neighbours_dead_alive(self, cell):
        alive = 0
        dead = 0
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
                              self.lattice[i_up][j], self.lattice[i_down][j],
                              self.lattice[i_up][j_up], self.lattice[i_up, j_down],
                              self.lattice[i_down][j_down], self.lattice[i_down][j_up]]
        for nn in nearest_neighbours:
            if nn == 1:
                alive += 1
            elif nn == 0:
                dead += 1
        return dead, alive

    # Parallel update of the lattice according to the rules of the game of life
    def update(self):
        lattice_copy = np.copy(self.lattice)
        # Loop through all cells
        for i in range(self.lx):
            for j in range(self.ly):
                dead_nn, alive_nn = self.nearest_neighbours_dead_alive([i, j])
                if self.lattice[i][j] == 1:
                    if alive_nn < 2 or alive_nn > 3:
                        lattice_copy[i][j] = 0
                elif self.lattice[i][j] == 0:
                    if alive_nn == 3:
                        lattice_copy[i][j] = 1
        return lattice_copy

    # Calculate the centre of mass of a glider
    # Note: the glider must be the only object in the lattice for this to work
    def glider_CoM(self):
        sum = np.array([0, 0]) # Initialise vector sum
        n_points = 0. # Number of centre of mass coords in the sum
        for i in range(self.lx):
            for j in range(self.ly):
                if self.lattice[i][j] == 1.:
                    sum += np.array([i, j])
                    n_points += 1.
        return sum / n_points

    # Checks if the glider is not crossing any boundaries
    # Only works for a single glider state of a lattice
    # Returns True if no, False if yes
    def glider_boundary_check(self):
        live_elements = np.nonzero(self.lattice) # Find positions of glider cells
        live_elements_x = live_elements[0]
        live_elements_y = live_elements[1]
        # If the alive glider cells are not crossing any boundaries their x or y coords will only differ by a maximum of 2
        if np.amax(live_elements_x) - np.amin(live_elements_x) <= 2 and \
            np.amax(live_elements_y) - np.amin(live_elements_y) <= 2:
            return True
        else:
            return False

    # Simulate a single lattice for N steps
    @staticmethod
    def simulate(lattice, N_steps, animate_freq, animate=False):
        for step in range(N_steps):
            print step
            if step % animate_freq == 0 and animate == True:
                plt.cla()
                im = plt.imshow(lattice.lattice, animated=True)
                plt.draw()
                plt.pause(0.0001)
            lattice.lattice = lattice.update()

    # Simulate a glider for N steps
    # Tracking the centre of mass against time
    # Returns the CoM locations against time
    @staticmethod
    def simulate_glider(lattice, N_steps, animate_freq, animate=False):
        glider_CoM_list = [] # List of CoM coords
        record = True # Boolean to indicate if CoM coords should be recorded
        for step in range(N_steps):
            print step
            # Stop recording when glider reaches boundary in opposite corner
            if lattice.glider_boundary_check() == False:
                record = False
            if record == True:
                print lattice.glider_CoM()
                glider_CoM_list.append(lattice.glider_CoM())
            # Animation
            if step % animate_freq == 0 and animate == True:
                plt.cla()
                im = plt.imshow(lattice.lattice, vmin=0., vmax=1., cmap='jet', animated=True)
                plt.draw()
                plt.pause(0.0001)
            lattice.lattice = lattice.update()

        glider_CoM_list = np.array(glider_CoM_list)
        return glider_CoM_list

def main():
    # Read probability values and system size from user
    if len(sys.argv) != 2:
        print "Wrong number of arguments."
        print "Usage: " + sys.argv[0] + " <input file>"
        quit()
    else:
        infileName = sys.argv[1]

    infile = open(infileName, 'r')

    line = infile.readline()
    tokens = line.split()

    init = str(tokens[0])
    size = int(tokens[1])
    sweeps = int(tokens[2])
    animate_freq = int(tokens[3])

    lattice = gameoflife(init, size, size)

    # Animation
    fig = plt.figure()
    im = plt.imshow(lattice.lattice, vmin=0., vmax=1., cmap='jet', animated=True)
    plt.ion()

    # Simulating
    if init == 'Glider':
        glider_CoM_list = gameoflife.simulate_glider(lattice, sweeps, animate_freq, animate=True)
        total_timesteps = len(glider_CoM_list) # Number of timesteps the CoM was recorded
        CoM_coord_1 = glider_CoM_list[0]
        CoM_coord_n = glider_CoM_list[total_timesteps-1]
        distance_travelled = np.linalg.norm(CoM_coord_n - CoM_coord_1) # Distance travelled in total_timesteps
        # Velocity is distance / time
        print "The velocity of the glider is %f" %(distance_travelled / total_timesteps)
    else:
        gameoflife.simulate(lattice, sweeps, animate_freq, animate=True)

main()
