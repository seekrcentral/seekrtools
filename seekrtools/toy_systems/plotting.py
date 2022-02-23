"""
plotting.py

Provide utilities to plot the toy systems.
"""
import re
import os
import glob
from math import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mdtraj
try:
    import openmm.unit as unit
except ModuleNotFoundError:
    import simtk.unit as unit
import seekr2.modules.common_base as base

class Toy_plot():
    def __init__(self, model, title, boundaries, landscape_resolution=100):
        """
        Object for plotting toy SEEKR calculations.
        """
        self.model = model
        self.title = title
        self.boundaries = boundaries
        self.landscape_resolution = landscape_resolution
        
        self.function_str = self.refine_function_str(
            model.toy_settings.potential_energy_expression)
        fig, ax = self.plot_energy_landscape()
        self.fig = fig
        self.ax = ax
        self.num_steps = 0
        
        return


    def refine_function_str(self, old_function_str):
        """
        Convert the energy landscape from a form used by OpenMM
        to a form used by Python.
        """
        new_function_str = re.sub("\^", "**", old_function_str)
        return new_function_str

    def plot_energy_landscape(self, max_Z=20):
        """
        Create the plot and plot the energy landscape.
        """
        z1 = 0.0
        min_x = self.boundaries[0,0]
        max_x = self.boundaries[0,1]
        min_y = self.boundaries[1,0]
        max_y = self.boundaries[1,1]
        bounds_x = np.linspace(min_x, max_x, self.landscape_resolution)
        bounds_y = np.linspace(min_y, max_y, self.landscape_resolution)
        X,Y = np.meshgrid(bounds_x, bounds_y)
        Z = np.zeros((self.landscape_resolution, self.landscape_resolution))
        for i, x1 in enumerate(bounds_x):
            for j, y1 in enumerate(bounds_y):
                # fill out landscape here
                Z[j,i] = eval(self.function_str)
                if Z[j,i] > max_Z:
                    Z[j,i] = max_Z
        
        fig, axs = plt.subplots(nrows=1, ncols=1)
        p = axs.pcolor(X, Y, Z, cmap=plt.cm.jet, vmin=Z.min(), 
                          vmax=Z.max())
        axs.set_title(self.title)
        axs.set_xlabel("$x_{1}$ (nm)")
        axs.set_ylabel("$y_{1}$ (nm)")
        cbar = plt.colorbar(p)
        cbar.set_label("Energy (kcal/mol)")
        return fig, axs
        
    def load_trajs(self, anchor_indices):
        """
        Load the trajectory files of the toy system and return arrays of 
        its positions over time.
        """
        rootdir = self.model.anchor_rootdir
        positions_list = []
        for anchor_index in anchor_indices:
            anchor = self.model.anchors[anchor_index]
            pdb_path = os.path.join(rootdir, anchor.directory, anchor.building_directory, "toy.pdb")
            frames_glob = os.path.join(rootdir, anchor.directory, anchor.production_directory, "mmvt1.*dcd")
            num_frames = len(glob.glob(frames_glob))
            assert num_frames > 0, "No DCD detected for anchor {}.".format(anchor_index)
            positions_frame_list = []
            for frame in range(num_frames):
                if num_frames == 1:
                    dcd_file_glob = "*.dcd".format(frame)
                else:
                    dcd_file_glob = "*.swarm_{}.dcd".format(frame)
                dcd_glob = os.path.join(rootdir, anchor.directory, anchor.production_directory, dcd_file_glob)
                dcd_files = glob.glob(dcd_glob)
                traj = mdtraj.load(dcd_files, top=pdb_path)
                positions_frame_list.append(traj.xyz)
                
            positions_list.append(positions_frame_list)
        
        self.num_steps = len(positions_list[0][0])
        return positions_list
    
    def traj_walk(self, positions_list):
        """
        Create the 'walk' that will be plotted - a series of points on 
        the 2D plot that will be animated
        """
        walks = []
        for positions_anchor in positions_list:
            for positions_swarm in positions_anchor:
                walk = []
                for positions_frame in positions_swarm:
                    walk.append(positions_frame[0,0:2])
                
                walks.append(np.array(walk))
            
        return walks
    
    def update_lines(self, num, walks, lines):
        """
        This function is called every step of the animation to determine
        where to draw the animated points and lines.
        """
        for i, (walk, line) in enumerate(zip(walks,lines)):
            line.set_data(walk[:num+1, 0], walk[:num+1, 1])
            circle1 = self.circle1_list[i]
            circle2 = self.circle2_list[i]
            circle1.center=(walk[num, 0], walk[num, 1])
            circle2.center=(walk[num, 0], walk[num, 1])
        
        return lines
    
    def make_graphical_objects(self, num_walks):
        """
        Create the circles and lines - graphical objects shown in the
        animation.
        """
        self.circle1_list = []
        self.circle2_list = []
        lines = []
        for i in range(num_walks):
            circle1 = plt.Circle((0,0), 0.05, color="r", zorder=2.5)
            circle2 = plt.Circle((0,0), 0.025, color="k", zorder=2.5)
            self.ax.add_patch(circle1)
            self.ax.add_patch(circle2)
            self.circle1_list.append(circle1)
            self.circle2_list.append(circle2)
            line, = self.ax.plot([], [], lw=1, color="w")
            lines.append(line)
        return lines
    
    #def animate_trajs(fig, walk, line, num_steps, circle1, circle2):
    def animate_trajs(self, animating_anchor_indices):
        """
        Animate the trajectories to view how the system progresses along
        the landscape.
        """
        positions_list = self.load_trajs(animating_anchor_indices)
        walks = self.traj_walk(positions_list)
        print("len(walks):", len(walks))
        lines = self.make_graphical_objects(len(walks))
        
        ani = animation.FuncAnimation(self.fig, self.update_lines, fargs=(walks, lines), frames=self.num_steps, interval=20, repeat=False)
        return ani

def draw_linear_milestones(toy_plot, milestone_function):
    """
    Create a series of milestones linear in shape.
    """
    model = toy_plot.model
    #x1 = toy_plot.boundaries[0,0] * 0.1
    #x2 = toy_plot.boundaries[0,1] * 0.1
    x1 = 0
    x2 = 0.1
    values = set()
    for anchor in model.anchors:
        for milestone in anchor.milestones:
            value = milestone.variables["value"]
            values.add(value)
    
    for value in values:
        x = x1
        y1 = eval(milestone_function)
        x = x2
        y2 = eval(milestone_function)
        plt.axline((x1, y1), (x2, y2), color="k", lw=3)
        
    return
    
def main():
    title = "Quadratic Potential System"
    model_file = "/home/lvotapka/toy_seekr_systems/simple_quadratic/model.xml"
    model = base.load_model(model_file)
    boundaries = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    toy_plot = Toy_plot(model, title, boundaries)
    milestone_function = "value"
    draw_linear_milestones(toy_plot, milestone_function)
    ani = toy_plot.animate_trajs(animating_anchor_indices=[0,1])
    plt.show()

if __name__ == "__main__":
    main()
