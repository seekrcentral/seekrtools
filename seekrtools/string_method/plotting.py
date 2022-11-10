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
#import seekr2.modules.common_base as base

class Toy_plot():
    def __init__(self, function_str, title, boundaries, landscape_resolution=100, stride=1,
                 timestep=0.002, steps_per_frame=100, model=None):
        """
        Object for plotting toy SEEKR calculations.
        """
        self.title = title
        self.boundaries = boundaries
        self.landscape_resolution = landscape_resolution
        self.function_str = self.refine_function_str(function_str)
        
        fig, ax = self.plot_energy_landscape()
        self.fig = fig
        self.ax = ax
        self.num_steps = 0
        self.stride = stride
        self.timestep = timestep
        self.steps_per_frame = steps_per_frame
        self.model = model
        self.smd_function = None
        self.smd_speed = None
        self.smd_line = None
        
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
        
    def load_trajs(self, pdb_path, topology=None):
        """
        Load the trajectory files of the toy system and return arrays of 
        its positions over time.
        """
        if topology is None:
            traj = mdtraj.load(pdb_path)
        else:
            traj = mdtraj.load(pdb_path, top=topology)
        positions = traj.xyz
        
        self.num_steps = len(positions)

        return positions
    
    def traj_walk(self, positions):
        """
        Create the 'walk' that will be plotted - a series of points on 
        the 2D plot that will be animated
        """
        walk = []
        for positions_frame in positions:
            walk.append(positions_frame[0,0:2])
        
        return np.array(walk)
    
    def update_lines(self, num, walk, line):
        """
        This function is called every step of the animation to determine
        where to draw the animated points and lines.
        """
        index = self.stride * num
        circle1 = self.circle1_list[0]
        circle2 = self.circle2_list[0]
        if index < len(walk):
            line.set_data(walk[:index+1, 0], walk[:index+1, 1])
            circle1.center=(walk[index, 0], walk[index, 1])
            circle2.center=(walk[index, 0], walk[index, 1])
        else:
            line.set_data(walk[:, 0], walk[:, 1])
            circle1.center=(walk[-1, 0], walk[-1, 1])
            circle2.center=(walk[-1, 0], walk[-1, 1])
        
        time_str = "{:.2f} ps".format((self.stride + index) * self.timestep * self.steps_per_frame)
        self.frame_text.set_text(time_str)
        
        if self.smd_speed is not None:
            x1 = -10.0
            x2 = 10.0
            value = self.model.anchors[0].variables["v_0"] + num*self.smd_speed
            x = x1
            y1 = eval(self.smd_function)
            x = x2
            y2 = eval(self.smd_function)
            #plt.axline((x1, y1), (x2, y2), color="r", lw=3, animated=True)
            self.smd_line.set_data([x1,x2],[y1,y2])
        
        return line
    
    def make_graphical_objects(self):
        """
        Create the circles and lines - graphical objects shown in the
        animation.
        """
        self.circle1_list = []
        self.circle2_list = []
        font = {"weight":"bold"}
        self.frame_text = plt.text(-1.4, 1.9, "0 ns", fontdict=font)
        circle1 = plt.Circle((0,0), 0.05, color="r", zorder=2.5)
        circle2 = plt.Circle((0,0), 0.025, color="k", zorder=2.5)
        self.ax.add_patch(circle1)
        self.ax.add_patch(circle2)
        self.circle1_list.append(circle1)
        self.circle2_list.append(circle2)
        line, = self.ax.plot([], [], lw=1, color="w")
        
        if self.smd_function is not None:
            #plt.axline((x1, y1), (x2, y2), color="r", lw=3, animated=True)
            self.smd_line, = self.ax.plot([], [], color="r",lw=3, zorder=1.5)
            
        return line
    
    #def animate_trajs(fig, walk, line, num_steps, circle1, circle2):
    def animate_trajs(self, pdb_path, topology=None):
        """
        Animate the trajectories to view how the system progresses along
        the landscape.
        """
        positions = self.load_trajs(pdb_path, topology)
        walk = self.traj_walk(positions)
        line = self.make_graphical_objects()
        num_frames = self.num_steps // self.stride + 1
        if self.smd_function is not None:
            first_value = self.model.anchors[0].variables["v_0"]
            last_value = self.model.anchors[-2].variables["v_0"]
            self.smd_speed = (last_value - first_value) / num_frames
        
        ani = animation.FuncAnimation(self.fig, self.update_lines, fargs=(walk, line), frames=num_frames, interval=2, repeat=False)
        return ani

def draw_linear_milestones(value, milestone_function):
    """
    Create a series of milestones linear in shape.
    """
    
    #x1 = toy_plot.boundaries[0,0] * 0.1
    #x2 = toy_plot.boundaries[0,1] * 0.1
    x1 = 0
    x2 = 0.1
    
    x = x1
    y1 = eval(milestone_function)
    x = x2
    y2 = eval(milestone_function)
    plt.axline((x1, y1), (x2, y2), color="k", lw=3)
        
    return

"""
def main():
    title = "Quadratic Potential System"
    boundaries = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    toy_plot = Toy_plot(model, title, boundaries)
    milestone_function = "value"
    draw_linear_milestones(toy_plot, milestone_function)
    ani = toy_plot.animate_trajs(animating_anchor_indices=[0,1])
    plt.show()

if __name__ == "__main__":
    main()
"""
