"""
Plot movie of trajectories on 2D CVs for each Voronoi tesselation anchor.
"""

import os
import glob
import argparse

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import mdtraj
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seekr2.modules.common_base as base
import seekr2.modules.mmvt_cvs.mmvt_cv_base as mmvt_cv_base
import seekr2.analyze as analyze

import seekrtools.visualize.plot_2d_cv as plot_2d_cv
from seekrtools.visualize.plot_2d_cv import PLOTS_DIRECTORY_NAME

def make_boundaries(anchor_cv_values):
    margin = 0.2
    min_x = 1e9
    max_x = -1e9
    min_y = 1e9
    max_y = -1e9
    for single_anchor_cv_values in anchor_cv_values:
        for anchor_value_pair in single_anchor_cv_values:
            val_x = anchor_value_pair[0]
            val_y = anchor_value_pair[1]
            if val_x < min_x:
                min_x = val_x
            if val_x > max_x:
                max_x = val_x
            if val_y < min_y:
                min_y = val_y
            if val_y > max_y:
                max_y = val_y
    
    boundaries = [min_x - margin, max_x + margin, min_y - margin, 
                  max_y + margin]
    return boundaries

def make_traj_cv_values(model):
    rootdir = model.anchor_rootdir
    voronoi_cv = model.collective_variables[0]
    anchor_cv_values = []
    for alpha, anchor in enumerate(model.anchors):
        if anchor.bulkstate:
            anchor_cv_values.append([])
            continue
        
        if anchor.__class__.__name__ in ["MMVT_toy_anchor", "Elber_toy_anchor"]:
            top_file_name = os.path.join(rootdir, anchor.directory, 
                                    anchor.building_directory, "toy.pdb")
            
        else:
            building_directory = os.path.join(
                model.anchor_rootdir, anchor.directory, 
                anchor.building_directory)
            prod_directory = os.path.join(
                model.anchor_rootdir, anchor.directory, 
                anchor.production_directory)
            if anchor.amber_params is not None:
                top_file_name = os.path.join(
                    building_directory, anchor.amber_params.prmtop_filename)
                mmvt_traj_basename = mmvt_cv_base.OPENMMVT_BASENAME+"*.dcd"
                mmvt_traj_glob = os.path.join(prod_directory, 
                                              mmvt_traj_basename)
                mmvt_traj_filenames = glob.glob(mmvt_traj_glob)
                if len(mmvt_traj_filenames) == 0:
                    anchor_cv_values.append([])
                    continue
            else:
                raise Exception("Only Amber inputs implemented at this time.")
    
        print(f"loading files: {top_file_name} and {mmvt_traj_filenames}")
        cv_values = []
        #for chunk in mdtraj.iterload(mmvt_traj_filenames, top=top_file_name, 
        #                             chunk=10):
        chunk = mdtraj.load(mmvt_traj_filenames, top=top_file_name)
        for frame_index in range(chunk.n_frames):
            sub_cv1 = voronoi_cv.child_cvs[0]
            sub_cv2 = voronoi_cv.child_cvs[1]
            value1 = sub_cv1.get_mdtraj_cv_value(chunk, frame_index)
            value2 = sub_cv2.get_mdtraj_cv_value(chunk, frame_index)
            values = [value1, value2]
            cv_values.append(values)
        
        anchor_cv_values.append(cv_values)
                
    return anchor_cv_values

class Movie_plot():
    def __init__(self, model, title, boundaries, stride=1, voronoi=False):
        """
        Object for plotting 2D CV SEEKR calculations.
        """
        self.model = model
        self.title = title
        self.boundaries = boundaries
        self.num_steps = 0
        self.stride = stride
        return
    
    def traj_walk(self, positions_list):
        """
        Create the 'walk' that will be plotted - a series of points on 
        the 2D plot that will be animated
        """
        walks = []
        for positions_anchor in positions_list:
            walk = []
            for positions_frame in positions_anchor:
                walk.append(positions_frame)
            
            walks.append(np.array(walk))
            if self.num_steps < len(positions_anchor):
                self.num_steps = len(positions_anchor)
            
        return walks
    
    def update_lines(self, num, walks, lines):
        """
        This function is called every step of the animation to determine
        where to draw the animated points and lines.
        """
        
        index = self.stride * num
        for i, (walk, line) in enumerate(zip(walks,lines)):
            if len(walk) == 0: 
                continue
            circle1 = self.circle1_list[i]
            circle2 = self.circle2_list[i]
            if index < len(walk):
                line.set_data(walk[:index+1, 0], walk[:index+1, 1])
                circle1.center=(walk[index, 0], walk[index, 1])
                circle2.center=(walk[index, 0], walk[index, 1])
            else:
                line.set_data(walk[:, 0], walk[:, 1])
                circle1.center=(walk[-1, 0], walk[-1, 1])
                circle2.center=(walk[-1, 0], walk[-1, 1])
        
        time_str = "{:.2f} ps".format(
            (self.stride + index) \
            * self.model.openmm_settings.langevin_integrator.timestep \
            * self.model.calculation_settings.trajectory_reporter_interval)
        self.frame_text.set_text(time_str)
        return lines
    
    def make_graphical_objects(self, num_walks):
        """
        Create the circles and lines - graphical objects shown in the
        animation.
        """
        self.circle1_list = []
        self.circle2_list = []
        font = {"weight":"bold"}
        text_x = self.boundaries[0]
        text_y = self.boundaries[2]
        self.frame_text = plt.text(text_x, text_y, "0 ns", fontdict=font)
        lines = []
        for i in range(num_walks):
            circle1 = plt.Circle((0,0), 0.005, color="r", zorder=2.5)
            circle2 = plt.Circle((0,0), 0.0025, color="k", zorder=2.5)
            self.ax.add_patch(circle1)
            self.ax.add_patch(circle2)
            self.circle1_list.append(circle1)
            self.circle2_list.append(circle2)
            line, = self.ax.plot([], [], lw=1, color="k")
            lines.append(line)
        return lines
    
    def plot_voronoi_tesselation(self, x_coordinate_title, y_coordinate_title):
        points = []
        num_variables = len(model.anchors[0].variables)
        for alpha, anchor in enumerate(model.anchors):
            #if anchor.bulkstate:
            #    continue
            values = []
            for i in range(num_variables):
                var_name = "value_0_{}".format(i)
                values.append(anchor.variables[var_name])
            
            points.append(values)
        
        points.append([-100, -100])
        points.append([100, 100])
        
        vor = Voronoi(points)
        fig = voronoi_plot_2d(vor, line_width=2, show_vertices=False)
        ax = plt.gca()
        ax.axis(self.boundaries)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(self.title)
        ax.set_xlabel(x_coordinate_title)
        ax.set_ylabel(y_coordinate_title)
        self.ax = ax
        self.fig = fig
        return
        
    def animate_trajs(self, positions_list, animating_anchor_indices):
        """
        Animate the trajectories to view how the system progresses along
        the landscape.
        """
        walks = self.traj_walk(positions_list)
        lines = self.make_graphical_objects(len(walks))
        
        num_frames = self.num_steps // self.stride
        ani = animation.FuncAnimation(
            self.fig, self.update_lines, fargs=(walks, lines), 
            frames=num_frames, interval=60, repeat=False)
        return ani

def movie_trajectories_2d_cv(
        model, plot_dir=None, x_coordinate_title="X-Coordinate", 
        y_coordinate_title="Y-Coordinate"):
    traj_cv_values = make_traj_cv_values(model)
    boundaries = make_boundaries(traj_cv_values)
    title = "Anchor Trajectories"
    movie_plot = Movie_plot(model, title, boundaries)
    animating_anchor_indices = list(range(model.num_anchors))
    movie_plot.plot_voronoi_tesselation(x_coordinate_title, y_coordinate_title)
    ani = movie_plot.animate_trajs(traj_cv_values, animating_anchor_indices)
    if plot_dir is None:
        plt.show()
    else:
        movie_filename = os.path.join(plot_dir, "movie_trajectory.mp4")
        print("Saving movie to:", movie_filename)
        writervideo = animation.FFMpegWriter(fps=60) 
        ani.save(movie_filename, writer=writervideo)
    
    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of model XML file for a SEEKR2 calculation. "\
        "One or more starting structures must be present in one or more of "\
        "the anchors.")
    #argparser.add_argument(
    #    "-t", "--title", dest="title", default="Voronoi Tesselation",
    #    type=str, help="The title of the plot")
    argparser.add_argument(
        "-x", "--x_coordinate_title", dest="x_coordinate_title", 
        default="X-Coordinate",
        type=str, help="The title of x-coordinate")
    argparser.add_argument(
        "-y", "--y_coordinate_title", dest="y_coordinate_title", 
        default="Y-Coordinate",
        type=str, help="The title of y-coordinate")
    argparser.add_argument(
        "-d", "--dpi", dest="dpi", default=200, type=int,
        help="The DPI (dots per inch) of resolution for plots.")
        
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    #title = args["title"]
    x_coordinate_title = args["x_coordinate_title"]
    y_coordinate_title = args["y_coordinate_title"]
    dpi = args["dpi"]
    model = base.load_model(model_file)
    plot_dir = os.path.join(model.anchor_rootdir, PLOTS_DIRECTORY_NAME)
    movie_trajectories_2d_cv(model, plot_dir, x_coordinate_title, 
        y_coordinate_title)
    