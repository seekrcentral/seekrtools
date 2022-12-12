"""
Plot the results of a 2D string method calculation.
"""

import os
import argparse
from math import exp
import re
import glob

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

import seekr2.modules.common_base as base

from seekrtools.string_method.base import STRING_LOG_GLOB

PLOTS_DIRECTORY_NAME = "string_method_plots"

def refine_function_str(old_function_str):
        """
        Convert the energy landscape from a form used by OpenMM
        to a form used by Python.
        """
        new_function_str = re.sub("\^", "**", old_function_str)
        return new_function_str

def plot_voronoi_tesselation(model, boundaries, anchor_values):
    points = []
    num_variables = len(model.anchors[0].variables)
    for alpha, anchor in enumerate(model.anchors):
        #if anchor.bulkstate:
        #    continue
        if alpha in anchor_values:
            points.append(anchor_values[alpha])
        else:
            values = []
            for i in range(num_variables):
                var_name = "value_0_{}".format(i)
                values.append(anchor.variables[var_name])
    
    points.append([-100, -100])
    points.append([100, 100])
    
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor, line_width=2, show_vertices=False)
    ax = plt.gca()
    ax.axis(boundaries)
    ax.set_aspect('equal', adjustable='box')
    
    return fig, ax, points[:-2]

def plot_potential(model, plot_dir, iteration, anchor_values,
                   trajectory_values, boundaries, title, x_coordinate_title, 
                   y_coordinate_title):
    fig, ax, points = plot_voronoi_tesselation(model, boundaries, anchor_values)
    min_x = boundaries[0]
    max_x = boundaries[1]
    min_y = boundaries[2]
    max_y = boundaries[3]
    if model.using_toy():
        my_color = "w"
        potential_energy_expression \
            = refine_function_str(
                model.toy_settings.potential_energy_expression)
        
        #boundaries = np.array([-1.5, 1.2, -0.2, 2.0])
        landscape_resolution=100
        max_Z=20
        bounds_x = np.linspace(min_x, max_x, landscape_resolution)
        bounds_y = np.linspace(min_y, max_y, landscape_resolution)
        X,Y = np.meshgrid(bounds_x, bounds_y)
        Z = np.zeros((landscape_resolution, landscape_resolution))
        for i, x1 in enumerate(bounds_x):
            for j, y1 in enumerate(bounds_y):
                # fill out landscape here
                Z[j,i] = eval(potential_energy_expression)
                if Z[j,i] > max_Z:
                    Z[j,i] = max_Z
        
        p = ax.pcolor(X, Y, Z, cmap=plt.cm.jet, vmin=Z.min(), vmax=Z.max())
        
        cbar = plt.colorbar(p)
        cbar.set_label("Energy (kcal/mol)")
    
    else:
        my_color = "k"
    
    ax.set_title(title)
    ax.set_xlabel(x_coordinate_title)
    ax.set_ylabel(y_coordinate_title)
    
    # Add trajectory points
    for alpha in trajectory_values:
        if model.anchors[alpha].bulkstate:
            continue
        for point in trajectory_values[alpha]:
            pass
            #circle = plt.Circle(point, 0.005, color=my_color, zorder=2.5, alpha=0.5)
            #ax.add_patch(circle)
            
    # Add lines to make string
    points_array = np.array(points).T
    plt.plot(points_array[0], points_array[1], "o", linestyle="-", linewidth=2)
    
    # Add iteration label in upper corner
    time_str = "iteration: {}".format(iteration)
    font = {"weight":"bold"}
    plt.text(min_x+0.1, max_y-0.2, time_str, fontdict=font)
    
    #plt.show()
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_filename = os.path.join(plot_dir, "string_{}.png".format(iteration))
    plt.savefig(plot_filename)
        
    return

def parse_log_file(model, log_file_glob):
    anchor_values_by_iter = []
    trajectory_values_by_iter = []
    counter = 0
    anchor_values = None
    trajectory_values = None
    log_file_list = base.order_files_numerically(glob.glob(log_file_glob))
    for log_file_name in log_file_list:
        with open(log_file_name, "r") as f:
            for line in f.readlines():
                if line.startswith("#"):
                    continue
                elif line.startswith("iteration"):
                    line_list = line.strip().split(" ")
                    iteration = int(line_list[1])
                    if anchor_values is not None:
                        anchor_values_by_iter.append(anchor_values)
                        trajectory_values_by_iter.append(trajectory_values)
                    anchor_values = {}
                    trajectory_values = {}
                    counter += 1
                elif line.startswith("anchor"):
                    line_list = line.strip().split("\t")
                    alpha = int(line_list[0].split(" ")[1])
                    cv_values_str = line_list[1].strip("[]").split(",")
                    #if model.anchors[alpha].bulkstate:
                    #    continue
                    cv_values_float = list(map(float, cv_values_str))
                    anchor_values[alpha] = cv_values_float
                    traj_pairs = line_list[2:]
                    traj_list = []
                    for traj_pair_str in traj_pairs:
                        traj_values_str = traj_pair_str.strip("[]").split(",")
                        traj_values_float = list(map(float, traj_values_str))
                        traj_list.append(traj_values_float)
                        
                    trajectory_values[alpha] = traj_list
                
            anchor_values_by_iter.append(anchor_values)
            trajectory_values_by_iter.append(trajectory_values)
    
    return anchor_values_by_iter, trajectory_values_by_iter, counter

def make_boundaries(anchor_values_by_iter):
    margin = 0.2
    
    min_x = 1e9
    max_x = -1e9
    min_y = 1e9
    max_y = -1e9
    for anchor_values in anchor_values_by_iter:
        for alpha in anchor_values:
            val_x = anchor_values[alpha][0]
            val_y = anchor_values[alpha][1]
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


def make_model_plot(model, plot_dir, title, x_coordinate_title, 
                    y_coordinate_title, boundaries=None):
    anchor_values = {}
    num_variables = len(model.anchors[0].variables)
    for alpha, anchor in enumerate(model.anchors):
        values = []
        for i in range(num_variables):
            var_name = "value_0_{}".format(i)
            values.append(anchor.variables[var_name])
        anchor_values[alpha] = values
    if boundaries is None:
        boundaries = make_boundaries([anchor_values])
    plot_potential(model, plot_dir, "model", anchor_values, [], boundaries, 
                   title, x_coordinate_title, y_coordinate_title)
    return boundaries

def make_plots_from_logs(model, plot_dir, log_file_glob, title, 
                        x_coordinate_title, y_coordinate_title):
    anchor_values_by_iter, trajectory_values_by_iter, max_iter \
        = parse_log_file(model, log_file_glob)
    if max_iter == 0:
        boundaries = None
    else:
        boundaries = make_boundaries(anchor_values_by_iter)
        
    for iteration in range(max_iter):
        plot_potential(model, plot_dir, iteration, 
                       anchor_values_by_iter[iteration],
                       trajectory_values_by_iter[iteration],
                       boundaries, title, x_coordinate_title, 
                       y_coordinate_title)
    return boundaries

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of model XML file for a SEEKR2 calculation. "\
        "One or more starting structures must be present in one or more of "\
        "the anchors.")
    argparser.add_argument(
        "-t", "--title", dest="title", default="String Method",
        type=str, help="The title of the plot")
    argparser.add_argument(
        "-x", "--x_coordinate_title", dest="x_coordinate_title", 
        default="X-Coordinate",
        type=str, help="The title of x-coordinate")
    argparser.add_argument(
        "-y", "--y_coordinate_title", dest="y_coordinate_title", 
        default="Y-Coordinate",
        type=str, help="The title of y-coordinate")
    
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    title = args["title"]
    x_coordinate_title = args["x_coordinate_title"]
    y_coordinate_title = args["y_coordinate_title"]
    
    model = base.load_model(model_file)
    plot_dir = os.path.join(model.anchor_rootdir, PLOTS_DIRECTORY_NAME)
    log_file_glob = os.path.join(model.anchor_rootdir, STRING_LOG_GLOB)
    boundaries = make_plots_from_logs(model, plot_dir, log_file_glob, title,
                                     x_coordinate_title, y_coordinate_title)
    make_model_plot(model, plot_dir, title, x_coordinate_title, 
                    y_coordinate_title, boundaries)
    #make_current_plot(model, plot_dir, boundaries)
    