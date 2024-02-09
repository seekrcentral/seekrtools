"""
Plot the results of a 2D string method calculation.
"""

import os
import argparse

import glob

import seekr2.modules.common_base as base

import seekrtools.visualize.plot_2d_cv as plot_2d_cv
from seekrtools.string_method.base import STRING_LOG_GLOB

PLOTS_DIRECTORY_NAME = "string_method_plots"

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

def make_plots_from_logs(model, plot_dir, log_file_glob, title, 
                        x_coordinate_title, y_coordinate_title, 
                        omit_iter_label=False, dpi=200):
    anchor_values_by_iter, trajectory_values_by_iter, max_iter \
        = parse_log_file(model, log_file_glob)
    if max_iter == 0:
        boundaries = plot_2d_cv.auto_boundary_by_anchor_values(model)
    else:
        all_point_values = [v for d in anchor_values_by_iter for v in d.values()]
        boundaries = plot_2d_cv.make_boundaries(all_point_values)
        
    for iteration in range(max_iter):
        plot_2d_cv.plot_anchors(model, plot_dir, iteration, 
                       anchor_values_by_iter[iteration],
                       trajectory_values_by_iter[iteration],
                       boundaries, title, x_coordinate_title, 
                       y_coordinate_title, omit_iter_label, dpi, 
                       file_base="string", draw_string=True)
    print("boundaries:", boundaries)
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
    argparser.add_argument(
        "-l", "--omit_iter_label", dest="omit_iter_label", default=False,
        help="Whether to omit the 'iteration' label in the top corner of the plot.",
        action="store_true")
    argparser.add_argument(
        "-d", "--dpi", dest="dpi", default=200, type=int,
        help="The DPI (dots per inch) of resolution for plots.")
        
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    title = args["title"]
    x_coordinate_title = args["x_coordinate_title"]
    y_coordinate_title = args["y_coordinate_title"]
    omit_iter_label = args["omit_iter_label"]
    dpi = args["dpi"]
    
    model = base.load_model(model_file)
    plot_dir = os.path.join(model.anchor_rootdir, PLOTS_DIRECTORY_NAME)
    log_file_glob = os.path.join(model.anchor_rootdir, STRING_LOG_GLOB)
    boundaries = make_plots_from_logs(model, plot_dir, log_file_glob, title,
                                     x_coordinate_title, y_coordinate_title,
                                     omit_iter_label, dpi)
    plot_2d_cv.make_model_plots(model, plot_dir, title, x_coordinate_title, 
                    y_coordinate_title, omit_iter_label, dpi, boundaries, 
                    base_name="string", draw_string=True)
    #make_current_plot(model, plot_dir, boundaries)
    
