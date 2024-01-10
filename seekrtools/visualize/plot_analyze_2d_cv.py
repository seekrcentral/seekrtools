"""
Plot 2D CVs for each Voronoi tesselation anchor, coloring by pi_alpha.
"""

import os
import argparse

import numpy as np

import seekrtools.visualize.plot_2d_cv as plot_2d_cv
from seekrtools.visualize.plot_2d_cv import PLOTS_DIRECTORY_NAME

import seekr2.modules.common_base as base
import seekr2.analyze as analyze

def make_analyze_plots(model, analysis, plot_dir=None,
                    x_coordinate_title="X-Coordinate", 
                    y_coordinate_title="Y-Coordinate", omit_iter_label=False, 
                    dpi=100, base_name="analyze", boundaries=None, traj_values=None,
                    draw_string=False):
    anchor_values = plot_2d_cv.find_anchor_points(model)
    if boundaries is None:
        #point_values = [anchor_values]
        point_values = list(anchor_values.values())
        if traj_values is None:
            traj_values = []
        else:
            point_values += traj_values[0]
        boundaries = plot_2d_cv.make_boundaries(point_values)
    
    anchor_free_energies = analysis.free_energy_anchors
    #fill_values = np.linspace(0.0, 20.0, len(model.anchors))
    for i, fill_value in enumerate(anchor_free_energies):
        if not np.isfinite(fill_value):
            anchor_free_energies[i] = np.max(anchor_free_energies[np.isfinite(anchor_free_energies)])
    
    
    plot_2d_cv.plot_anchors(
        model, plot_dir, None, traj_values, boundaries, 
        "Anchor Free Energies", x_coordinate_title, y_coordinate_title, omit_iter_label, dpi, 
        base_name, draw_string, fill_values=anchor_free_energies, 
        colorbar_label="Energy (kcal/mol)")
    
    log_times = []
    min_time = 1.0 / max(analysis.main_data_sample.k_alpha)
    for k_alpha in analysis.main_data_sample.k_alpha:
        if k_alpha > 0.0:
            log_times.append(np.log10(1.0 / k_alpha))
        else:
            log_times.append(min_time)
    log_times.append(min_time)
    
    plot_2d_cv.plot_anchors(
        model, plot_dir, None, traj_values, boundaries, 
        "Time Between Bounces For Each Anchor", x_coordinate_title, y_coordinate_title, omit_iter_label, dpi, 
        base_name+"_time_alpha", draw_string, fill_values=log_times, 
        colorbar_label="$\log_{10}(time_{\\alpha}/ps)$")
    
    plot_2d_cv.plot_anchors(
        model, plot_dir, None, traj_values, boundaries, 
        "Anchor $N_{\\alpha,\\beta}$", x_coordinate_title, y_coordinate_title, 
        omit_iter_label, dpi, 
        base_name+"_N_alpha_beta", draw_string, fill_values=None, 
        edge_dictionary=analysis.main_data_sample.N_alpha_beta, edge_labels=True)
    
    plot_2d_cv.plot_anchors(
        model, plot_dir, None, traj_values, boundaries, 
        "Anchor $k_{\\alpha,\\beta}$", x_coordinate_title, y_coordinate_title, 
        omit_iter_label, dpi, 
        base_name+"_k_alpha_beta", draw_string, fill_values=None, 
        edge_dictionary=analysis.main_data_sample.k_alpha_beta, edge_labels=True)
    
    milestone_free_energies = analysis.free_energy_profile
    plot_2d_cv.plot_milestones(model, plot_dir, None, boundaries, 
                 "Milestone Free Energies", x_coordinate_title, y_coordinate_title, 
                 omit_iter_label, dpi, base_name+"_milestones", 
                 draw_string, fill_values=milestone_free_energies)
    
    plot_2d_cv.plot_milestones(model, plot_dir, None, boundaries, 
                 "Milestone $N_{i,j}$", x_coordinate_title, y_coordinate_title, 
                 omit_iter_label, dpi, base_name+"_milestones_N_ij", 
                 draw_string, fill_values=None, 
                 edge_dictionary=analysis.main_data_sample.N_ij_unmodified, edge_labels=True)
    
    return boundaries

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
    
    analysis = analyze.analyze(model, num_error_samples=0, skip_checks=True)
    
    #plot_2d_cv.make_model_plots(model, plot_dir, x_coordinate_title, 
    #                y_coordinate_title, True, dpi, base_name="anchor_energy",
    #                fill_values=fill_values)
    make_analyze_plots(model, analysis, plot_dir, x_coordinate_title, 
                    y_coordinate_title, True, dpi, base_name="analyze")