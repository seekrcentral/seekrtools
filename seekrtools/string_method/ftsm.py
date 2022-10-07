"""
ftsm.py

Applies the Finite-Temperature String Method as described in the paper:

Vanden-Eijnden E, Venturoli M. Revisiting the finite temperature string 
method for the calculation of reaction tubes and free energies. J Chem 
Phys. 2009 May 21;130(19):194103. doi: 10.1063/1.3130083. 
PMID: 19466817.


"""
import os
import glob
import argparse
from math import exp
from collections import defaultdict
from shutil import copyfile

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

import seekr2.modules.common_base as base
import seekr2.modules.mmvt_base as mmvt_base
import seekr2.modules.common_cv as common_cv
import seekr2.modules.mmvt_sim_openmm as mmvt_sim_openmm
import seekr2.run as run
import seekr2.modules.runner_openmm as runner_openmm
import seekr2.modules.check as check

PLOTS_DIRECTORY_NAME = "string_method_plots"
STRING_MODEL_GLOB = "model_pre_string_*.xml"
STRING_MODEL_BASE = "model_pre_string_{}.xml"

def plot_voronoi_tesselation(model, boundaries):
    points = []
    for alpha, anchor in enumerate(model.anchors):
        values = []
        assert len(anchor.variables) == 2
        for i in range(2):
            var_name = "value_0_{}".format(i)
            value_i = anchor.variables[var_name]
            values.append(value_i)
            
        points.append(values)
    
    points.append([-100, -100])
    points.append([100, 100])
    
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor, line_width=2, show_vertices=False)
    ax = plt.gca()
    ax.axis(boundaries)
    ax.set_aspect('equal', adjustable='box')
    
    return fig, ax

def plot_muller_potential(model, plot_dir, iteration, anchor_cv_values):
    potential_energy_expression = "-20*exp(-1 * (x1 - 1)**2 + 0 * (x1 - 1) "\
        "* (y1 - 0) - 10 * (y1 - 0)**2) - 10*exp(-1 * (x1 - 0)**2 + "\
        "0 * (x1 - 0) * (y1 - 0.5) - 10 * (y1 - 0.5)**2) - "\
        "17*exp(-6.5 * (x1 + 0.5)**2 + 11 * (x1 + 0.5) * (y1 - 1.5) "\
        "- 6.5 * (y1 - 1.5)**2) + 1.5*exp(0.7 * (x1 + 1)**2 + 0.6 * (x1 + 1) "\
        "* (y1 - 1) + 0.7 * (y1 - 1)**2)"
    title = "Muller Potential System"
    boundaries = np.array([-1.5, 1.2, -0.2, 2.0])
    landscape_resolution=100
    max_Z=20
    fig, ax = plot_voronoi_tesselation(model, boundaries)
    min_x = boundaries[0]
    max_x = boundaries[1]
    min_y = boundaries[2]
    max_y = boundaries[3]
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
    ax.set_title(title)
    ax.set_xlabel("$x_{1}$ (nm)")
    ax.set_ylabel("$y_{1}$ (nm)")
    cbar = plt.colorbar(p)
    cbar.set_label("Energy (kcal/mol)")
    
    # Add trajectory points
    for alpha in anchor_cv_values:
        for point in anchor_cv_values[alpha]:
            circle = plt.Circle(point, 0.005, color="w", zorder=2.5, alpha=0.5)
            ax.add_patch(circle)
    
    # Add iteration label in upper corner
    time_str = "iteration: {}".format(iteration)
    font = {"weight":"bold"}
    plt.text(-1.4, 1.9, time_str, fontdict=font)
    
    #plt.show()
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_filename = os.path.join(plot_dir, "muller_{}.png".format(iteration))
    plt.savefig(plot_filename)
        
    return

def get_cv_values(model, anchor, voronoi_cv):
    values = []
    traj = check.load_structure_with_mdtraj(model, anchor, mode="mmvt_traj")
    num_frames = traj.n_frames
    for i in range(num_frames):
        values.append(voronoi_cv.get_mdtraj_cv_value(traj, i))
    return values

def move_anchors(model, anchor_cv_values, convergence_factor, smoothing_factor):
    old_anchor_values = []
    num_variables = len(model.anchors[0].variables)
    num_anchors = len(model.anchors)
    for alpha, anchor in enumerate(model.anchors):
        values = []
        for i in range(num_variables):
            var_name = "value_0_{}".format(i)
            value_i = anchor.variables[var_name]
            values.append(value_i)
            
        old_anchor_values.append(values)
    old_anchor_values = np.array(old_anchor_values)
    
    avg_anchor_values = []
    for alpha, anchor in enumerate(model.anchors):
        this_anchor_values = np.array(anchor_cv_values[alpha])
        avg_values = []
        for i in range(num_variables):
            avg_value = np.mean(this_anchor_values[:,i])
            avg_values.append(avg_value)
        avg_anchor_values.append(avg_values)
    
    avg_anchor_values = np.array(avg_anchor_values)
    
    adjusted_anchor_values = (1.0-convergence_factor)*old_anchor_values \
        + convergence_factor*avg_anchor_values
    
    vector_array_list = []
    for i in range(num_variables):
        variable_i_list = []
        for value in adjusted_anchor_values[:,i]:
            variable_i_list.append(value)
        vector_array_list.append(variable_i_list)
        
    tck, u, = splprep(vector_array_list, k=1, s=smoothing_factor)
    u2 = np.linspace(0, 1, num_anchors)
    new_points = splev(u2, tck)
    
    for alpha, anchor in enumerate(model.anchors):
        for i in range(num_variables):
            var_name = "value_0_{}".format(i)
            anchor.variables[var_name] = new_points[i][alpha]
    
    return

def save_new_model(model, save_old_model=True):
    """
    At the end of a string method calculation, generate a new model file. The
    old model file(s) will be renamed.
    
    
    Parameters
    ----------
    model : Model()
        The unfilled Seekr2 Model object.
        
    """
    model_path = os.path.join(model.anchor_rootdir, "model.xml")
    if os.path.exists(model_path) and save_old_model:
        # This is expected, because this old model was loaded
        hidr_model_glob = os.path.join(model.anchor_rootdir, STRING_MODEL_GLOB)
        num_globs = len(glob.glob(hidr_model_glob))
        new_pre_hidr_model_filename = STRING_MODEL_BASE.format(num_globs)
        new_pre_hidr_model_path = os.path.join(model.anchor_rootdir, 
                                               new_pre_hidr_model_filename)
        print("Renaming model.xml to {}".format(new_pre_hidr_model_filename))
        copyfile(model_path, new_pre_hidr_model_path)
        
    print("Saving new model.xml")
    old_rootdir = model.anchor_rootdir
    model.anchor_rootdir = "."
    base.save_model(model, model_path)
    model.anchor_rootdir = old_rootdir
    return

def redefine_anchor_neighbors(model, voronoi_cv):
    neighbor_anchor_indices = common_cv.find_voronoi_anchor_neighbors(
        model.anchors)
    milestone_index = 0
    for alpha, anchor in enumerate(model.anchors):
        anchor.milestones = []
        neighbor_anchor_alphas = neighbor_anchor_indices[alpha]
        assert len(neighbor_anchor_alphas) < 31, \
            "Only up to 31 neighbors allowed by the SEEKR2 plugin."
        for neighbor_anchor_alpha in neighbor_anchor_alphas:
            neighbor_anchor = model.anchors[neighbor_anchor_alpha]
            
            milestone_index \
                = common_cv.make_mmvt_milestone_between_two_voronoi_anchors(
                    anchor, alpha, neighbor_anchor, neighbor_anchor_alpha,
                    milestone_index, voronoi_cv.index, 
                    len(voronoi_cv.child_cvs))
    return
                
def define_new_starting_states(model, voronoi_cv, anchor_cv_values):
    for alpha, anchor in enumerate(model.anchors):
        traj = check.load_structure_with_mdtraj(model, anchor, mode="mmvt_traj")
        num_frames = traj.n_frames
        anchor_points = []
        for i in range(len(anchor.variables)):
            var_name = "value_0_{}".format(i)
            anchor_point = anchor.variables[var_name]
            anchor_points.append(anchor_point)
        
        anchor_points_array = np.array(anchor_points)
        dists = []
        for frame_id, anchor_cv_value in enumerate(anchor_cv_values[alpha]):
            # find dist between anchor_point and cv_value
            anchor_cv_value_array = np.array(anchor_cv_value)
            dist = np.linalg.norm(anchor_points_array - anchor_cv_value_array)
            dists.append(dist)
            
        sort_index = np.argsort(np.array(dists))        
        starting_frame = None
        for index in sort_index:
            anchor_cv_value = anchor_cv_values[alpha][index]
            in_boundary = True
            for milestone in anchor.milestones:
                if not voronoi_cv.check_value_within_boundary(anchor_cv_value,
                                                          milestone.variables):
                    in_boundary = False
            if in_boundary:
                starting_frame = index
                break
        
        if starting_frame is None:
            raise Exception(
                "Suitable next state not found in anchor {}".format(alpha))
        
        new_positions = traj.xyz[starting_frame, :,:]
        if model.using_toy():
            anchor.starting_positions = np.array([new_positions])
        else:
            positions_filename = os.path.join(
                model.anchor_rootdir, anchor.directory, 
                anchor.building_directory, "string_output.pdb")
            traj[starting_frame].save_pdb(
                positions_filename, force_overwrite=True)
    
    return

def ftsm(model, cuda_device_index, iterations, points_per_iter, steps_per_iter, 
         stationary_states, convergence_factor, smoothing_factor):
    assert isinstance(model.collective_variables[0], mmvt_base.MMVT_Voronoi_CV)
    assert len(model.collective_variables) == 1
    assert steps_per_iter % points_per_iter == 0, \
        "points_per_iter must be a multiple of steps_per_iter"
    voronoi_cv = model.collective_variables[0]
    plot_dir = os.path.join(model.anchor_rootdir, PLOTS_DIRECTORY_NAME)
    states = defaultdict(lambda: None)
    anchor_cv_values = defaultdict(list)
    
    frame_interval = steps_per_iter // points_per_iter
    model.calculation_settings.energy_reporter_interval = frame_interval
    model.calculation_settings.trajectory_reporter_interval = frame_interval
    model.calculation_settings.restart_checkpoint_interval = frame_interval
    
    for iteration in range(iterations):
        for alpha, anchor in enumerate(model.anchors):
            run.run(model, str(alpha), save_state_file=False,
                    load_state_file=states[alpha], 
                    force_overwrite=True, 
                    min_total_simulation_length=steps_per_iter, 
                    cuda_device_index=cuda_device_index)
            anchor_cv_values[alpha] = get_cv_values(model, anchor, voronoi_cv)
            
        plot_muller_potential(model, plot_dir, iteration, anchor_cv_values)
        move_anchors(model, anchor_cv_values, convergence_factor, 
                     smoothing_factor)
        redefine_anchor_neighbors(model, voronoi_cv)
        define_new_starting_states(model, voronoi_cv, anchor_cv_values)
        
    
    save_new_model(model)
    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of model XML file for a SEEKR2 calculation. "\
        "One or more starting structures must be present in one or more of "\
        "the anchors.")
    argparser.add_argument(
        "-c", "--cuda_device_index", dest="cuda_device_index", default=None,
        help="modify which cuda_device_index to run the simulation on. For "\
        "example, the number 0 or 1 would suffice. To run on multiple GPU "\
        "indices, simply enter comma separated indices. Example: '0,1'. If a "\
        "value is not supplied, the value in the MODEL_FILE will be used by "\
        "default.", type=str)
    argparser.add_argument(
        "-I", "--iterations", dest="iterations", default=100,
        type=int, help="The number of iterations to take, per anchor")
    argparser.add_argument(
        "-P", "--points_per_iter", dest="points_per_iter", default=100,
        type=int, help="The number of timesteps to take per iteration. "\
        "Default: 100")
    argparser.add_argument(
        "-S", "--steps_per_iter", dest="steps_per_iter", default=10000,
        type=int, help="The number of timesteps to take per iteration. "\
        "Default: 10000")
    argparser.add_argument(
        "-s", "--stationary_states", dest="stationary_states", default="",
        type=str, help="A comma-separated list of anchor indices that "
        "will not be moved through the course of the simulations.")
    argparser.add_argument(
        "-C", "--convergence_factor", dest="convergence_factor", default=0.2,
        type=float, help="The aggressiveness of convergence. This value "\
        "should be between 0 and 1. A value too high, and the string method "\
        "might become numerically unstable. A value too low, and convergence "\
        "will take a very long time. Default: 0.1")
    argparser.add_argument(
        "-K", "--smoothing_factor", dest="smoothing_factor", default=0.0,
        type=float, help="The degree to smoothen the curve describing the "\
        "string going through each anchor. Default: 0.0")
    
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    cuda_device_index = args["cuda_device_index"]
    iterations = args["iterations"]
    steps_per_iter = args["steps_per_iter"]
    points_per_iter = args["points_per_iter"]
    stationary_states = args["stationary_states"]
    convergence_factor = args["convergence_factor"]
    smoothing_factor = args["smoothing_factor"]
    
    model = base.load_model(model_file)
    ftsm(model, cuda_device_index, iterations, points_per_iter, steps_per_iter, 
         stationary_states, convergence_factor, smoothing_factor)