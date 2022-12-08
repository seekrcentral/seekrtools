"""
Base functions for string methods.

"""

import os
import sys
import glob
from  shutil import copyfile
import re


import numpy as np
from scipy.interpolate import splprep, splev

import seekr2.modules.common_base as base
import seekr2.modules.check as check
import seekr2.modules.common_cv as common_cv

import seekrtools.hidr.hidr_base as hidr_base

STRING_MODEL_GLOB = "model_pre_string_*.xml"
STRING_MODEL_BASE = "model_pre_string_{}.xml"

STRING_LOG_FILENAME = "string_{}.log"
STRING_LOG_GLOB = "string*.log"
STRING_OUTPUT = "string_output.pdb"

def log_string_results(model, iteration, anchor_cv_values, log_filename):
    #log_filename = os.path.join(model.anchor_rootdir, STRING_LOG_FILENAME)
    if iteration == 0:
        log_file = open(log_filename, "w")
        log_file.write("#anchor_id\tcv_value\tsampled_points\n")
    else:
        log_file = open(log_filename, "a")
    log_file.write("iteration: {}\n".format(iteration))
    
    for alpha, anchor in enumerate(model.anchors):
        
        log_file.write("anchor: {}\t[".format(alpha))
        for i in range(len(anchor.variables)):
            if i == 0:
                sep1 = ""
            else:
                sep1 = ", "
            var_name = "value_0_{}".format(i)
            value_i = anchor.variables[var_name]
            #values.append(value_i)
            log_file.write("{}{:.3f}".format(sep1, value_i))
        
        if alpha not in anchor_cv_values:
            log_file.write("]\n")
            continue
        
        log_file.write("]\t[")
        #log_file.write(",{}".format(values))
        for i, point in enumerate(anchor_cv_values[alpha]):
            if i == 0:
                sep1 = ""
            else:
                sep1 = "\t"
            log_file.write("{}[".format(sep1))
            for j, val in enumerate(point):
                if j == 0:
                    sep2 = ""
                else:
                    sep2 = ", "
                log_file.write("{}{:.3f}".format(sep2, val))
            log_file.write("]")
        
        log_file.write("]\n")
    
    log_file.close()
    
    return

def get_cv_values(model, anchor, voronoi_cv, mode="mmvt_traj"):
    values = []
    traj = check.load_structure_with_mdtraj(model, anchor, mode=mode)
    num_frames = traj.n_frames
    curdir = os.getcwd()
    os.chdir(model.anchor_rootdir)
    for i in range(num_frames):
        values.append(voronoi_cv.get_mdtraj_cv_value(traj, i))
    os.chdir(curdir)
    return values

def interpolate_points(model, anchor_cv_values, convergence_factor, smoothing_factor):
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
        if alpha in anchor_cv_values:
            this_anchor_values = np.array(anchor_cv_values[alpha])
            avg_values = []
            for i in range(num_variables):
                avg_value = np.mean(this_anchor_values[:,i])
                avg_values.append(avg_value)
        else:
            avg_values = np.array(old_anchor_values[alpha])
        
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
    
    return np.array(new_points).T

def save_new_model(model, save_old_model=True, overwrite_log=False):
    """
    At the end of a string method calculation, generate a new model file. The
    old model file(s) will be renamed.
    
    
    Parameters
    ----------
    model : Model()
        The unfilled Seekr2 Model object.
        
    """
    model_path = os.path.join(model.anchor_rootdir, "model.xml")
    string_model_glob = os.path.join(
        model.anchor_rootdir, STRING_MODEL_GLOB)
    num_globs = len(glob.glob(string_model_glob))
    log_filename = os.path.join(model.anchor_rootdir, STRING_LOG_FILENAME.format(num_globs))
    if save_old_model:
        # This is expected, because this old model was loaded
        if os.path.exists(model_path):
            new_pre_string_model_filename = STRING_MODEL_BASE.format(num_globs)
            new_pre_string_model_path = os.path.join(model.anchor_rootdir, 
                                                     new_pre_string_model_filename)
            print("Renaming model.xml to {}".format(new_pre_string_model_filename))
            copyfile(model_path, new_pre_string_model_path)
    
    if overwrite_log:
        # Then see if an old log file exists below the correct number, and 
        # if so, save it. But delete all numbered log files above the correct 
        # number
        log_glob = os.path.join(model.anchor_rootdir, STRING_LOG_GLOB)
        log_files = glob.glob(log_glob)
        for existing_log_file in log_files:
            existing_log_basename = os.path.basename(existing_log_file)
            log_file_index = int(re.findall(r"\d+", existing_log_basename)[0])
            if log_file_index >= num_globs:
                print("deleting log file:", existing_log_file)
                os.remove(existing_log_file)
        
    print("Saving new model.xml")
    old_rootdir = model.anchor_rootdir
    model.anchor_rootdir = "."
    base.save_model(model, model_path)
    model.anchor_rootdir = old_rootdir
    return log_filename

def define_new_starting_states(model, voronoi_cv, anchor_cv_values, 
                               ideal_points, stationary_alphas):
    
    # If using a centroid to define the new state, not an average.
    num_variables = len(model.anchors[0].variables)
    for alpha, anchor in enumerate(model.anchors):
        ideal_point = ideal_points[alpha]
        if anchor.bulkstate or alpha in stationary_alphas:
            # TODO: assign anchor location here??
            continue
        
        best_dist = 1e99
        best_alpha2 = None
        best_index = None
        for alpha2, anchor2 in enumerate(model.anchors):
            if alpha2 not in anchor_cv_values:
                continue
            for index, anchor_cv_value in enumerate(anchor_cv_values[alpha2]):
                anchor_cv_value_array = np.array(anchor_cv_value)
                dist = np.linalg.norm(ideal_point - anchor_cv_value_array)
                if dist < best_dist:
                    best_dist = dist
                    best_alpha2 = alpha2
                    best_index = index
        
        best_anchor = model.anchors[best_alpha2]
        for i in range(num_variables):
            var_name = "value_0_{}".format(i)
            anchor.variables[var_name] \
                = anchor_cv_values[best_alpha2][best_index][i]
        
        traj = check.load_structure_with_mdtraj(
            model, best_anchor, mode="mmvt_traj")
        if model.using_toy():
            new_positions = traj.xyz[best_index, :,:]
            anchor.starting_positions = np.array([new_positions])
        else:
            positions_filename = os.path.join(
                model.anchor_rootdir, anchor.directory, 
                anchor.building_directory, STRING_OUTPUT)
            traj[best_index].save_pdb(
                positions_filename, force_overwrite=True)
            hidr_base.change_anchor_pdb_filename(anchor, STRING_OUTPUT)
            box_vectors = base.get_box_vectors_from_pdb(positions_filename)
            hidr_base.change_anchor_box_vectors(anchor, box_vectors)
    
    return

def redefine_anchor_neighbors(model, voronoi_cv, skip_checks=False):
    neighbor_anchor_indices = common_cv.find_voronoi_anchor_neighbors(
        model.anchors)
    milestone_index = 0
    for alpha, anchor in enumerate(model.anchors):
        anchor.milestones = []
    
    for alpha, anchor in enumerate(model.anchors):
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
    
    model.num_milestones = milestone_index
    if not skip_checks:
        check.check_pre_simulation_all(model)
    return

def initialize_stationary_states(model, stationary_states):
    if stationary_states == "":
        stationary_alphas = []
    else:
        stationary_alphas = stationary_states.split(",")
        for stationary_alpha in stationary_alphas:
            stationary_alpha_int = int(stationary_alpha)
            assert stationary_alpha_int >= 0, "stationary_states must only "\
                "include integers greater than or equal to zero."
            assert stationary_alpha_int < len(model.anchors), "stationary_states "\
                "must only include integers less than the number of anchors."
    
    return list(map(int, stationary_alphas))

def signal_handler(signum, frame):
    print("Abort detected")
    sys.exit(0)
