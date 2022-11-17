"""
ftsm.py

Applies the Finite-Temperature String Method as described in the paper:

Vanden-Eijnden E, Venturoli M. Revisiting the finite temperature string 
method for the calculation of reaction tubes and free energies. J Chem 
Phys. 2009 May 21;130(19):194103. doi: 10.1063/1.3130083. 
PMID: 19466817.


"""
# Enables global fork() followed by execve()
#from multiprocessing import set_start_method
#set_start_method("spawn")

# Instead, this method will spawn using fork() and execve() for
# just our own pools.

import os
import sys
import glob
import argparse
from collections import defaultdict
from collections.abc import Iterable
from shutil import copyfile
import multiprocessing
import signal
import traceback
import ast
import time
import re

import numpy as np
from scipy.interpolate import splprep, splev

import seekr2.modules.common_base as base
import seekr2.modules.mmvt_base as mmvt_base
import seekr2.modules.common_cv as common_cv
import seekr2.modules.mmvt_sim_openmm as mmvt_sim_openmm
import seekr2.run as run
import seekr2.modules.runner_openmm as runner_openmm
import seekr2.modules.check as check

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
                
def define_new_starting_states(model, voronoi_cv, anchor_cv_values, 
                               ideal_points, stationary_alphas):
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

def create_swarm(model, alpha, swarm_size):
    anchor = model.anchors[alpha]
    if model.using_toy():
        starting_coords_shape = anchor.starting_positions.shape
        assert starting_coords_shape[0] == 1, \
            "Swarms may not already exist when using the string method."
            
    else:
        pass
        
    if swarm_size == 1:
        return
    else:
        if model.using_toy():
            new_starting_coords = np.zeros(
                (swarm_size, starting_coords_shape[1], 3))
            for i in range(swarm_size):
                new_starting_coords[i,:,:] = anchor.starting_positions[0,:,:]
            
            anchor.starting_positions = new_starting_coords
            
        else:
            pdb_file = hidr_base.get_anchor_pdb_filename(anchor)
            assert isinstance(pdb_file, str), \
                "Swarms may not already exist when using the string method."
            pdb_file_list = [pdb_file] * swarm_size
            hidr_base.change_anchor_pdb_filename(anchor, pdb_file_list)
        
    return

def make_process_instructions(model, steps_per_iter, cuda_device_args, states,
                              stationary_alphas):
    process_instructions = []
    if not isinstance(cuda_device_args, Iterable):
        cuda_device_args = [cuda_device_args]
    num_processes = len(cuda_device_args)
    process_task_set = []
    counter = 0
    for alpha, anchor in enumerate(model.anchors):
        if alpha in stationary_alphas or anchor.bulkstate:
            continue
        state = states[alpha]
        idx = counter % num_processes
        if idx == 0 and len(process_task_set) > 0:
            process_instructions.append(process_task_set)
            process_task_set = []
        if cuda_device_args[idx] is None:
            cuda_device_index = None
        else:
            cuda_device_index = str(cuda_device_args[idx])
            
        process_task = [model, alpha, state, steps_per_iter, cuda_device_index]
        process_task_set.append(process_task)
        counter += 1
        
    process_instructions.append(process_task_set)
    process_task_set = []
    return process_instructions

def signal_handler(signum, frame):
    print("Abort detected")
    sys.exit(0)

def run_anchor_in_parallel(process_task):
    model, alpha, state, steps_per_iter, cuda_device_index = process_task
    run.run(model, str(alpha), save_state_file=False,
            load_state_file=state, 
            force_overwrite=True, 
            min_total_simulation_length=steps_per_iter, 
            cuda_device_index=cuda_device_index)
    return

def ftsm(model, cuda_device_args=None, iterations=100, points_per_iter=100, 
         steps_per_iter=10000, stationary_states="", convergence_factor=0.2, 
         smoothing_factor=0.0, swarm_size=1):
    #signal.signal(signal.SIGHUP, signal_handler)
    assert isinstance(model.collective_variables[0], mmvt_base.MMVT_Voronoi_CV)
    assert len(model.collective_variables) == 1
    assert steps_per_iter % points_per_iter == 0, \
        "points_per_iter must be a multiple of steps_per_iter"
    voronoi_cv = model.collective_variables[0]
    
    states = defaultdict(lambda: None)
    anchor_cv_values = {} #defaultdict(list)
    frame_interval = steps_per_iter // points_per_iter
    model.calculation_settings.energy_reporter_interval = frame_interval
    model.calculation_settings.trajectory_reporter_interval = frame_interval
    model.calculation_settings.restart_checkpoint_interval = frame_interval
    stationary_alphas = initialize_stationary_states(model, stationary_states)
    log_filename = save_new_model(model, save_old_model=True, overwrite_log=True)
    
    starting_anchor_cv_values = defaultdict(list)
    for alpha, anchor in enumerate(model.anchors):
        if anchor.bulkstate or alpha in stationary_alphas:
            continue
        starting_anchor_cv_values[alpha] = get_cv_values(
            model, anchor, voronoi_cv, mode="pdb")
        
    for iteration in range(iterations):
        for alpha, anchor in enumerate(model.anchors):
            if alpha in stationary_alphas or anchor.bulkstate:
                continue
            create_swarm(model, alpha, swarm_size)
        
        process_instructions = make_process_instructions(
            model, steps_per_iter, cuda_device_args, states,
            stationary_alphas)
        for process_task_set in process_instructions:
            # loop through the serial list of parallel tasks
            num_processes = len(process_task_set)
            with multiprocessing.get_context("spawn").Pool(num_processes) as p:
                p.map(run_anchor_in_parallel, process_task_set)
            
            # Serial run
            #run_anchor_in_parallel(process_task_set[0])
            
        """
        for alpha, anchor in enumerate(model.anchors):
            if alpha in stationary_alphas or anchor.bulkstate:
                continue
            create_swarm(model, alpha, swarm_size)
            #run.run(model, str(alpha), save_state_file=False,
            #        load_state_file=states[alpha], 
            #        force_overwrite=True, 
            #        min_total_simulation_length=steps_per_iter, 
            #        cuda_device_index=cuda_device_index)
            anchor_cv_values[alpha] = get_cv_values(model, anchor, voronoi_cv)
        """
        for alpha, anchor in enumerate(model.anchors):
            if alpha in stationary_alphas or anchor.bulkstate:
                continue
            anchor_cv_values[alpha] = get_cv_values(model, anchor, voronoi_cv)
        log_string_results(model, iteration, anchor_cv_values, log_filename)
        ideal_points = interpolate_points(
            model, anchor_cv_values, convergence_factor, smoothing_factor)
        define_new_starting_states(model, voronoi_cv, anchor_cv_values, 
                                   ideal_points, stationary_alphas)
        
        redefine_anchor_neighbors(model, voronoi_cv, skip_checks=True)
        save_new_model(model, save_old_model=False, overwrite_log=False)
        assert check.check_systems_within_Voronoi_cells(model)
        
    
    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of model XML file for a SEEKR2 calculation. "\
        "All starting structures must be present in all anchors.")
    argparser.add_argument(
        "-c", "--cuda_device_string", dest="cuda_device_string", default="None",
        help="Specify a string defining how to handle parallel processing "\
        "for the anchors in the string method. One could use a simple int "\
        "or an entire list of ints. One may use default in the model file "\
        "by passing None, or multiple CPUs by passing a list of Nones. "\
        "Python syntax should be used.", type=str)
    argparser.add_argument(
        "-I", "--iterations", dest="iterations", default=100,
        type=int, help="The number of iterations to take, per anchor")
    argparser.add_argument(
        "-P", "--points_per_iter", dest="points_per_iter", default=100,
        type=int, help="The number of position points to save per iteration. "\
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
        "will take a very long time. Default: 0.2")
    argparser.add_argument(
        "-K", "--smoothing_factor", dest="smoothing_factor", default=0.0,
        type=float, help="The degree to smoothen the curve describing the "\
        "string going through each anchor. Default: 0.0")
    argparser.add_argument(
        "-w", "--swarm_size", dest="swarm_size", default=1,
        type=int, help="The size of the swarm - that is, the number of "\
        "replicas of the system moving concurrently. Default: 1")
    
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    cuda_device_args = ast.literal_eval(args["cuda_device_string"])
    iterations = args["iterations"]
    steps_per_iter = args["steps_per_iter"]
    points_per_iter = args["points_per_iter"]
    stationary_states = args["stationary_states"]
    convergence_factor = args["convergence_factor"]
    smoothing_factor = args["smoothing_factor"]
    swarm_size = args["swarm_size"]
    
    model = base.load_model(model_file)
    start_time = time.time()
    ftsm(model, cuda_device_args, iterations, points_per_iter, steps_per_iter, 
         stationary_states, convergence_factor, smoothing_factor, swarm_size)
    print("Time elapsed:", time.time() - start_time)
    