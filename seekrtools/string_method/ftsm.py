"""
ftsm.py

Applies the Finite-Temperature String Method as described in the paper:

Vanden-Eijnden E, Venturoli M. Revisiting the finite temperature string 
method for the calculation of reaction tubes and free energies. J Chem 
Phys. 2009 May 21;130(19):194103. doi: 10.1063/1.3130083. 
PMID: 19466817.

The main difference with smst, is that this string method enforces the 
MMVT boundaries.

"""
# Enables global fork() followed by execve()
#from multiprocessing import set_start_method
#set_start_method("spawn")

# Instead, this method will spawn using fork() and execve() for
# just our own pools.

import argparse
from collections import defaultdict
from collections.abc import Iterable
import multiprocessing
import ast
import time

import numpy as np
import seekr2.modules.common_base as base
import seekr2.modules.mmvt_cvs.mmvt_voronoi_cv as mmvt_voronoi_cv
import seekr2.run as run
import seekr2.modules.check as check

import seekrtools.hidr.hidr_base as hidr_base
import seekrtools.string_method.base as string_base

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

def run_anchor_in_parallel(process_task):
    model, alpha, state, steps_per_iter, cuda_device_index = process_task
    run.run(model, str(alpha), save_state_file=False,
            load_state_file=state, 
            force_overwrite=True, 
            min_total_simulation_length=steps_per_iter, 
            cuda_device_index=cuda_device_index)
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


def ftsm(model, cuda_device_args=None, iterations=100, points_per_iter=100, 
         steps_per_iter=10000, stationary_states="", convergence_factor=0.2, 
         smoothing_factor=0.0, swarm_size=1):
    #signal.signal(signal.SIGHUP, signal_handler)
    assert isinstance(model.collective_variables[0], mmvt_voronoi_cv.MMVT_Voronoi_CV)
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
    stationary_alphas = string_base.initialize_stationary_states(model, stationary_states)
    log_filename = string_base.save_new_model(model, save_old_model=True, overwrite_log=True)
    
    starting_anchor_cv_values = defaultdict(list)
    for alpha, anchor in enumerate(model.anchors):
        if anchor.bulkstate or alpha in stationary_alphas:
            continue
        starting_anchor_cv_values[alpha] = string_base.get_cv_values(
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
            
        for alpha, anchor in enumerate(model.anchors):
            if alpha in stationary_alphas or anchor.bulkstate:
                continue
            anchor_cv_values[alpha] = string_base.get_cv_values(model, anchor, voronoi_cv)
        string_base.log_string_results(model, iteration, anchor_cv_values, log_filename)
        ideal_points = string_base.interpolate_points(
            model, anchor_cv_values, convergence_factor, smoothing_factor)
        string_base.define_new_starting_states(model, voronoi_cv, anchor_cv_values, 
                                   ideal_points, stationary_alphas)
        
        string_base.redefine_anchor_neighbors(model, voronoi_cv, skip_checks=True)
        string_base.save_new_model(model, save_old_model=False, overwrite_log=False)
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
    