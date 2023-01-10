"""
smst.py

Applies the String Method with Swarms of Trajectories as described in the 
paper:

Finding Transition Pathways Using the String Method with Swarms of Trajectories
Albert C. Pan, Deniz Sezer, and Benoît Roux
The Journal of Physical Chemistry B 2008 112 (11), 3432-3440
DOI: 10.1021/jp0777059

The main difference with ftsm, is that this string method does not enforce
the MMVT boundaries, but allows the system to explore space freely.
"""

import os
import argparse
import ast
import time
from collections import defaultdict
from collections.abc import Iterable
import multiprocessing

import numpy as np
import parmed
import openmm
import openmm.app as openmm_app
from openmm import unit
import seekr2.modules.common_base as base
import seekr2.modules.mmvt_cvs.mmvt_voronoi_cv as mmvt_voronoi_cv
import seekr2.modules.common_sim_openmm as common_sim_openmm
import seekr2.modules.check as check

import seekrtools.hidr.hidr_base as hidr_base
import seekrtools.hidr.hidr_simulation as hidr_simulation
import seekrtools.string_method.base as string_base

STRING_NAME = "string_method_at_{}.pdb"
DELTA = 0.0001
CUTOFF_TO_USE_CPU_MINIMIZER = 1e12 * unit.kilojoules / unit.moles
MAX_CPU_MINIMIZER_ITERATIONS = 10

def make_cv_id_list(model, anchor):
    cv_id_list = []
    value_list = []
    for variable_key in anchor.variables:
        var_cv = int(variable_key.split("_")[1])
        cv = model.collective_variables[var_cv]
        if isinstance(cv, mmvt_voronoi_cv.MMVT_Voronoi_CV):
            var_child_cv = int(variable_key.split("_")[2])
            cv_id_list.append(var_child_cv)
        else:
            cv_id_list.append(var_cv)
        
        value = anchor.variables[variable_key]
        value_list.append(value)
        
    return cv_id_list, value_list

def make_single_simulation(model, anchor, restraint_force_constant, 
                           skip_minimization, equilibration_steps, 
                           cuda_device_index="0"):
    cv_id_list, value_list = make_cv_id_list(model, anchor)
    sim_openmm = hidr_simulation.HIDR_sim_openmm()
    
    system, topology, positions, box_vectors, num_frames \
        = common_sim_openmm.create_openmm_system(sim_openmm, model, anchor)
    
    assert positions is not None,\
        "Anchor {} is missing starting positions".format(anchor.index)
    
    sim_openmm.system = system
    time_step = hidr_simulation.add_integrator(sim_openmm, model)
    common_sim_openmm.add_barostat(system, model)
    # TODO: modify platform to run on different GPUs?
    if cuda_device_index is not None:
        model.openmm_settings.cuda_platform_settings.cuda_device_index \
            = cuda_device_index
    common_sim_openmm.add_platform(sim_openmm, model)
    
    if equilibration_steps > 0 and restraint_force_constant > 0.0:
        forces = hidr_simulation.add_forces(
            sim_openmm, model, anchor, restraint_force_constant, 
            cv_id_list, value_list)
        #forces = hidr_simulation.add_forces(
        #    sim_openmm, model, anchor, 0.0, 
        #    cv_id_list, value_list)
    
    hidr_simulation.add_simulation(
        sim_openmm, model, topology, positions, box_vectors, 
        skip_minimization=True)
    return sim_openmm, positions, box_vectors

def make_simulation_set(model, stationary_alphas, restraint_force_constant, 
                        skip_minimization):
    sim_openmm_list = []
    total_num_alphas = len(model.anchors)-len(stationary_alphas)
    print("making a total of", total_num_alphas, "systems.")
    for alpha, anchor in enumerate(model.anchors):
        if anchor.bulkstate or alpha in stationary_alphas:
            print("skipping simulation object:", alpha, "out of:", len(model.anchors))
            sim_openmm_list.append(None)
            continue
        print("making simulation object:", alpha, "out of:", len(model.anchors))
        sim_openmm, positions = make_single_simulation(
            model, anchor, restraint_force_constant, skip_minimization, 
            equilibration_steps)
        sim_openmm_list.append(sim_openmm)
        
    return sim_openmm_list

def make_process_instructions(
        model, swarm_size, steps_per_iter, cuda_device_args, stationary_alphas,
        use_centroid, skip_minimization, skip_equilibration, 
        restraint_force_constant):
    process_instructions = []
    if not isinstance(cuda_device_args, Iterable):
        cuda_device_args = [cuda_device_args]
        
    num_processes = len(cuda_device_args)
    process_task_set = []
    counter = 0
    for alpha, anchor in enumerate(model.anchors):
        if alpha in stationary_alphas or anchor.bulkstate:
            continue
        
        idx = counter % num_processes
        if idx == 0 and len(process_task_set) > 0:
            process_instructions.append(process_task_set)
            process_task_set = []
        if cuda_device_args[idx] is None:
            cuda_device_index = None
        else:
            cuda_device_index = str(cuda_device_args[idx])
            
        process_task = [model, alpha, swarm_size, steps_per_iter, 
                        cuda_device_index, use_centroid, skip_minimization, 
                        skip_equilibration, restraint_force_constant]
        process_task_set.append(process_task)
        counter += 1
        
    process_instructions.append(process_task_set)
    process_task_set = []
    return process_instructions

def save_avg_pdb_structure(model, anchor, sim_openmm, positions, box_vectors):
    
    if anchor.__class__.__name__ in ["MMVT_toy_anchor"]:
        anchor.starting_positions = np.array(positions)
        return None
    else:
        var_string = hidr_base.make_var_string(anchor)
        string_output_pdb_name = STRING_NAME.format(var_string)
        hidr_base.change_anchor_box_vectors(anchor, box_vectors)
        hidr_base.change_anchor_pdb_filename(anchor, string_output_pdb_name)
        output_pdb_file = os.path.join(
            model.anchor_rootdir, anchor.directory, anchor.building_directory,
            string_output_pdb_name)
        parm = parmed.openmm.load_topology(sim_openmm.simulation.topology, 
                                           sim_openmm.system)
        parm.positions = positions
        parm.box_vectors = box_vectors
        print("saving file:", output_pdb_file)
        parm.save(output_pdb_file, overwrite=True)
        return string_output_pdb_name

def save_positions_dcd(model, anchor, positions_list, box_vector_list):
    
    traj = check.load_structure_with_mdtraj(model, anchor, mode="pdb")
    traj.xyz = positions_list
    traj.unitcell_vectors = np.array(box_vector_list)
    swarm_dcd_filename = os.path.join(
        model.anchor_rootdir, anchor.directory, anchor.production_directory,
        "mmvt1.dcd")
    traj.save(swarm_dcd_filename)
    return

def set_anchor_cv_values(anchor, cv_values, as_positions=False):
    num_variables = len(anchor.variables)
    if as_positions:
        new_cv_values = []
        for i in range(num_variables):
            var_name = "value_0_{}".format(i)
            j = i // 3
            anchor.variables[var_name] = cv_values[j,i].value_in_unit(unit.nanometers)
            new_cv_values.append(anchor.variables[var_name])
        return new_cv_values
    else:
        
        for i in range(num_variables):
            var_name = "value_0_{}".format(i)
            anchor.variables[var_name] = cv_values[i]
        return cv_values
    
def get_anchor_cv_values(anchor):
    num_variables = len(anchor.variables)
    cv_values = []
    for i in range(num_variables):
        var_name = "value_0_{}".format(i)
        cv_values.append(anchor.variables[var_name])
    return cv_values

def load_string_positions(model, alpha):
    anchor = model.anchors[alpha]
    if model.using_toy():
        positions = anchor.starting_positions[0] * unit.nanometers
        box_vectors = None
    else:
        #var_string = hidr_base.make_var_string(anchor)
        #string_output_pdb_name = STRING_NAME.format(var_string)
        string_output_pdb_name = hidr_base.get_anchor_pdb_filename(anchor)
        pdb_filename = os.path.join(
            model.anchor_rootdir, anchor.directory, anchor.building_directory,
            string_output_pdb_name)
        print("attempting to load:", pdb_filename)
        pdb_structure = parmed.load_file(pdb_filename)
        positions = pdb_structure.coordinates * unit.angstroms
        box_vectors_tuple = pdb_structure.box_vectors.value_in_unit(unit.nanometers)
        box_vectors = np.array(box_vectors_tuple) * unit.nanometers
    
    return positions, box_vectors

def interpolate_positions(model, anchor_cv_values, stationary_alphas):
    curdir = os.getcwd()
    os.chdir(model.anchor_rootdir)
    voronoi_cv = model.collective_variables[0]
    total_distance = 0.0
    assert len(model.anchors) >= 2
    positions_list = []
    box_vectors_list = []
    total_num_anchors_to_interpolate = 0
    start_index = None
    last_index = None
    maintain_continuity_flag = False
    
    for alpha, anchor in enumerate(model.anchors):
        if anchor.bulkstate:
            break
        
        positions, box_vectors = load_string_positions(model, alpha)
        positions_list.append(positions)
        box_vectors_list.append(box_vectors)
        
        if alpha in stationary_alphas:
            if start_index is not None:
                maintain_continuity_flag = True
            continue
        
        assert not maintain_continuity_flag, \
            "stationary states may only exist at the beginning or end of "\
            "the string."
        
        if start_index is None:
            start_index =  alpha
        last_index = alpha
        total_num_anchors_to_interpolate += 1
        
    start_span_index = max(0, start_index-1)
    last_span_index = min(last_index+1, len(model.anchors)-1)
    total_num_anchors_to_span = last_span_index - start_span_index + 1
    
    for alpha in range(start_span_index, last_span_index):
        start_point = np.array(anchor_cv_values[alpha][0])
        end_point = np.array(anchor_cv_values[alpha+1][0])
        segment_distance = np.linalg.norm(end_point-start_point)
        total_distance += segment_distance
    
    even_spacing_distances = np.linspace(0, total_distance, total_num_anchors_to_span)
    assert total_num_anchors_to_interpolate > 0
    counter = 0
    even_spacing_counter = start_index - start_span_index
    current_ideal_index = start_index
    
    total_distance = 0.0
    
    ideal_cv_values = []
    all_done = False
    for alpha in range(start_index):
        ideal_cv_values.append(anchor_cv_values[alpha][0])
    
    sim_openmm, positions, box_vectors_DUMMY = make_single_simulation(
        model, model.anchors[last_index], restraint_force_constant=0.0, 
        skip_minimization=True, equilibration_steps=0, cuda_device_index=None)
    
    for alpha, anchor in enumerate(model.anchors):
        if alpha < start_span_index:
            continue
        
        if all_done:
            break
        
        if alpha == len(model.anchors) - 1:
            # skip the last one
            break
        
        start_point = np.array(anchor_cv_values[alpha][0])
        end_point = np.array(anchor_cv_values[alpha+1][0])
        start_box_vectors = box_vectors_list[alpha]
        start_positions = positions_list[alpha]
        start_box_vectors = box_vectors_list[alpha]
        if model.anchors[alpha+1].bulkstate:
            end_positions = start_positions
            end_box_vectors = start_box_vectors
        else:
            end_positions = positions_list[alpha+1]
            end_box_vectors = box_vectors_list[alpha+1]
        segment_distance = np.linalg.norm(end_point-start_point)
        current_ideal_total = even_spacing_distances[even_spacing_counter]
        current_ideal_distance = current_ideal_total - total_distance
        assert current_ideal_distance >= 0.0
        while (current_ideal_distance < segment_distance + DELTA) \
                and (counter <= total_num_anchors_to_interpolate):
            progress_between_points = current_ideal_distance / segment_distance
            interp_values = (1.0-progress_between_points) * start_point \
                                  + progress_between_points * end_point
            interp_positions = (1.0-progress_between_points) * start_positions \
                                  + progress_between_points * end_positions
            interp_positions = interp_positions #* unit.angstroms
            
            interp_cv_val = voronoi_cv.get_openmm_context_cv_value(
                None, interp_positions, sim_openmm.system)
            if start_box_vectors is not None and end_box_vectors is not None:
                interp_box_vecs = progress_between_points * start_box_vectors \
                           + (1.0-progress_between_points) * end_box_vectors
            else:
                interp_box_vecs = None
            current_ideal_anchor = model.anchors[current_ideal_index]
            
            if model.using_toy():
                #set_anchor_cv_values(current_ideal_anchor, interp_positions,
                #                     as_positions=True)
                set_anchor_cv_values(current_ideal_anchor, interp_values)
                #save_avg_pdb_structure(model, current_ideal_anchor, sim_openmm, 
                #                   [interp_positions], interp_box_vecs)
            else:
                #set_anchor_cv_values(current_ideal_anchor, interp_cv_val)
                set_anchor_cv_values(current_ideal_anchor, interp_values)
                #save_avg_pdb_structure(model, current_ideal_anchor, sim_openmm, 
                #                   interp_positions, interp_box_vecs)
            
            #ideal_cv_values.append(interp_cv_val)
            ideal_cv_values.append(interp_values)
            
            current_ideal_index += 1
            counter += 1
            even_spacing_counter += 1
            #if current_ideal_index >= len(model.anchors) - 1:
            if counter >= total_num_anchors_to_interpolate:
                all_done = True
                break
            current_ideal_total = even_spacing_distances[even_spacing_counter]
            current_ideal_distance = current_ideal_total - total_distance
        
        total_distance += segment_distance
    
    if last_index+1 < len(model.anchors):
        for alpha in range(last_index+1, len(model.anchors)):
            ideal_cv_values.append(anchor_cv_values[alpha][0])
    
    assert len(model.anchors) == len(ideal_cv_values)
    os.chdir(curdir)
    return ideal_cv_values

def assert_stationary_alphas_continuity(model, stationary_alphas):
    start_index = None
    maintain_continuity_flag = False
    for alpha, anchor in enumerate(model.anchors):
        if alpha in stationary_alphas or anchor.bulkstate:
            if start_index is not None:
                maintain_continuity_flag = True
            continue
        
        assert not maintain_continuity_flag, \
            "stationary states may only exist at the beginning or end of "\
            "the string."
        
        if start_index is None:
            start_index =  alpha

# TODO: remove?
def minimize_with_CPU(sim_openmm, model, box_vectors, positions, 
                      enforcePeriodicBox):
    # The GPU minimizer cannot handle the steric clashes caused by interpolation
    #  due to its fixed-point approach. The CPU platform uses the more-
    #  versatile floating-point approach - minimize with CPU.
    print("minimizing with CPU")
    sim_openmm_cpu = hidr_simulation.HIDR_sim_openmm()
    sim_openmm_cpu.system = sim_openmm.system
    time_step_cpu = hidr_simulation.add_integrator(sim_openmm_cpu, model)
    sim_openmm_cpu.platform = \
        openmm.Platform.getPlatformByName('Reference')
    sim_openmm_cpu.properties = {}
    hidr_simulation.add_simulation(
        sim_openmm_cpu, model, sim_openmm.simulation.topology, positions, 
        box_vectors, skip_minimization=True)
    sim_openmm_cpu.simulation.minimizeEnergy(
        maxIterations=MAX_CPU_MINIMIZER_ITERATIONS)
    state = sim_openmm_cpu.simulation.context.getState(
        getPositions=True, getVelocities=False, getEnergy=True,
        enforcePeriodicBox=enforcePeriodicBox)
    post_cpu_min_potential_energy = state.getPotentialEnergy()
    post_cpu_min_kinetic_energy = state.getKineticEnergy()
    new_positions = state.getPositions(asNumpy=True)
    return new_positions
    
def run_anchor_in_parallel(process_task):
    model, alpha, swarm_size, steps_per_iter, cuda_device_index, \
        use_centroid, skip_minimization, equilibration_steps, \
        restraint_force_constant = process_task
    
    curdir = os.getcwd()
    os.chdir(model.anchor_rootdir)
    
    voronoi_cv = model.collective_variables[0]
    anchor = model.anchors[alpha]
    
    if anchor.__class__.__name__ in ["MMVT_toy_anchor"]:
        enforcePeriodicBox = False
    else:
        enforcePeriodicBox = True
    
    start_time = time.time()
    sim_openmm, positions, box_vectors = make_single_simulation(
        model, anchor, restraint_force_constant, skip_minimization, 
        equilibration_steps, cuda_device_index)
    
    cv_before_min = voronoi_cv.get_openmm_context_cv_value(
                    context=None, positions=positions, 
                    system=sim_openmm.system)
    state = sim_openmm.simulation.context.getState(
        getEnergy=True,
        enforcePeriodicBox=enforcePeriodicBox)
    pre_min_potential_energy = state.getPotentialEnergy()
    pre_min_kinetic_energy = state.getKineticEnergy()
    
    if not skip_minimization:
        sim_openmm.simulation.minimizeEnergy()
        state = sim_openmm.simulation.context.getState(
            getPositions=True, getVelocities=False, getEnergy=True,
            enforcePeriodicBox=enforcePeriodicBox)
        post_min_positions = state.getPositions(asNumpy=True)
        positions_cv_val = voronoi_cv.get_openmm_context_cv_value(
                    context=None, positions=post_min_positions, 
                    system=sim_openmm.system)
        set_anchor_cv_values(anchor, positions_cv_val)
        post_gpu_min_potential_energy = state.getPotentialEnergy()
        post_gpu_min_kinetic_energy = state.getKineticEnergy()
    
    if equilibration_steps > 0:
        print("performing equilibration for image:", alpha)
        cv_id_list, value_list = make_cv_id_list(model, anchor)
        hidr_simulation.update_forces(
            sim_openmm, sim_openmm.forces, model, anchor, 
            restraint_force_constant, cv_list=cv_id_list, 
            window_values=value_list)
        
        # DEBUG:
        traj_filename = "/tmp/equil.dcd"
        debug_reporter = openmm_app.DCDReporter(traj_filename, 1, 
                enforcePeriodicBox=False)
        #sim_openmm.simulation.reporters.append(debug_reporter)
        
        sim_openmm.simulation.context.reinitialize(preserveState=True)
        sim_openmm.simulation.context.setVelocitiesToTemperature(
            model.openmm_settings.initial_temperature * unit.kelvin)
        sim_openmm.simulation.step(equilibration_steps)
        # Now turn off the forces for the swarms
        hidr_simulation.update_forces(
            sim_openmm, sim_openmm.forces, model, anchor, 
            0.0, cv_list=cv_id_list, window_values=value_list)
        sim_openmm.simulation.context.reinitialize(preserveState=True)
        #sim_openmm.simulation.reporters.pop()
    
    # Now run swarms
    state = sim_openmm.simulation.context.getState(
        getPositions=True, getVelocities=False, 
        enforcePeriodicBox=enforcePeriodicBox)
    reset_positions = state.getPositions(asNumpy=True)
    
    # DEBUG:
    post_equil_cv_val = voronoi_cv.get_openmm_context_cv_value(
        context=None, positions=reset_positions, 
        system=sim_openmm.system)
    
    reset_box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
    swarm_positions_list = []
    box_vectors_list = []
    print("creating a swarm of", swarm_size, "for image:", alpha)
    for swarm_index in range(swarm_size):
        if swarm_index > 0:
            sim_openmm.simulation.context.setPositions(reset_positions)
        sim_openmm.simulation.context.setVelocitiesToTemperature(
            model.openmm_settings.initial_temperature * unit.kelvin)
        sim_openmm.simulation.step(steps_per_iter)
        # extract/save swarm positions and reset
        state = sim_openmm.simulation.context.getState(
        getPositions=True, getVelocities=False, 
        enforcePeriodicBox=enforcePeriodicBox)
        swarm_positions = state.getPositions(asNumpy=True)
        swarm_positions_list.append(swarm_positions)
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        box_vectors_list.append(box_vectors)
        
    if use_centroid:
        save_positions_dcd(model, anchor, swarm_positions_list, 
                           box_vectors_list)
    
    avg_positions = np.mean(swarm_positions_list, axis=0) * unit.nanometers
    sim_openmm.simulation.context.setPositions(avg_positions)
    # Write the averaged PDB and assign the new CV values to the model
    if model.using_toy():
        avg_values = set_anchor_cv_values(anchor, avg_positions, as_positions=True)
        #save_avg_pdb_structure(model, anchor, sim_openmm, [avg_positions], None)
        pdb_filename = save_avg_pdb_structure(model, anchor, sim_openmm, [reset_positions], None)
        positions_vec = [reset_positions]
        #reset_vec = reset_positions
        #set_anchor_cv_values(anchor, reset_vec, as_positions=True)
    else:
        avg_values = voronoi_cv.get_openmm_context_cv_value(
            None, avg_positions, sim_openmm.system)
        avg_box_vectors = np.mean(box_vectors_list, axis=0) * unit.nanometers
        set_anchor_cv_values(anchor, avg_values)
        #save_avg_pdb_structure(model, anchor, sim_openmm, avg_positions, avg_box_vectors)
        pdb_filename = save_avg_pdb_structure(model, anchor, sim_openmm, reset_positions, reset_box_vectors)
        positions_vec = reset_box_vectors
        
    time_step = hidr_simulation.get_timestep(model)
    total_time = time.time() - start_time
    total_steps = equilibration_steps + swarm_size * steps_per_iter
    simulation_in_ns = total_steps * time_step.value_in_unit(
        unit.picoseconds) * 1e-3
    total_time_in_days = total_time / (86400.0)
    ns_per_day = simulation_in_ns / total_time_in_days
    print("Benchmark:", ns_per_day, "ns/day")
    os.chdir(curdir)
    return avg_values, pdb_filename, positions_vec

def smst(model, cuda_device_args=None, iterations=100, swarm_size=10, 
         steps_per_iter=100, stationary_states="", smoothing_factor=0.0, 
         use_centroid=False, skip_minimization=False, equilibration_steps=10000,
         restraint_force_constant=4184.0):
    assert isinstance(model.collective_variables[0], mmvt_voronoi_cv.MMVT_Voronoi_CV)
    assert len(model.collective_variables) == 1
    voronoi_cv = model.collective_variables[0]
    anchor_cv_values = {} #defaultdict(list)
    log_filename = string_base.save_new_model(model, save_old_model=True, overwrite_log=True)
    stationary_alphas = string_base.initialize_stationary_states(model, stationary_states)
    #simulation_set = make_simulation_set(model, stationary_alphas,
    #    restraint_force_constant, skip_minimization)
    
    assert_stationary_alphas_continuity(model, stationary_alphas)
    
    for iteration in range(iterations):
        print("Iteration:", iteration)
        process_instructions = make_process_instructions(
            model, swarm_size, steps_per_iter, cuda_device_args, 
            stationary_alphas, use_centroid, skip_minimization, 
            equilibration_steps, restraint_force_constant)
        for process_task_set in process_instructions:
            # loop through the serial list of parallel tasks
            num_processes = len(process_task_set)
            with multiprocessing.get_context("spawn").Pool(num_processes) as p:
                for i, result in enumerate(p.map(run_anchor_in_parallel, process_task_set)):
                    alpha = process_task_set[0][1]
                    cv_values = result[0]
                    pdb_filename = result[1]
                    positions_vec_or_box_vec = result[2]
                    set_anchor_cv_values(model.anchors[alpha], cv_values)
                    if model.using_toy:
                        model.anchors[alpha].starting_positions \
                            = np.array(positions_vec_or_box_vec)
                    else:
                        hidr_base.change_anchor_pdb_filename(
                            model.anchors[alpha], pdb_filename)
                        hidr_base.change_anchor_box_vectors(
                            model.anchors[alpha], positions_vec_or_box_vec)
                    
            # Serial run - to start with
            #run_anchor_in_parallel(process_task_set[0])
        
        for alpha, anchor in enumerate(model.anchors):
            #if alpha in stationary_alphas or anchor.bulkstate:
            #    continue
            anchor_cv_values[alpha] = [get_anchor_cv_values(anchor)]
                
        if use_centroid:
            raise Exception("Centroid mode not yet supported.")
        else:
            ideal_cv_values = interpolate_positions(
                model, anchor_cv_values, stationary_alphas)
            
            for alpha, anchor in enumerate(model.anchors):
                if alpha in stationary_alphas or anchor.bulkstate:
                    continue
                anchor_cv_values[alpha] = [ideal_cv_values[alpha]]
                    
        string_base.log_string_results(model, iteration, anchor_cv_values, 
                                       log_filename)
            
        string_base.redefine_anchor_neighbors(model, voronoi_cv, skip_checks=True)
        string_base.save_new_model(model, save_old_model=False, overwrite_log=False)
        #assert check.check_systems_within_Voronoi_cells(model)
        
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
        "-w", "--swarm_size", dest="swarm_size", default=32,
        type=int, help="The size of the swarm - that is, the number of "\
        "replicas of the system moving concurrently. Default: 32")
    argparser.add_argument(
        "-S", "--steps_per_iter", dest="steps_per_iter", default=5000,
        type=int, help="The number of timesteps to take per iteration. "\
        "Default: 5000")
    argparser.add_argument(
        "-s", "--stationary_states", dest="stationary_states", default="",
        type=str, help="A comma-separated list of anchor indices that "
        "will not be moved through the course of the simulations.")
    #argparser.add_argument(
    #    "-C", "--convergence_factor", dest="convergence_factor", default=0.2,
    #    type=float, help="The aggressiveness of convergence. This value "\
    #    "should be between 0 and 1. A value too high, and the string method "\
    #    "might become numerically unstable. A value too low, and convergence "\
    #    "will take a very long time. Default: 0.2")
    argparser.add_argument(
        "-K", "--smoothing_factor", dest="smoothing_factor", default=0.0,
        type=float, help="The degree to smoothen the curve describing the "\
        "string going through each anchor. Default: 0.0")
    #NOTE: Benoit Roux uses a smoothing constant of 0.1 !!!
    #argparser.add_argument(
    #    "-C", "--use_centroid", dest="use_centroid", default=False,
    #    help="Whether to assign the displacement to the centroid "\
    #    "of the sampled swarm. If left at False, then the average of the "\
    #    "swarm is used by default as the CV point, and the previous iter's "\
    #    "structure is used as a starting point.", action="store_true")
    argparser.add_argument(
        "-m", "--skip_minimization", dest="skip_minimization", default=False,
        help="Whether to skip minimization when the starting "\
        "structure is assigned. By default, minimization will be performed.",
        action="store_true")
    argparser.add_argument(
        "-e", "--equilibration_steps", dest="equilibration_steps", 
        default=10000, type=int,
        help="How many equilibration steps to run when the starting "\
        "structure is assigned, right before the swarm is spawned. "\
        "Default: 10000.")
    argparser.add_argument(
        "-k", "--restraint_force_constant", dest="restraint_force_constant",
        type=float, default=4184.0, 
        help="The force constant to use for restraints in units of "\
        "kilojoules per mole per nanometer**2. Default: 4184.")
    
    
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    cuda_device_args = ast.literal_eval(args["cuda_device_string"])
    iterations = args["iterations"]
    swarm_size = args["swarm_size"]
    steps_per_iter = args["steps_per_iter"]
    stationary_states = args["stationary_states"]
    smoothing_factor = args["smoothing_factor"]
    #use_centroid = args["use_centroid"]
    use_centroid = False
    skip_minimization = args["skip_minimization"]
    equilibration_steps = args["equilibration_steps"]
    restraint_force_constant = args["restraint_force_constant"]
    
    model = base.load_model(model_file)
    start_time = time.time()
    smst(model, cuda_device_args, iterations, swarm_size, steps_per_iter, 
         stationary_states,  smoothing_factor, use_centroid, skip_minimization,
         equilibration_steps, restraint_force_constant)
    print("Time elapsed:", time.time() - start_time)
