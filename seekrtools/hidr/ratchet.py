"""
ratchet.py

Use the ratchet method to sample the multidimensional Voronoi
cells of a MMVT calculation in SEEKR.
"""

import os
import argparse
import glob
import math
from collections import defaultdict
import tempfile
import ast

import seekr2.modules.common_base as base
import seekr2.modules.mmvt_sim_openmm as mmvt_sim_openmm
import seekr2.run as run
import seekr2.modules.runner_openmm as runner_openmm

import seekrtools.hidr.hidr_base as hidr_base

MAX_COUNTER = 10000000

def uniform_select_from_list(mylist, count):
    if len(mylist) < count:
        return mylist
    interval = int(math.floor(len(mylist) / count))
    saving_indices = range(0, len(mylist), interval)[:count]
    saving_state_files = []
    for saving_index in saving_indices:
        saving_filename = mylist[saving_index]
        saving_state_files.append(saving_filename)
    assert len(saving_state_files) == count, \
        "len(saving_state_files): {}, count: {}".format(
            len(saving_state_files), count)
    return saving_state_files

def get_state_glob_all_boundaries(model, anchor):
    state_glob = os.path.join(
        model.anchor_rootdir, anchor.directory, 
        anchor.production_directory, 
        runner_openmm.SAVE_STATE_DIRECTORY, 
        runner_openmm.SAVE_STATE_PREFIX+"*")
    return state_glob
    
def get_state_glob_certain_boundary(model, anchor, alias_index):
    state_glob = os.path.join(
        model.anchor_rootdir, anchor.directory, 
        anchor.production_directory, 
        runner_openmm.SAVE_STATE_DIRECTORY, 
        runner_openmm.SAVE_STATE_PREFIX+"_*_{}".format(alias_index))
    return state_glob

def get_min_state_count(model, anchor):
    state_glob = get_state_glob_all_boundaries(model, anchor)
    min_state_count = runner_openmm.min_number_of_states_per_boundary(
        state_glob, anchor)
    return min_state_count
    
def get_state_files_in_anchor_by_adj(model, anchor, alias_index):
    state_glob = get_state_glob_certain_boundary(
        model, anchor, alias_index)
    state_files_for_adj_anchor = glob.glob(state_glob)
    return state_files_for_adj_anchor

def delete_extra_state_files(model, anchor, max_states_per_boundary, 
                             states_per_anchor):
    for milestone in anchor.milestones:
        alias_index = milestone.alias_index
        boundary_state_files = get_state_files_in_anchor_by_adj(
            model, anchor, alias_index)
        saving_state_files = uniform_select_from_list(boundary_state_files, 
                                                  max_states_per_boundary)
        for deleting_state_file in boundary_state_files:
            if deleting_state_file in saving_state_files:
                continue
            #print("removing file:", deleting_state_file)
            assert len(saving_state_files) >= states_per_anchor, \
                "Too many files being deleted."
            os.remove(deleting_state_file)
        
    return

def extract_states_not_in_anchor(model, anchor_index, state_files):
    
    # negative tolerance because we want to make sure that the system is
    #  in the expected Voronoi cell.
    TOL = -0.001
    anchor = model.anchors[anchor_index]
    new_state_files_list = []
    for state_file in state_files:
        dummy_file = tempfile.NamedTemporaryFile()
        sim_openmm_obj = mmvt_sim_openmm.create_sim_openmm(
            model, anchor, dummy_file.name,
            load_state_file=state_file)
        context = sim_openmm_obj.simulation.context
        in_boundary = True
        for cv in model.collective_variables:
            for milestone in anchor.milestones:
                if milestone.cv_index == cv.index:
                    in_milestone = cv.check_openmm_context_within_boundary(
                        context, milestone.variables, tolerance=TOL)
                    if not in_milestone:
                        in_boundary = False
                    #    error_msg = "State file {} not ".format(state_file) \
                    #        +"in boundary of anchor {}, ".format(anchor.index) \
                    #        +"milestone alias_id {}".format(milestone.alias_id)
                    #    raise Exception(error_msg)
                    
        if in_boundary:
            new_state_files_list.append(state_file)
    
    return new_state_files_list

def make_states_dict_one_anchor(model, anchor):
    states_dict = {}
    for milestone in anchor.milestones:
        state_files_for_adj_anchor = get_state_files_in_anchor_by_adj(
            model, anchor, milestone.alias_index)
        state_files_for_adj_anchor = extract_states_not_in_anchor(
            model, milestone.neighbor_anchor_index, state_files_for_adj_anchor)
        states_dict[milestone.neighbor_anchor_index] = state_files_for_adj_anchor
    
    return states_dict

def make_states_dict_all_anchors(model):
    states_dict = defaultdict(list)
    for alpha, anchor in enumerate(model.anchors):
        this_anchor_states_dict = make_states_dict_one_anchor(model, anchor)
        for anchor2 in this_anchor_states_dict:
            states_dict[anchor2] += this_anchor_states_dict[anchor2]
    
    return states_dict

def obtain_required_states(model):
    """
    Required states include the bound anchor(s), as well as anchors adjacent to
    bulk, and named anchors.
    """
    required_states = set()
    for anchor in model.anchors:
        if anchor.endstate:
            required_states.add(anchor.index)
        
    return list(required_states)

def ratchet(model, cuda_device_index, pdb_files, states_per_anchor, 
            max_states_per_boundary, steps_per_iter, 
            minimum_timesteps_per_anchor=0, toy_coordinates=None, 
            force_overwrite=False, finish_on_endstates=False,
            frames_per_anchor=100):
    """
    Use the ratchet method to move the system across the CV space.
    """
    complete_anchors = []
    incomplete_anchors = []
    required_states = obtain_required_states(model)
    if model.get_bulk_index() is None:
        reached_bulk_state = True
    else:
        reached_bulk_state = False
    
    if frames_per_anchor > 0 and minimum_timesteps_per_anchor > 0:
        frame_interval = minimum_timesteps_per_anchor // frames_per_anchor
        model.calculation_settings.energy_reporter_interval = frame_interval
        model.calculation_settings.trajectory_reporter_interval = frame_interval
        model.calculation_settings.restart_checkpoint_interval = frame_interval
    
    # Initialize a defaultdict with force_overwrite as default value
    local_force_overwrite = defaultdict(lambda:force_overwrite)
    
    states_dict = {}
    assert states_per_anchor <= max_states_per_boundary, \
        "states_per_anchor cannot be more than max_states_per_boundary."
    if model.calculation_settings.restart_checkpoint_interval > steps_per_iter:
        model.calculation_settings.restart_checkpoint_interval = steps_per_iter
    
    first_anchors = []
    if model.using_toy():
        print("toy_coordinates:", toy_coordinates)
        for toy_coordinate in toy_coordinates:
            assert len(toy_coordinate) == 3
        first_anchor_index = hidr_base.assign_toy_coords_to_model(
            model, toy_coordinates)
        incomplete_anchors.append(first_anchor_index)
        first_anchors.append(first_anchor_index)
    else:
        for pdb_file in pdb_files:
            first_anchor_index = hidr_base.assign_pdb_file_to_model(
                model, pdb_file)
            incomplete_anchors.append(first_anchor_index)
            first_anchors.append(first_anchor_index)
    
    counter = 0
    # TODO: need a function to identify states existing before the simulation
    #  and to populate states_dict (or create a different dictionary of any
    #  existing states on this boundary)
    starting_states_dict = make_states_dict_all_anchors(model)
    anchor_counter = {}
    for alpha, anchor in enumerate(model.anchors):
        if force_overwrite:
            anchor_counter[alpha] = 0
            continue
        
        if alpha in starting_states_dict:
            starting_states_list = starting_states_dict[alpha]
        else:
            anchor_counter[alpha] = 0
            continue
                
        if len(starting_states_list) == 0:
            swarm_frame = None
            load_state_file = None
        elif states_per_anchor == 1:
            swarm_frame = None
            load_state_file = starting_states_list[0]
        else:
            swarm_frame = 0
            load_state_file = starting_states_list[0]
        currentStep = run.get_current_step_openmm(
            model, anchor, load_state_file_list=load_state_file, 
            swarm_frame=swarm_frame)
        anchor_counter[alpha] = currentStep // steps_per_iter
    
    anchors_to_run_sorted = sorted(required_states)
    
    ratchet_finished = False
    while not ratchet_finished:
        next_incomplete_anchors = set()
        for incomplete_anchor in incomplete_anchors:
            # TODO: find a way to enable some of the first_anchor states to run
            anchor = model.anchors[incomplete_anchor]
            if (incomplete_anchor in states_dict) and \
                    (not incomplete_anchor in first_anchors):
                all_state_for_anchor = states_dict[incomplete_anchor]
                states_to_run = uniform_select_from_list(all_state_for_anchor, 
                                                         states_per_anchor)
            else:
                states_to_run = None
            
            print("running anchor:", incomplete_anchor)
            total_simulation_length = steps_per_iter\
                *(anchor_counter[incomplete_anchor]+1)
            print("total_simulation_length:", total_simulation_length)
            print("steps_per_iter:", steps_per_iter)
            print("anchor_counter[incomplete_anchor]:", anchor_counter[incomplete_anchor])
            print("states_to_run:", states_to_run)
            run.run(model, str(incomplete_anchor), save_state_file=True,
                    load_state_file=states_to_run,
                    force_overwrite=local_force_overwrite[incomplete_anchor], 
                    min_total_simulation_length=total_simulation_length, 
                    cuda_device_index=cuda_device_index)
            
            anchor_counter[incomplete_anchor] += 1
            if incomplete_anchor in anchors_to_run_sorted:
                anchors_to_run_sorted.remove(incomplete_anchor)
            
            # Make sure we don't force overwrite after the first time
            local_force_overwrite[incomplete_anchor] = False
            min_state_count = get_min_state_count(model, anchor)
            this_anchor_completed = False
            if minimum_timesteps_per_anchor > 0:
                if total_simulation_length >= minimum_timesteps_per_anchor:
                    this_anchor_completed = True
            else:
                if min_state_count >= states_per_anchor:
                    this_anchor_completed = True
            
            if this_anchor_completed:
                assert incomplete_anchor not in complete_anchors
                complete_anchors.append(incomplete_anchor)
            elif (incomplete_anchor not in complete_anchors):
                next_incomplete_anchors.add(incomplete_anchor)
            delete_extra_state_files(model, anchor, max_states_per_boundary, 
                states_per_anchor)
            this_anchor_states_dict = make_states_dict_one_anchor(model, anchor)
            #states_dict.update(this_anchor_states_dict)
            for anchor2 in this_anchor_states_dict:
                states_dict[anchor2] = this_anchor_states_dict[anchor2]
            
        for anchor2_index in states_dict:
            if len(states_dict[anchor2_index]) >= states_per_anchor \
                    and (anchor2_index not in complete_anchors) \
                    and (anchor2_index not in incomplete_anchors) \
                    and not model.anchors[anchor2_index].bulkstate:
                next_incomplete_anchors.add(anchor2_index)
            if model.anchors[anchor2_index].bulkstate:
                reached_bulk_state = True
                    
        incomplete_anchors = list(next_incomplete_anchors)
        
        if finish_on_endstates:
            if (len(anchors_to_run_sorted) == 0) and reached_bulk_state:
                ratchet_finished = True
        
        if len(incomplete_anchors) == 0:
            ratchet_finished = True
        
        counter += 1
        if counter == MAX_COUNTER:
            print("Max counter exceeded!")
            break
    
    print("next_incomplete_anchors:", next_incomplete_anchors)
    print("complete_anchors:", complete_anchors)
    print("incomplete_anchors:", incomplete_anchors)
    print("both:", complete_anchors+incomplete_anchors)
    print("counter:", counter)
        
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
        "-p", "--pdb_files", dest="pdb_files", default=[], nargs="*", type=str,
        metavar="FILE1 FILE2 ...", help="One or more PDB files which will be "\
        "placed into the correct anchors. NOTE: the parameter/topology files "\
        "must already be assigned into an anchor for this to work.")
    argparser.add_argument(
        "-t", "--toy_coordinates", dest="toy_coordinates", default="[]", 
        metavar="[[x1, y1, z1], [x2, y2, z2], ...]", help="Enter the X, Y, Z "\
        "coordinates for toy system's starting position. It will be "\
        "automatically assigned to the correct anchor.")
    argparser.add_argument(
        "-f", "--force_overwrite", dest="force_overwrite", default=False,
        help="Toggle whether to overwrite existing simulation output files "\
        "within any anchor that might have existed in an old model that would "\
        "be overwritten by generating this new model. If not toggled, this "\
        "program will skip the stage instead of performing any such "\
        "overwrite.", action="store_true")
    argparser.add_argument(
        "-s", "--states_per_anchor", dest="states_per_anchor", default=1,
        type=int, help="The number of states that must be attained for "\
        "each anchor before ratchet can proceed to the next anchor.")
    argparser.add_argument(
        "-m", "--max_states_per_boundary", dest="max_states_per_boundary", 
        default=0, type=int,
        help="The maximum number of states to save per boundary to avoid "\
        "Running out of hard drive space. The default value of 0 indicates "\
        "That it is the same value as states_per_anchor.")
    argparser.add_argument(
        "-T", "--minimum_timesteps_per_anchor", 
        dest="minimum_timesteps_per_anchor", default=0,
        type=int, help="The number of steps to take, per anchor, before "\
        "an anchor is considered completed. A value of 0 indicates that "\
        "there is no minimum timesteps per anchor. Instead, an anchor is "\
        "assumed completed if all the boundaries have been hit.")
    argparser.add_argument(
        "-S", "--steps_per_iter", dest="steps_per_iter", default=10000,
        type=int, help="The number of steps to take per iteration. Default: "\
        "10000")
    argparser.add_argument(
        "-e", "--finish_on_endstates", dest="finish_on_endstates", 
        default=False, help="Toggle whether to end the ratchet simulation "\
        "once every end state is reached.", action="store_true")
    argparser.add_argument(
        "-F", "--frames_per_anchor", dest="frames_per_anchor", default=100,
        type=int, help="The number of frames per anchor. Default: 100")
    
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    cuda_device_index = args["cuda_device_index"]
    pdb_files = args["pdb_files"]
    toy_coordinates = ast.literal_eval(args["toy_coordinates"])
    force_overwrite = args["force_overwrite"]
    states_per_anchor = args["states_per_anchor"]
    max_states_per_boundary = args["max_states_per_boundary"]
    minimum_timesteps_per_anchor = args["minimum_timesteps_per_anchor"]
    steps_per_iter = args["steps_per_iter"]
    finish_on_endstates = args["finish_on_endstates"]
    frames_per_anchor = args["frames_per_anchor"]
    if max_states_per_boundary == 0:
        max_states_per_boundary = states_per_anchor
    
    model = base.load_model(model_file)
    ratchet(model, cuda_device_index, pdb_files, states_per_anchor, 
            max_states_per_boundary, steps_per_iter, 
            minimum_timesteps_per_anchor, toy_coordinates, force_overwrite, 
            finish_on_endstates, frames_per_anchor)
