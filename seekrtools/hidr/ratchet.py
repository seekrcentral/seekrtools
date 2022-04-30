"""
ratchet.py

Use the ratchet method to sample the multidimensional Voronoi
cells of a MMVT calculation in SEEKR.
"""

import os
import argparse
import glob
import random

import seekr2.modules.common_base as base
import seekr2.modules.mmvt_sim_openmm as mmvt_sim_openmm
import seekr2.run as run

import seekrtools.hidr.hidr_base as hidr_base

RUN_STATES_PER_ANCHOR = 1

def get_states_dict(model, anchor):
    states_dict = {}
    states_directory = os.path.join(model.anchor_rootdir, anchor.directory, 
                                    anchor.production_directory, "states")
    for milestone in anchor.milestones:
        state_glob = "*_{}".format(milestone.alias_index)
        state_path = os.path.join(states_directory, state_glob)
        state_file_list = glob.glob(state_path)
        if len(state_file_list) == 0:
            continue
        states_dict[milestone.neighbor_anchor_index] = state_file_list
        
    return states_dict

def sort_anchors_by_number_of_states(states_dict, anchors_run):
    sorted_anchors = sorted(
        states_dict.keys(), key = lambda ele: len(states_dict[ele]))
    for anchor in anchors_run:
        if anchor in sorted_anchors:
            sorted_anchors.remove(anchor)
    return sorted_anchors

def ratchet(model, pdb_files, toy_coordinates=None, force_overwrite=False):
    """
    
    """
    MAX_ITER = 200
    anchors_run = []
    first_anchors = []
    states_dict = {}
    
    if model.using_toy():
        assert len(toy_coordinates) == 3
        first_anchor_index = hidr_base.assign_toy_coords_to_model(
            model, toy_coordinates)
        first_anchors.append(first_anchor_index)
    else:
        for pdb_file in pdb_files:
            first_anchor_index = hidr_base.assign_pdb_file_to_model(model, pdb_file)
            first_anchors.append(first_anchor_index)
    
    anchors_to_run_sorted = []
    for first_anchor_index in first_anchors:
        print("first_anchor_index:", first_anchor_index)
        run.run(model, str(first_anchor_index), save_state_file=True, 
                force_overwrite=force_overwrite)
        anchors_run.append(first_anchor_index)
        first_anchor = model.anchors[first_anchor_index]
        this_states_dict = get_states_dict(model, first_anchor)
        for key, value in this_states_dict.items():
            if key in states_dict:
                states_dict[key].extend(value)
            else:
                states_dict[key] = value

    anchors_to_run_sorted = sort_anchors_by_number_of_states(
                states_dict, anchors_run)
        
        
    counter = 0
    while len(anchors_to_run_sorted) > 0:
        print("anchors_to_run_sorted:", anchors_to_run_sorted)
        #for anchor_index in anchors_to_run_sorted:
        anchor_index = anchors_to_run_sorted.pop()
        anchor = model.anchors[anchor_index]
        state_files = random.sample(states_dict[anchor_index], 
                                    RUN_STATES_PER_ANCHOR)
        for state_file in state_files:
            in_anchor = mmvt_sim_openmm.check_if_state_in_anchor(
                model, anchor, state_file)
            if not in_anchor:
                # handle the situation if the state is not in the anchor
                print("State not in anchor")
                exit()
            
            print("loading state file:", state_file)
            run.run(model, str(anchor_index), save_state_file=True, 
                    load_state_file=state_file, force_overwrite=force_overwrite)
            anchors_run.append(anchor_index)
            states_dict_this = get_states_dict(model, anchor)
            
            for key, value in states_dict_this.items():
                if key in states_dict:
                    states_dict[key].extend(value)
                else:
                    states_dict[key] = value
                
        anchors_to_run_sorted = sort_anchors_by_number_of_states(
            states_dict, anchors_run)
        counter += 1
        if counter > MAX_ITER:
            print("maximum iterations exceeded.")
            break
    
    print("anchors_run:", anchors_run)
    
    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of model XML file for a SEEKR2 calculation. "\
        "One or more starting structures must be present in one or more of "\
        "the anchors.")
    argparser.add_argument(
        "-p", "--pdb_files", dest="pdb_files", default=[], nargs="*", type=str,
        metavar="FILE1 FILE2 ...", help="One or more PDB files which will be "\
        "placed into the correct anchors. NOTE: the parameter/topology files "\
        "must already be assigned into an anchor for this to work.")
    argparser.add_argument(
        "-t", "--toy_coordinates", dest="toy_coordinates", default=[], 
        nargs=3, type=float, metavar="x y z", help="Enter the X, Y, Z "\
        "coordinates for toy system's starting position. It will be "\
        "automatically assigned to the correct anchor.")
    argparser.add_argument(
        "-f", "--force_overwrite", dest="force_overwrite", default=False,
        help="Toggle whether to overwrite existing simulation output files "\
        "within any anchor that might have existed in an old model that would "\
        "be overwritten by generating this new model. If not toggled, this "\
        "program will skip the stage instead of performing any such "\
        "overwrite.", action="store_true")
    
    args = argparser.parse_args() # parse the args into a dictionary
    args = vars(args)
    model_file = args["model_file"]
    pdb_files = args["pdb_files"]
    toy_coordinates = args["toy_coordinates"]
    force_overwrite = args["force_overwrite"]
    
    model = base.load_model(model_file)
    model.calculation_settings.energy_reporter_interval = None
    ratchet(model, pdb_files, toy_coordinates, force_overwrite)
