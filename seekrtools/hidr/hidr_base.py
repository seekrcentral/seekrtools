"""
Base classes, objects, functions, and constants for the HIDR program.
"""

import os
import glob
from shutil import copyfile
import tempfile

import numpy as np
import mdtraj
import seekr2.modules.common_base as base
import seekr2.modules.mmvt_sim_openmm as mmvt_sim_openmm
import seekr2.modules.check as check
try:
    import openmm.unit as unit
except ModuleNotFoundError:
    import simtk.unit as unit
    
try:
    import openmm.app as openmm_app
except ImportError:
    import simtk.openmm.app as openmm_app

HIDR_MODEL_GLOB = "model_pre_hidr_*.xml"
HIDR_MODEL_BASE = "model_pre_hidr_{}.xml"

EQUILIBRATED_NAME = "hidr_equilibrated.pdb"
EQUILIBRATED_TRAJ_NAME = "hidr_traj_equilibrated.dcd"
SMD_NAME = "hidr_smd_at_{}.pdb"
RAMD_NAME = "hidr_ramd_at_{}_{}.pdb"
RAMD_TRAJ_NAME = "hidr_traj_ramd.dcd"
METADYN_NAME = "hidr_metadyn_at_{}_{}.pdb"
METADYN_TRAJ_NAME = "hidr_traj_metadyn.dcd"
SETTLED_FINAL_STRUCT_NAME = "hidr_settled_at_{}.pdb"
SETTLED_TRAJ_NAME = "hidr_traj_settled_at_{}.dcd"

def find_anchors_with_starting_structure(model):
    """
    Search the model for anchors which have a defined starting 
    structure.
    
    Parameters
    ----------
    model : Model()
        A Seekr2 Model object with insufficiently filled anchor
        starting structures.
        
    Returns
    -------
    anchors_with_starting_structures : str
        A list of integers representing anchor indices where a
        starting structure was found.
    """
    anchors_with_starting_structures = []
    for i, anchor in enumerate(model.anchors):
        assert i == anchor.index
        if anchor.__class__.__name__ in ["MMVT_toy_anchor"]:
            if anchor.starting_positions is not None \
                    and len(anchor.starting_positions) > 0:
                anchors_with_starting_structures.append(anchor.index)
        
        
        elif anchor.amber_params is not None:
            if anchor.amber_params.pdb_coordinates_filename:
                anchors_with_starting_structures.append(anchor.index)
        
        elif anchor.forcefield_params is not None:
            if anchor.charmm_params.pdb_coordinates_filename:
                anchors_with_starting_structures.append(anchor.index)
        
        elif anchor.charmm_params is not None:
            if anchor.charmm_params.pdb_coordinates_filename:
                anchors_with_starting_structures.append(anchor.index)
        
    
    assert len(anchors_with_starting_structures) > 0, "No anchors with "\
        "starting structures were found in this model."
    return anchors_with_starting_structures

def find_destinations(model, destination_str, anchors_with_starting_structures):
    """
    Given the destination anchor instructions, search the model to
    find all possible destination anchor indices.
    
    Parameters
    ----------
    model : Model()
        The unfilled Seekr2 Model object.
        
    destination_str : str
        The string representing a given anchor index, or the word
        "any", which would indicate that all destinations should be
        searched.
        
    anchors_with_starting_structures : str
        A list of integers representing anchor indices where a
        starting structure was found.
        
    Returns
    -------
    destination_list : list
        A list of integers representing anchor indices that HIDR
        will generate starting structures for.
    
    """
    destination_list = []
    complete_anchor_list = []
    if destination_str == "any":
        for i, anchor in enumerate(model.anchors):
            if anchor.bulkstate:
                continue
            if i not in anchors_with_starting_structures:
                destination_list.append(i)
            complete_anchor_list.append(i)
        
    else:
        destination_list = [int(destination_str)]
        complete_anchor_list = [int(destination_str)]
        
    return destination_list, complete_anchor_list
    
def check_destinations(model, anchors_with_starting_structures, 
                       destination_list, force_overwrite=False):
    """
    Check to make sure that at least one destination anchor is
    adjacent to one anchor which has a starting structure. If
    there is one, return the indices of the anchors with the
    structures we can start from.
    
    Parameters
    ----------
    model : Model()
        The unfilled Seekr2 Model object.
        
    anchors_with_starting_structures : str
        A list of integers representing anchor indices where a
        starting structure was found.
        
    destination_list : list
        A list of integers representing anchor indices that HIDR
        will generate starting structures for.
        
    Returns
    -------
    relevant_starting_anchor_indices : list
        A list of integers representing anchor indices that HIDR
        will generate starting structures for.
    """
    if not len(destination_list) > 0:
        print("No destinations found. Do all destination anchors already "\
              "have starting structures?")
        
    relevant_starting_anchor_indices = []
    for starting_anchor_index in anchors_with_starting_structures:
        anchor_relevant = False
        starting_anchor = model.anchors[starting_anchor_index]
        for destination_index in destination_list:
            for milestone in starting_anchor.milestones:
                if destination_index == milestone.neighbor_anchor_index:
                    # found a way to this destination
                    anchor_relevant = True
                    break
            
            if anchor_relevant:
                break
            
        if anchor_relevant:
            relevant_starting_anchor_indices.append(starting_anchor_index)
    
    if not len(relevant_starting_anchor_indices) > 0:
        print("There is not at least one destination anchor that is "\
              "immediately adjacent to an anchor with a starting structure. "\
              "HIDR will not run.")
    
    if force_overwrite:
        return relevant_starting_anchor_indices
    else:
        relevant_starting_anchor_indices_no_eq = []
        for starting_anchor_index in relevant_starting_anchor_indices:
            anchor = model.anchors[starting_anchor_index]
            directory = os.path.join(
                model.anchor_rootdir, anchor.directory, 
                anchor.building_directory)
            equilibrated_path = os.path.join(directory, EQUILIBRATED_NAME)
            if not os.path.exists(equilibrated_path):
                relevant_starting_anchor_indices_no_eq.append(
                    starting_anchor_index)
            else:
                print("Anchor {} has already ".format(starting_anchor_index)\
                      +"been equilibrated. Skipping equilibration.")
                  
        return relevant_starting_anchor_indices_no_eq

def make_var_string(anchor):
    """
    
    """
    var_list = []
    for variable_key in anchor.variables:
        if variable_key in anchor.variables:
            value = anchor.variables[variable_key]
            
        var_list.append("{:.3f}".format(value))
        
    var_string = "_".join(var_list)
    return var_string

def make_settling_names(model, anchor_index):
    """
    
    """
    var_string = make_var_string(
        model.anchors[anchor_index])
    settled_final_filename = SETTLED_FINAL_STRUCT_NAME.format(var_string)
    settled_traj_filename = SETTLED_TRAJ_NAME.format(var_string)
    return settled_final_filename, settled_traj_filename

def check_settling_anchors(model, complete_anchor_list, force_overwrite=False):
    """
    
    """
    settling_anchor_list = []
    for i, anchor in enumerate(model.anchors):
        assert i == anchor.index
        if i not in complete_anchor_list: continue
        settled_final_filename, settled_traj_filename \
            = make_settling_names(model, i)
        output_final_pdb_file = os.path.join(
            model.anchor_rootdir, anchor.directory, anchor.building_directory,
            settled_final_filename)
        output_traj_pdb_file = os.path.join(
            model.anchor_rootdir, anchor.directory, anchor.building_directory,
            settled_traj_filename)
        
        
        if anchor.__class__.__name__ in ["MMVT_toy_anchor"]:
            if anchor.starting_positions is not None:
                continue
        
        elif anchor.amber_params is not None:
            if anchor.amber_params.pdb_coordinates_filename:
                pdb_coords_filename \
                    = anchor.amber_params.pdb_coordinates_filename
        
        elif anchor.forcefield_params is not None:
            if anchor.forcefield_params.pdb_coordinates_filename:
                pdb_coords_filename \
                    = anchor.forcefield_params.pdb_coordinates_filename
        
        elif anchor.charmm_params is not None:
            if anchor.charmm_params.pdb_coordinates_filename:
                pdb_coords_filename \
                    = anchor.charmm_params.pdb_coordinates_filename
        
        if force_overwrite:
            settling_anchor_list.append(i)
        else:
            if not os.path.exists(output_final_pdb_file) \
                    or not os.path.exists(output_traj_pdb_file) \
                    or pdb_coords_filename != settled_traj_filename:
                settling_anchor_list.append(i)
    
    return settling_anchor_list

def save_new_model(model, save_old_model=True):
    """
    At the end of a HIDR calculation, generate a new model file. The
    old model file(s) will be renamed with a numerical index.
    
    
    Parameters
    ----------
    model : Model()
        The unfilled Seekr2 Model object.
        
    """
    if model.openmm_settings.cuda_platform_settings is not None:
        cuda_device_index = model.openmm_settings.cuda_platform_settings\
            .cuda_device_index
        model.openmm_settings.cuda_platform_settings.cuda_device_index = "0"
        
    model_path = os.path.join(model.anchor_rootdir, "model.xml")
    if os.path.exists(model_path) and save_old_model:
        # This is expected, because this old model was loaded
        hidr_model_glob = os.path.join(model.anchor_rootdir, HIDR_MODEL_GLOB)
        num_globs = len(glob.glob(hidr_model_glob))
        new_pre_hidr_model_filename = HIDR_MODEL_BASE.format(num_globs)
        new_pre_hidr_model_path = os.path.join(model.anchor_rootdir, 
                                               new_pre_hidr_model_filename)
        print("Renaming model.xml to {}".format(new_pre_hidr_model_filename))
        copyfile(model_path, new_pre_hidr_model_path)
        
    print("Saving new model.xml")
    old_rootdir = model.anchor_rootdir
    model.anchor_rootdir = "."
    base.save_model(model, model_path)
    model.anchor_rootdir = old_rootdir
    if model.openmm_settings.cuda_platform_settings is not None:
        model.openmm_settings.cuda_platform_settings.cuda_device_index\
            = cuda_device_index
    return

def change_anchor_pdb_filename(anchor, new_pdb_filename):
    """
    Reassign an anchor's starting structure filename.
    
    Parameters
    ----------
    anchor : Anchor()
        For a given Anchor object, assign a new PDB file name into the
        proper field, depending on whether the input parameters are
        Amber, Charmm, etc.
    """
    if anchor.amber_params is not None:
        anchor.amber_params.pdb_coordinates_filename = new_pdb_filename
    
    if anchor.forcefield_params is not None:
        anchor.forcefield_params.pdb_coordinates_filename = new_pdb_filename
        
    if anchor.charmm_params is not None:
        anchor.charmm_params.pdb_coordinates_filename = new_pdb_filename
        
    return

def get_anchor_pdb_filename(anchor):
    return base.get_anchor_pdb_filename(anchor)
        
def change_anchor_box_vectors(anchor, new_box_vectors):
    """
    Reassign an anchor's starting box vectors.
    
    Parameters
    ----------
    anchor : Anchor()
        For a given Anchor object, assign new box vectors into the proper 
        field, depending on whether the input parameters are Amber, 
        Charmm, etc.
    """
    if new_box_vectors is None:
        box_vectors = new_box_vectors
    else:
        box_vectors = base.Box_vectors()
        box_vectors.from_quantity(new_box_vectors)
        
    if anchor.amber_params is not None:
        anchor.amber_params.box_vectors = box_vectors
    
    if anchor.forcefield_params is not None:
        anchor.forcefield_params.box_vectors = box_vectors
        
    if anchor.charmm_params is not None:
        anchor.charmm_params.box_vectors = box_vectors
        
    return

def assign_pdb_file_to_model(model, pdb_file, skip_checks=False, dry_run=False):
    """
    Given a pdb file, assign it to an anchor where it fits between the 
    milestones of the anchor.
    
    Parameters:
    -----------
    model : Model()
        The (partially?) unfilled Seekr2 Model object.
    pdb_file : str
        A path to a PDB file to assign to this model.
        
    """
    for anchor in model.anchors:
        if anchor.bulkstate:
            break
        
        if not skip_checks:
            try:
                traj = mdtraj.load(pdb_file)
                assert traj.n_frames == 1, "More than one frame detected in PDB "\
                    f"file {pdb_file}. PDB files assigned using HIDR ought to "\
                    "have a single frame, or else problems may occur in later "\
                    "calculation stages. If you wish to proceed anyways, you may "\
                    "skip checks with the '-s' or '--skip_checks' argument."
            
            except ValueError:
                print("Warning: unable to test input PDB for multiple frames.")
        
        tmp_path = tempfile.NamedTemporaryFile()
        output_file = tmp_path.name
        my_sim_openmm = mmvt_sim_openmm.create_sim_openmm(
            model, anchor, output_file, use_only_reference=True)
        context = my_sim_openmm.simulation.context
        positions_obj = openmm_app.PDBFile(pdb_file)
        positions = positions_obj.getPositions()
        state = context.getState(getPositions = True, enforcePeriodicBox = True)
        context_positions = state.getPositions()
        assert len(context_positions) == len(positions), \
            "Mismatch between atom numbers in anchor parameter file and "\
            "provided pdb file {}. Incorrect pdb file?".format(pdb_file)
        context.setPositions(positions)
        
        between_milestones = True
        for milestone in anchor.milestones:
            cv = model.collective_variables[milestone.cv_index]
            curdir = os.getcwd()
            os.chdir(model.anchor_rootdir)
            #result = cv.check_mdtraj_within_boundary(traj, milestone.variables, 
            #                                         verbose=False)
            result = cv.check_openmm_context_within_boundary(
                context, milestone.variables, verbose=False)
            os.chdir(curdir)
            if not result:
                between_milestones = False
                break
        
        if between_milestones:
            # Then this is the correct anchor
            anchor_building_dir = os.path.join(
                model.anchor_rootdir, anchor.directory, 
                anchor.building_directory)
            pdb_base = os.path.basename(pdb_file)
            new_pdb_filename = os.path.join(anchor_building_dir, 
                                            pdb_base)
            if not dry_run:
                copyfile(os.path.expanduser(pdb_file), new_pdb_filename)
            print("Assigning pdb file {} to anchor {}".format(
                pdb_file, anchor.index))
            change_anchor_pdb_filename(anchor, pdb_base)
            box_vectors = base.get_box_vectors_from_pdb(pdb_file)
            change_anchor_box_vectors(anchor, box_vectors)
            anchor.endstate = True
            
            break
        
    return anchor.index

def assign_toy_coords_to_model(model, toy_coordinates):
    """
    
    """
    assert model.using_toy(), \
        "assign_toy_coords_to_model may only be used for a toy system"
    for anchor in model.anchors:
        between_milestones = True
        positions = unit.Quantity(toy_coordinates, unit=unit.nanometers)
        for milestone in anchor.milestones:
            cv = model.collective_variables[milestone.cv_index]
            result = cv.check_positions_within_boundary(
                positions, milestone.variables)
            if not result:
                between_milestones = False
                break
            
        if between_milestones:
            anchor.starting_positions = np.array(
                [positions.value_in_unit(unit.nanometers)])
            print("Assigning toy coord {} to anchor {}".format(
                toy_coordinates, anchor.index))
            return anchor.index
    raise Exception("No starting anchor found for coordinates.")
