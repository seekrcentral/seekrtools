"""
Base classes, objects, functions, and constants for the HIDR program.
"""

import os
import glob
from shutil import copyfile

import mdtraj
import seekr2.modules.common_base as base
import seekr2.modules.check as check

HIDR_MODEL_GLOB = "model_pre_hidr_*.xml"
HIDR_MODEL_BASE = "model_pre_hidr_{}.xml"

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
        if anchor.amber_params is not None:
            if anchor.amber_params.pdb_coordinates_filename:
                anchors_with_starting_structures.append(anchor.index)
        
        elif anchor.forcefield_params is not None:
            anchors_with_starting_structures.append(anchor.index)
        
        elif anchor.charmm_params is not None:
            raise Exception("Charmm systems not yet implemented")
        
    
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
    if destination_str == "any":
        for i, anchor in enumerate(model.anchors):
            if anchor.bulkstate:
                continue
            if i not in anchors_with_starting_structures:
                destination_list.append(i)
        
    else:
        destination_list = [int(destination_str)]
        
    return destination_list
    
def check_destinations(model, anchors_with_starting_structures, 
                       destination_list):
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
    
    assert len(relevant_starting_anchor_indices) > 0, "There must be at least "\
        "one destination anchor that is immediately adjacent to an anchor "\
        "with a starting structure."
    return relevant_starting_anchor_indices
        
def save_new_model(model):
    """
    At the end of a HIDR calculation, generate a new model file. The
    old model file(s) will be renamed with a numerical index.
    
    
    Parameters
    ----------
    model : Model()
        The unfilled Seekr2 Model object.
        
    """
    model_path = os.path.join(model.anchor_rootdir, "model.xml")
    if os.path.exists(model_path):
        # This is expected, because this old model was loaded
        hidr_model_glob = os.path.join(model.anchor_rootdir, HIDR_MODEL_GLOB)
        num_globs = len(glob.glob(hidr_model_glob))
        new_pre_hidr_model_filename = HIDR_MODEL_BASE.format(num_globs)
        new_pre_hidr_model_path = os.path.join(model.anchor_rootdir, 
                                               new_pre_hidr_model_filename)
        print("Renaming model.xml to {}".format(new_pre_hidr_model_filename))
        copyfile(model_path, new_pre_hidr_model_path)
        
    print("Saving new model.xml")
    base.save_model(model, model_path)
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

def assign_pdb_file_to_model(model, pdb_file):
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
        traj = mdtraj.load(pdb_file)
        between_milestones = True
        for milestone in anchor.milestones:
            cv = model.collective_variables[milestone.cv_index]
            result = cv.check_mdtraj_within_boundary(traj, milestone.variables, 
                                                     verbose=False)
            if not result:
                between_milestones = False
                break
        
        if between_milestones:
            # Then this is the correct anchor
            print("Assigning pdb file {} to anchor {}".format(
                pdb_file, anchor.index))
            change_anchor_pdb_filename(anchor, pdb_file)
            break
    return