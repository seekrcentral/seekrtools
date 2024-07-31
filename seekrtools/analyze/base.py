"""
Base functions for seekrtools analyze functions.
"""

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import MDAnalysis as mda

import seekr2.modules.common_base as seekr2_base
import seekr2.modules.mmvt_cvs.mmvt_cv_base as mmvt_cv_base

def load_structure_with_mdanalysis(model, anchor, mode="starting"):
    """
    Given the simulation inputs, load an anchor's structure for one of
    the checks and return the MDAnalysis Universe object.
    
    Parameter "mode" can be "starting" or "trajectory".
    """
    assert mode in ["starting", "trajectory"], \
        "mode param must be either 'starting' or 'trajectory'"
    building_directory = os.path.join(
        model.anchor_rootdir, anchor.directory, anchor.building_directory)
    prod_directory = os.path.join(
        model.anchor_rootdir, anchor.directory, anchor.production_directory)
    assert model.get_type() == "mmvt", "Only MMVT supported at this time"
    
    if mode == "trajectory":
        mmvt_traj_basename = mmvt_cv_base.OPENMMVT_BASENAME+"*.dcd"
        mmvt_traj_glob = os.path.join(prod_directory, mmvt_traj_basename)
        mmvt_traj_filenames = seekr2_base.order_files_numerically(
            glob.glob(mmvt_traj_glob))
        if len(mmvt_traj_filenames) == 0:
            return None
        
        # search for and remove any empty trajectory files
        indices_to_pop = []
        for i, mmvt_traj_filename in enumerate(mmvt_traj_filenames):
            if os.path.getsize(mmvt_traj_filename) == 0:
                indices_to_pop.append(i)
        
        for i in indices_to_pop[::-1]:
            mmvt_traj_filenames.pop(i)
    
        
    if model.using_toy():
        pdb_filename = os.path.join(building_directory, "toy.pdb")
        if mode == "starting":
            u = mda.Universe(pdb_filename)
            
        else:
            if not len(mmvt_traj_filenames) > 0:
                warnings.warn("Empty mmvt trajectories were found in "\
                      "anchor: {}.".format(anchor.index))
                return None
                
            u = mda.Universe(pdb_filename, *mmvt_traj_filenames)
            
        return u
    
    elif anchor.amber_params is not None:
        if (anchor.amber_params.prmtop_filename is None) \
                or (anchor.amber_params.prmtop_filename == ""):
            return None
        else:
            prmtop_filename = os.path.join(
                building_directory, anchor.amber_params.prmtop_filename)
            
        if mode == "starting":
            if anchor.amber_params.pdb_coordinates_filename is not None \
                    and anchor.amber_params.pdb_coordinates_filename != "":
                pdb_filename = os.path.join(
                    building_directory, 
                    anchor.amber_params.pdb_coordinates_filename)
                u = mda.Universe(prmtop_filename, pdb_filename)
            
            else:
                return None
        
        else:
            if not len(mmvt_traj_filenames) > 0:
                warnings.warn("Empty mmvt trajectories were found in "\
                      "anchor: {}.".format(anchor.index))
                return None
                
            u = mda.Universe(prmtop_filename, *mmvt_traj_filenames)
            
        return u
        
    elif anchor.forcefield_params is not None:
        pdb_filename = os.path.join(building_directory, 
                               anchor.forcefield_params.pdb_filename)
        if mode == "starting":
            u = mda.Universe(pdb_filename)
        else:
            assert len(mmvt_traj_filenames) > 0, "Only empty mmvt " \
                "trajectories were found. You can force SEEKR to skip these "\
                "checks by using the --skip_checks (-s) argument"
            u = mda.Universe(pdb_filename, *mmvt_traj_filenames)
            
        return u
    
    elif anchor.charmm_params is not None:
        if anchor.charmm_params.psf_filename is not None:
            psf_filename = os.path.join(
                building_directory, anchor.charmm_params.psf_filename)
        else:
            return None
        
        if mode == "starting":
            if anchor.charmm_params.pdb_coordinates_filename is not None \
                    and anchor.charmm_params.pdb_coordinates_filename != "":
                pdb_filename = os.path.join(
                    building_directory, 
                    anchor.charmm_params.pdb_coordinates_filename)
                u = mda.Universe(psf_filename, pdb_filename)
            else:
                # anchor has no structure files
                return None
        
        else:
            if not len(mmvt_traj_filenames) > 0:
                warnings.warn("Empty mmvt trajectories were found in "\
                      "anchor: {}.".format(anchor.index))
                return None
                
            u = mda.Universe(psf_filename, *mmvt_traj_filenames)
            
        return u
        
    else:
        return None