#!/usr/bin/env python
"""
HIDR - Holo Insertion by Directed Restraints

HIDR takes one or more starting structures, as well as a SEEKR Input Model
and, closer to further, pulls the system towards all the anchors within the 
SEEKR model until starting structures exist in all of them. This is 
accomplished using steered molecular dynamics (SMD) simulations.
"""
import os
import argparse
import tempfile
import glob
import ast
import shutil

try:
    import openmm.unit as unit
except ModuleNotFoundError:
    import simtk.unit as unit
import seekr2.modules.common_base as base
import seekr2.modules.common_prepare as common_prepare
import seekr2.modules.check as check

import seekrtools.hidr.hidr_base as hidr_base
import seekrtools.hidr.hidr_network as hidr_network
import seekrtools.hidr.hidr_simulation as hidr_simulation
from seekrtools.hidr.hidr_base import SETTLED_FINAL_STRUCT_NAME
from seekrtools.hidr.hidr_base import SETTLED_TRAJ_NAME

kJ_per_mol = unit.kilojoules / unit.mole

def catch_erroneous_destination(destination):
    """
    Catch instructions that are not valid and throw an error.
    """
    error_msg = "Available instructions are: 'any', or '#', where '#' "\
        "is an integer."
    if destination not in ["any"]:
        try:
            dummy_integer = int(destination)
        except ValueError:
            print(error_msg)
            return False
            
    return True

def assign_pdb_or_toy_coords(model, pdb_files=None, toy_coordinates=None,
                             skip_checks=False, dry_run=False):
    if model.using_toy():
        for toy_coordinate in toy_coordinates:
            assert len(toy_coordinate) == 3
        hidr_base.assign_toy_coords_to_model(model, toy_coordinates)
        
    else:
        for pdb_file in pdb_files:
            hidr_base.assign_pdb_file_to_model(model, pdb_file, skip_checks, 
                                               dry_run)
    return

def hidr(model, destination, pdb_files=[], toy_coordinates=None, dry_run=False,
         equilibration_steps=0, skip_minimization=False, mode="SMD",
         restraint_force_constant=90000.0*unit.kilojoules_per_mole/unit.nanometers**2,
         ramd_force_magnitude=14.0*unit.kilocalories_per_mole/unit.nanometers,
         translation_velocity=0.01*unit.nanometers/unit.nanoseconds,
         settling_steps=0, settling_frames=1, skip_checks=False,
         force_overwrite=False, traj_mode=False, smd_dcd_interval=None,
         ligand_atom_indices=None, receptor_atom_indices=None,
         steps_per_algorithm_update=250, steps_per_anchor_check=250,
         RAMD_cutoff_distance=0.0025, keeping_starting=True, ignore_cv=None,
         metadyn_sigma=0.05*unit.nanometers, metadyn_biasfactor=10.0,
         metadyn_height=1.0*kJ_per_mol):
    """
    Run the full HIDR calculation for a model.
    
    Parameters:
    -----------
    model : Model()
        The model object containing the relevant anchors for simulation.
    destination : str
        A string representing the 'instruction' for HIDR. Specifically,
        either an index of a reachable destination anchor or the 
        string "any", where all anchors are sought.
    dry_run : bool, Default False
        Whether to do a 'dry run', where information is merely printed,
        but no simulations are actually performed.
    equilibration_steps : int, Default 0
        The number of steps to run equilibrating starting anchor 
        simulations.
    skip_minimization : bool, Default False
        Whether to skip minimizations before doing equilibrations.
    restrain_force_constant : Quantity, Default 90000.0 kJ/mol*nm**2
        The force constant to use for the force which restrains the
        system to the anchor or the SMD window.
    translation_velocity : Quantity, Default 0.01 nm/ns
        The target velocity to move the system between anchors.
    skip_checks : bool
        Whether to skip checks after HIDR, which are normally used to
        ensure that HIDR placed the anchors correctly.
    force_overwrite : bool
        Whether to overwrite any simulation files instead of skipping over them
        if detected.
    """
    assert catch_erroneous_destination(destination)
    
    mode = mode.lower()
    
    assert mode in ["smd", "ramd", "metadyn", "meta", "metad"], \
        "Incorrect mode option: {}. ".format(mode)\
        +"Available options are: 'SMD', 'RAMD', and 'metadyn'/'meta'."
    
    assign_pdb_or_toy_coords(model, pdb_files, toy_coordinates, skip_checks,
                             dry_run)
    
    # Find all anchors with a starting structure
    anchors_with_starting_structures \
        = hidr_base.find_anchors_with_starting_structure(model)
    
    # Find destination(s) and see whether any starting anchors can reach any
    #  destination(s).
    destination_list, complete_anchor_list = hidr_base.find_destinations(
        model, destination, anchors_with_starting_structures)
    relevant_anchors_with_starting_structures = hidr_base.check_destinations(
        model, anchors_with_starting_structures, destination_list, 
        force_overwrite)
    print("anchors_with_starting_structures:", anchors_with_starting_structures)
    settling_anchor_list = hidr_base.check_settling_anchors(
        model, complete_anchor_list, force_overwrite)
    
    smd_dcd_filename = os.path.join(
        model.anchor_rootdir, hidr_simulation.SMD_DCD_NAME)
    if force_overwrite and os.path.exists(smd_dcd_filename):
        print("Force deletion of file:", smd_dcd_filename)
        os.remove(smd_dcd_filename)
    
    metadyn_bias_dir = os.path.join(model.anchor_rootdir, 
                                    hidr_simulation.METADYN_BIAS_DIR_NAME)
    if force_overwrite and os.path.exists(metadyn_bias_dir):
        print("Force deletion of directory:", metadyn_bias_dir)
        shutil.rmtree(metadyn_bias_dir)
    
    # Given the destination command, generate a recipe of instructions for
    #  reaching all destination anchors
    if mode == "smd":
        procedure = hidr_network.get_procedure(
            model, anchors_with_starting_structures, destination_list)
        estimated_smd_time = hidr_network.estimate_simulation_time(
        model, procedure, translation_velocity)
    elif (mode == "ramd") or (mode in ["metadyn", "meta", "metad"]):
        #procedure = [[anchors_with_starting_structures[-1], destination_list]]
        smd_procedure = hidr_network.get_procedure(
            model, anchors_with_starting_structures, destination_list)
        if len(smd_procedure) == 0:
            print("Nothing to run. Exiting.")
            exit()
        starting_anchor = smd_procedure[0][0]
        #procedure = [[starting_anchor, destination_list]]
        procedure = [[starting_anchor, complete_anchor_list]]
        estimated_smd_time = 100.0 * unit.nanoseconds
        
    
    estimated_equilibration_time = equilibration_steps \
        * hidr_simulation.get_timestep(model)
    
    estimated_total_time = estimated_equilibration_time + estimated_smd_time
    
    # Print the entire procedure for the user.
    print("The following procedure will be executed:")
    if equilibration_steps > 0:
        if skip_minimization:
            print("Minimization will be skipped.")
        else:
            print("Minimization will be performed.")
        for starting_anchor_index in relevant_anchors_with_starting_structures:
            print("Equilibration on anchor {} for {} steps.".format(
                starting_anchor_index, equilibration_steps))
    
    
    for step in procedure:
        source_anchor_index = step[0]
        destination_anchor_index = step[1]
        if mode == "smd":
            force_constant = restraint_force_constant
            
        elif mode == "ramd":
            force_constant = ramd_force_magnitude
        
        elif mode in ["metadyn", "meta", "metad"]:
            force_constant = None
        
        if force_constant is not None:
            print("{} force constant is {}".format(mode, force_constant))
            print("{} from anchor {} to anchor {}".format(
                mode, source_anchor_index, destination_anchor_index))
    
    if settling_steps > 0:
        for anchor_index in settling_anchor_list:
            print("Settling MD in anchor "\
                  +"{} for {} steps. {} frame(s) will be saved.".format(
                      anchor_index, settling_steps, settling_frames))
    
    est_time_str = "{:.3f} ns".format(estimated_total_time.value_in_unit(
        unit.nanoseconds))
    print("Estimated total simulation time:", est_time_str)
    if dry_run:
        print("Dry run; exiting.")
        exit()
    
    # Run equilibration for all anchors with a starting structure
    if equilibration_steps > 0:
        for starting_anchor_index in relevant_anchors_with_starting_structures:
            print("running equilibration for anchor {}".format(
                starting_anchor_index))
            ns_per_day = hidr_simulation.run_min_equil_anchor(
                model, starting_anchor_index, equilibration_steps, 
                skip_minimization, restraint_force_constant)
            
            print("Performance:", ns_per_day, "ns per day")
    
    hidr_base.save_new_model(model)
        
    # Run the recipe of SMD instructions
    for step in procedure:
        if mode.lower() == "smd":
            source_anchor_index = step[0]
            destination_anchor_index = step[1]
            print("running SMD from anchor {} to anchor {}".format(
                source_anchor_index, destination_anchor_index))
            hidr_simulation.run_SMD_simulation(
                model, source_anchor_index, destination_anchor_index, 
                restraint_force_constant, translation_velocity, 
                smd_dcd_interval, ignore_cv)
            
            # save the new model file and check the generated structures
            hidr_base.save_new_model(model, save_old_model=False)
            
        elif mode.lower() == "ramd":
            source_anchor_index = step[0]
            destination_anchor_indices = step[1]
            print("running RAMD from anchor {} to anchor {}".format(
                source_anchor_index, destination_anchor_indices))
            
            if ligand_atom_indices is None:
                # TODO: REDO HACK !!!
                rec_indices = model.collective_variables[0].get_atom_groups()[0]
                lig_indices = model.collective_variables[0].get_atom_groups()[1]
            else:
                rec_indices = receptor_atom_indices
                lig_indices = ligand_atom_indices
            
            ns_per_day = hidr_simulation.run_RAMD_simulation(
                model, ramd_force_magnitude, source_anchor_index, 
                destination_anchor_indices, lig_indices, rec_indices, 
                traj_mode=traj_mode, 
                steps_per_RAMD_update=steps_per_algorithm_update, 
                steps_per_anchor_check=steps_per_anchor_check,
                RAMD_cutoff_distance_nanometers=RAMD_cutoff_distance)
            # save the new model file and check the generated structures
            print("Benchmark:", ns_per_day, "ns/day")
            hidr_base.save_new_model(model, save_old_model=False)
            
        elif mode.lower() in ["metadyn", "meta", "metad"]:
            source_anchor_index = step[0]
            destination_anchor_indices = step[1]
            print("running Metadynamics from anchor {} to anchor {}".format(
                source_anchor_index, destination_anchor_indices))
            ns_per_day = hidr_simulation.run_Metadyn_simulation(
                model, source_anchor_index,
                destination_anchor_indices=destination_anchor_indices,
                steps_per_metadyn_update=steps_per_algorithm_update, 
                steps_per_anchor_check=steps_per_anchor_check, 
                metadyn_npoints=None, metadyn_sigma=metadyn_sigma, 
                metadyn_biasfactor=metadyn_biasfactor, 
                metadyn_height=metadyn_height, ignore_cv=None, 
                anchors_with_starting_structures=anchors_with_starting_structures)
            # save the new model file and check the generated structures
            print("Benchmark:", ns_per_day, "ns/day")
            hidr_base.save_new_model(model, save_old_model=False)
            
        else:
            print("mode not allowed: {}".format(mode))
    
    if settling_steps > 0:
        for anchor_index in settling_anchor_list:
            print("Settling MD in anchor "\
                  +"{} for {} steps. {} frame(s) will be saved.".format(
                      anchor_index, settling_steps, settling_frames))
            #var_string = hidr_simulation.make_var_string(
            #    model.anchors[anchor_index])
            #settled_final_filename = SETTLED_FINAL_STRUCT_NAME.format(var_string)
            #settled_traj_filename = SETTLED_TRAJ_NAME.format(var_string)
            settled_final_filename, settled_traj_filename \
                = hidr_base.make_settling_names(model, anchor_index)
            ns_per_day = hidr_simulation.run_min_equil_anchor(
                model, anchor_index, settling_steps, skip_minimization=True, 
                restraint_force_constant=restraint_force_constant, 
                equilibrated_name=settled_final_filename, 
                trajectory_name=settled_traj_filename,
                assign_trajectory_to_model=True)
            hidr_base.save_new_model(model, save_old_model=False)
    
    if keeping_starting:
        assign_pdb_or_toy_coords(model, pdb_files, toy_coordinates, dry_run)
    
    if not skip_checks:
        print("Running pre-simulation checks...")
        check.check_pre_simulation_all(model)
    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "destination", metavar="DESTINATION_ANCHORS", type=str,
        help="The index of which anchor(s) to next generate a structure for. "\
        "If the word 'any' is entered here, then all accessible anchors will "\
        "be populated one by one. If an integer is provided, that anchor "\
        "will be populated.")
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of model XML file for a SEEKR2 calculation. "\
        "One or more starting structures must be present in one or more of "\
        "the anchors.")
    
    argparser.add_argument(
        "-M", "--mode", dest="mode", default="SMD", type=str,
        metavar="MODE", help="The 'mode' or type of enhanced sampling method"\
        "to use for generating starting structures for HIDR. At this time, "\
        "the options 'SMD', 'RAMD', and 'MetaD' have been implemented. "\
        "Default: SMD.")
    
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
        "-d", "--dry_run", dest="dry_run", default=False,
        help="Toggle a dry run to print information about HIDR's planned "\
        "actions, and then quit. Default: False.", action="store_true")
    argparser.add_argument(
        "-e", "--equilibration_steps", dest="equilibration_steps", 
        metavar="EQUILIBRATION_STEPS", type=int, default=0,
        help="Enter the number of steps of equilibration to run for the "\
        "starting structure. Default: 0.")
    argparser.add_argument(
        "-m", "--skip_minimization", dest="skip_minimization", 
        default=False,
        help="Toggle this setting to skip minimizations before the "\
        "equilibration of any starting structures. Default: False.", 
        action="store_true")
    argparser.add_argument(
        "-k", "--restraint_force_constant", dest="restraint_force_constant",
        type=float, default=90000.0, 
        help="The force constant to use for restraints in units of "\
        "kilojoules per mole per nanometer**2. Default: 90000.")
    argparser.add_argument(
        "-K", "--ramd_force_magnitude", dest="ramd_force_magnitude",
        type=float, default=14.0, 
        help="The force constant to use for the RAMD force in units of "\
        "kilocalories per mole per angstrom. Default: 14.0.")
    argparser.add_argument(
        "-v", "--translation_velocity", dest="translation_velocity",
        type=float, default=0.01, 
        help="The velocity (in nm/ns) to pull the system along "\
        "an SMD trajectory for distance-based CVs. The default is 0.01 nm/ns.")
    argparser.add_argument(
        "-S", "--settling_steps", dest="settling_steps",
        metavar="SETTLING_STEPS", type=int, default=0,
        help="Enter the number of post-pulling stage 'settling' steps, where "\
        "the system is held at the location of the anchor in an umbrella "\
        "simulation. This feature can be used to generate a 'starting swarm' "\
        "for MMVT simulations.")
    argparser.add_argument(
        "-F", "--settling_frames", dest="settling_frames", 
        metavar="SETTLING_FRAMES", type=int, default=1,
        help="If there is a nonzero number of --settling_steps, a swarm of "\
        "starting conformations can be generated for MMVT by entering a "\
        "number here greater than 1. Default: 1.")
    argparser.add_argument(
        "-s", "--skip_checks", dest="skip_checks", default=False, 
        help="By default, pre-simulation checks will be run after the "\
        "preparation is complete, and if the checks fail, the SEEKR2 "\
        "model will not be saved. This argument bypasses those "\
        "checks and allows the model to be generated anyways. Default: False.", 
        action="store_true")
    argparser.add_argument(
        "-c", "--cuda_device_index", dest="cuda_device_index", default=None,
        help="modify which cuda_device_index to run the simulation on. For "\
        "example, the number 0 or 1 would suffice. To run on multiple GPU "\
        "indices, simply enter comma separated indices. Example: '0,1'. If a "\
        "value is not supplied, the value in the MODEL_FILE will be used by "\
        "default.", type=str)
    argparser.add_argument(
        "-f", "--force_overwrite", dest="force_overwrite", default=False,
        help="Toggle whether to overwrite existing simulation output files "\
        "within any anchor that might have existed in an old model that would "\
        "be overwritten by generating this new model. If not toggled, this "\
        "program will skip the stage instead of performing any such "\
        "overwrite.", action="store_true")
    argparser.add_argument(
        "-T", "--traj_mode", dest="traj_mode", default=False,
        help="Toggle whether to enable trajectory mode for RAMD, which will "\
        "save all crossing events to be simulated in an MMVT swarm.", 
        action="store_true")
    argparser.add_argument(
        "-l", "--ligand_atom_indices", dest="ligand_atom_indices", default=None, 
        metavar="[i1, i2, i3, ...]", help="Enter the atom indices "\
        "of the ligand molecule for RAMD simulations. Default: None.")
    argparser.add_argument(
        "-r", "--receptor_atom_indices", dest="receptor_atom_indices", 
        default=None, metavar="[r1, r2, r3, ...]", help="Enter the atom "\
        "indices of the receptor molecule for RAMD simulations. Default: None.")
    argparser.add_argument(
        "-R", "--steps_per_algorithm_update", dest="steps_per_algorithm_update", 
        metavar="STEPS_PER_ALGORITHM_UPDATE", type=int, default=250,
        help="How many simulation timesteps to take in RAMD or Metadynamics "\
        "simulations before checking whether the forces should be recomputed. "\
        "Default: 250.")
    argparser.add_argument(
        "-u", "--steps_per_anchor_check", dest="steps_per_anchor_check", 
        metavar="STEPS_PER_ANCHOR_CHECK", type=int, default=250,
        help="How many simulation timesteps to wait before checking if new "\
        "anchors have been reached. Default: 250.")
    argparser.add_argument(
        "-D", "--RAMD_cutoff_distance", dest="RAMD_cutoff_distance",
        type=float, default=0.0025, 
        help="The distance (in nm) at which if the ligand COMs of two "\
        "consecutive updates have not exceeded, change the force direction. "\
        "Default: 0.0025 nm.")
    argparser.add_argument(
        "-i", "--ignore_cv", dest="ignore_cv", default=None,
        metavar="[c1, c2, ...]", help="Enter the CV indices to ignore when "\
        "making forces, also applies to sub-cvs for MMVT. Default: None. ")
    argparser.add_argument(
        "-w", "--metadyn_sigma", dest="metadyn_sigma", default=None,
        metavar="w or [w1, w2, ...]", help="The standard deviations of the "\
        "Gaussians added to the bias in metadynamics for each CV. Can be "\
        "a float or a list of floats. Default: 0.05 nm. ")
    argparser.add_argument(
        "-b", "--metadyn_biasfactor", dest="metadyn_biasfactor",
        type=float, default=10.0, 
        help="The biasFactor used to scale the height of the Gaussians added "\
        "to the bias. The CVs are sampled as if the effective temperature "\
        "of the simulation were temperature*biasFactor. Default: 10.0.")
    argparser.add_argument(
        "-H", "--metadyn_height", dest="metadyn_height", type=float, 
        default=1.0, help="The initial heights of the metadynamics Gaussians. "\
        "Default: 1.0 kJ/mol.")
    
    args = argparser.parse_args() # parse the args into a dictionary
    args = vars(args)
    destination = args["destination"]
    model_file = args["model_file"]
    mode = args["mode"]
    pdb_files = args["pdb_files"]
    toy_coordinates = ast.literal_eval(args["toy_coordinates"])
    dry_run = args["dry_run"]
    equilibration_steps = args["equilibration_steps"]
    skip_minimization = args["skip_minimization"]
    restraint_force_constant = args["restraint_force_constant"] \
        * unit.kilojoules_per_mole / unit.nanometers**2
    ramd_force_magnitude = args["ramd_force_magnitude"] \
        * unit.kilocalories_per_mole / unit.angstroms
    translation_velocity = args["translation_velocity"] * unit.nanometers \
        / unit.nanoseconds
    settling_steps = args["settling_steps"]
    settling_frames = args["settling_frames"]
    skip_checks = args["skip_checks"]
    cuda_device_index = args["cuda_device_index"]
    force_overwrite = args["force_overwrite"]
    traj_mode = args["traj_mode"]
    ligand_atom_indices = args["ligand_atom_indices"]
    receptor_atom_indices = args["receptor_atom_indices"]
    steps_per_algorithm_update = args["steps_per_algorithm_update"]
    steps_per_anchor_check = args["steps_per_anchor_check"]
    RAMD_cutoff_distance = args["RAMD_cutoff_distance"]
    ignore_cv = args["ignore_cv"]
    metadyn_sigma = args["metadyn_sigma"]
    metadyn_biasfactor = args["metadyn_biasfactor"]
    metadyn_height = args["metadyn_height"] * kJ_per_mol
    
    if ligand_atom_indices is None:
        assert receptor_atom_indices is None, \
            "if ligand_atom_indices is None, receptor_atom_indices must also "\
            "be None."
    else:
        ligand_atom_indices = ast.literal_eval(ligand_atom_indices)
        if receptor_atom_indices is not None:
            receptor_atom_indices = ast.literal_eval(receptor_atom_indices)
    
    if ignore_cv is not None:
        ignore_cv = ast.literal_eval(ignore_cv)
    
    model = base.load_model(model_file)
    if cuda_device_index is not None:
        assert model.openmm_settings.cuda_platform_settings is not None
        model.openmm_settings.cuda_platform_settings.cuda_device_index = \
            cuda_device_index
    hidr(model, destination, pdb_files, toy_coordinates, dry_run, 
         equilibration_steps, skip_minimization, mode, restraint_force_constant, 
         ramd_force_magnitude, translation_velocity, 
         settling_steps, settling_frames, skip_checks, force_overwrite, 
         traj_mode, ligand_atom_indices=ligand_atom_indices, 
         receptor_atom_indices=receptor_atom_indices,
         steps_per_algorithm_update=steps_per_algorithm_update, 
         steps_per_anchor_check=steps_per_anchor_check,
         RAMD_cutoff_distance=RAMD_cutoff_distance, ignore_cv=ignore_cv,
         metadyn_sigma=metadyn_sigma, metadyn_biasfactor=metadyn_biasfactor,
         metadyn_height=metadyn_height)