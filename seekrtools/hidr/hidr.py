#!/usr/bin/env python
"""
HIDR - Holo Insertion by Directed Restraints

HIDR takes one or more starting structures, as well as a SEEKR Input Model
and, closer to further, pulls the system towards all the anchors within the 
SEEKR model until starting structures exist in all of them. This is 
accomplished using steered molecular dynamics (SMD) simulations.
"""

import argparse
import tempfile
import glob

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

def catch_erroneous_destination(destination):
    """
    Catch instructions that are not valid and throw an error.
    """
    error_msg = "Available instructions are: 'any', or '#', where '#' "\
        "is an integer."
    if destination not in ["any"]:
        try:
            integer_destination = int(destination)
        except ValueError:
            print(error_msg)
            return False
            
    return True

def hidr(model, destination, pdb_files=[], dry_run=False, equilibration_steps=0, 
         skip_minimization=False, mode="SMD",
         restraint_force_constant=90000.0*unit.kilojoules_per_mole/unit.nanometers**2, 
         ramd_force_magnitude=14.0*unit.kilocalories_per_mole/unit.nanometers, 
         translation_velocity=0.01*unit.nanometers/unit.nanoseconds, 
         settling_steps=0, settling_frames=1, skip_checks=False, 
         force_overwrite=False, traj_mode=False):
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
    
    assert mode in ["SMD", "RAMD"], "Incorrect mode option: {}. ".format(mode)\
        +"Available options are: 'SMD' and 'RAMD'."
    
    for pdb_file in pdb_files:
        hidr_base.assign_pdb_file_to_model(model, pdb_file)
    
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
    settling_anchor_list = hidr_base.check_settling_anchors(
        model, complete_anchor_list, force_overwrite)
    
    # Given the destination command, generate a recipe of instructions for
    #  reaching all destination anchors
    if mode == "SMD":
        procedure = hidr_network.get_procedure(
            model, anchors_with_starting_structures, destination_list)
        estimated_smd_time = hidr_network.estimate_simulation_time(
        model, procedure, translation_velocity)
    elif mode == "RAMD":
        #procedure = [[anchors_with_starting_structures[-1], destination_list]]
        smd_procedure = hidr_network.get_procedure(
            model, anchors_with_starting_structures, destination_list)
        starting_anchor = smd_procedure[0][0]
        #procedure = [[starting_anchor, destination_list]]
        procedure = [[starting_anchor, complete_anchor_list]]
        estimated_smd_time = 20.0 * unit.nanoseconds
        
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
        if mode == "SMD":
            force_constant = restraint_force_constant
            
        elif mode == "RAMD":
            force_constant = ramd_force_magnitude
        
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
        if mode == "SMD":
            source_anchor_index = step[0]
            destination_anchor_index = step[1]
            print("running SMD from anchor {} to anchor {}".format(
                source_anchor_index, destination_anchor_index))
            hidr_simulation.run_SMD_simulation(
                model, source_anchor_index, destination_anchor_index, 
                restraint_force_constant, translation_velocity)
            
            # save the new model file and check the generated structures
            hidr_base.save_new_model(model, save_old_model=False)
            
        elif mode == "RAMD":
            source_anchor_index = step[0]
            destination_anchor_indices = step[1]
            print("running RAMD from anchor {} to anchor {}".format(
                source_anchor_index, destination_anchor_indices))
            
            # TODO: REDO HACK !!!
            rec_indices = model.collective_variables[0].group1
            lig_indices = model.collective_variables[0].group2
            
            hidr_simulation.run_RAMD_simulation(
                model, ramd_force_magnitude, source_anchor_index, 
                destination_anchor_indices, lig_indices, rec_indices, 
                traj_mode=traj_mode)
            # save the new model file and check the generated structures
            hidr_base.save_new_model(model, save_old_model=True)
            
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
        "the options 'SMD' and 'RAMD' have been implemented.")
    
    argparser.add_argument(
        "-p", "--pdb_files", dest="pdb_files", default=[], nargs="*", type=str,
        metavar="FILE1 FILE2 ...", help="One or more PDB files which will be "\
        "placed into the correct anchors. NOTE: the parameter/topology files "\
        "must already be assigned into an anchor for this to work.")
    argparser.add_argument(
        "-d", "--dry_run", dest="dry_run", default=False,
        help="Toggle a dry run to print information about HIDR's planned "\
        "actions, and then quit.", action="store_true")
    argparser.add_argument(
        "-e", "--equilibration_steps", dest="equilibration_steps", 
        metavar="EQUILIBRATION_STEPS", type=int, default=0,
        help="Enter the number of steps of equilibration to run for the "\
        "starting structure.")
    argparser.add_argument(
        "-m", "--skip_minimization", dest="skip_minimization", 
        default=False,
        help="Toggle this setting to skip minimizations before the "\
        "equilibration of any starting structures.", action="store_true")
    argparser.add_argument(
        "-k", "--restraint_force_constant", dest="restraint_force_constant",
        type=float, default=90000.0, 
        help="The force constant to use for restraints in units of "\
        "kilojoules per mole per nanometer**2")
    argparser.add_argument(
        "-K", "--ramd_force_magnitude", dest="ramd_force_magnitude",
        type=float, default=14.0, 
        help="The force constant to use for the RAMD force in units of "\
        "kilocalories per mole per angstrom.")
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
        "number here greater than 1.")
    argparser.add_argument(
        "-s", "--skip_checks", dest="skip_checks", default=False, 
        help="By default, pre-simulation checks will be run after the "\
        "preparation is complete, and if the checks fail, the SEEKR2 "\
        "model will not be saved. This argument bypasses those "\
        "checks and allows the model to be generated anyways.", 
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
    
    args = argparser.parse_args() # parse the args into a dictionary
    args = vars(args)
    destination = args["destination"]
    model_file = args["model_file"]
    mode = args["mode"]
    pdb_files = args["pdb_files"]
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
    
    model = base.load_model(model_file)
    if cuda_device_index is not None:
        assert model.openmm_settings.cuda_platform_settings is not None
        model.openmm_settings.cuda_platform_settings.cuda_device_index = \
            cuda_device_index
    hidr(model, destination, pdb_files, dry_run, equilibration_steps, 
         skip_minimization, mode, restraint_force_constant, 
         ramd_force_magnitude, translation_velocity, 
         settling_steps, settling_frames, skip_checks, force_overwrite, 
         traj_mode)