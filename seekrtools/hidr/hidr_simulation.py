"""
Run minimizations and equilibrations for a given structure in an anchor.
"""

import sys
import os
import time
from shutil import copyfile
from copy import deepcopy

from icecream import ic
import numpy as np
import parmed
import mdtraj
import matplotlib.pyplot as plt
try:
    import openmm.app as openmm_app
    import openmm
    
except ImportError:
    import simtk.openmm.app as openmm_app
    import simtk.openmm as openmm

try:
    import openmm.unit as unit
except ModuleNotFoundError:
    import simtk.unit as unit

try:
    from openmm_ramd import openmm_ramd
except ModuleNotFoundError:
    print("openmm_ramd module not found. You will be unable to run RAMD.")

import seekr2.modules.common_base as base
import seekr2.modules.mmvt_cvs.mmvt_cv_base as mmvt_cv_base
import seekr2.modules.mmvt_cvs.mmvt_spherical_cv as mmvt_spherical_cv
import seekr2.modules.mmvt_cvs.mmvt_tiwary_cv as mmvt_tiwary_cv
import seekr2.modules.mmvt_cvs.mmvt_planar_cv as mmvt_planar_cv
import seekr2.modules.mmvt_cvs.mmvt_rmsd_cv as mmvt_rmsd_cv
import seekr2.modules.mmvt_cvs.mmvt_closest_pair_cv as mmvt_closest_pair_cv
import seekr2.modules.mmvt_cvs.mmvt_count_contacts_cv as mmvt_count_contacts_cv
import seekr2.modules.mmvt_cvs.mmvt_external_cv as mmvt_external_cv
import seekr2.modules.mmvt_cvs.mmvt_voronoi_cv as mmvt_voronoi_cv
import seekr2.modules.common_sim_openmm as common_sim_openmm
import seekr2.modules.common_analyze as common_analyze

import seekrtools.hidr.hidr_base as hidr_base

NUM_WINDOWS=10
NUM_EQUIL_FRAMES=10
EQUIL_UPDATE_INTERVAL=5000
SMD_DCD_NAME="smd.dcd"
RAMD_LOG_FILENAME = "ramd.log"
#EQUILIBRATED_NAME = "hidr_equilibrated.pdb"
#EQUILIBRATED_TRAJ_NAME = "hidr_traj_equilibrated.pdb"
#SMD_NAME = "hidr_smd_at_{}.pdb"
#SETTLED_FINAL_STRUCT_NAME = "hidr_settled_at_{}.pdb"
#SETTLED_TRAJ_NAME = "hidr_traj_settled_at_{}.pdb"
from seekrtools.hidr.hidr_base import EQUILIBRATED_NAME
from seekrtools.hidr.hidr_base import EQUILIBRATED_TRAJ_NAME
from seekrtools.hidr.hidr_base import SMD_NAME
from seekrtools.hidr.hidr_base import RAMD_NAME
from seekrtools.hidr.hidr_base import RAMD_TRAJ_NAME
from seekrtools.hidr.hidr_base import METADYN_NAME
from seekrtools.hidr.hidr_base import METADYN_TRAJ_NAME
#from seekrtools.hidr.hidr_base import SETTLED_FINAL_STRUCT_NAME
#from seekrtools.hidr.hidr_base import SETTLED_TRAJ_NAME
DEFAULT_METADYN_NPOINTS = 181
DEFAULT_METADYN_SIGMA = 0.5
MAX_METADYN_STEPS = 1000000000
METADYN_BIAS_DIR_NAME = "metadyn_bias_dir"
kcal_per_mol = unit.kilocalories / unit.mole

class HIDR_sim_openmm(common_sim_openmm.Common_sim_openmm):
    """
    system : The OpenMM system object for this simulation.
    
    integrator : the OpenMM integrator object for this simulation.
    
    simulation : the OpenMM simulation object.
    
    traj_reporter : openmm.PDBReporter
        The OpenMM Reporter object to which the trajectory will be
        written.
        
    energy_reporter : openmm.StateDataReporter
        The OpenMM StateDataReporter to which the energies and other
        state data will be reported.
        
    """
    def __init__(self):
        super(HIDR_sim_openmm, self).__init__()
        self.system = None
        self.integrator = None
        self.simulation = None
        self.traj_reporter = openmm_app.DCDReporter
        self.energy_reporter = openmm_app.StateDataReporter
        self.forces = None
        return

def get_timestep(model):
    """
    Given the integrator settings in model, return the length of the
    simulation timestep in picoseconds.
    
    Parameters:
    -----------
    model : Model()
        The model object containing the relevant anchors.
        
    Returns:
    --------
    timestep : Quantity
        The length of this model's simulation timestep.
    """
    if model.openmm_settings.langevin_integrator is not None:
        timestep = \
            model.openmm_settings.langevin_integrator.timestep
    else:
        raise Exception("Settings not provided for available "\
                        "integrator type(s).")
    return timestep*unit.picoseconds

def add_integrator(sim_openmm, model):
    """
    Assign the proper integrator to this OpenMM simulation.
    """
    if model.openmm_settings.langevin_integrator is not None:
        target_temperature = \
            model.openmm_settings.langevin_integrator.target_temperature
        friction_coefficient = \
            model.openmm_settings.langevin_integrator.friction_coefficient
        random_seed = \
            model.openmm_settings.langevin_integrator.random_seed
        timestep = \
            model.openmm_settings.langevin_integrator.timestep
        rigid_constraint_tolerance = \
            model.openmm_settings.langevin_integrator\
            .rigid_tolerance
            
        sim_openmm.timestep = timestep
        
        sim_openmm.integrator = openmm.LangevinIntegrator(
            target_temperature*unit.kelvin, 
            friction_coefficient/unit.picoseconds, 
            timestep*unit.picoseconds)
        
        if random_seed is not None:
            sim_openmm.integrator.setRandomNumberSeed(random_seed)
            
        if rigid_constraint_tolerance is not None:
            sim_openmm.integrator.setConstraintTolerance(
                rigid_constraint_tolerance)
        
    else:
        raise Exception("Settings not provided for available "\
                        "integrator type(s).")
    return timestep*unit.picoseconds

def add_simulation(sim_openmm, model, topology, positions, box_vectors, 
                   skip_minimization):
    """
    Assign the OpenMM simulation object.
    """
    sim_openmm.simulation = openmm_app.Simulation(
        topology, sim_openmm.system, 
        sim_openmm.integrator, sim_openmm.platform, 
        sim_openmm.properties)
    
    if positions is not None:
        sim_openmm.simulation.context.setPositions(positions)
        # For an unknown reason, assigning velocities caused numerical 
        #  instability
        #sim_openmm.simulation.context.setVelocitiesToTemperature(
        #    model.openmm_settings.initial_temperature * unit.kelvin)
        
    if box_vectors is not None:
        sim_openmm.simulation.context.setPeriodicBoxVectors(
            *box_vectors.to_quantity())
    if not skip_minimization:
        sim_openmm.simulation.minimizeEnergy()    
    
    assert sim_openmm.timestep is not None
    return

def handle_reporters(model, anchor, sim_openmm, trajectory_reporter_interval, 
                     energy_reporter_interval, 
                     traj_filename_base=EQUILIBRATED_TRAJ_NAME, 
                     smd_dcd_filename=None, smd_dcd_interval=None):
    """
    If relevant, add the necessary state and trajectory reporters to
    the simulation object.
    """
    directory = os.path.join(
        model.anchor_rootdir, anchor.directory, anchor.building_directory)
    traj_filename = os.path.join(directory, traj_filename_base)
    simulation = sim_openmm.simulation
    traj_reporter = sim_openmm.traj_reporter
    if trajectory_reporter_interval is not None:
        simulation.reporters.append(traj_reporter(
            traj_filename, trajectory_reporter_interval))
    
    if energy_reporter_interval is not None:
        simulation.reporters.append(
            sim_openmm.energy_reporter(
                sys.stdout, energy_reporter_interval, step=True, 
                potentialEnergy=True, temperature=True, volume=True))
    
    if smd_dcd_filename is not None:
        if os.path.exists(smd_dcd_filename):
            append = True
        else:
            append = False
        simulation.reporters.append(openmm_app.DCDReporter(
            smd_dcd_filename, smd_dcd_interval, append))
        
    return

def make_restraining_force(cv, variables_values_list):
    """
    Take a Collective_variable object and a particular milestone and
    return an OpenMM Force() object that the plugin can use to restrain
    the system.
    
    Parameters
    ----------
    cv : Collective_variable()
        A Collective_variable object which contains all the information
        for the collective variable describing this variable. In fact,
        the boundaries are contours of the function described by cv.
        This variable contains information like the groups of atoms
        involved with the CV, and the expression which describes the
        function.
        
    variables_values_list : list
        A list of values for each of the variables in the force object.
        
    Returns
    -------
    myforce : openmm.Force()
        An OpenMM force object which does not affect atomic motion, but
        allows us to conveniently monitor a function of atomic 
        position.
    """
    alias_index = 1 # TODO: wrong??
    myforce = cv.make_restraining_force(alias_index)
    myforce.setForceGroup(1)
    variables_names_list = cv.add_parameters(myforce)
    cv.add_groups_and_variables(myforce, variables_values_list, alias_index)
    return myforce

def make_meta_force_bias(cv, min_value, max_value, bias_width, grid_width):
    """
    Take a Collective_variable object return an OpenMM BiasVariable() 
    object that the plugin can use to run a metadynamics sim.
    
    Parameters
    ----------
    cv : Collective_variable()
        A Collective_variable object which contains all the information
        for the collective variable describing this variable. In fact,
        the boundaries are contours of the function described by cv.
        This variable contains information like the groups of atoms
        involved with the CV, and the expression which describes the
        function.
        
    Returns
    -------
    my_bias_variable : openmm.BiasVariable()
        An OpenMM BiasVariable object used to propagate a metadynamics sim.
    """
    alias_index = 1 # TODO: wrong??
    my_meta_force = cv.make_cv_force(alias_index)
    my_meta_force.setForceGroup(1)
    variables_names_list = cv.add_groups(my_meta_force)
    cv.add_groups_and_variables(my_meta_force, [], alias_index)
    
    # Check cv type to see whether periodic should be accepted
    
    ic(my_meta_force)
    
    myforce1_bias = openmm_app.BiasVariable(
        my_meta_force, minValue=min_value, maxValue=max_value,
        biasWidth=bias_width, periodic=False, gridWidth=grid_width)
    
    return myforce1_bias

def update_restraining_force(cv, variables_values_list, force, context):
    """
    Update the restraining force variables in a CV.
    """
    alias_index = 1 # TODO: wrong??
    cv.update_groups_and_variables(force, variables_values_list, alias_index, context)
    return

def add_forces(sim_openmm, model, anchor, restraint_force_constant, 
               cv_list=None, window_values=None, ignore_cv=None):
    """
    Add the proper forces for this restrained simulation.
    
    Parameters:
    -----------
    sim_openmm : HIDR_sim_openmm()
        The HIDR_sim_openmm object which contains all simulation
        settings and information needed.
    model : Model()
        The model object containing the relevant anchors.
    anchor : Anchor()
        The anchor object to generate forces for.
    restrain_force_constant : float
        The restraint force constant (in units of kcal/mol*nm**2.
    """
    value_dict = {}
    if ignore_cv is None:
        ignore_cv = []
    if cv_list is not None and window_values is not None:
        for cv_index, value in zip(cv_list, window_values):
            value_dict[cv_index] = value
        
    forces = []
    for variable_key in anchor.variables:
        var_name = variable_key.split("_")[0]
        var_cv = int(variable_key.split("_")[1])
        if cv_list is not None:
            if var_cv not in cv_list:
                continue
        
        cv = model.collective_variables[var_cv]
        
        curdir = os.getcwd()
        os.chdir(model.anchor_rootdir)
        
        if isinstance(cv, mmvt_voronoi_cv.MMVT_Voronoi_CV):
            var_child_cv = int(variable_key.split("_")[2])
            child_cv = cv.child_cvs[var_child_cv]
            if window_values is None:
                var_value = anchor.variables[variable_key]
            else:
                var_value = value_dict[var_child_cv]
            cv_variables = child_cv.get_variable_values()
            if var_child_cv in ignore_cv:
                this_restraint_force_constant = 0.0
            else:
                this_restraint_force_constant = restraint_force_constant
                
            variables_values_list = [1] + cv_variables \
                + [this_restraint_force_constant, var_value]
            myforce = make_restraining_force(child_cv, variables_values_list)
        else:
            if window_values is None:
                var_value = anchor.variables[variable_key]
            else:
                var_value = value_dict[var_cv]
            cv_variables = cv.get_variable_values()
            if var_cv in ignore_cv:
                this_restraint_force_constant = 0.0
            else:
                this_restraint_force_constant = restraint_force_constant
                
            variables_values_list = [1] + cv_variables \
            + [this_restraint_force_constant, var_value]
            myforce = make_restraining_force(cv, variables_values_list)
                
        os.chdir(curdir)
        forcenum = sim_openmm.system.addForce(myforce)
        forces.append(myforce)
    
    sim_openmm.forces = forces
    return forces

def update_forces(sim_openmm, forces, model, anchor, restraint_force_constant, 
               cv_list=None, window_values=None, ignore_cv=None):
    """
    Update the forces for this restrained simulation.
    
    Parameters:
    -----------
    sim_openmm : HIDR_sim_openmm()
        The HIDR_sim_openmm object which contains all simulation
        settings and information needed.
    model : Model()
        The model object containing the relevant anchors.
    anchor : Anchor()
        The anchor object to generate forces for.
    restrain_force_constant : float
        The restraint force constant (in units of kcal/mol*nm**2.
    """
    value_dict = {}
    if ignore_cv is None:
        ignore_cv = []
    if cv_list is not None and window_values is not None:
        for cv_index, value in zip(cv_list, window_values):
            value_dict[cv_index] = value
    
    for force, variable_key in zip(forces, anchor.variables):
        var_name = variable_key.split("_")[0]
        var_cv = int(variable_key.split("_")[1])
        if cv_list is not None:
            if var_cv not in cv_list:
                continue
        
        cv = model.collective_variables[var_cv]
        curdir = os.getcwd()
        os.chdir(model.anchor_rootdir)
        
        if isinstance(cv, mmvt_voronoi_cv.MMVT_Voronoi_CV):
            var_child_cv = int(variable_key.split("_")[2])
            child_cv = cv.child_cvs[var_child_cv]
            if window_values is None:
                var_value = anchor.variables[variable_key]
            else:
                var_value = value_dict[var_child_cv]
            cv_variables = child_cv.get_variable_values()
            if var_child_cv in ignore_cv:
                this_restraint_force_constant = 0.0
            else:
                this_restraint_force_constant = restraint_force_constant
            variables_values_list = [1] + cv_variables \
                + [this_restraint_force_constant, var_value]
            update_restraining_force(child_cv, variables_values_list, force,
                                     sim_openmm.simulation.context)
        else:
            if window_values is None:
                var_value = anchor.variables[variable_key]
            else:
                var_value = value_dict[var_cv]
            cv_variables = cv.get_variable_values()
            if var_cv in ignore_cv:
                this_restraint_force_constant = 0.0
            else:
                this_restraint_force_constant = restraint_force_constant
            variables_values_list = [1] + cv_variables \
                + [this_restraint_force_constant, var_value]
            update_restraining_force(cv, variables_values_list, force,
                                     sim_openmm.simulation.context)
        
        os.chdir(curdir)
        
    return

def add_metadyn_cvs(model, bias_widths, grid_widths, ignore_cv=None):
    """
    
    """
    value_dict = {}
    if ignore_cv is None:
        ignore_cv = []
        
    meta_force_biases = []
    for i, variable_key in enumerate(model.anchors[0].variables):
        var_name = variable_key.split("_")[0]
        var_cv = int(variable_key.split("_")[1])
        
        variable_values = []
        cv = model.collective_variables[var_cv]
        for anchor in model.anchors:
            value = anchor.variables[variable_key]
            variable_values.append(value)
        
        #min_value = min(variable_values)
        min_value = 0.0
        max_value = max(variable_values)
        
        curdir = os.getcwd()
        os.chdir(model.anchor_rootdir)
        
        if isinstance(cv, mmvt_voronoi_cv.MMVT_Voronoi_CV):
            var_child_cv = int(variable_key.split("_")[2])
            if var_child_cv in ignore_cv:
                continue
            child_cv = cv.child_cvs[var_child_cv]
            myforce_bias = make_meta_force_bias(
                child_cv, min_value, max_value, bias_widths[i], grid_widths[i])
        else:
            if var_cv in ignore_cv:
                continue
            myforce_bias = make_meta_force_bias(
                cv, min_value, max_value, bias_widths[i], grid_widths[i])
                
        os.chdir(curdir)
        meta_force_biases.append(myforce_bias)
    
    return meta_force_biases

def add_barostat(sim_openmm, model):
    """
    Optionally add a barostat to the simulation to maintain constant
    pressure.
    """
    if model.openmm_settings.barostat.membrane:
        barostat = openmm.MonteCarloMembraneBarostat(
                1.0*unit.bar, 
                0.0*unit.bar*unit.nanometers, 
                model.openmm_settings.langevin_integrator.target_temperature, 
                openmm.MonteCarloMembraneBarostat.XYIsotropic, 
                openmm.MonteCarloMembraneBarostat.ZFixed)
    else:
        barostat = openmm.MonteCarloBarostat(
            1.0*unit.bar,
            model.openmm_settings.langevin_integrator.target_temperature, 25)
    
    sim_openmm.system.addForce(barostat)

def run_min_equil_anchor(model, anchor_index, equilibration_steps, 
                         skip_minimization, restraint_force_constant,
                         equilibrated_name=EQUILIBRATED_NAME, 
                         trajectory_name=EQUILIBRATED_TRAJ_NAME,
                         num_equil_frames=NUM_EQUIL_FRAMES,
                         assign_trajectory_to_model=False):
    """
    Run minimizations and equilibrations for a given anchor.
    
    Parameters:
    -----------
    model : Model()
        The model object containing the relevant anchors.
    anchor_index : int
        The index of the anchor object to simulate.
    equilibration_steps : int
        The number of steps to run for equilibration
    skip_minimization : bool
        Whether or not to skip minimizations before the equilibration.
    restrain_force_constant : float
        The restraint force constant (in units of kcal/mol*nm**2.
        
    Returns:
    --------
    ns_per_day : float
        The performance benchmark of the simulation for this anchor,
        in nanoseconds per day.
    """
    
    trajectory_reporter_interval = equilibration_steps // num_equil_frames
    energy_reporter_interval = trajectory_reporter_interval
    total_number_of_steps = equilibration_steps
    
    anchor = model.anchors[anchor_index]
    sim_openmm = HIDR_sim_openmm()
    system, topology, positions, box_vectors, num_frames \
        = common_sim_openmm.create_openmm_system(sim_openmm, model, anchor)
    sim_openmm.system = system
    time_step = add_integrator(sim_openmm, model)
    # TODO: cannot have barostat for settling stage!!
    #add_barostat(sim_openmm, model)
    common_sim_openmm.add_platform(sim_openmm, model)
    add_forces(sim_openmm, model, anchor, restraint_force_constant)
    add_simulation(sim_openmm, model, topology, positions, box_vectors, 
                   skip_minimization)
    handle_reporters(model, anchor, sim_openmm, trajectory_reporter_interval, 
                     energy_reporter_interval, 
                     traj_filename_base=trajectory_name)
    
    output_pdb_file = os.path.join(
        model.anchor_rootdir, anchor.directory, anchor.building_directory,
        equilibrated_name)
    
    start_time = time.time()
    sim_openmm.simulation.step(total_number_of_steps)
    total_time = time.time() - start_time
    
    state = sim_openmm.simulation.context.getState(
        getPositions = True, getVelocities = False, enforcePeriodicBox = True)
    positions = state.getPositions()
    #amber_parm = parmed.load_file(prmtop_filename, inpcrd_filename)
    
    # TODO: this might cause errors with older versions of parmed. 
    # adapt the openmm import if necessary.
    parm = parmed.openmm.load_topology(topology, system)
    parm.positions = positions
    parm.box_vectors = state.getPeriodicBoxVectors()
    parm.save(output_pdb_file, overwrite=True)
    
    hidr_base.change_anchor_box_vectors(anchor, state.getPeriodicBoxVectors())
    if assign_trajectory_to_model:
        hidr_base.change_anchor_pdb_filename(anchor, trajectory_name)
    else:
        hidr_base.change_anchor_pdb_filename(anchor, equilibrated_name)

    simulation_in_ns = total_number_of_steps * time_step.value_in_unit(
        unit.picoseconds) * 1e-3
    total_time_in_days = total_time / (86400.0)
    ns_per_day = simulation_in_ns / total_time_in_days
    
    return ns_per_day

def run_window(model, anchor, sim_openmm, restraint_force_constant, cv_list, 
               window_values, steps_in_window, smd_dcd_filename, 
               smd_dcd_interval):
    """
    Run the window of an SMD simulation, where the restraint equilibrium
    value has been moved incrementally to a new position for a short
    time to draw the system towards a new anchor.
    """
    # TODO: make one sim_openmm object for all windows, but update
    #  parameters in context
    total_number_of_steps = steps_in_window
    if anchor.__class__.__name__ in ["MMVT_toy_anchor"]:
        enforcePeriodicBox = False
    else:
        enforcePeriodicBox = True
    start_time = time.time()
    sim_openmm.simulation.step(total_number_of_steps)
    total_time = time.time() - start_time
    state = sim_openmm.simulation.context.getState(
        getPositions=False, getVelocities=False, 
        enforcePeriodicBox=enforcePeriodicBox)
    #positions = state.getPositions()
    box_vectors = state.getPeriodicBoxVectors()
    
    # DEBUG
    timestep = get_timestep(model)
    simulation_in_ns = total_number_of_steps \
        * timestep.value_in_unit(unit.picoseconds) * 1e-3
    total_time_in_days = total_time / 86400.0
    ns_per_day = simulation_in_ns / total_time_in_days
    print("window benchmark:", ns_per_day, "ns/day")
    return box_vectors
    
def run_SMD_simulation(model, source_anchor_index, destination_anchor_index, 
                         restraint_force_constant, translation_velocity,
                         smd_dcd_interval=None, ignore_cv=None):
    """
    Run a steered molecular dynamics (SMD) simulation between a source 
    anchor and a destination anchor. The resulting structure will be
    saved within the destination anchor of the model's directory.
    
    Parameters
    ----------
    model : Model()
        The model object from the source anchor.
    source_anchor_index : int
        The index, within the model, of the anchor where the starting
        structure is known, and where the SMD simulation will begin.
    destination_anchor_index : int
        The index, within the model, of the anchor which is the final
        destination of this SMD simulation.
    restraint_force_constant : Quantity
        The value of the force constant used in the restraint that
        pulls the system between anchors.
    translation_velocity : Quantity
        The target velocity of the restraint as it travels between
        anchors.
        
    """
    source_anchor = model.anchors[source_anchor_index]
    destination_anchor = model.anchors[destination_anchor_index]
    
    if smd_dcd_interval is not None:
        smd_dcd_filename = os.path.join(
            model.anchor_rootdir, SMD_DCD_NAME)
    else:
        smd_dcd_filename = None
    
    cv_id_list = []
    windows_list_unzipped = []
    
    # Get positions
    dummy_sim_openmm = HIDR_sim_openmm()
    system, dummy_topology, positions, dummy_box_vectors, \
        dummy_num_frames = common_sim_openmm.create_openmm_system(
            dummy_sim_openmm, model, source_anchor)
    
    total_sq_distance = 0.0
    for variable_key in source_anchor.variables:
        var_name = variable_key.split("_")[0]
        var_cv = int(variable_key.split("_")[1])
        cv = model.collective_variables[var_cv]
        skip_increment = False
        # the two anchors must share a common variable
        if variable_key in destination_anchor.variables:
            if source_anchor.__class__.__name__ in ["MMVT_toy_anchor"]:
                start_values = cv.get_cv_value(positions)
            else:
                start_values = cv.get_openmm_context_cv_value(None, positions, system)
                        
            if isinstance(cv, mmvt_voronoi_cv.MMVT_Voronoi_CV):
                var_child_cv = int(variable_key.split("_")[2])
                if ignore_cv is not None:
                    if var_child_cv in ignore_cv:
                        skip_increment = True
                start_value = start_values[var_child_cv]
                cv_id_list.append(var_child_cv)
            else:
                #start_value = cv.get_openmm_context_cv_value(None, positions, system)
                if ignore_cv is not None:
                    if var_cv in ignore_cv:
                        skip_increment = True
                start_value = start_values
                cv_id_list.append(var_cv)
                
            last_value = destination_anchor.variables[variable_key]
            if not skip_increment:
                increment = (last_value - start_value)/NUM_WINDOWS
                total_sq_distance += increment ** 2
            windows = np.linspace(start_value, last_value, NUM_WINDOWS)
            #windows = np.arange(start_value, last_value+0.0001*increment, 
            #                    increment)
            windows_list_unzipped.append(windows)
            
    var_string = hidr_base.make_var_string(destination_anchor)
    hidr_output_pdb_name = SMD_NAME.format(var_string)
    windows_list_zipped = list(zip(*windows_list_unzipped))
    
    
    timestep = get_timestep(model)
    distance = np.sqrt(total_sq_distance) * unit.nanometers
    steps_in_window = int(abs(distance) / (translation_velocity * timestep))
    assert steps_in_window > 0
    
    start_time = time.time()
    
    # Make the system and simulation
    energy_reporter_interval = max(steps_in_window // 10, 1)
    sim_openmm = HIDR_sim_openmm()
    system, topology, positions, box_vectors, num_frames \
        = common_sim_openmm.create_openmm_system(sim_openmm, model, 
                                                 source_anchor)
    sim_openmm.system = system
    time_step = add_integrator(sim_openmm, model)
    common_sim_openmm.add_platform(sim_openmm, model)
    forces = add_forces(
        sim_openmm, model, source_anchor, restraint_force_constant, 
        cv_id_list, windows_list_zipped[0], ignore_cv)
    add_simulation(sim_openmm, model, topology, positions, box_vectors, 
                   skip_minimization=True)
    handle_reporters(
        model, source_anchor, sim_openmm, 
        trajectory_reporter_interval=steps_in_window, 
        energy_reporter_interval=energy_reporter_interval, 
        smd_dcd_filename=smd_dcd_filename,
        smd_dcd_interval=smd_dcd_interval)
    for window_values in windows_list_zipped:
        print("running_window:", window_values, "steps_in_window:", steps_in_window)
        update_forces(
            sim_openmm, forces, model, source_anchor, restraint_force_constant, 
            cv_list=cv_id_list, window_values=window_values, 
            ignore_cv=ignore_cv)
        sim_openmm.simulation.context.reinitialize(preserveState=True)
        box_vectors = run_window(
            model, source_anchor, sim_openmm, restraint_force_constant, cv_id_list, 
            window_values, steps_in_window, smd_dcd_filename, smd_dcd_interval)
    
    total_time = time.time() - start_time
    simulation_in_ns = steps_in_window * len(list(windows_list_zipped)) \
        * timestep.value_in_unit(unit.picoseconds) * 1e-3
    total_time_in_days = total_time / 86400.0
    ns_per_day = simulation_in_ns / total_time_in_days
    print("Benchmark:", ns_per_day, "ns/day")
    
    # assign the new model attributes, and copy over the building files
    state = sim_openmm.simulation.context.getState(
        getPositions=True, enforcePeriodicBox=True)
    positions = state.getPositions()
    
    if source_anchor.__class__.__name__ in ["MMVT_toy_anchor"]:
        # Get current position
        new_positions = np.array([positions.value_in_unit(unit.nanometers)])
        print("new_positions:", new_positions)
        destination_anchor.starting_positions = new_positions
        
    else:
        destination_anchor.amber_params = deepcopy(source_anchor.amber_params)
        if destination_anchor.amber_params is not None:
            src_prmtop_filename = os.path.join(
                model.anchor_rootdir, source_anchor.directory, 
                source_anchor.building_directory,
                source_anchor.amber_params.prmtop_filename)
            dest_prmtop_filename = os.path.join(
                model.anchor_rootdir, destination_anchor.directory, 
                destination_anchor.building_directory,
                destination_anchor.amber_params.prmtop_filename)
            if os.path.exists(dest_prmtop_filename):
                os.remove(dest_prmtop_filename)
            copyfile(src_prmtop_filename, dest_prmtop_filename)
            #destination_anchor.amber_params.box_vectors = base.Box_vectors()
            #destination_anchor.amber_params.box_vectors.from_quantity(box_vectors)
            
        destination_anchor.forcefield_params = deepcopy(source_anchor.forcefield_params)
        if destination_anchor.forcefield_params is not None:
            pass
            # TODO: more here for forcefield
        destination_anchor.charmm_params = deepcopy(source_anchor.charmm_params)
        if destination_anchor.charmm_params is not None:
            pass
            # TODO: more here for charmm
    
        hidr_base.change_anchor_box_vectors(
            destination_anchor, box_vectors)
        
        hidr_base.change_anchor_pdb_filename(
            destination_anchor, hidr_output_pdb_name)
        
        output_pdb_file = os.path.join(
            model.anchor_rootdir, destination_anchor.directory,
            destination_anchor.building_directory,hidr_output_pdb_name)
        
        parm = parmed.openmm.load_topology(topology, system)
        parm.positions = positions
        parm.box_vectors = box_vectors
        
        print("saving new PDB file:", output_pdb_file)
        parm.save(output_pdb_file, overwrite=True)
    return

def run_RAMD_simulation(model, force_constant, source_anchor_index, 
                        destination_anchor_indices, lig_indices, rec_indices,
                        max_num_steps=1000000, traj_mode=False,
                        steps_per_RAMD_update=50, steps_per_anchor_check=250,
                        RAMD_cutoff_distance_nanometers=0.0025):
    """
    Run a random accelerated molecular dynamics (SMD) simulation 
    until every destination anchor index has been reached. The 
    resulting structure will be saved within the destination anchor 
    of the model's directory.
    
    Parameters
    ----------
    model : Model()
        The model object from the source anchor.
    force_constant : Quantity
        The value of the force constant used in the random force that
        pushes the system between anchors.
    destination_anchor_index : int
        The index, within the model, of the anchor which is the final
        destination of this SMD simulation.
        
    """
    #steps_per_RAMD_update = 50
    #steps_per_anchor_check = 250
    # If True, then remove any starting structures if anchors are skipped
    #  during RAMD.
    removing_starting_from_skipped_structures = True
    assert steps_per_anchor_check % steps_per_RAMD_update == 0, \
        "steps_per_anchor_check must be a multiple of steps_per_RAMD_update."
    RAMD_cutoff_distance = RAMD_cutoff_distance_nanometers * unit.nanometer
    trajectory_reporter_interval = 50
    energy_reporter_interval = 50
    source_anchor = model.anchors[source_anchor_index]
    sim_openmm = HIDR_sim_openmm()
    system, topology, positions, box_vectors, num_frames \
        = common_sim_openmm.create_openmm_system(sim_openmm, model, 
                                                 source_anchor)    
    sim_openmm.system = system
    time_step = add_integrator(sim_openmm, model)
    common_sim_openmm.add_platform(sim_openmm, model)
    
    source_directory = os.path.join(
        model.anchor_rootdir, source_anchor.directory, 
        source_anchor.building_directory)
    log_file_name = os.path.join(source_directory, RAMD_LOG_FILENAME)
    simulation = openmm_ramd.RAMDSimulation(
        topology, system, sim_openmm.integrator, force_constant, lig_indices, 
        rec_indices, ramdSteps=steps_per_RAMD_update, 
        rMinRamd=RAMD_cutoff_distance.value_in_unit(unit.angstroms),
        platform=sim_openmm.platform, properties=sim_openmm.properties, 
        log_file_name=log_file_name)
    
    simulation.RAMD_start()
    
    simulation.context.setPositions(positions)
    
    anchor_pdb_counters = []
    anchor_pdb_filenames = []
    for i, anchor in enumerate(model.anchors):
        anchor_pdb_counters.append(0)
        anchor_pdb_filenames.append([])
    
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(
            *box_vectors.to_quantity())
    
    sim_openmm.simulation = simulation
    handle_reporters(model, source_anchor, sim_openmm, 
                     trajectory_reporter_interval, 
                     energy_reporter_interval, 
                     traj_filename_base=RAMD_TRAJ_NAME)
    
    if model.using_toy():
        tolerance = 0.0
    else:
        tolerance = 0.0 #-0.001
    
    new_com = openmm_ramd.base.get_ligand_com(system, positions, lig_indices)
    start_time = time.time()
    counter = 0
    old_positions = None
    old_anchor_index = source_anchor_index
    found_bulk_state = False
    destination_anchor_index = source_anchor_index
    while counter < max_num_steps:
        simulation.RAMD_step(steps_per_RAMD_update)
        #old_com = new_com
        #simulation.step(steps_per_RAMD_update)
        state = simulation.context.getState(getPositions=True, 
                                            enforcePeriodicBox=True)
        positions = state.getPositions()
        if old_positions is None:
            old_positions = positions
        #new_com = openmm_ramd.base.get_ligand_com(system, positions, lig_indices)
        #com_com_distance = np.linalg.norm(old_com.value_in_unit(unit.nanometers) \
        #                              - new_com.value_in_unit(unit.nanometers))
        
        #if com_com_distance*unit.nanometers < RAMD_cutoff_distance:
        #    print("recomputing force at step:", counter)
        #    simulation.recompute_RAMD_force()
        
        found_anchor = False
        if counter % steps_per_anchor_check == 0:
            popping_indices = []
            for i, anchor in enumerate(model.anchors):
                in_anchor = True
                for milestone in anchor.milestones:
                    cv = model.collective_variables[milestone.cv_index]
                    # TODO fix error: chdir to model rootdir
                    curdir = os.getcwd()
                    os.chdir(model.anchor_rootdir)
                    
                    result = cv.check_openmm_context_within_boundary(
                        simulation.context, milestone.variables, positions, 
                        tolerance=tolerance) #tolerance=-0.001)
                    os.chdir(curdir)
                    if not result:
                        in_anchor = False
                
                if in_anchor:
                    assert not found_anchor, "Found system in two different "\
                        "anchors in the same step."
                    found_anchor = True
                    if i in destination_anchor_indices:
                        if i == destination_anchor_index:
                            continue
                        print("Entered anchor {}. step: {}".format(i, counter))
                        destination_anchor_index = i
                        destination_anchor = model.anchors[destination_anchor_index]
                        
                        if destination_anchor_index != source_anchor_index:
                            if not model.using_toy():
                                destination_anchor.amber_params = deepcopy(source_anchor.amber_params)
                                if destination_anchor.amber_params is not None:
                                    src_prmtop_filename = os.path.join(
                                        model.anchor_rootdir, source_anchor.directory, 
                                        source_anchor.building_directory,
                                        source_anchor.amber_params.prmtop_filename)
                                    dest_prmtop_filename = os.path.join(
                                        model.anchor_rootdir, destination_anchor.directory, 
                                        destination_anchor.building_directory,
                                        destination_anchor.amber_params.prmtop_filename)
                                    if os.path.exists(dest_prmtop_filename):
                                        os.remove(dest_prmtop_filename)
                                    copyfile(src_prmtop_filename, dest_prmtop_filename)
                                    #destination_anchor.amber_params.box_vectors = base.Box_vectors()
                                    #destination_anchor.amber_params.box_vectors.from_quantity(box_vectors)
                                    
                                destination_anchor.forcefield_params = deepcopy(source_anchor.forcefield_params)
                                if destination_anchor.forcefield_params is not None:
                                    raise Exception("forcefield not yet implemented for RAMD")
                                    # TODO: more here for forcefield
                                    
                                destination_anchor.charmm_params = deepcopy(source_anchor.charmm_params)
                                if destination_anchor.charmm_params is not None:
                                    raise Exception("charmm not yet implemented for RAMD")
                                    # TODO: more here for charmm
                            
                                hidr_base.change_anchor_box_vectors(
                                    destination_anchor, box_vectors.to_quantity())
                        
                        var_string = hidr_base.make_var_string(destination_anchor)
                        hidr_output_pdb_name = RAMD_NAME.format(var_string, 0)
                        
                        if model.using_toy():
                            new_positions = np.array([positions.value_in_unit(unit.nanometers)])
                            destination_anchor.starting_positions = new_positions
                            
                        else:
                            output_pdb_file = os.path.join(
                                model.anchor_rootdir, destination_anchor.directory,
                                destination_anchor.building_directory,
                                hidr_output_pdb_name)
                            
                            if not destination_anchor.bulkstate: # \
                                #    and not os.path.exists(output_pdb_file):
                                hidr_base.change_anchor_pdb_filename(
                                    destination_anchor, hidr_output_pdb_name)
                                parm = parmed.openmm.load_topology(topology, system)
                                parm.positions = positions
                                parm.box_vectors = box_vectors.to_quantity()
                                print("saving preliminary PDB file:", 
                                      output_pdb_file)
                                parm.save(output_pdb_file, overwrite=True)
                            
                        popping_indices.append(destination_anchor_index)
                        
                        old_anchor = model.anchors[old_anchor_index]
                        var_string = hidr_base.make_var_string(old_anchor)
                        
                        skipped_anchor_indices = list(range(
                            min(old_anchor_index, destination_anchor_index)+1, 
                            max(old_anchor_index, destination_anchor_index)))
                        
                        if model.using_toy():
                            prev_positions = np.array([old_positions.value_in_unit(unit.nanometers)])
                            old_anchor.starting_positions = prev_positions
                            assert not traj_mode, \
                                "Traj mode not currently allowed for toy systems."
                            if removing_starting_from_skipped_structures:
                                # Then remove all starting structures
                                for skipped_anchor_index in skipped_anchor_indices:
                                    print("removing starting structure from anchor:", skipped_anchor_index)
                                    skipped_anchor = model.anchors[skipped_anchor_index]
                                    skipped_anchor.starting_positions = None
                                    
                        else:
                            hidr_output_pdb_name = RAMD_NAME.format(var_string, anchor_pdb_counters[old_anchor_index])
                            hidr_base.change_anchor_pdb_filename(
                                old_anchor, hidr_output_pdb_name)
                            
                            output_pdb_file = os.path.join(
                                model.anchor_rootdir, old_anchor.directory,
                                old_anchor.building_directory, hidr_output_pdb_name)
                            
                            parm = parmed.openmm.load_topology(topology, system)
                            parm.positions = old_positions
                            parm.box_vectors = box_vectors.to_quantity()
                            print("saving previous anchor PDB file:", output_pdb_file)
                            parm.save(output_pdb_file, overwrite=True)
                            if traj_mode:
                                anchor_pdb_counters[old_anchor_index] += 1
                                anchor_pdb_filenames[old_anchor_index].append(output_pdb_file)
                            else:
                                anchor_pdb_filenames[old_anchor_index] = [output_pdb_file]
                                
                            if removing_starting_from_skipped_structures:
                                # Then remove all starting structures
                                for skipped_anchor_index in skipped_anchor_indices:
                                    print("removing starting structure from anchor:", skipped_anchor_index)
                                    skipped_anchor = model.anchors[skipped_anchor_index]
                                    hidr_base.change_anchor_pdb_filename(
                                        skipped_anchor, "")
                        
                        old_anchor_index = i
                        old_positions = positions
                    
                    if anchor.bulkstate:
                        found_bulk_state = True
                    
            #for popping_index in popping_indices:
            #    destination_anchor_indices.remove(popping_index)
                
            if len(destination_anchor_indices) == 0:
                # We've reached all destinations
                break
            
            if found_bulk_state:
                break
            
        counter += steps_per_RAMD_update
    
    total_time = time.time() - start_time
    simulation_in_ns = counter * time_step.value_in_unit(unit.picosecond)  * 1e-3
    total_time_in_days = total_time / (86400.0)
    ns_per_day = simulation_in_ns / total_time_in_days
    
    # TODO: add a check to make sure that anchors aren't crossed in RAMD - 
    # give warning about decreasing steps between RAMD evals, or not having 
    # so many anchor.
    
    for i, anchor in enumerate(model.anchors):
        if anchor.bulkstate:
            continue
        # TODO: fix hack
        if model.using_toy():
            if anchor.starting_positions is None:
                print("Warning: anchor {} has no starting positions."\
                      .format(i))
        else:
            if hidr_base.get_anchor_pdb_filename(anchor) == "":
                print("Warning: anchor {} has no starting PDB structures."\
                      .format(i))
    
    if traj_mode:
        for i, anchor in enumerate(model.anchors):
            if anchor.bulkstate or len(anchor_pdb_filenames[i]) == 0:
                continue
            directory = os.path.join(
                model.anchor_rootdir, anchor.directory, 
                anchor.building_directory)
            #pdb_file_list = anchor_pdb_filenames[i]
            #pdb_swarm_name = combine_pdb_files_into_traj(directory, pdb_file_list)
            var_string = hidr_base.make_var_string(anchor)
            pdb_swarm_name = RAMD_NAME.format(var_string, "swarm")
            
            os.chdir(directory)
            
            stride = 1
            if len(anchor_pdb_filenames[i]) > 10:
                stride = len(anchor_pdb_filenames[i]) // 10
            
            #traj = mdtraj.load(anchor_pdb_filenames[i][::-1])
            anchor_pdb_filenames_culled = anchor_pdb_filenames[i][::-1][::stride]
            print("anchor_pdb_filenames_culled", anchor_pdb_filenames_culled)
            exit()
            traj = mdtraj.load(anchor_pdb_filenames_culled)
            traj.save_pdb(pdb_swarm_name)
            for filename in anchor_pdb_filenames[i]:
                os.remove(filename)
            
            hidr_base.change_anchor_pdb_filename(anchor, pdb_swarm_name)
    
    return ns_per_day

def run_Metadyn_simulation(model, source_anchor_index, 
                           destination_anchor_indices,
                           max_num_steps=MAX_METADYN_STEPS,
                           steps_per_metadyn_update=250, 
                           steps_per_anchor_check=250,
                           metadyn_npoints=None, metadyn_sigma=None, 
                           metadyn_biasfactor=10.0, metadyn_height=1.0, 
                           ignore_cv=None, 
                           anchors_with_starting_structures=None):
    """
    Run a metadynamics  simulation 
    until every destination anchor index has been reached. The 
    resulting structures will be saved in each anchor
    
    Parameters
    ----------
    model : Model()
        The model object from the source anchor.
    
        
    """
    print(f"Running metadyn. steps_per_metadyn_update: {steps_per_metadyn_update}")
    print(f"steps_per_anchor_check: {steps_per_anchor_check}")
    print(f"metadyn_biasfactor: {metadyn_biasfactor}")
    print(f"metadyn_height: {metadyn_height}")
    # If True, then remove any starting structures if anchors are skipped
    #  during Metadyn.
    save_structures_as_we_go = False
    save_final_structure = True # Vs. saving the first structure when a new
    # anchor is encountered.
    save_plot = True
    anchor_positions = {}
    if anchors_with_starting_structures is None:
        visited_anchors = set()
    else:
        visited_anchors = set(anchors_with_starting_structures)
        
    if save_final_structure:
        removing_starting_from_skipped_structures = True
    else:
        removing_starting_from_skipped_structures = False
        
    assert steps_per_anchor_check % steps_per_metadyn_update == 0, \
        "steps_per_anchor_check must be a multiple of steps_per_RAMD_update."
    trajectory_reporter_interval = 10000
    energy_reporter_interval = 10000
    source_anchor = model.anchors[source_anchor_index]
    #destination_anchor = model.anchors[destination_anchor_index]
    sim_openmm = HIDR_sim_openmm()
    system, topology, positions, box_vectors, num_frames \
        = common_sim_openmm.create_openmm_system(sim_openmm, model, 
                                                 source_anchor)    
    sim_openmm.system = system
    time_step = add_integrator(sim_openmm, model)
    common_sim_openmm.add_platform(sim_openmm, model)
    
    source_directory = os.path.join(
        model.anchor_rootdir, source_anchor.directory, 
        source_anchor.building_directory)
    
    cv_id_list = []
    for variable_key in source_anchor.variables:
        var_name = variable_key.split("_")[0]
        var_cv = int(variable_key.split("_")[1])
        cv = model.collective_variables[var_cv]
        # the two anchors must share a common variable
        if variable_key in source_anchor.variables:
            if isinstance(cv, mmvt_voronoi_cv.MMVT_Voronoi_CV):
                var_child_cv = int(variable_key.split("_")[2])
                cv_id_list.append(var_child_cv)
            else:
                #start_value = cv.get_openmm_context_cv_value(None, positions, system)
                cv_id_list.append(var_cv)
            
    # Make Metadyn CV
    num_cvs = len(cv_id_list)
    if metadyn_npoints is None:
        metadyn_npoints = [DEFAULT_METADYN_NPOINTS] * num_cvs
    elif not (type(metadyn_npoints) == list):
        metadyn_npoints = [metadyn_npoints]
    
    if metadyn_sigma is None:
        metadyn_sigma = [DEFAULT_METADYN_SIGMA] * num_cvs
    elif not (type(metadyn_sigma) == list):
        metadyn_sigma = [metadyn_sigma]
        
    print(f"metadyn_npoints: {metadyn_npoints}")
    print(f"metadyn_sigma: {metadyn_sigma}")
    if save_final_structure:
        print("Saving structures from final time anchor is entered.")
    else:
        print("Saving structures from first time anchor is entered.")
    
    metadyn_cvs = add_metadyn_cvs(model, metadyn_sigma, metadyn_npoints, ignore_cv)
    
    metadyn_bias_dir = os.path.join(model.anchor_rootdir, METADYN_BIAS_DIR_NAME)
    if not os.path.exists(metadyn_bias_dir):
        os.mkdir(metadyn_bias_dir)
    
    meta = openmm_app.Metadynamics(
        system, variables=metadyn_cvs, temperature=model.temperature,
        biasFactor=metadyn_biasfactor, height=metadyn_height, 
        frequency=steps_per_metadyn_update, 
        saveFrequency=steps_per_metadyn_update, biasDir=metadyn_bias_dir)
    
    add_simulation(sim_openmm, model, topology, positions, box_vectors, 
                   skip_minimization=True)
    simulation = sim_openmm.simulation
    
    anchor_pdb_counters = []
    anchor_pdb_filenames = []
    for i, anchor in enumerate(model.anchors):
        anchor_pdb_counters.append(0)
        anchor_pdb_filenames.append([])
    
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(
            *box_vectors.to_quantity())
    
    handle_reporters(model, source_anchor, sim_openmm, 
                     trajectory_reporter_interval, 
                     energy_reporter_interval, 
                     traj_filename_base=METADYN_TRAJ_NAME)
    
    if model.using_toy():
        tolerance = 0.0
    else:
        tolerance = 0.0 #-0.001
    
    start_time = time.time()
    counter = 0
    old_positions = None
    old_anchor_index = source_anchor_index
    found_bulk_state = False
    destination_anchor_index = source_anchor_index
    while counter < max_num_steps:
        meta.step(simulation, steps_per_anchor_check)
        state = simulation.context.getState(getPositions=True, 
                                            enforcePeriodicBox=True)
        positions = state.getPositions()
        if old_positions is None:
            old_positions = positions
        
        found_anchor = False
        if counter % steps_per_anchor_check == 0:
            popping_indices = []
            for i, anchor in enumerate(model.anchors):
                in_anchor = True
                for milestone in anchor.milestones:
                    cv = model.collective_variables[milestone.cv_index]
                    # TODO fix error: chdir to model rootdir
                    curdir = os.getcwd()
                    os.chdir(model.anchor_rootdir)
                    
                    result = cv.check_openmm_context_within_boundary(
                        simulation.context, milestone.variables, positions, 
                        tolerance=tolerance) #tolerance=-0.001)
                    os.chdir(curdir)
                    if not result:
                        in_anchor = False
                
                if in_anchor:
                    assert not found_anchor, "Found system in two different "\
                        "anchors in the same step."
                    found_anchor = True
                    if i in destination_anchor_indices:
                        if i == destination_anchor_index:
                            continue
                        print("Entered anchor {}. step: {}".format(i, counter))
                        destination_anchor_index = i
                        destination_anchor = model.anchors[destination_anchor_index]
                        
                        if save_final_structure or not (destination_anchor_index in visited_anchors):
                            if destination_anchor_index != source_anchor_index:
                                if not model.using_toy():
                                    destination_anchor.amber_params = deepcopy(source_anchor.amber_params)
                                    if destination_anchor.amber_params is not None:
                                        src_prmtop_filename = os.path.join(
                                            model.anchor_rootdir, source_anchor.directory, 
                                            source_anchor.building_directory,
                                            source_anchor.amber_params.prmtop_filename)
                                        dest_prmtop_filename = os.path.join(
                                            model.anchor_rootdir, destination_anchor.directory, 
                                            destination_anchor.building_directory,
                                            destination_anchor.amber_params.prmtop_filename)
                                        if os.path.exists(dest_prmtop_filename):
                                            os.remove(dest_prmtop_filename)
                                        copyfile(src_prmtop_filename, dest_prmtop_filename)
                                        #destination_anchor.amber_params.box_vectors = base.Box_vectors()
                                        #destination_anchor.amber_params.box_vectors.from_quantity(box_vectors)
                                        
                                    destination_anchor.forcefield_params = deepcopy(source_anchor.forcefield_params)
                                    if destination_anchor.forcefield_params is not None:
                                        raise Exception("forcefield not yet implemented for RAMD")
                                        # TODO: more here for forcefield
                                        
                                    destination_anchor.charmm_params = deepcopy(source_anchor.charmm_params)
                                    if destination_anchor.charmm_params is not None:
                                        raise Exception("charmm not yet implemented for RAMD")
                                        # TODO: more here for charmm
                                
                                    hidr_base.change_anchor_box_vectors(
                                        destination_anchor, box_vectors.to_quantity())
                            
                            var_string = hidr_base.make_var_string(destination_anchor)
                            hidr_output_pdb_name = METADYN_NAME.format(var_string, 0)
                            
                            if model.using_toy():
                                new_positions = np.array([positions.value_in_unit(unit.nanometers)])
                                destination_anchor.starting_positions = new_positions
                                
                            else:
                                output_pdb_file = os.path.join(
                                    model.anchor_rootdir, destination_anchor.directory,
                                    destination_anchor.building_directory,
                                    hidr_output_pdb_name)
                                
                                if not destination_anchor.bulkstate:
                                    if save_structures_as_we_go:
                                        #    and not os.path.exists(output_pdb_file):
                                        hidr_base.change_anchor_pdb_filename(
                                            destination_anchor, hidr_output_pdb_name)
                                        parm = parmed.openmm.load_topology(topology, system)
                                        parm.positions = positions
                                        parm.box_vectors = box_vectors.to_quantity()
                                        print("saving preliminary PDB file:", 
                                              output_pdb_file)
                                        parm.save(output_pdb_file, overwrite=True)
                                    
                                    else:
                                        anchor_positions[destination_anchor_index] =\
                                            positions
                                            
                            if save_structures_as_we_go:
                                hidr_base.save_new_model(model, save_old_model=False)
                                
                            visited_anchors.add(destination_anchor_index)
                            
                            popping_indices.append(destination_anchor_index)
                            
                        old_anchor = model.anchors[old_anchor_index]
                        var_string = hidr_base.make_var_string(old_anchor)
                        
                        skipped_anchor_indices = list(range(
                            min(old_anchor_index, destination_anchor_index)+1, 
                            max(old_anchor_index, destination_anchor_index)))
                    
                        if save_final_structure or not (old_anchor_index in visited_anchors):
                            if model.using_toy():
                                prev_positions = np.array([old_positions.value_in_unit(unit.nanometers)])
                                old_anchor.starting_positions = prev_positions
                                if removing_starting_from_skipped_structures:
                                    # Then remove all starting structures
                                    for skipped_anchor_index in skipped_anchor_indices:
                                        print("removing starting structure from anchor:", skipped_anchor_index)
                                        skipped_anchor = model.anchors[skipped_anchor_index]
                                        skipped_anchor.starting_positions = None
                                        
                            else:
                                if save_structures_as_we_go:
                                    hidr_output_pdb_name = METADYN_NAME.format(var_string, anchor_pdb_counters[old_anchor_index])
                                    hidr_base.change_anchor_pdb_filename(
                                        old_anchor, hidr_output_pdb_name)
                                    output_pdb_file = os.path.join(
                                        model.anchor_rootdir, old_anchor.directory,
                                        old_anchor.building_directory, hidr_output_pdb_name)
                                    
                                    parm = parmed.openmm.load_topology(topology, system)
                                    parm.positions = old_positions
                                    parm.box_vectors = box_vectors.to_quantity()
                                    print("saving previous anchor PDB file:", output_pdb_file)
                                    parm.save(output_pdb_file, overwrite=True)
                                    anchor_pdb_filenames[old_anchor_index] = [output_pdb_file]
                                
                                else:
                                    anchor_positions[old_anchor_index] =\
                                            old_positions
                                
                                if removing_starting_from_skipped_structures:
                                    # Then remove all starting structures
                                    for skipped_anchor_index in skipped_anchor_indices:
                                        print("removing starting structure from anchor:", skipped_anchor_index)
                                        skipped_anchor = model.anchors[skipped_anchor_index]
                                        hidr_base.change_anchor_pdb_filename(
                                            skipped_anchor, "")
                                        
                            if save_structures_as_we_go:
                                hidr_base.save_new_model(model, save_old_model=False)
                        
                        visited_anchors.add(old_anchor_index)
                        
                        old_anchor_index = i
                        old_positions = positions
                    
                    if anchor.bulkstate:
                        found_bulk_state = True
                    
            #for popping_index in popping_indices:
            #    destination_anchor_indices.remove(popping_index)
                
            if len(destination_anchor_indices) == 0:
                # We've reached all destinations
                break
            
            if found_bulk_state:
                break
            
        counter += steps_per_anchor_check
    
    total_time = time.time() - start_time
    simulation_in_ns = counter * time_step.value_in_unit(unit.picosecond)  * 1e-3
    total_time_in_days = total_time / (86400.0)
    ns_per_day = simulation_in_ns / total_time_in_days
    
    # Save structures
    if not save_structures_as_we_go:
        for anchor_index in anchor_positions:
            anchor = model.anchors[anchor_index]
            positions = anchor_positions[anchor_index]
            var_string = hidr_base.make_var_string(anchor)
            hidr_output_pdb_name = METADYN_NAME.format(var_string, 0)
            hidr_base.change_anchor_pdb_filename(
                anchor, hidr_output_pdb_name)
            
            output_pdb_file = os.path.join(
                model.anchor_rootdir, anchor.directory,
                anchor.building_directory, hidr_output_pdb_name)
            
            parm = parmed.openmm.load_topology(topology, system)
            parm.positions = positions
            parm.box_vectors = box_vectors.to_quantity()
            print("saving anchor PDB file:", output_pdb_file)
            parm.save(output_pdb_file, overwrite=True)
            anchor_pdb_filenames[anchor_index] = [output_pdb_file]
    
    for i, anchor in enumerate(model.anchors):
        if anchor.bulkstate:
            continue
        # TODO: fix hack
        if model.using_toy():
            if anchor.starting_positions is None:
                print("Warning: anchor {} has no starting positions."\
                      .format(i))
        else:
            if hidr_base.get_anchor_pdb_filename(anchor) == "":
                print("Warning: anchor {} has no starting PDB structures."\
                      .format(i))
    
    if save_plot:
        image_directory = common_analyze.make_image_directory(
            model, None)
        print(f"saving metadynamics plot to directory: {image_directory}")
        dG = meta.getFreeEnergy().value_in_unit(kcal_per_mol)
        anchor_indices = np.zeros(len(model.anchors), dtype=np.int8)
        anchor_values = np.zeros(len(model.anchors), dtype=np.float64)
        if len(model.anchors[0].variables) > 1:
            # cannot make plots for multidimensional CVs
            return
        for i, anchor in enumerate(model.anchors):
            anchor_indices[i] = anchor.index
            anchor_values[i] = list(anchor.variables.values())[0]
        
        x = np.linspace(anchor_values[0], anchor_values[-1], 
                        DEFAULT_METADYN_NPOINTS)
        pi_fig, ax = plt.subplots()
        plt.plot(x, dG)
        plt.xticks(anchor_values, anchor_values, rotation=90)
        plt.ylabel("Metadynamics \u0394G (kJ/mol)")
        plt.xlabel("anchor value")
        plt.title("Metadynamics Free Energy Profile")
        plt.tight_layout()
        pi_fig.savefig(os.path.join(image_directory, "metadynamics.png"))
    
    return ns_per_day