"""
Run minimizations and equilibrations for a given structure in an anchor.
"""

import sys
import os
import time
from shutil import copyfile
from copy import deepcopy

import numpy as np
import parmed

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

import seekr2.modules.common_base as base
import seekr2.modules.common_sim_openmm as common_sim_openmm

import seekrtools.hidr.hidr_base as hidr_base

NUM_WINDOWS=10
NUM_EQUIL_FRAMES=10
EQUIL_UPDATE_INTERVAL=5000
EQUILIBRATED_NAME = "hidr_equilibrated.pdb"
SMD_NAME = "hidr_smd_at_{}.pdb"


class HIDR_sim_openmm(common_sim_openmm.Common_sim_openmm):
    """
    system : The OpenMM system object for this simulation.
    
    integrator : the OpenMM integrator object for this simulation.
    
    simulation : the OpenMM simulation object.
    
    traj_reporter : openmm.DCDReporter
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
        topology.topology, sim_openmm.system, 
        sim_openmm.integrator, sim_openmm.platform, 
        sim_openmm.properties)
    
    if positions is not None:
        sim_openmm.simulation.context.setPositions(positions.positions)
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
                     energy_reporter_interval):
    """
    If relevant, add the necessary state and trajectory reporters to
    the simulation object.
    """
    directory = os.path.join(
        model.anchor_rootdir, anchor.directory, anchor.production_directory)
    traj_filename = os.path.join(directory, "hidr.dcd")
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
        for the collective variable describine this variable. In fact,
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
    myforce = cv.make_restraining_force()
    myforce.setForceGroup(1)
    variables_names_list = cv.add_parameters(myforce)
    cv.add_groups_and_variables(myforce, variables_values_list)
    return myforce

def add_forces(sim_openmm, model, anchor, restraint_force_constant, 
               cv_list=None, window_values=None):
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
    if cv_list is not None and window_values is not None:
        for cv_index, value in zip(cv_list, window_values):
            value_dict[cv_index] = value
    
    for variable_key in anchor.variables:
        var_name = variable_key.split("_")[0]
        var_cv = int(variable_key.split("_")[1])
        if cv_list is not None:
            if var_cv not in cv_list:
                continue
        
        if window_values is None:
            var_value = anchor.variables[variable_key]
        else:
            var_value = value_dict[var_cv]
        
        
        cv = model.collective_variables[var_cv]
        cv_variables = cv.get_variable_values()
        variables_values_list = [1] + cv_variables \
            + [restraint_force_constant, var_value]
        myforce = make_restraining_force(cv, variables_values_list)
        forcenum = sim_openmm.system.addForce(myforce)
        
    return

def add_barostat(sim_openmm, model):
    """
    Optionally add a barostat to the simulation to maintain constant
    pressure.
    """
    barostat = openmm.MonteCarloBarostat(
        1.0*unit.bar,
        model.openmm_settings.langevin_integrator.target_temperature, 25)
    sim_openmm.system.addForce(barostat)

def run_min_equil_anchor(model, anchor_index, equilibration_steps, 
                         skip_minimization, restraint_force_constant):
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
    
    # TODO: fill out these quantities
    trajectory_reporter_interval = equilibration_steps // NUM_EQUIL_FRAMES
    energy_reporter_interval = EQUIL_UPDATE_INTERVAL
    total_number_of_steps = equilibration_steps
    
    anchor = model.anchors[anchor_index]
    sim_openmm = HIDR_sim_openmm()
    system, topology, positions, box_vectors \
        = common_sim_openmm.create_openmm_system(sim_openmm, model, anchor)
    sim_openmm.system = system
    time_step = add_integrator(sim_openmm, model)
    add_barostat(sim_openmm, model)
    common_sim_openmm.add_platform(sim_openmm, model)
    add_forces(sim_openmm, model, anchor, restraint_force_constant)
    add_simulation(sim_openmm, model, topology, positions, box_vectors, 
                   skip_minimization)
    handle_reporters(model, anchor, sim_openmm, trajectory_reporter_interval, 
                     energy_reporter_interval)
    
    
    output_pdb_file = os.path.join(
        model.anchor_rootdir, anchor.directory, anchor.building_directory,
        EQUILIBRATED_NAME)
    
    start_time = time.time()
    sim_openmm.simulation.step(total_number_of_steps)
    total_time = time.time() - start_time
    
    state = sim_openmm.simulation.context.getState(
        getPositions = True, getVelocities = False, enforcePeriodicBox = True)
    positions = state.getPositions()
    #amber_parm = parmed.load_file(prmtop_filename, inpcrd_filename)
    
    # TODO: this might cause errors with older versions of parmed. 
    # adapt the openmm import if necessary.
    parm = parmed.openmm.load_topology(topology.topology, system)
    parm.positions = positions
    parm.box_vectors = state.getPeriodicBoxVectors()
    parm.save(output_pdb_file, overwrite=True)
    
    hidr_base.change_anchor_box_vectors(anchor, state.getPeriodicBoxVectors())
    hidr_base.change_anchor_pdb_filename(anchor, EQUILIBRATED_NAME)

    simulation_in_ns = total_number_of_steps * time_step.value_in_unit(
        unit.picoseconds) * 1e-3
    total_time_in_days = total_time / (86400.0)
    ns_per_day = simulation_in_ns / total_time_in_days
    
    return ns_per_day

def run_window(model, anchor, restraint_force_constant, cv_list, window_values,
               steps_in_window):
    """
    Run the window of an SMD simulation, where the restraint equilibrium
    value has been moved incrementally to a new position for a short
    time to draw the system towards a new anchor.
    """
    energy_reporter_interval = None
    total_number_of_steps = steps_in_window
    
    sim_openmm = HIDR_sim_openmm()
    system, topology, positions, box_vectors \
        = common_sim_openmm.create_openmm_system(sim_openmm, model, 
                                                 anchor)
    sim_openmm.system = system
    time_step = add_integrator(sim_openmm, model)
    common_sim_openmm.add_platform(sim_openmm, model)
    add_forces(sim_openmm, model, anchor, restraint_force_constant, cv_list,
               window_values)
    add_simulation(sim_openmm, model, topology, positions, box_vectors, 
                   skip_minimization=True)
    handle_reporters(
        model, anchor, sim_openmm, trajectory_reporter_interval=None, 
        energy_reporter_interval=energy_reporter_interval)
    sim_openmm.simulation.step(total_number_of_steps)
    state = sim_openmm.simulation.context.getState(
        getPositions = True, getVelocities = False, enforcePeriodicBox = True)
    positions = state.getPositions()
    box_vectors = state.getPeriodicBoxVectors()
    return system, topology, positions, box_vectors
    

def run_SMD_simulation(model, source_anchor_index, destination_anchor_index, 
                         restraint_force_constant, translation_velocity):
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
    
    cv_id_list = []
    windows_list_unzipped = []
    
    var_list = []
    for variable_key in source_anchor.variables:
        var_name = variable_key.split("_")[0]
        var_cv = int(variable_key.split("_")[1])
        # the two anchors must share a common variable
        if variable_key in destination_anchor.variables:
            start_value = source_anchor.variables[variable_key]
            last_value = destination_anchor.variables[variable_key]
            increment = (last_value - start_value)/NUM_WINDOWS
            windows = np.arange(start_value, last_value+0.0001*increment, 
                                increment)
            windows_list_unzipped.append(windows)
            cv_id_list.append(var_cv)
        var_list.append("{:.3f}".format(last_value))
        
    var_string = "_".join(var_list)
    hidr_output_pdb_name = SMD_NAME.format(var_string)
    windows_list_zipped = zip(*windows_list_unzipped)
    
    timestep = get_timestep(model)
    distance = increment * unit.nanometers
    steps_in_window = int(abs(distance) / (translation_velocity * timestep))
    assert steps_in_window > 0
    
    for i, window_values in enumerate(windows_list_zipped):
        print("running_window:", window_values, "steps_in_window:", steps_in_window)
        system, topology, positions, box_vectors = run_window(
            model, source_anchor, restraint_force_constant, cv_id_list, 
            window_values, steps_in_window)
    
    # assign the new model attributes, and copy over the building files
    
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
    
    parm = parmed.openmm.load_topology(topology.topology, system)
    parm.positions = positions
    parm.box_vectors = box_vectors
    print("saving new PDB file:", output_pdb_file)
    parm.save(output_pdb_file, overwrite=True)
    return
    
    