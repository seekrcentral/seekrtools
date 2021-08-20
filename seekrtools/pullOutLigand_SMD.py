"""
This script uses steered molecular dynamics (SMD) in order to pull a ligand
out of a binding site, saving structures along the way for input to SEEKR2.
"""

import sys
import time

import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as unit
import numpy as np
import parmed

#########################
# MODIFY THESE VARIABLES
#########################

rec_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
                31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
                45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
                59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 
                73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 
                87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 
                101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 
                112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 
                123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 
                134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 
                145, 146]
lig_indices = [147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 
                158, 159, 160, 161]

prmtop_filename = "equilibrated.parm7"
inpcrd_filename = "equilibrated.rst7"
pdb_filename = "equilibrated.pdb"

temperature = 298.15*unit.kelvin
spring_constant = 90000.0*unit.kilojoules_per_mole / unit.nanometer**2
cuda_device_index = "0"
nonbonded_cutoff = 0.9*unit.nanometer

# time step of simulation 
time_step = 0.002 * unit.picoseconds

#optionally print a trajectory
trajectory_filename = "smd_trajectory.pdb"
trajectory_interval = 500000

# total number of timesteps to take in the CMD simulation
total_num_steps = 50000000 # 100 ns

# How many "windows" from the bound state to the unbound state
num_windows = 100
show_state_output = False

# Radii of anchor locations. Warning: don't include zero, nor the lowest anchor
#  because the structure defined by pdb_filename should already be the lowest 
#  bound state.
target_radii = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 
                1.15, 1.25, 1.35]

########################################################
# DO NOT MODIFY BELOW UNLESS YOU KNOW WHAT YOU'RE DOING
########################################################

state_filename = "state.xml"
basename = "smd_at"
steps_per_window = total_num_steps // num_windows

prmtop = app.AmberPrmtopFile(prmtop_filename)
inpcrd = app.AmberInpcrdFile(inpcrd_filename)
mypdb = app.PDBFile(pdb_filename)

parmed_struct = parmed.load_file(prmtop_filename, xyz=pdb_filename)

def get_lig_rec_distance(parmed_struct, positions, lig_atom_list, rec_atom_list):
    parmed_struct.coordinates = positions
    center_of_mass_1 = np.array([[0., 0., 0.]])
    total_mass = 0.0
    for atom_index in lig_atom_list:
        atom_pos = parmed_struct.coordinates[atom_index,:]
        atom_mass = parmed_struct.atoms[atom_index].mass
        center_of_mass_1 += atom_mass * atom_pos
        total_mass += atom_mass
    center_of_mass_1 = center_of_mass_1 / total_mass
    
    center_of_mass_2 = np.array([[0., 0., 0.]])
    total_mass = 0.0
    for atom_index in rec_atom_list:
        atom_pos = parmed_struct.coordinates[atom_index,:]
        atom_mass = parmed_struct.atoms[atom_index].mass
        center_of_mass_2 += atom_mass * atom_pos
        total_mass += atom_mass
    center_of_mass_2 = center_of_mass_2 / total_mass
    distance = 0.1*np.linalg.norm(center_of_mass_2 - center_of_mass_1)
    return distance

def run_window(target_radius, save_state_filename, index):
    system = prmtop.createSystem(
        nonbondedMethod=app.PME, nonbondedCutoff=nonbonded_cutoff, 
        constraints=app.HBonds)
    myforce1 = mm.CustomCentroidBondForce(
        2, "0.5*k*(distance(g1, g2) - radius)^2")
    mygroup1a = myforce1.addGroup(rec_indices)
    mygroup2a = myforce1.addGroup(lig_indices)
    myforce1.setForceGroup(1)
    myforce1.addPerBondParameter("k")
    myforce1.addPerBondParameter("radius")
    myforce1.addBond([mygroup1a, mygroup2a], [spring_constant, 
                                              target_radius*unit.nanometers])
    forcenum1 = system.addForce(myforce1)

    integrator = mm.LangevinIntegrator(temperature, 1/unit.picosecond, 
        time_step)
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {"CudaDeviceIndex": cuda_device_index, "CudaPrecision": "mixed"}
    simulation = app.Simulation(prmtop.topology, system, integrator, platform, 
                            properties)
                            
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    
    if trajectory_filename and trajectory_interval:
        simulation.reporters.append(pdb_reporter)
    
    if index == 0:
        simulation.context.setPositions(mypdb.positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.minimizeEnergy()
        
    else:
        simulation.loadState(save_state_filename)
        
    if show_state_output:
        simulation.reporters.append(app.StateDataReporter(sys.stdout, steps_per_window, step=True,
            potentialEnergy=True, temperature=True, volume=True))
    
    
    simulation.step(steps_per_window)
    state = simulation.context.getState(getPositions = True, 
                                        getVelocities = True,
                                        enforcePeriodicBox = True)
    positions = state.getPositions()
    distance = get_lig_rec_distance(parmed_struct, positions, lig_indices, rec_indices)
    simulation.saveState(state_filename)
    return distance, positions
    
if trajectory_filename and trajectory_interval:
    pdb_reporter = app.PDBReporter(trajectory_filename, trajectory_interval)

start_radius = get_lig_rec_distance(parmed_struct, mypdb.positions, lig_indices, rec_indices)
print("start_radius:", start_radius)
start_window = start_radius
last_window = target_radii[-1] + 0.1
increment = (last_window - start_window)/num_windows
print("simulating steered MD in windows from", start_window, "to", last_window, 
    "in increments of", increment)
windows = np.arange(start_window, last_window, increment)
old_distance = start_radius
distance = start_radius
goal_radius_index = 0

for i, window_radius in enumerate(windows):
    print("running window:", window_radius)
    if goal_radius_index >= len(target_radii):
        break
    goal_radius = target_radii[goal_radius_index]
    assert goal_radius > start_radius, "Error: your system is starting with "\
        "a ligand-receptor distance of %f, but a target radius is "\
        "listed at a distance of %f. This may never be reached. Please choose "\
        "a list of target radii that are larger than the starting ligand-"\
        "receptor distance."
    old_distance = distance
    distance, positions = run_window(window_radius, state_filename, i)
    print("current ligand-receptor distance:", distance)
    if (distance-goal_radius)*(old_distance-goal_radius) < 0:
        # then a crossing event has taken place
        print("radius crossed:", goal_radius, "saving state")
        amber_parm = parmed.amber.AmberParm(prmtop_filename, inpcrd_filename)
        amber_parm.positions = positions
        pdb_save_filename = basename+"%.2f.pdb" % goal_radius
        amber_parm.save(pdb_save_filename, overwrite=True)
        goal_radius_index += 1
    
