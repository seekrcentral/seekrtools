"""
This sample script provides a template that one can use to minimize and
equilibrate a molecular system in molecular dynamics (MD) for a SEEKR2
calculation.

In this script, the system is minimized, and then allowed to equilibrate
while the ligand is kept restrained inside the binding site.
"""

import time
from sys import stdout

import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as unit
import parmed
import mdtraj
import numpy as np

def get_site_ligand_distance(pdb_filename, group1, group2):
    """
    Compute the distance between the centers of masses of two groups of
    atom indices (group1 and group2) in the pdb_filename structure.
    """
    traj = mdtraj.load(pdb_filename)
    traj1 = traj.atom_slice(group1)
    traj2 = traj.atom_slice(group2)
    com1_array = mdtraj.compute_center_of_mass(traj1)
    com2_array = mdtraj.compute_center_of_mass(traj2)
    com1 = com1_array[0,:]
    com2 = com2_array[0,:]
    distance = np.linalg.norm(com2-com1)
    return distance * unit.nanometer

#########################
# MODIFY THESE VARIABLES
#########################

# These are the files produced by the LEAP script
prmtop_filename = "molecule.prmtop"
inpcrd_filename = "molecule.inpcrd"
input_pdb_file = "molecule.pdb"

# Output equilibration trajectory
trajectory_filename = "equilibration_trajectory.pdb"

# The interval between updates to the equilibration trajectory
steps_per_trajectory_update = 300000

# Final structure output
output_pdb_file = "equilibrated.pdb"

# Whether to minimize
minimize = True

# The total number of equilibration MD steps to take
num_steps = 30000000 # 60 nanoseconds

# The interval between energy printed to standard output
steps_per_energy_update = 5000

# time step of simulation 
time_step = 0.002 * unit.picoseconds

# Enter the atom indices whose center of mass defines the receptor binding site
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
                
# Enter the atom indices of the ligand molecule
lig_indices = [147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 
                158, 159, 160, 161]
                
# To hold the ligand in place during the equilibration, a harmonic force 
#  keeps the center of mass of the ligand and binding site at a constant
#  distance
spring_constant = 9000.0 * unit.kilojoules_per_mole * unit.nanometers**2

# simulation initial and target temperature
temperature = 298.15 * unit.kelvin

# If constant pressure is desired
constant_pressure = True
target_pressure = 1.0 * unit.bar

# Define which GPU to use
cuda_index = "0"

# Nonbonded cutoff
nonbonded_cutoff = 0.9 * unit.nanometer

########################################################
# DO NOT MODIFY BELOW UNLESS YOU KNOW WHAT YOU'RE DOING
########################################################

target_distance = get_site_ligand_distance(input_pdb_file, rec_indices, 
                                           lig_indices)

print("Starting ligand-site distance:", target_distance)

tmp_prmtop_filename = "molecule_TMP.parm7"
tmp_inpcrd_filename = "molecule_TMP.rst7"
inpcrd_final = "equilibrated.rst7"

# parmed can process the prmtop/inpcrd to get ready for OpenMM
amber_parm = parmed.amber.AmberParm(prmtop_filename, inpcrd_filename)
amber_parm.save(tmp_prmtop_filename, overwrite=True)
amber_parm.save(tmp_inpcrd_filename, overwrite=True)

prmtop = app.AmberPrmtopFile(tmp_prmtop_filename)
inpcrd = app.AmberInpcrdFile(tmp_inpcrd_filename)
mypdb = app.PDBFile(input_pdb_file)
system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=nonbonded_cutoff,
        constraints=app.HBonds)
myforce1 = mm.CustomCentroidBondForce(
    2, "0.5*k*(distance(g1, g2) - radius)^2")
mygroup1a = myforce1.addGroup(rec_indices)
mygroup2a = myforce1.addGroup(lig_indices)
myforce1.setForceGroup(1)
myforce1.addPerBondParameter("k")
myforce1.addPerBondParameter("radius")
myforce1.addBond([mygroup1a, mygroup2a], [spring_constant, target_distance])
forcenum1 = system.addForce(myforce1)
if constant_pressure:
    barostat = mm.MonteCarloBarostat(target_pressure, temperature, 25)
    
system.addForce(barostat)
integrator = mm.LangevinIntegrator(temperature, 1/unit.picosecond, time_step)
platform = mm.Platform.getPlatformByName('CUDA')
properties = {"CudaDeviceIndex": cuda_index, "CudaPrecision": "mixed"}
simulation = app.Simulation(prmtop.topology, system, integrator, platform, properties)
simulation.context.setPositions(mypdb.positions)
simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
if minimize:
    simulation.minimizeEnergy()
    
simulation.context.setVelocitiesToTemperature(temperature)
simulation.reporters.append(app.StateDataReporter(stdout, steps_per_energy_update, step=True,
            potentialEnergy=True, temperature=True, volume=True))
pdb_reporter = app.PDBReporter(trajectory_filename, steps_per_trajectory_update)
simulation.reporters.append(pdb_reporter)
start_time = time.time()
simulation.step(num_steps)
total_time = time.time() - start_time

simulation_in_ns = num_steps * time_step.value_in_unit(unit.picoseconds) * 1e-3
total_time_in_days = total_time / (86400.0)
ns_per_day = simulation_in_ns / total_time_in_days
print("Equilibration benchmark:", ns_per_day, "ns/day")

state = simulation.context.getState(getPositions = True, getVelocities = True, enforcePeriodicBox = True)
positions = state.getPositions()
amber_parm = parmed.amber.AmberParm(prmtop_filename, inpcrd_filename)
amber_parm.positions = positions
amber_parm.box_vectors = state.getPeriodicBoxVectors()
amber_parm.save(inpcrd_final, overwrite=True)
amber_parm.save(output_pdb_file, overwrite=True)

end_distance = get_site_ligand_distance(output_pdb_file, rec_indices, 
                                                         lig_indices)
print("Final ligand-site distance:", end_distance)

