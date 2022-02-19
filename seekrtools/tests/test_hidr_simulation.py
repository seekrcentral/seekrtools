"""
test_hidr_simulation.py
"""

import os
import pytest

import numpy as np
try:
    import openmm.unit as unit
except ModuleNotFoundError:
    import simtk.unit as unit
import seekr2.modules.common_sim_openmm as common_sim_openmm

import seekrtools.hidr.hidr_simulation as hidr_simulation

def set_model_to_reference(model):
    """
    Set the model to use the reference platform for simulations.
    """
    model.openmm_settings.cuda_platform_settings = None
    model.openmm_settings.reference_platform = True
    return

def create_sim(model):
    """
    Tests a number of different functions within hidr_simulation.
    """
    test_steps = 10000
    trajectory_reporter_interval = 1000
    energy_reporter_interval = 1000
    total_number_of_steps = test_steps
    anchor_index = 0
    restraint_force_constant = 900.0
    skip_minimization = False
    anchor = model.anchors[anchor_index]
    sim_openmm = hidr_simulation.HIDR_sim_openmm()
    system, topology, positions, box_vectors, num_frames \
        = common_sim_openmm.create_openmm_system(
            sim_openmm, model, anchor)
    sim_openmm.system = system
    time_step = hidr_simulation.add_integrator(sim_openmm, model)
    hidr_simulation.add_barostat(sim_openmm, model)
    common_sim_openmm.add_platform(sim_openmm, model)
    hidr_simulation.add_forces(
        sim_openmm, model, anchor, restraint_force_constant)
    hidr_simulation.add_simulation(
        sim_openmm, model, topology, positions, box_vectors, skip_minimization)
    hidr_simulation.handle_reporters(
        model, anchor, sim_openmm, trajectory_reporter_interval, 
        energy_reporter_interval)
    output_pdb_file = os.path.join(
        model.anchor_rootdir, anchor.directory, anchor.building_directory,
        "hidr_equilibrated.pdb")
    return sim_openmm

def test_create_sim(host_guest_mmvt_model):
    """
    Merely create a simulation object without simulating to see how
    it goes.
    """
    sim_openmm = create_sim(host_guest_mmvt_model)
    return

def test_get_timestep(host_guest_mmvt_model):
    """
    Test the get_timestep function.
    """
    timestep = hidr_simulation.get_timestep(host_guest_mmvt_model)
    expected_timestep = 0.002*unit.picoseconds
    assert np.isclose(timestep.value_in_unit(unit.picoseconds), 
                      expected_timestep.value_in_unit(unit.picoseconds))
    return

def test_run_min_equil_reference(host_guest_mmvt_model):
    """
    Test running a minimization-equilibration simulation.
    """
    anchor_index = 0
    equilibration_steps = 10
    skip_minimization = False
    restraint_force_constant = 900.0
    set_model_to_reference(host_guest_mmvt_model)
    hidr_simulation.run_min_equil_anchor(
        host_guest_mmvt_model, anchor_index, equilibration_steps, 
        skip_minimization, restraint_force_constant)
    return

@pytest.mark.needs_cuda
def test_run_min_equil_cuda(host_guest_mmvt_model):
    """
    Test running a minimization-equilibration simulation.
    """
    anchor_index = 0
    equilibration_steps = 100
    skip_minimization = False
    restraint_force_constant = 900.0
    hidr_simulation.run_min_equil_anchor(
        host_guest_mmvt_model, anchor_index, equilibration_steps, 
        skip_minimization, restraint_force_constant)
    return

def test_run_SMD_simulation_reference(host_guest_mmvt_model):
    """
    Run a test SMD simulation.
    """
    source_anchor_index = 0
    destination_anchor_index = 1
    equilibration_steps = 10
    skip_minimization = False
    restraint_force_constant = 900.0
    translation_velocity = 1000.0 * unit.nanometers / unit.nanoseconds
    set_model_to_reference(host_guest_mmvt_model)
    hidr_simulation.run_SMD_simulation(
        host_guest_mmvt_model, source_anchor_index, destination_anchor_index, 
        restraint_force_constant, translation_velocity)
    return