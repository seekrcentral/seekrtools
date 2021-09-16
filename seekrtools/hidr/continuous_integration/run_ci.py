"""
Run full HIDR calculations to test for problems in the pipeline.
"""
import os
import sys
import time
import tempfile

import openmm.unit as unit
import seekr2.tests.create_model_input as create_model_input
import seekr2.prepare as prepare
import seekr2.modules.check as check
from seekr2.modules.common_prepare import Browndye_settings_input, \
    MMVT_input_settings, Elber_input_settings
from seekr2.modules.common_base import Ion, Amber_params, Forcefield_params, \
    Box_vectors
from seekr2.modules.common_cv import Spherical_cv_anchor, Spherical_cv_input

import seekrtools.hidr.hidr as hidr

def create_host_guest_mmvt_model_input(temp_dir):
    """
    
    """
    host_guest_mmvt_model_input_persisent_obj \
        = create_model_input.create_host_guest_mmvt_model_input(
            temp_dir, bd=True)
    for i, input_anchor in enumerate(host_guest_mmvt_model_input_persisent_obj\
            .cv_inputs[0].input_anchors[1:]):
        if input_anchor.starting_amber_params is not None:
            input_anchor.starting_amber_params.pdb_coordinates_filename = ""
        if input_anchor.starting_forcefield_params is not None:
            input_anchor.starting_forcefield_params.pdb_coordinates_filename = ""
    return host_guest_mmvt_model_input_persisent_obj

def create_sod_mmvt_model_input(temp_dir):
    """
    
    """
    sod_mmvt_model_input_persisent_obj \
        = create_model_input.create_sod_mmvt_model_input(
            temp_dir)
    for i, cv_input in enumerate(
            sod_mmvt_model_input_persisent_obj.cv_inputs):
        for j, input_anchor in enumerate(cv_input.input_anchors):
            if j == 0:
                # bound states
                continue
            if input_anchor.starting_amber_params is not None:
                input_anchor.starting_amber_params.pdb_coordinates_filename = ""
            if input_anchor.starting_forcefield_params is not None:
                input_anchor.starting_forcefield_params.pdb_coordinates_filename = ""

    return sod_mmvt_model_input_persisent_obj

def run_short_ci(model_input, cuda_device_index):
    """
    
    """
    start_dir = os.getcwd()
    model, xml_path = prepare.generate_seekr2_model_and_filetree(
        model_input, force_overwrite=False)
    
    model_dir = os.path.dirname(xml_path)
    model.anchor_rootdir = os.path.abspath(model_dir)
    #check.check_pre_simulation_all(model)
    hidr.hidr(model, "any", dry_run=False, equilibration_steps=1000,  
         translation_velocity=1.0*unit.nanometers/unit.nanoseconds)
    os.chdir(start_dir)
    return

def run_generic_hostguest_ci(cuda_device_index):
    with tempfile.TemporaryDirectory() as temp_dir:
        host_guest_model_input \
            = create_host_guest_mmvt_model_input(temp_dir)
        run_short_ci(host_guest_model_input, cuda_device_index)
    return

def run_multisite_sod_ci(cuda_device_index):
    with tempfile.TemporaryDirectory() as temp_dir:
        sod_model_input  = create_sod_mmvt_model_input(temp_dir)
        run_short_ci(sod_model_input, cuda_device_index)
    return

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        argument = "short"
    else:
        argument = sys.argv[1]
        
    if len(sys.argv) == 3:
        cuda_device_index = sys.argv[2]
    else:
        cuda_device_index = None
    
    starttime = time.time()
    if argument == "short":
        run_generic_hostguest_ci(cuda_device_index)
        
    elif argument == "multisite":
        run_multisite_sod_ci(cuda_device_index)
        
    elif argument == "long":
        run_generic_hostguest_ci(cuda_device_index)
        run_multisite_sod_ci(cuda_device_index)
    
    print("Time elapsed: {:.3f}".format(time.time() - starttime))
    print("Continuous Integration Tests Passed Successfully.")