"""
Run a test of the order parameter CV.
"""

import os

import seekr2.modules.common_base as base
import seekr2.modules.common_prepare as common_prepare
import seekr2.modules.common_cv as common_cv
import seekr2.prepare as prepare
import seekr2.run as run

import seekrtools.hidr.ratchet as ratchet

starting_pdb_file = "/home/lvotapka/seekr2_systems/systems/trypsin_benzamidine_files/mmvt/tryp_ben_at0.pdb"

def assign_amber_params(input_anchor, prmtop_filename, pdb_filename):
    input_anchor.starting_amber_params = base.Amber_params()
    input_anchor.starting_amber_params.prmtop_filename = prmtop_filename
    input_anchor.starting_amber_params.pdb_coordinates_filename = pdb_filename
    return

def create_multidimensional_tryp_ben_model_input(root_dir):
    """
    Create a bond order CV host-guest model input object.
    """
    model_input = common_prepare.Model_input()
    model_input.calculation_type = "mmvt"
    model_input.calculation_settings = common_prepare.MMVT_input_settings()
    model_input.calculation_settings.md_output_interval = 100
    model_input.calculation_settings.md_steps_per_anchor = 10000 #1000000
    model_input.temperature = 298.15
    model_input.pressure = 1.0
    model_input.ensemble = "nvt"
    model_input.root_directory = root_dir
    model_input.md_program = "openmm"
    model_input.constraints = "HBonds"
    model_input.rigidWater = True
    model_input.hydrogenMass = None
    model_input.timestep = 0.002
    model_input.nonbonded_cutoff = 0.9
    
    cv_input1 = common_cv.Spherical_cv_input()
    cv_input1.group1 = [2478, 2489, 2499, 2535, 2718, 2745, 2769, 2787, 2794, 2867, 2926]
    cv_input1.group2 = [3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229]
    cv_input1.input_anchors = []
    
    cv_input2 = common_cv.RMSD_cv_input()
    cv_input2.group = [3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229]
    cv_input2.ref_structure = starting_pdb_file
    cv_input2.input_anchors = []
    
    radius_values_list = [0.05, 0.15, 0.25, 0.35, 0.45, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
    rmsd_values_list = [0.1, 0.3, 0.5, 0.7]
    amber_prmtop_filename = "/home/lvotapka/seekr2_systems/systems/trypsin_benzamidine_files/tryp_ben.prmtop"
    
    pdb_filenames = []
    for i, value in enumerate(radius_values_list):
        input_anchor = common_cv.Spherical_cv_anchor()
        input_anchor.radius = value
        assign_amber_params(input_anchor, amber_prmtop_filename, 
                                None)
        if i == 0:
            input_anchor.bound_state = True
        else:
            input_anchor.bound_state = False
            
        if i == len(radius_values_list)-1:
            input_anchor.bulk_anchor = True
        else:
            input_anchor.bulk_anchor = False
        #input_anchor.bulk_anchor = False
    
        cv_input1.input_anchors.append(input_anchor)
    
    for j, value in enumerate(rmsd_values_list):
        input_anchor = common_cv.RMSD_cv_anchor()
        input_anchor.value = value
        
        #if i == 0:
        #    input_anchor.bound_state = True
        #else:
        #    input_anchor.bound_state = False
            
        #if i == len(values_list)-1:
        #    input_anchor.bulk_anchor = True
        #else:
        #    input_anchor.bulk_anchor = False
        input_anchor.bulk_anchor = False
    
        cv_input2.input_anchors.append(input_anchor)
    
    combo = common_cv.Grid_combo()
    combo.cv_inputs = [cv_input1, cv_input2]
    
    model_input.cv_inputs = [combo]
    
    model_input.browndye_settings_input = None
    
    return model_input
    
if __name__ == "__main__":
    root_dir = "/home/lvotapka/multidim_tryp_ben"
    model_input = create_multidimensional_tryp_ben_model_input(root_dir)
    model, xml_path = prepare.prepare(model_input, force_overwrite=True)
    model_dir = os.path.dirname(xml_path)
    model.anchor_rootdir = os.path.abspath(model_dir)
    
    ratchet.ratchet(model, [starting_pdb_file], toy_coordinates=None, force_overwrite = True)
    #run.run(model, "0")
