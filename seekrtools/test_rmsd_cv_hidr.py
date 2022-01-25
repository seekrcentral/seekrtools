"""
Run a test of the order parameter CV.
"""

import os

try:
    import openmm.unit as unit
except ModuleNotFoundError:
    import simtk.unit as unit
import seekr2.modules.common_base as base
import seekr2.modules.common_prepare as common_prepare
import seekr2.modules.common_cv as common_cv
import seekr2.modules.check as check
import seekr2.prepare as prepare
import seekr2.run as run

import seekrtools.hidr.hidr as hidr

def assign_amber_params(input_anchor, prmtop_filename, pdb_filename):
    input_anchor.starting_amber_params = base.Amber_params()
    input_anchor.starting_amber_params.prmtop_filename = prmtop_filename
    input_anchor.starting_amber_params.pdb_coordinates_filename = pdb_filename
    return

def create_trp_cage_mmvt_rmsd_CV_model_input(
        root_dir, ff="amber"):
    """
    Create a bond order CV host-guest model input object.
    """
    model_input = common_prepare.Model_input()
    model_input.calculation_type = "mmvt"
    model_input.calculation_settings = common_prepare.MMVT_input_settings()
    model_input.calculation_settings.md_output_interval = 10000
    model_input.calculation_settings.md_steps_per_anchor = 100000 #1000000
    model_input.temperature = 298.15
    model_input.pressure = 1.0
    model_input.ensemble = "nvt"
    model_input.root_directory = root_dir
    model_input.md_program = "openmm"
    model_input.constraints = "HBonds"
    model_input.rigidWater = True
    model_input.hydrogenMass = None
    model_input.timestep = 0.002
    
    model_input.nonbonded_cutoff = None
    cv_input1 = common_cv.RMSD_cv_input()
    cv_input1.group = [4, 16, 26, 47, 57, 74, 98, 117, 139, 151, 158, 173, 
                       179, 190, 201, 208, 240, 254, 268, 274]
    cv_input1.ref_structure = "/home/lvotapka/seekr2/seekr2/data/trp_cage_files/trp_cage.pdb"
    cv_input1.input_anchors = []
    
    values_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    amber_prmtop_filename = "/home/lvotapka/seekr2/seekr2/data/trp_cage_files/trp_cage.prmtop"
    forcefield_built_in_ff_list = ["amber14/tip3pfb.xml"]
    forcefield_custom_ff_list = ["../data/hostguest_files/hostguest.xml"]
    
    pdb_filenames = ["", "", "", "", "", ""]
    for i, (value, pdb_filename) in enumerate(zip(values_list, pdb_filenames)):
        input_anchor = common_cv.Tiwary_cv_anchor()
        input_anchor.value = value
        if ff == "amber":
            assign_amber_params(input_anchor, amber_prmtop_filename, 
                                pdb_filename)
        elif ff == "forcefield":
            assign_forcefield_params(input_anchor, forcefield_built_in_ff_list, 
                                     forcefield_custom_ff_list, pdb_filename)
        else:
            raise Exception("ff type not supported: {}".format(ff))
        
        if i == 0:
            input_anchor.bound_state = True
        else:
            input_anchor.bound_state = False
            
        if i == len(values_list)-1:
            input_anchor.bulk_anchor = True
        else:
            input_anchor.bulk_anchor = False
        #input_anchor.bulk_anchor = False
    
        cv_input1.input_anchors.append(input_anchor)
    
    model_input.cv_inputs = [cv_input1]
    model_input.browndye_settings_input = None
    
    return model_input
    
if __name__ == "__main__":
    root_dir = "/home/lvotapka/rmsd_test"
    model_input = create_trp_cage_mmvt_rmsd_CV_model_input(root_dir)
    model, xml_path = prepare.prepare(model_input, force_overwrite=True)
    model_dir = os.path.dirname(xml_path)
    model.anchor_rootdir = os.path.abspath(model_dir)
    check.check_pre_simulation_all(model)
    #hidr.hidr(model, "any", dry_run=False, equilibration_steps=1000,  
    #     translation_velocity=1.0*unit.nanometers/unit.nanoseconds)
    hidr.hidr(
        model, "any", 
        pdb_files=["/home/lvotapka/seekr2/seekr2/data/trp_cage_files/trp_cage.pdb"], 
        dry_run=False, equilibration_steps=1000,  
        translation_velocity=1.0*unit.nanometers/unit.nanoseconds)
    
