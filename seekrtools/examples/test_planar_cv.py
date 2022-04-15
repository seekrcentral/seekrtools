"""
Run a test of the order parameter CV.
"""

import os

import seekr2.modules.common_base as base
import seekr2.modules.common_prepare as common_prepare
import seekr2.modules.common_cv as common_cv
import seekr2.prepare as prepare
import seekr2.run as run

def assign_amber_params(input_anchor, prmtop_filename, pdb_filename):
    input_anchor.starting_amber_params = base.Amber_params()
    input_anchor.starting_amber_params.prmtop_filename = prmtop_filename
    input_anchor.starting_amber_params.pdb_coordinates_filename = pdb_filename
    return

def create_host_guest_mmvt_planar_CV_model_input(
        root_dir, bd=True, ff="amber"):
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
    model_input.nonbonded_cutoff = 0.9
    cv_input1 = common_cv.Planar_cv_input()
    cv_input1.start_group = list(range(130))
    cv_input1.end_group = list(range(130, 162))
    cv_input1.mobile_group = list(range(147, 162))
    cv_input1.input_anchors = []
    
    values_list = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
                   1.05, 1.15, 1.25, 1.35]
    amber_prmtop_filename = "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest.parm7"
    forcefield_built_in_ff_list = ["amber14/tip3pfb.xml"]
    forcefield_custom_ff_list = ["../data/hostguest_files/hostguest.xml"]
    pdb_filenames = ["/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_at0.5.pdb",
                     "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_at1.5.pdb",
                     "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_at2.5.pdb",
                     "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_at3.5.pdb",
                     "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_at4.5.pdb",
                     "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_at5.5.pdb",
                     "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_at6.5.pdb",
                     "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_at7.5.pdb",
                     "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_at8.5.pdb",
                     "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_at9.5.pdb",
                     "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_at10.5.pdb",
                     "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_at11.5.pdb",
                     "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_at12.5.pdb",
                     ""]
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
            
        #if i == len(values_list)-1:
        #    input_anchor.bulk_anchor = True
        #else:
        #    input_anchor.bulk_anchor = False
        input_anchor.bulk_anchor = False
    
        cv_input1.input_anchors.append(input_anchor)
    
    model_input.cv_inputs = [cv_input1]
    
    if bd:
        model_input.browndye_settings_input \
            = common_prepare.Browndye_settings_input()
        model_input.browndye_settings_input.binary_directory = ""
        model_input.browndye_settings_input.receptor_pqr_filename \
            = "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_receptor.pqr"
        model_input.browndye_settings_input.ligand_pqr_filename \
            = "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_ligand.pqr"
        model_input.browndye_settings_input.apbs_grid_spacing = 0.5
        model_input.browndye_settings_input.receptor_indices = list(range(147))
        model_input.browndye_settings_input.ligand_indices = list(range(15))
        
        ion1 = base.Ion()
        ion1.radius = 1.2
        ion1.charge = -1.0
        ion1.conc = 0.0
        ion2 = base.Ion()
        ion2.radius = 0.9
        ion2.charge = 1.0
        ion2.conc = 0.0
        model_input.browndye_settings_input.ions = [ion1, ion2]
        model_input.browndye_settings_input.num_bd_milestone_trajectories = 100
        model_input.browndye_settings_input.num_b_surface_trajectories = 10000
        model_input.browndye_settings_input.max_b_surface_trajs_to_extract = 100
        model_input.browndye_settings_input.n_threads = 1
    else:
        model_input.browndye_settings_input = None
    
    return model_input
    
if __name__ == "__main__":
    root_dir = "/home/lvotapka/planar_test"
    model_input = create_host_guest_mmvt_planar_CV_model_input(
        root_dir, bd=False)
    model, xml_path = prepare.prepare(model_input, force_overwrite=True)
    model_dir = os.path.dirname(xml_path)
    model.anchor_rootdir = os.path.abspath(model_dir)
    run.run(model, "0")
