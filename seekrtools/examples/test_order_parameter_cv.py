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

def create_host_guest_mmvt_bond_order_CV_model_input(
        root_dir, order_parameters, order_parameter_weights, bd=True, ff="amber"):
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
    cv_input1 = common_cv.Tiwary_cv_input()
    cv_input1.order_parameters = order_parameters
    cv_input1.order_parameter_weights = order_parameter_weights
    cv_input1.input_anchors = []
    
    values_list = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
                   1.05, 1.15, 1.25, 1.35]
    amber_prmtop_filename = "/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest.parm7"
    forcefield_built_in_ff_list = ["amber14/tip3pfb.xml"]
    forcefield_custom_ff_list = ["../data/hostguest_files/hostguest.xml"]
    """
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
    """
    pdb_filenames = ["/home/lvotapka/seekr2/seekr2/data/hostguest_files/hostguest_at0.5.pdb",
                     "", "", "", "", "", "", "", "", "", "", "", "", ""]
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
    
    model_input.cv_inputs[0].input_anchors[-1].bulk_anchor = False
    
    if bd:
        model_input.cv_inputs[0].input_anchors[-1].connection_flags = [1]
        cv_input2 = common_cv.Spherical_cv_input()
        cv_input2.input_anchors = []
        cv_input2.group1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
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
        cv_input2.group2 = [147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 
                        158, 159, 160, 161]
        spherical_input_anchor1 = common_cv.Spherical_cv_anchor()
        spherical_input_anchor1.radius = 1.35
        spherical_input_anchor1.bound_state = False
        spherical_input_anchor1.bulk_anchor = False
        spherical_input_anchor1.connection_flags = [1]
        cv_input2.input_anchors.append(spherical_input_anchor1)
        
        spherical_input_anchor2 = common_cv.Spherical_cv_anchor()
        spherical_input_anchor2.radius = 1.45
        spherical_input_anchor2.bound_state = False
        spherical_input_anchor2.bulk_anchor = False
        cv_input2.input_anchors.append(spherical_input_anchor2)
        
        spherical_input_anchor3 = common_cv.Spherical_cv_anchor()
        spherical_input_anchor3.radius = 1.55
        spherical_input_anchor3.bound_state = False
        spherical_input_anchor3.bulk_anchor = True
        cv_input2.input_anchors.append(spherical_input_anchor3)
        
        model_input.cv_inputs.append(cv_input2)
        
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
    root_dir = "/home/lvotapka/tiwary_test"
    
    order_parameter1 = common_cv.Tiwary_cv_distance_order_parameter()
    order_parameter1.group1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
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
    order_parameter1.group2 = [147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 
                        158, 159, 160, 161]
    order_parameter2 = common_cv.Tiwary_cv_angle_order_parameter()
    order_parameter2.group1 = [0]
    order_parameter2.group2 = [147]
    order_parameter2.group3 = [161]
    order_parameters = [order_parameter1, order_parameter2]
    order_parameter_weights = [0.99, 0.01]
    
    model_input = create_host_guest_mmvt_bond_order_CV_model_input(
        root_dir, order_parameters, order_parameter_weights, bd=True)
    model, xml_path = prepare.prepare(model_input, force_overwrite=True)
    model_dir = os.path.dirname(xml_path)
    model.anchor_rootdir = os.path.abspath(model_dir)
    run.run(model, "0")
