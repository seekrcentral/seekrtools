"""
Run a test of the order parameter CV.
"""

import os

import numpy as np
import seekr2.modules.common_base as base
import seekr2.modules.common_prepare as common_prepare
import seekr2.modules.common_cv as common_cv
import seekr2.prepare as prepare
import seekr2.run as run

import seekrtools.hidr.ratchet as ratchet

def create_multidimensional_muller_potential_model_input(root_dir):
    """
    Create a bond order CV host-guest model input object.
    """
    model_input = common_prepare.Model_input()
    model_input.calculation_type = "mmvt"
    model_input.calculation_settings = common_prepare.MMVT_input_settings()
    model_input.calculation_settings.md_output_interval = 50
    model_input.calculation_settings.md_steps_per_anchor = 50000
    model_input.temperature = 298.15
    model_input.pressure = 1.0
    model_input.ensemble = "nvt"
    model_input.root_directory = root_dir
    model_input.md_program = "openmm"
    model_input.constraints = "none"
    model_input.rigidWater = True
    model_input.hydrogenMass = None
    model_input.timestep = 0.002
    model_input.nonbonded_cutoff = 0.9
    
    cv_input1 = common_cv.Toy_cv_input()
    cv_input1.groups = [[0]]
    cv_input1.variable_name = "value"
    cv_input1.cv_expression = "x1"
    cv_input1.openmm_expression = "step(k*(x1 - value))"
    cv_input1.input_anchors = []
    
    cv_input2 = common_cv.Toy_cv_input()
    cv_input2.groups = [[0]]
    cv_input2.variable_name = "value"
    cv_input2.cv_expression = "y1"
    cv_input2.openmm_expression = "step(k*(y1 - value))"
    cv_input2.input_anchors = []
    
    values_list1 = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    values_list2 = [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    for i, value in enumerate(values_list1):
        input_anchor = common_cv.Toy_cv_anchor()
        input_anchor.value = value
        input_anchor.bound_state = False
        input_anchor.bound_state = False
        cv_input1.input_anchors.append(input_anchor)
    
    for j, value in enumerate(values_list2):
        input_anchor = common_cv.Toy_cv_anchor()
        input_anchor.value = value
        input_anchor.bound_state = False
        input_anchor.bound_state = False
        cv_input2.input_anchors.append(input_anchor)
    
    combo = common_cv.Grid_combo()
    combo.cv_inputs = [cv_input1, cv_input2]
    
    state_point1 = common_cv.State_point()
    state_point1.name = "stateA"
    state_point1.location = [0.5, 0.0, 0.0]
    state_point2 = common_cv.State_point()
    state_point2.name = "stateB"
    state_point2.location = [-0.5, 1.5, 0.0]
    combo.state_points = [state_point1, state_point2]
    
    model_input.cv_inputs = [combo]
    
    model_input.browndye_settings_input = None
    
    model_input.toy_settings_input = common_prepare.Toy_settings_input()
    model_input.toy_settings_input.potential_energy_expression \
        = "-20*exp(-1 * (x1 - 1)^2 + 0 * (x1 - 1) * (y1 - 0) - 10 * (y1 - 0)^2) - 10*exp(-1 * (x1 - 0)^2 + 0 * (x1 - 0) * (y1 - 0.5) - 10 * (y1 - 0.5)^2) - 17*exp(-6.5 * (x1 + 0.5)^2 + 11 * (x1 + 0.5) * (y1 - 1.5) - 6.5 * (y1 - 1.5)^2) + 1.5*exp(0.7 * (x1 + 1)^2 + 0.6 * (x1 + 1) * (y1 - 1) + 0.7 * (y1 - 1)^2)"
    model_input.toy_settings_input.num_particles = 1
    model_input.toy_settings_input.masses = np.array([10.0])
    
    return model_input
    
if __name__ == "__main__":
    root_dir = "/home/lvotapka/multidim_muller_potential"
    model_input = create_multidimensional_muller_potential_model_input(root_dir)
    model, xml_path = prepare.prepare(model_input, force_overwrite=True)
    model_dir = os.path.dirname(xml_path)
    model.anchor_rootdir = os.path.abspath(model_dir)
    
    ratchet.ratchet(model, None, toy_coordinates=[0.5, 0.0, 0.0], force_overwrite = True)
    #run.run(model, "0")
