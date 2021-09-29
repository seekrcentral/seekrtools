"""
convergence_window.py

Given a span of time, the convergence of kinetic/thermodynamic 
quantities will be computed for transition data within a moving window
of that length of time
"""
import os
import argparse

from openmm import unit
import seekr2.modules.common_base as base
import seekr2.modules.common_analyze as common_analyze
import seekr2.modules.common_converge as common_converge
import seekr2.analyze as analyze

def convergence_window(model, image_directory, time_window, num_windows=100):
    window_time_list = []
    start_time = 0.0
    # TODO: fix - hardcoded timestep: implement generic timestep method
    dt = model.openmm_settings.langevin_integrator.timestep
    timestep_in_ns = dt * 0.001
    max_step_list = common_converge.get_mmvt_max_steps(model, dt)
    end_time = 1e9
    for max_step in max_step_list[:-1]:
        if max_step[-1] < end_time:
            end_time = float(max_step[-1]) * dt
            
    interval = (end_time - start_time - time_window) / num_windows
    k_off_list = []
    
    time_list = []
    for i in range(num_windows):
        min_time = i * interval
        max_time = min_time + time_window
        print("running window {} of {}".format(i, num_windows))
        window_time_list.append((min_time, max_time))
        try:
            analysis = analyze.analyze(model, num_error_samples=0, 
                skip_checks=True, min_time=min_time, max_time=max_time)
            k_off_list.append(analysis.k_off)
        except:
            k_off_list.append(0.0)
        time_list.append(min_time)
    
    figure_path = os.path.join(image_directory, "k_off_windows.png")
    print("Saving figure to {}.".format(image_directory))    
    k_off_fig, ax = common_converge.plot_scalar_conv(
        k_off_list, time_list, title="$k_{off}$ Windows", 
        label="k_{off} (s^{-1})", timestep_in_ns=timestep_in_ns)
    k_off_fig.savefig(figure_path)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="name of model file for SEEKR2 calculation. This would be the "\
        "XML file generated in the prepare stage.")
    argparser.add_argument(
        "-s", "--k_on_state", dest="k_on_state", default=None, type=int,
        help="Define the bound state that will be used to compute k-on. If "\
        "left blank, and k-on statistics exist, then the first end state will "\
        "be chosen by default.")
    argparser.add_argument(
        "-d", "--image_directory", dest="image_directory", 
        default=None, type=str,
        help="Define the directory where all plots and images will be saved. "\
            "By default, graphics will be saved to the "\
            "'%s' directory in the model's anchor root directory."\
            % common_analyze.DEFAULT_IMAGE_DIR)
    argparser.add_argument(
        "-t", "--time_window", dest="time_window", default=50000.0, type=float, 
        help="The span of time (in ps) that is the 'width' of the window.")
    
    args = argparser.parse_args() # parse the args into a dictionary
    args = vars(args)
    model_file = args["model_file"]
    k_on_state = args["k_on_state"]
    image_directory = args["image_directory"]
    time_window = args["time_window"]
    
    model = base.load_model(model_file)
    
    image_directory = common_analyze.make_image_directory(
        model, image_directory)
    convergence_window(model, image_directory, time_window)