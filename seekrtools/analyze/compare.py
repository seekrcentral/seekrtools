"""
Perform a series of analyses that plots and shows quantities of interest
from a HIDR/SEEKR calculation, comparing a calculation to itself or to another
calculation.

- Plot how internal RMSD of ligand and site structures changes across
  anchors as HIDR is run.
- Plot how RMSD of ligand and site within SEEKR anchor runs change from the 
  starting HIDR structure.
- ...
"""

import os
import argparse
from textwrap import wrap

import numpy as np
import matplotlib.pyplot as plt
import mdtraj
import seekr2.modules.common_base as seekr2_base
import seekr2.modules.check as check
import seekr2.modules.mmvt_cvs.mmvt_spherical_cv as mmvt_spherical_cv

import seekrtools.analyze.base as analyze_base

PLOTS_DIRECTORY_NAME = "images_and_plots"
SELF_COMPARE_PLOT_DIRECTORY_NAME = "compare"
A_PER_NM = 10.0
NS_PER_PS = 1e-3

STARTING_RMSD_CAPTION = "The RMSD of each anchor's starting structure is "\
    "shown in comparison to the 'bound state' anchor's starting structure. "\
    "First, the alpha carbons (or site, if no alpha carbons present) is "\
    "superimposed, then the RMSD of the {} is found compared to the reference "\
    "structure. "
TRAJ_RMSD_CAPTION = "The RMSD over the span of the MMVT trajectory is shown "\
    "in comparison to the starting structure at the beginning of the "\
    "trajectory. First, the alpha carbons (or site, if no alpha carbons "\
    "present) is superimposed, then the RMSD of the {} is found compared to "\
    "the reference structure."

class Comparison_analysis_model():
    def __init__(self, model):
        self.model = model
        # Construct MDAnalysis universes for each anchor
        self.start_trajs = []
        self.mmvt_trajs = []
        self.num_anchors = 0
        self.anchor_indices = np.zeros(len(self.model.anchors), dtype=np.int8)
        anchor_values = []
        self.end_state_index = None
        for i, anchor in enumerate(model.anchors):
            if anchor.bulkstate:
                continue
            if anchor.endstate and self.end_state_index is None:
                self.end_state_index = i
            self.start_trajs.append(
                check.load_structure_with_mdtraj(model, anchor, mode="pdb"))
            self.mmvt_trajs.append(
                check.load_structure_with_mdtraj(
                    model, anchor, mode="mmvt_traj"))
            self.anchor_indices[i] = anchor.index
            anchor_values.append(list(anchor.variables.values())[0]*A_PER_NM)
            self.num_anchors += 1
            
        self.anchor_values = np.array(anchor_values)
        if self.end_state_index is None:
            self.end_state_index = 0
        
        return
    
    def compare_internal_starting_rmsds(self):
        cv = self.model.collective_variables[0]
        assert isinstance(model.collective_variables[0], 
                          mmvt_spherical_cv.MMVT_spherical_CV), \
            "At present, this script only works for 1D spherical anchors."
        site_indices = cv.group1
        ligand_indices = cv.group2
        site_selection = "index "+" ".join(map(str, site_indices))
        ligand_selection = "index "+" ".join(map(str, ligand_indices))
        anchor0_traj = self.start_trajs[self.end_state_index]
        site_indices = anchor0_traj.top.select(site_selection)
        ligand_indices = anchor0_traj.top.select(ligand_selection)
        site_start_rmsd_array = np.zeros(self.num_anchors)
        lig_start_rmsd_array = np.zeros(self.num_anchors)
        align_indices = anchor0_traj.top.select("name CA")
        if len(align_indices) == 0:
            align_indices = site_indices
        for i, start_traj in enumerate(self.start_trajs):
            start_traj.superpose(anchor0_traj, atom_indices=align_indices, 
                                 ref_atom_indices=align_indices)
            site_rmsd = np.sqrt(3*np.mean(np.square(
                start_traj.xyz[0,site_indices] \
                - anchor0_traj.xyz[0,site_indices]))) # , axis=(1,2)))
            site_start_rmsd_array[i] = site_rmsd*A_PER_NM
            ligand_rmsd = np.sqrt(3*np.mean(
                np.square(start_traj.xyz[0,ligand_indices] \
                          - anchor0_traj.xyz[0,ligand_indices]))) # , axis=(1,2)))
            lig_start_rmsd_array[i] = ligand_rmsd*A_PER_NM
            
        site_traj_rmsds = []
        lig_traj_rmsds = []
        for i, mmvt_traj in enumerate(self.mmvt_trajs):
            mmvt_traj.superpose(mmvt_traj, atom_indices=align_indices, 
                                 ref_atom_indices=align_indices)
            
            site_rmsd = np.sqrt(3*np.mean(np.square(
                mmvt_traj.xyz[:,site_indices] \
                - mmvt_traj.xyz[0,site_indices]), axis=(1,2)))
            site_traj_rmsds.append(site_rmsd*A_PER_NM)
            ligand_rmsd = np.sqrt(3*np.mean(
                np.square(mmvt_traj.xyz[:,ligand_indices] \
                          - mmvt_traj.xyz[0,ligand_indices]), axis=(1,2)))
            lig_traj_rmsds.append(ligand_rmsd*A_PER_NM)
        
        return site_start_rmsd_array, lig_start_rmsd_array, site_traj_rmsds, \
            lig_traj_rmsds

def plot_data_by_anchor(poses, data_frames, title, labels,
                        x_axis_name, y_axis_name, filename=None, caption=""):
    
    fig, ax = plt.subplots()
    for pos, data_frame, label in zip(poses, data_frames, labels):
        plt.plot(pos, data_frame, label=label)
        
    #plt.xticks(anchor_pos, anchor_pos, rotation=90)
    plt.ylabel(y_axis_name)
    wrapped_caption = "\n".join(wrap(caption))
    x_axis_name_txt = x_axis_name + "\n\n" + wrapped_caption
    ax.set_xlabel(x_axis_name_txt)
    plt.title(title)
    #plt.yscale("log", nonpositive="mask")
    plt.tight_layout()
    plt.legend()
    #plt.figtext(0.5, -0.2, caption, wrap=True, horizontalalignment='center', 
    #            fontsize=12)
    #fig.set_size_inches(7, 8, forward=True)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)
        
    return

def compare_all(models, compare_plot_dir, model_directories):
    self_site_starting_rmsds = []
    self_lig_starting_rmsds = []
    site_traj_rmsds_list = []
    lig_traj_rmsds_list = []
    labels = []
    anchor_locations_by_model = []
    anchor_locations_first_model = None
    running_time_series = True
    times_by_model = []
    for i, model in enumerate(models):
        if i == 0:
            label = "This SEEKR model"
        else:
            label = f"SEEKR model {i}"
        labels.append(label)
        comp_model = Comparison_analysis_model(model)
        if not os.path.exists(compare_plot_dir):
            os.mkdir(compare_plot_dir)
        self_site_starting_rmsd, self_lig_starting_rmsd, \
            site_traj_rmsds, lig_traj_rmsds\
            = comp_model.compare_internal_starting_rmsds()
        self_site_starting_rmsds.append(self_site_starting_rmsd)
        self_lig_starting_rmsds.append(self_lig_starting_rmsd)
        site_traj_rmsds_list.append(site_traj_rmsds)
        lig_traj_rmsds_list.append(lig_traj_rmsds)
        if anchor_locations_first_model is None:
            anchor_locations_first_model = comp_model.anchor_values
        else:
            if (comp_model.anchor_values != anchor_locations_first_model).any():
                running_time_series = False
                print("Model anchor positions do not align. MMVT trajectory "\
                      "time series will not be plotted.")
        
        anchor_locations_by_model.append(comp_model.anchor_values)
        stride_time = model.calculation_settings.trajectory_reporter_interval \
            * model.get_timestep() * NS_PER_PS
        times_by_anchor = []
        for site_traj_rmsd in site_traj_rmsds:
            times = stride_time * np.arange(len(site_traj_rmsd))
            times_by_anchor.append(times)
            
        times_by_model.append(times_by_anchor)
    
    site_starting_fig_filename = os.path.join(
        compare_plot_dir, "site_starting.png")
    model_explanation_list = []
    for label, model_directory in zip(labels, model_directories):
        model_explanation_list.append(
            f"{label} came from directory: {model_directory}")
        
    model_explanation_str = ", ".join(model_explanation_list)
    
    site_starting_caption = STARTING_RMSD_CAPTION.format(
        "site", model_explanation_str)
    lig_starting_fig_filename = os.path.join(
        compare_plot_dir, "lig_starting.png")
    lig_starting_caption = STARTING_RMSD_CAPTION.format(
        "ligand", model_explanation_str)
    plot_data_by_anchor(anchor_locations_by_model, self_site_starting_rmsds,
                        "Site starting RMSD by anchor", labels,
                        "anchor value ($\AA$)", "RMSD ($\AA$)",
                        filename=site_starting_fig_filename, 
                        caption=site_starting_caption)
    plot_data_by_anchor(anchor_locations_by_model, self_lig_starting_rmsds,
                        "Ligand starting RMSD by anchor", labels,
                        "anchor value ($\AA$)", "RMSD ($\AA$)",
                        filename=lig_starting_fig_filename, 
                        caption=lig_starting_caption)
    site_starting_caption = TRAJ_RMSD_CAPTION.format(
        "site", model_explanation_str)
    lig_starting_caption = TRAJ_RMSD_CAPTION.format(
        "ligand", model_explanation_str)
    if running_time_series:
        #for i, site_traj_rmsd in enumerate(site_traj_rmsds):
        for i in range(len(anchor_locations_first_model)):
            site_traj_rmsds_this_anchor = []
            times_over_models = []
            for j, site_traj_rmsds_this_model in enumerate(site_traj_rmsds_list):
                site_traj_rmsds_this_anchor.append(
                    site_traj_rmsds_this_model[i])
                times_over_models.append(times_by_model[j][i])
                
            lig_traj_rmsds_this_anchor = []
            for j, lig_traj_rmsds_this_model in enumerate(lig_traj_rmsds_list):
                lig_traj_rmsds_this_anchor.append(
                    lig_traj_rmsds_this_model[i])
            
            site_traj_fig_filename = os.path.join(
                compare_plot_dir, f"site_trajectory_anchor_{i}.png")
            plot_data_by_anchor(times_over_models, site_traj_rmsds_this_anchor,
                            "Site MMVT trajectory RMSD over time", labels,
                            "MMVT simulation time (ns)", 
                            "RMSD ($\AA$)", filename=site_traj_fig_filename,
                            caption=site_starting_caption)
            
            lig_traj_fig_filename = os.path.join(
                compare_plot_dir, f"lig_trajectory_anchor_{i}.png")
            plot_data_by_anchor(times_over_models, lig_traj_rmsds_this_anchor,
                            "Ligand MMVT trajectory RMSD over time", labels,
                            "MMVT simulation time (ns)", 
                            "RMSD ($\AA$)", filename=lig_traj_fig_filename,
                            caption=lig_starting_caption)
            
    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_files", metavar="MODEL_FILES", nargs="+", 
        help="The names of model XML files for a SEEKR2 calculation."\
            "List the primary model first - images will be written there.")
    
    args = argparser.parse_args()
    args = vars(args)
    model_files = args["model_files"]
    models = []
    model_directories = []
    for model_file in model_files:
        model = seekr2_base.load_model(model_file)
        models.append(model)
        model_directories.append(os.path.dirname(model_file))
    
    primary_model = models[0]
    plot_dir = os.path.join(primary_model.anchor_rootdir, PLOTS_DIRECTORY_NAME)
    compare_plot_dir = os.path.join(plot_dir, SELF_COMPARE_PLOT_DIRECTORY_NAME)
    compare_all(models, compare_plot_dir, model_directories)
    print("All plots being saved to:", compare_plot_dir)
    