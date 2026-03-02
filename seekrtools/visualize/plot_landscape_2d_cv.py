"""
This program is a combination of movie_trajectories_2d_cv.py and 
plot_analyze_2d_cv.py, in that it performs an analysis of a 2D 
CV SEEKR calculation, and then loads the trajectory for each anchor,
then uses a k-nearest-neighbors approach to assign a density, and thus,
a free energy to each point so that the energy landscape can be 
plotted in more detail.
"""

import os
import argparse

import numpy as np
from scipy.spatial import KDTree
import matplotlib
import matplotlib.pyplot as plt
import seekr2.modules.common_base as base
import seekr2.analyze as analyze

import seekrtools.visualize.plot_2d_cv as plot_2d_cv
from seekrtools.visualize.plot_2d_cv import PLOTS_DIRECTORY_NAME
import seekrtools.visualize.movie_trajectories_2d_cv as movie_trajectories_2d_cv

GAS_CONSTANT = 0.001987204  # kcal/(mol*K)

def estimate_density_knn(points, k=20):
    """Estimate the local density of each 2D point using k-nearest neighbors.
    
    For each point, the density is estimated as:
        rho_i = k / (N * pi * r_k^2)
    where r_k is the distance to the k-th nearest neighbor.
    
    Parameters
    ----------
    points : numpy.ndarray, shape (N, 2)
        Array of 2D points.
    k : int
        Number of nearest neighbors to use for density estimation.
    
    Returns
    -------
    densities : numpy.ndarray, shape (N,)
        Estimated density at each point.
    """
    n = len(points)
    if n <= 1:
        return np.ones(n)
    # Clamp k if there aren't enough points
    k_use = min(k, n - 1)
    tree = KDTree(points)
    # Query k_use+1 neighbors because each point is its own nearest neighbor
    distances, _ = tree.query(points, k=k_use + 1)
    # The k-th neighbor distance is the last column (index k_use)
    r_k = distances[:, -1]
    # Replace zero distances with a small epsilon to avoid division by zero
    r_k = np.where(r_k > 0, r_k, np.finfo(float).eps)
    densities = k_use / (n * np.pi * r_k ** 2)
    return densities

def make_landscape_plot(model, analysis, plot_dir=None,
                    x_coordinate_title="X-Coordinate", 
                    y_coordinate_title="Y-Coordinate", omit_iter_label=False, 
                    dpi=100, base_name="landscape", boundaries=None, traj_values=None,
                    draw_string=False, k=20, max_energy_cap=None):
    """Create a fine-grained free energy landscape by combining per-anchor
    trajectory densities (via KNN) with SEEKR2 anchor free energies.
    
    Parameters
    ----------
    model : object
        SEEKR2 model object.
    analysis : object
        SEEKR2 analysis object with free_energy_anchors attribute.
    plot_dir : str or None
        Directory to save the plot. If None, uses the model's anchor_rootdir.
    x_coordinate_title : str
        Label for the x-axis.
    y_coordinate_title : str
        Label for the y-axis.
    omit_iter_label : bool
        Whether to omit iteration labels.
    dpi : int
        Resolution of the saved plot.
    base_name : str
        Base filename for the saved plot.
    boundaries : list or None
        Plot boundaries [xmin, xmax, ymin, ymax].
    traj_values : list or None
        Pre-computed trajectory CV values. If None, they are loaded from disk.
    draw_string : bool
        Whether to draw the string path.
    k : int
        Number of nearest neighbors for density estimation.
    max_energy_cap : float or None
        Maximum free energy to display. Points above this are clamped.
        If None, defaults to max anchor free energy + 5 kcal/mol.
    """
    anchor_values = plot_2d_cv.find_anchor_points(model)
    print("Anchor values:", anchor_values)
    traj_cv_values = movie_trajectories_2d_cv.make_traj_cv_values(model)
    print("Trajectory CV values computed for", len(traj_cv_values), "anchors")
    boundaries = movie_trajectories_2d_cv.make_boundaries(traj_cv_values)
    print("Boundaries:", boundaries)

    anchor_free_energies = analysis.free_energy_anchors.copy()
    for i, fill_value in enumerate(anchor_free_energies):
        if not np.isfinite(fill_value):
            anchor_free_energies[i] = np.max(
                anchor_free_energies[np.isfinite(anchor_free_energies)])
    
    # Estimate density of each trajectory point and convert to free energy
    temperature = model.temperature
    kBT = GAS_CONSTANT * temperature  # kcal/mol
    
    all_xy = []
    all_free_energies = []
    for alpha, anchor_points in enumerate(traj_cv_values):
        if len(anchor_points) == 0:
            continue
        points = np.array(anchor_points)
        densities = estimate_density_knn(points, k=k)
        rho_max = np.max(densities)
        # Free energy offset within this anchor: F = -kBT * ln(rho / rho_max)
        # The densest point gets F_offset = 0, sparser points are higher
        with np.errstate(divide='ignore'):
            free_energy_offset = -kBT * np.log(densities / rho_max)
        # Shift by the anchor's global free energy
        point_free_energies = anchor_free_energies[alpha] + free_energy_offset
        all_xy.append(points)
        all_free_energies.append(point_free_energies)
    
    if len(all_xy) == 0:
        print("No trajectory data found. Cannot create landscape plot.")
        return
    
    all_xy = np.concatenate(all_xy, axis=0)
    all_free_energies = np.concatenate(all_free_energies)
    
    # Cap extreme free energy values
    if max_energy_cap is None:
        max_energy_cap = np.max(anchor_free_energies[np.isfinite(anchor_free_energies)]) + 5.0
    all_free_energies = np.clip(all_free_energies, None, max_energy_cap)
    
    print(f"Plotting {len(all_xy)} points, free energy range: "
          f"{np.min(all_free_energies):.2f} to {np.max(all_free_energies):.2f} kcal/mol")
    
    # Draw Voronoi tessellation without cell fills
    fig, ax, vor, anchor_pts = plot_2d_cv.plot_voronoi_tesselation(
        model, boundaries, fill_values=None, anchor_values=anchor_values)
    
    # Overlay scatter plot of trajectory points colored by free energy
    vmin = np.min(all_free_energies)
    vmax = np.max(all_free_energies)
    sc = ax.scatter(all_xy[:, 0], all_xy[:, 1], c=all_free_energies,
                    cmap=plt.cm.jet, s=2, edgecolors='none',
                    vmin=vmin, vmax=vmax, zorder=2)
    
    # Add colorbar
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet)
    cbar = plt.colorbar(mappable, ax=ax)
    cbar.set_label("Free Energy (kcal/mol)")
    
    ax.set_xlabel(x_coordinate_title)
    ax.set_ylabel(y_coordinate_title)
    ax.set_title("Free Energy Landscape")
    
    # Save
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = os.path.join(plot_dir, f"{base_name}.png")
        plt.savefig(plot_filename, dpi=dpi)
        print(f"Landscape plot saved to: {plot_filename}")
    
    plt.close(fig)
    return all_xy, all_free_energies

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of model XML file for a SEEKR2 calculation. "\
        "One or more starting structures must be present in one or more of "\
        "the anchors.")
    #argparser.add_argument(
    #    "-t", "--title", dest="title", default="Voronoi Tesselation",
    #    type=str, help="The title of the plot")
    argparser.add_argument(
        "-x", "--x_coordinate_title", dest="x_coordinate_title", 
        default="X-Coordinate",
        type=str, help="The title of x-coordinate")
    argparser.add_argument(
        "-y", "--y_coordinate_title", dest="y_coordinate_title", 
        default="Y-Coordinate",
        type=str, help="The title of y-coordinate")
    argparser.add_argument(
        "-d", "--dpi", dest="dpi", default=200, type=int,
        help="The DPI (dots per inch) of resolution for plots.")
    argparser.add_argument(
        "-k", "--knn", dest="k", default=20, type=int,
        help="Number of nearest neighbors for density estimation. "
        "Smaller values give noisier but more detailed landscapes; "
        "larger values give smoother landscapes.")
    argparser.add_argument(
        "-m", "--max_energy", dest="max_energy_cap", default=None, 
        type=float,
        help="Maximum free energy (kcal/mol) to display. Points above "
        "this value are clamped. Default: max anchor free energy + 5.")
        
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    #title = args["title"]
    x_coordinate_title = args["x_coordinate_title"]
    y_coordinate_title = args["y_coordinate_title"]
    dpi = args["dpi"]
    k = args["k"]
    max_energy_cap = args["max_energy_cap"]
    
    model = base.load_model(model_file)
    plot_dir = os.path.join(model.anchor_rootdir, PLOTS_DIRECTORY_NAME)
    
    analysis = analyze.analyze(model, num_error_samples=0, skip_checks=True)
    
    make_landscape_plot(model, analysis, plot_dir, x_coordinate_title, 
                    y_coordinate_title, True, dpi, base_name="landscape",
                    k=k, max_energy_cap=max_energy_cap)