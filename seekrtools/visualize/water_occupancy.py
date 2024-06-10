"""
Analyze and display the number of waters within an area of the site
by milestone.
"""
import os
import glob
import argparse
from collections import defaultdict

import numpy as np
import mdtraj
import matplotlib.pyplot as plt
import seekr2.modules.common_base as base
import seekr2.modules.mmvt_cvs.mmvt_cv_base as mmvt_cv_base

def water_occupancy_data(model, cutoff):
    water_count_by_anchor_and_time = defaultdict(list)
    cv0 = model.collective_variables[0]
    site_atom_indices = cv0.group1
    #longest_traj = 0
    shortest_traj = 1e99
    for alpha, anchor in enumerate(model.anchors):
        print("processing anchor:", alpha, "of", len(model.anchors)-1)
        if anchor.bulkstate:
            continue
        
        building_directory = os.path.join(
            model.anchor_rootdir, anchor.directory, 
            anchor.building_directory)
        prod_directory = os.path.join(
            model.anchor_rootdir, anchor.directory, 
            anchor.production_directory)
        if anchor.amber_params is not None:
            top_file_name = os.path.join(
                building_directory, anchor.amber_params.prmtop_filename)
            mmvt_traj_basename = mmvt_cv_base.OPENMMVT_BASENAME+"*.dcd"
            mmvt_traj_glob = os.path.join(prod_directory, 
                                          mmvt_traj_basename)
            mmvt_traj_filenames = base.order_files_numerically(
                glob.glob(mmvt_traj_glob))
        
        else:
            raise Exception("Only Amber inputs implemented at this time.")
        
        try:
            chunk = mdtraj.load(mmvt_traj_filenames, top=top_file_name)
        except OSError:
            print("Error while loading file(s):", mmvt_traj_filenames, 
                  ". Skipping.")
        
        chunk.image_molecules(inplace=True)
        # DEBUG
        #chunk.save_dcd(os.path.join(prod_directory, "chunk.dcd"))
        
        
        all_water_indices = chunk.topology.select("water and element == O")
        all_waters = chunk.atom_slice(all_water_indices)
        #all_waters.save_pdb(os.path.join(prod_directory, "waters.pdb"))
        
        traj_site = chunk.atom_slice(site_atom_indices)
        #traj_site.save_pdb(os.path.join(prod_directory, "site.pdb"))
        com1_array = mmvt_cv_base.traj_center_of_mass(traj_site)
        #if chunk.n_frames > longest_traj:
        #    longest_traj = chunk.n_frames
        if chunk.n_frames < shortest_traj:
            shortest_traj = chunk.n_frames
            
        for frame_index in range(chunk.n_frames):
            site_com = com1_array[frame_index,:]
            num_waters_near_site = 0
            for atom_index in range(all_waters.n_atoms):
                water_coord = all_waters.xyz[frame_index, atom_index,:]
                if np.linalg.norm(water_coord - site_com) < cutoff:
                    num_waters_near_site += 1
                
            water_count_by_anchor_and_time[alpha].append(num_waters_near_site)
            
    assert shortest_traj < 1e7
    
    n = len(water_count_by_anchor_and_time)
    #water_count_2d_array = np.zeros((n, longest_traj))
    water_count_2d_array = np.zeros((n, shortest_traj))
    for i in range(n):
        #for j in range(len(water_count_by_anchor_and_time[i])):
        for j in range(shortest_traj):
            water_count_2d_array[i, j] = water_count_by_anchor_and_time[i][j]
    
    return water_count_2d_array        
            
def plot_water_occupancy(cutoff, water_count_2d_array):
    num_anchors = water_count_2d_array.shape[0]
    num_frames = water_count_2d_array.shape[1]
    print("water_count_2d_array:", water_count_2d_array)
    #ax = plt.figure().add_subplot(projection='3d')
    #ax.plot_surface(X, Y, water_count_2d_array, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
    #            alpha=0.3)
    #ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
    #   xlabel='X', ylabel='Y', zlabel='Z')
    im = plt.imshow(water_count_2d_array, vmin=water_count_2d_array.min(), 
                      vmax=water_count_2d_array.max(), extent=[0, num_frames, num_anchors, 0],
                   cmap=plt.cm.jet)
    plt.xlabel("Frame")
    plt.ylabel("Anchor")
    plt.title(f"Number of waters within \n{cutoff} nm of site")
    plt.colorbar()
    plt.show()
    return
     
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of model XML file for a SEEKR2 calculation. "\
        "One or more starting structures must be present in one or more of "\
        "the anchors.")
    argparser.add_argument(
        "-c", "--cutoff", dest="cutoff", default=0.5, type=float,
        help="The cutoff distance (in nm) from the center of the site within "\
        "which to find waters.")
    
    
    args = argparser.parse_args()
    args = vars(args)
    model_file = args["model_file"]
    cutoff = args["cutoff"]
    model = base.load_model(model_file)
    water_count_2d_array = water_occupancy_data(model, cutoff)
    plot_water_occupancy(cutoff, water_count_2d_array)
