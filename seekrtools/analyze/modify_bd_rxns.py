"""
modify_bd_rxns.py

Modify the BD reaction criteria for a SEEKR2 calculation in order to 
better capture the binding kinetics if there is a bad choice of 
anchor radii or choice of binding site.

1. 
If the HIDR stage of a SEEKR calculation are complete, find the starting
structures for each anchor, determine which anchor is located closest to
the convex hull of the protein, and return this position.

2. Run BD simulations with no reaction criteria to determine the closest
approach location of the ligand to the point outside the convex hull.
Add 0.1 nm to this distance to find the secondary point.

3. Determine which milestone is just outside this secondary point, which
will identify the anchor then outside of this milestone. This will be the 
new outermost anchor, and the BD reaction criteria will be updated to 
encompass this anchor. A secondary spherical reaction criteria will be
imposed on the secondary point. 

4. The model needs to be modified to exclude any anchors outside of this.
The BD simulations will be rerun with the new reaction criteria to 
determine the new k_on estimate.

"""

import os
import glob
import argparse
import subprocess
from tqdm import tqdm
from multiprocessing import Pool
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import transform, Delaunay
import quaternion
import mdtraj
import parmed

import seekr2.modules.common_base as base
import seekr2.modules.common_sim_browndye2 as sim_browndye2
import seekr2.modules.runner_browndye2 as runner_browndye2
import seekr2.analyze as analyze

BD_REDO_MODEL_GLOB = "model_pre_bd_redo_*.xml"
BD_REDO_MODEL_BASE = "model_pre_bd_redo_{}.xml"

def make_browndye_empty_reaction_xml(abs_reaction_path):
    rxnroot = sim_browndye2.Reaction_root()
    rxnroot.first_state = "b_surface"
    rxnroot.write(abs_reaction_path)
    return

def extract_fates_xml(trajectory_file):
    tree = ET.parse(trajectory_file)
    root = tree.getroot()
    fates_string = []
    for i, traj in enumerate(root.findall(".//trajectory")):
        fate = traj.findtext('fate', default='').strip()
        fates_string.append(fate)
    # Convert 'bound' to 1 and 'escaped' to 0
    fates = [1 if fate == 'reacted' else 0 for fate in fates_string]
    # fates_np = np.array(fates, dtype=int).reshape(-1, 1)
    return fates

def get_traj_xml(trajectory_file, n_traj):
    index_file = trajectory_file.replace(".xml", ".index.xml")
    command = f"process_trajectories -traj {trajectory_file} -index {index_file} -n {n_traj} "
    try:
        atom_traj_xml_raw = subprocess.check_output(command, shell=True, text=True)
        atom_traj_xml = ET.fromstring(atom_traj_xml_raw)
        return atom_traj_xml
    except subprocess.CalledProcessError as e:
        print(f"Error processing trajectories: {e}")
        return None
    
def get_traj_list(atom_traj_xml):
    atom_time_list = []
    atom_traj_list = []
    for item in atom_traj_xml.findall(".//core"):
        pos_quat = [float(i) for i in item.text.strip().split()]
        atom_traj_list.append(pos_quat)
    #dt_item = atom_traj_xml.find(".//dt")
    for item in atom_traj_xml.findall(".//dt"):
        dt = float(item.text.strip())
        atom_time_list.append(dt)
    atom_traj_list = atom_traj_list[:-2]  # remove last frame
    atom_time_list = atom_time_list[:-1]  # remove last frame
    return atom_traj_list, atom_time_list

def get_quaternion_data(atom_traj_list, lig_centroid_to_ghost_vector):
    lig_xyz_centroid = [0.1 * np.array(pos_quat[:3]) for pos_quat in atom_traj_list[1::2]]  # Convert Angstrom to nm
    lig_quat = [quaternion.quaternion(pos_quat[3], pos_quat[4], pos_quat[5], pos_quat[6]) \
                for pos_quat in atom_traj_list[1::2]]
    lig_centroid_to_ghost_vector_rotated = np.array([quaternion.rotate_vectors(
        lig_q, lig_centroid_to_ghost_vector) for lig_q in lig_quat])
    lig_xyz = np.array(lig_xyz_centroid) + lig_centroid_to_ghost_vector_rotated
    #rec_xyz = [pos_quat[:3] for pos_quat in atom_traj_list[0::2]] # Not necessary since rec is fixed at origin, but included for completeness
    #lig_quat = [quaternion.quaternion(pos_quat[3], pos_quat[4], pos_quat[5], pos_quat[6]) for pos_quat in atom_traj_list[1::2]] # Also not necessary, rotational alignment is on receptor
    rec_quat = [quaternion.quaternion(pos_quat[3], pos_quat[4], pos_quat[5], pos_quat[6]) for pos_quat in atom_traj_list[0::2]]
    return np.array(lig_xyz), np.array(rec_quat)

def do_quaternion_alignment(lig_xyz_array, rec_quat_array): #, alignment_quaternion):
    #q_ref = quaternion.quaternion(1, 0, 0, 0)  # Reference quaternion
    #rot = alignment_quaternion * np.conjugate(rec_quat_array)
    rot = np.conjugate(rec_quat_array)
    lig_xyz_aligned = np.array([quaternion.rotate_vectors(r, v) for r, v in zip(rot, lig_xyz_array)])
    return lig_xyz_aligned

def get_3d_traj_from_fate_aligned(
        trajectory_file, n_traj, fate, lig_centroid_to_ghost_vector):
    #    trajectory_file, alignment_quaternion, n_traj, fate):
    traj_xml = get_traj_xml(trajectory_file, n_traj)
    if traj_xml is None:
        return np.array([])  # Return empty array if there was an error
    traj_list, time_list = get_traj_list(traj_xml)
    lig_traj, rec_quat = get_quaternion_data(traj_list, lig_centroid_to_ghost_vector)
    lig_traj_aligned = do_quaternion_alignment(
        lig_traj, rec_quat)
        #lig_traj, rec_quat, alignment_quaternion)
    traj_with_time = np.insert(lig_traj_aligned, 3, 0.0, axis=1)
    # Now fill out the time column with the values from time_list
    for i in range(traj_with_time.shape[0]):
        traj_with_time[i, 3] = time_list[i]
    traj_with_fate = np.insert(traj_with_time, 4, fate, axis=1)
    traj_with_fate[-1, 4] = fate + 2  # Ensure we indicate the last frame
    return traj_with_fate

# 1. Find the first starting structure outside of the convex hull of the receptor.
def get_starting_structure_convex_hull(model, atom_name=None, residue_name=None):
    """
    Align the structures of the model anchors to the receptor.pqr 
    reference structure and find the first structure that lies 
    outside of the convex hull of the receptor.
    """
    starting_lig_centroids = []
    starting_site_centroids = []
    receptor_pqr_filename = os.path.join(
        model.anchor_rootdir, model.k_on_info.b_surface_directory,
        model.browndye_settings.receptor_pqr_filename)
    rec_pqr_struct = parmed.load_file(receptor_pqr_filename)
    if atom_name is not None:
        rec_pqr_indices = [i for i, atom in enumerate(rec_pqr_struct.atoms) \
                           if atom.name == atom_name]
    else:
        assert residue_name is not None
        rec_pqr_indices = [i for i, atom in enumerate(rec_pqr_struct.atoms) \
                           if atom.residue.name == residue_name]
    rec_pqr_ca_coords = np.array(
        [0.1*rec_pqr_struct.coordinates[i] for i in rec_pqr_indices])
    rec_pqr_centroid = np.mean(rec_pqr_ca_coords, axis=0)
    #print("rec_pqr_centroid:", rec_pqr_centroid)
    rec_pqr_hull = Delaunay(rec_pqr_ca_coords)
    gho_indices = model.collective_variables[0].group1
    lig_indices = model.collective_variables[0].group2
    first_distance_outside_hull = 0.0
    last_distance_outside_hull = 0.0
    first_lig_centroid_outside_hull = None
    first_gho_centroid_outside_hull = None
    first_gho_starting_structure_outside_hull = None
    starting_struct = None
    for anchor in model.anchors:
        starting_pdb_filename = base.get_anchor_pdb_filename(anchor)
        if starting_pdb_filename == "":
            continue
        starting_pdb_fullpath = os.path.join(
            model.anchor_rootdir, anchor.directory, anchor.building_directory,
            starting_pdb_filename)
        starting_struct = mdtraj.load(starting_pdb_fullpath)
        if atom_name is not None:
            starting_struct_indices = [i for i, atom in enumerate(starting_struct.topology.atoms) \
                                       if atom.name == atom_name]
        else:
            assert residue_name is not None
            starting_struct_indices = [i for i, atom in enumerate(starting_struct.topology.atoms) \
                                       if atom.residue.name == residue_name]
        starting_struct_coords = starting_struct.xyz[0, starting_struct_indices, :]
        # Calculate centroids
        starting_struct_centroid = np.mean(starting_struct_coords, axis=0)
        # Center the CA coordinates before alignment
        starting_struct_coords_centered = starting_struct_coords - starting_struct_centroid
        rec_pqr_coords_centered = rec_pqr_ca_coords - rec_pqr_centroid
        # Align centered structures
        rotation, rmsd = transform.Rotation.align_vectors(
            rec_pqr_coords_centered, starting_struct_coords_centered)
        # Apply: (1) translate to origin, (2) rotate, (3) keep centered at origin
        coords = starting_struct.xyz[0, :, :]
        coords_centered = coords - starting_struct_centroid  # Center at origin
        coords_rotated = rotation.apply(coords_centered)  # Rotate
        starting_struct.xyz = coords_rotated
        #starting_struct.save_pdb(f"DELETE_ME_{anchor.index}.pdb")
        lig_coords = starting_struct.xyz[0, lig_indices, :]
        gho_coords = starting_struct.xyz[0, gho_indices, :]
        lig_centroid = np.mean(lig_coords, axis=0)
        # Compute the convex hull of rec_coords and check if the ligand 
        # centroid is outside of it
        lig_centroid_outside_hull = rec_pqr_hull.find_simplex(lig_centroid) < 0
        gho_centroid = np.mean(gho_coords, axis=0)
        starting_lig_centroids.append(lig_centroid)
        starting_site_centroids.append(gho_centroid)
        if lig_centroid_outside_hull:
            distance_outside_hull = np.linalg.norm(lig_centroid - rec_pqr_centroid)
            if first_distance_outside_hull == 0.0:
                first_distance_outside_hull = distance_outside_hull
                first_lig_centroid_outside_hull = lig_centroid
                first_gho_centroid_outside_hull = gho_centroid
                first_gho_starting_structure_outside_hull = starting_struct

            last_distance_outside_hull = distance_outside_hull
            print(f"Anchor {anchor.index} is outside of the convex hull of the receptor.")
            print(f"Distance outside hull: {distance_outside_hull:.3f} nm")

    assert starting_struct is not None, \
        "No starting structures found in any anchors."
    distance_first_to_last = last_distance_outside_hull - first_distance_outside_hull
    #print("first_lig_centroid_outside_hull:", first_lig_centroid_outside_hull)
    #print("first_gho_centroid_outside_hull:", first_gho_centroid_outside_hull)
    if first_gho_centroid_outside_hull is None:
        # Assign the last anchor's structure to be the "first outside" hull.
        distance_outside_hull = np.linalg.norm(lig_centroid - rec_pqr_centroid)
        first_distance_outside_hull = distance_outside_hull
        first_lig_centroid_outside_hull = lig_centroid
        first_gho_centroid_outside_hull = gho_centroid

    return distance_first_to_last, first_lig_centroid_outside_hull, \
        first_gho_centroid_outside_hull, starting_struct

# 2. Use the milestones to rewrite the rxns.xml file and re-run a 
#    few BD simulations with no reaction criteria to see which
#    milestones are actually reachable.
def rerun_bd_no_reaction_criteria(
        model: base.Model,
        num_trajectories: int
        ):
    b_surface_dir = os.path.join(
        model.anchor_rootdir, model.k_on_info.b_surface_directory)
    root_directory = os.path.abspath(
        os.path.expanduser(model.anchor_rootdir))
    receptor_pqr_filename = os.path.join(
        b_surface_dir, model.browndye_settings.receptor_pqr_filename)
    receptor_xml_filename = os.path.splitext(receptor_pqr_filename)[0] + ".xml"
    ligand_pqr_filename = os.path.join(
        b_surface_dir, model.browndye_settings.ligand_pqr_filename)
    ligand_xml_filename = os.path.splitext(ligand_pqr_filename)[0] + ".xml"
    NEW_RXNS_NAME = "rxns_postprocess_k_on.xml"
    rxns_full_filename = os.path.join(
        b_surface_dir, NEW_RXNS_NAME)
    model.k_on_info.reactions_filename = NEW_RXNS_NAME
    #rxns_backup_filename = os.path.join(
    #    b_surface_dir, "rxns_backup.xml")
    # Create a backup of the existing rxns.xml file.
    #shutil.copyfile(rxns_full_filename, rxns_backup_filename)
    # Rewrite the rxns.xml file to have no reaction criteria.
    make_browndye_empty_reaction_xml(rxns_full_filename)
    empty_result_filename = "empty_results.xml"
    model.k_on_info.bd_output_glob = "empty_results.xml"
    model.k_on_info.bd_milestones[0].bd_output_glob = "empty_results.xml"
    #runner_browndye2.cleanse_bd_outputs(b_surface_dir, check_mode=False)
    runner_browndye2.make_browndye_input_xml(
        model, root_directory, receptor_xml_filename, ligand_xml_filename,
        num_trajectories)
    runner_browndye2.run_bd_top(model.browndye_settings.browndye_bin_dir, 
                                b_surface_dir, force_overwrite=True)
    runner_browndye2.modify_variables(b_surface_dir, model.k_on_info.bd_output_glob, 
                                      num_trajectories, 
                                      output_file=empty_result_filename,
                                      n_steps_per_output=1, 
                                      desolvation_parameter=0.07957747)
    runner_browndye2.run_nam_simulation(model.browndye_settings.browndye_bin_dir, 
                                        b_surface_dir, model.k_on_info.bd_output_glob)

#    Examine the closeness of the BD encounter points to the MD starting
#    structures in each anchor. Error out if there's not a close overlap.
#    If there's overlap, choose which milestones to actually make the
#    outermost milestone based on the BD encounter points.
def get_and_align_bd_traj_structures(model, atom_name=None, residue_name=None):
    bd_full_directory = os.path.join(
        model.anchor_rootdir, model.k_on_info.b_surface_directory)
    trajectory_files = glob.glob(os.path.join(bd_full_directory, "traj*[0-9].xml"))
    receptor_pqr_filename = os.path.join(
        model.anchor_rootdir, model.k_on_info.b_surface_directory,
        model.browndye_settings.receptor_pqr_filename)
    ligand_pqr_filename = os.path.join(
        model.anchor_rootdir, model.k_on_info.b_surface_directory,
        model.browndye_settings.ligand_pqr_filename)
    rec_pqr_struct = parmed.load_file(receptor_pqr_filename)
    lig_pqr_struct = parmed.load_file(ligand_pqr_filename)
    lig_ghost_atoms = lig_pqr_struct["@GHO"]
    lig_ghost_atom_coords = 0.1*np.mean(lig_ghost_atoms.coordinates, axis=0)
    
    # Select alpha carbons from structure
    if atom_name is not None:
        rec_pqr_indices = [i for i, atom in enumerate(rec_pqr_struct.atoms) if atom.name == atom_name]
    else:
        assert residue_name is not None
        rec_pqr_indices = [i for i, atom in enumerate(rec_pqr_struct.atoms) if atom.residue.name == residue_name]
    rec_pqr_ca_coords = np.array(
        [0.1*rec_pqr_struct.coordinates[i] for i in rec_pqr_indices])
    lig_pqr_coords = np.array(
        [0.1*lig_pqr_struct.coordinates[i] for i in range(len(lig_pqr_struct.atoms))])
    rec_pqr_centroid = np.mean(rec_pqr_ca_coords, axis=0)
    lig_pqr_centroid = np.mean(lig_pqr_coords, axis=0)
    lig_centroid_to_ghost_vector = lig_ghost_atom_coords - lig_pqr_centroid
    rec_pqr_coords_centered = rec_pqr_ca_coords - rec_pqr_centroid
    total_traj_list = []
    for trajectory_file in tqdm(trajectory_files):
        fates = extract_fates_xml(trajectory_file)
        with Pool() as pool:
            #traj_list = pool.starmap(get_3d_traj_from_fate_aligned, [
            #    (trajectory_file, alignment_quaternion, n_traj, fate)
            #        for n_traj, fate in enumerate(fates)])
            traj_list = pool.starmap(get_3d_traj_from_fate_aligned, [
                (trajectory_file, n_traj, fate, lig_centroid_to_ghost_vector)
                    for n_traj, fate in enumerate(fates)])
        total_traj_list.extend(traj_list)

    return total_traj_list

# Find closest approach location
def find_closest_approach_point(
        model,
        lig_gho_vector_from_md,
        aligned_bd_traj_structures,
    ):
    # Load receptor PQR file to find receptor center
    # And location of ghost atom in receptor
    receptor_pqr_filename = os.path.join(
        model.anchor_rootdir, model.k_on_info.b_surface_directory,
        model.browndye_settings.receptor_pqr_filename)
    rec_pqr_struct = parmed.load_file(receptor_pqr_filename)
    rec_ghost_atoms = rec_pqr_struct["@GHO"]
    rec_first_ghost_atom = rec_ghost_atoms[0]
    rec_ghost_atoms = rec_pqr_struct[f"@GHO and :{rec_first_ghost_atom.residue.number}"]
    rec_ghost_atom_coords = 0.1*np.mean(rec_ghost_atoms.coordinates, axis=0)
    print("rec_ghost_atom_coords:", rec_ghost_atom_coords)
    rec_centroid = np.mean(0.1*rec_pqr_struct.coordinates, axis=0)
    rec_centroid_to_ghost_vector = rec_ghost_atom_coords - rec_centroid
    print("rec_centroid_to_ghost_vector:", rec_centroid_to_ghost_vector)
    closest_distance_to_lig_starting_structure = 9e9
    closest_location = None

    for bd_traj in aligned_bd_traj_structures:
        for bd_frame in bd_traj:
            # Convert from Angstroms to nm
            # Already converted!
            bd_lig_xyz = bd_frame[:3]
            
            # Calculate distance in plane perpendicular to radial direction
            bd_projected = bd_lig_xyz - (lig_gho_vector_from_md + rec_centroid_to_ghost_vector)
            #print("bd_projected:", bd_projected)
            lig_distance_to_md_start = np.linalg.norm(bd_projected)
            #print("lig_distance_to_md_start:", lig_distance_to_md_start)
            if lig_distance_to_md_start \
                    <= closest_distance_to_lig_starting_structure:
                closest_distance_to_lig_starting_structure = lig_distance_to_md_start
                closest_location = bd_lig_xyz
            #exit()
    
    assert closest_location is not None, \
        "No BD trajectories found to determine closest approach point."
    site_to_lig_vector = closest_location - rec_centroid_to_ghost_vector
    site_to_lig_distance = np.linalg.norm(site_to_lig_vector)
    print(f"Closest approach distance: {site_to_lig_distance:.3f} nm")
    # Extend length of site_to_lig_vector by 0.1 nm to get the secondary point
    # NOTE: tried using this value as the secondary point, but that put it way out
    # into the solvent, where it could occlude the binding of the ligand.
    extended_site_to_lig_vector = site_to_lig_vector * (1 + 0.1 / site_to_lig_distance)
    extended_site_to_lig_dist = np.linalg.norm(extended_site_to_lig_vector)
    #secondary_point = rec_centroid_to_ghost_vector + extended_site_to_lig_vector
    #secondary_distance = np.linalg.norm(extended_site_to_lig_vector)
    # NOTE: instead, we are going to set the ghost atom to be at the place where
    # the convex hull is encountered in the MD.
    secondary_point = rec_centroid_to_ghost_vector + lig_gho_vector_from_md
    secondary_distance = np.linalg.norm(lig_gho_vector_from_md)

    return closest_location, secondary_point, extended_site_to_lig_dist, secondary_distance

def find_new_bd_anchor_and_update_model(
        model, 
        outermost_distance):
    # Find which milestone is just outside the secondary point, which will
    # identify the new outermost anchor. Update the BD reaction criteria to
    # encompass this anchor and add a secondary spherical reaction criteria on
    # the secondary point.
    outermost_anchor_index = None
    for alpha, anchor in enumerate(model.anchors):
        if anchor.bulkstate:
            continue
        milestone_radii = [ms.variables["radius"] for ms in anchor.milestones]
        if len(milestone_radii) < 2:
            #print(f"Anchor {anchor.index} has less than 2 milestones, skipping.")
            continue
        outer_milestone_radius = max(milestone_radii)
        inner_milestone_radius = min(milestone_radii)
        if inner_milestone_radius > outermost_distance:
            outermost_anchor_index = alpha
            break

    #assert outermost_anchor_index is not None, \
    #    "No suitable outermost anchor found to update BD reaction criteria."
    return outermost_anchor_index

def modify_model_for_new_bd_anchor(
        model, 
        outermost_anchor_index, 
        ):
    if outermost_anchor_index is None:
        return
    # Update the BD reaction criteria to encompass the new outermost anchor and
    # add a secondary spherical reaction criteria on the secondary point.
    #model.k_on_info.bd_milestones[0].variables["radius"] = secondary_distance
    for alpha, anchor in enumerate(model.anchors):
        if alpha == outermost_anchor_index:
            inner_milestone = min(
                anchor.milestones, key=lambda ms: ms.variables["radius"])
            outer_milestone = max(
                anchor.milestones, key=lambda ms: ms.variables["radius"])
            model.k_on_info.bd_milestones[0]\
                .outer_milestone = outer_milestone
            model.k_on_info.bd_milestones[0]\
                .inner_milestone = inner_milestone

        elif alpha == outermost_anchor_index + 1:
            # Keep smallest milestone and assign bulk_state to True
            inner_milestone = min(
                anchor.milestones, key=lambda ms: ms.variables["radius"])
            anchor.milestones = [inner_milestone]
            anchor.bulkstate = True
            anchor.amber_params = None
            anchor.charmm_params = None
            anchor.forcefield_params = None
            break
        elif alpha > outermost_anchor_index + 1:
            # Pop the anchor out of the anchors list.
            model.anchors.pop(alpha)
            model.num_anchors -= 1
            # TODO: not true in general
            model.num_milestones -= 1
    return

def add_ghost_atom_to_pqr(pqr_filename, secondary_point, 
                          new_pqr_filename=None, center_molecule=True):
    if new_pqr_filename is None:
        new_pqr_filename = pqr_filename
    pqr_struct = parmed.load_file(pqr_filename, skip_bonds=True)
    if center_molecule:
        # Compute the center of mass of the entire molecule to be transposed
        mol_center_of_mass = np.array([[0., 0., 0.]])
        mol_total_mass = 0.0
        for atom_index, atom in enumerate(pqr_struct.atoms):
            atom_pos = pqr_struct.coordinates[atom_index,:]
            atom_mass = atom.mass
            if atom_mass == 0.0:
                atom_mass = 0.0001
            mol_center_of_mass += atom_mass * atom_pos
            mol_total_mass += atom_mass
        mol_center_of_mass = mol_center_of_mass / mol_total_mass
    
    ghost_atom = parmed.Atom(name="GHO", mass=0.0, charge=0.0, solvent_radius=0.0)
    ghost_structure = parmed.Structure()
    ghost_structure.add_atom(ghost_atom, "GHO", 1)
    ghost_structure.coordinates = np.array(10.0 * secondary_point)
    pqr_complex = pqr_struct + ghost_structure
    for residue in pqr_complex.residues:
        residue.chain = ""
    
    if center_molecule:
        new_coordinates = np.zeros(pqr_complex.coordinates.shape)
        for atom_index in range(len(pqr_complex.atoms)):
            new_coordinates[atom_index,:] = pqr_complex.coordinates[atom_index,:] \
                - mol_center_of_mass[0,:]
                
        pqr_complex.coordinates = new_coordinates
    
    pqr_complex.save(new_pqr_filename, overwrite=True)
    ghost_index = len(pqr_complex.atoms)
    
    return ghost_index

def save_new_rxn_file(model, secondary_ghost_atom, secondary_distance):
    b_surface_dir = os.path.join(
        model.anchor_rootdir, model.k_on_info.b_surface_directory)
    NEW_RXNS_NAME = "rxns_bd_redo.xml"
    rxns_full_filename = os.path.join(
        b_surface_dir, NEW_RXNS_NAME)
    model.k_on_info.reactions_filename = NEW_RXNS_NAME
    model.k_on_info.bd_output_glob = "results*.xml"
    model.k_on_info.bd_milestones[0].bd_output_glob = "results*.xml"
    runner_browndye2.make_browndye_reaction_xml(
        model, rxns_full_filename, secondary_ghost_atom, 10.0*secondary_distance,
        inner_surface_smaller_by=1.0)
    root_directory = os.path.abspath(
        os.path.expanduser(model.anchor_rootdir))
    receptor_pqr_filename = os.path.join(
        b_surface_dir, model.browndye_settings.receptor_pqr_filename)
    receptor_xml_filename = os.path.splitext(receptor_pqr_filename)[0] + ".xml"
    ligand_pqr_filename = os.path.join(
        b_surface_dir, model.browndye_settings.ligand_pqr_filename)
    ligand_xml_filename = os.path.splitext(ligand_pqr_filename)[0] + ".xml"
    runner_browndye2.make_browndye_input_xml(
        model, root_directory, receptor_xml_filename, ligand_xml_filename,
        num_trajectories)
    sim_browndye2.make_pqrxml(
        receptor_pqr_filename, output_xml_filename=receptor_xml_filename)

def write_bd_traj_to_pdb(
        lig_rec_vector_from_md,
        aligned_bd_traj_structures,
        output_filename,
        ):
    """
    
    """
    with open(output_filename, "w") as f:
        atom_serial = 0
        atom_serial += 1
        x, y, z = lig_rec_vector_from_md * 10.0
        pdb_line = (
            f"ATOM  {atom_serial:5d}  STT STT  {0:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{0.0:6.2f}{0.0:6.2f}          H \n"
        )
        f.write(pdb_line)
        
        for bd_traj_index, bd_traj in enumerate(aligned_bd_traj_structures):
            for bd_frame_index, bd_frame in enumerate(bd_traj):
                # Convert from Angstroms to nm
                #bd_lig_xyz = 0.1 * bd_frame[:3]
                # Already converted!
                bd_lig_xyz = bd_frame[:3]
                atom_serial += 1
                if atom_serial > 99999:
                    atom_serial_string = "*****"
                else:
                    atom_serial_string = f"{atom_serial:5d}"
                x, y, z = bd_lig_xyz * 10.0
                pdb_line = (
                    f"ATOM  {atom_serial_string}  FRM TRA  {bd_traj_index+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{0.0:6.2f}{0.0:6.2f}          C \n"
                )
                f.write(pdb_line)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="The name of model XML file for a SEEKR2 calculation. "\
        "One or more starting structures must be present in one or more of "\
        "the anchors.")
    argparser.add_argument(
        "-d", "--distance_min", metavar="DISTANCE_MIN", type=float, default=1.0,
        help="The minimum distance for BD reaction criteria."
        " Default: 1.0 nm")
    argparser.add_argument(
        "-N", "--num_trajectories", metavar="N", type=int, default=10000,
        help="The number of trajectories to run in BD simulations."
        " Default: 1000")
    
    args = argparser.parse_args() # parse the args into a dictionary
    args = vars(args)
    model_file = args["model_file"]
    distance_min = args["distance_min"]
    num_trajectories = args["num_trajectories"]
    model = base.load_model(model_file)

    # 1. Find starting point just outside the convex hull of the receptor.
    distance_first_to_last, first_lig_centroid_outside_hull, \
        first_gho_centroid_outside_hull, starting_structure \
            = get_starting_structure_convex_hull(model, atom_name="CA")
                #model, residue_name="MGO")
    lig_gho_vector_from_md = first_lig_centroid_outside_hull - first_gho_centroid_outside_hull
    
    # 2. Rerun BD simulations with no reaction criteria to determine the 
    # closest approach location of the ligand to the point outside the convex hull.
    rerun_bd_no_reaction_criteria(model, num_trajectories)
    aligned_bd_traj_structures = get_and_align_bd_traj_structures(
            model, atom_name="CA")
    closest_location, secondary_point, site_to_milestone_distance, secondary_distance \
        = find_closest_approach_point(
            model, lig_gho_vector_from_md, aligned_bd_traj_structures)
    
    # 3. Determine which milestone is just outside this secondary point, which
    # will identify the anchor then outside of this milestone. This will be the
    # new outermost anchor, and the BD reaction criteria will be updated to
    # encompass this anchor. A secondary spherical reaction criteria will be
    # imposed on the secondary point.
    #write_bd_traj_to_pdb(
    #    lig_gho_vector_from_md, aligned_bd_traj_structures, "bd_traj.pdb")
    outermost_anchor_index = find_new_bd_anchor_and_update_model(
        model, site_to_milestone_distance)
    
    modify_model_for_new_bd_anchor(model, outermost_anchor_index)
    receptor_pqr_filename = os.path.join(
        model.anchor_rootdir, model.k_on_info.b_surface_directory,
        model.browndye_settings.receptor_pqr_filename)
    secondary_ghost_atom = add_ghost_atom_to_pqr(receptor_pqr_filename, secondary_point)
    save_new_rxn_file(model, secondary_ghost_atom, distance_min)
    base.save_new_model(model, BD_REDO_MODEL_GLOB, BD_REDO_MODEL_BASE, 
                        save_old_model=True)
    
