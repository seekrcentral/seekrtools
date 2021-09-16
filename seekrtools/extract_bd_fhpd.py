"""

"""

import re
import os
import glob
import subprocess
import argparse
import tempfile
import xml.etree.ElementTree as ET

import parmed
import mdtraj

import seekr2.modules.common_base as base
import seekr2.modules.runner_browndye2 as runner_browndye2

MAX_B_SURFACE_FHPD_STRUCTURES = 1000


def make_proc_file_last_frame(input_filename, output_filename, 
                              pqrxml_path_1, pqrxml_path_2):
    """
    Extract the last frame from a process_trajectories output XML
    and write a new XML containing only the last frame.
    
    input_filename : str
        The path to the initial (long) process_trajectories XML file.
    
    output_filename : str
        The path to the output (short) process_trajectories XML file,
        which will have only a single frame written.
        
    pqrxml_path_1 : str
        The path to the first pqrxml file representing the atomic
        information of the molecule, probably the receptor molecule.
        
    pqrxml_path_2 : str
        The path to the second pqrxml file representing the atomic
        information of the molecule, probably the ligand molecule.
    """
    input_tree = ET.parse(input_filename)
    output_trajectory = ET.Element("trajectory")
    output_trajectory.text = "\n  "
    input_trajectory = input_tree.getroot()
    last_state = None
    for item in input_trajectory:
        if item.tag == "state":
            last_state = item
        elif item.tag == "atom_files":
            new_atom_files = ET.SubElement(output_trajectory, "atom_files")
            new_atom_files.text = " %s %s " % (pqrxml_path_1, pqrxml_path_2)
            new_atom_files.tail = "\n  "
        else:
            output_trajectory.append(item)
    assert last_state is not None
    output_trajectory.append(last_state)
    xmlstr = ET.tostring(output_trajectory).decode("utf-8")
    with open(output_filename, "w") as f:
        f.write(xmlstr)
        
    return

def extract_bd_surface(model, bd_milestone, extract_directory, 
                       max_b_surface_trajs_to_extract, force_overwrite=False, 
                       restart=False, silent=True):
    """
    Given the processed trajectories output XML files, extract the
    encounter complex PQR files from the b-surface simulation(s).
    
    Parameters
    ----------
    model : Model()
        The model object contains all information needed for a SEEKR2
        calculation.
        
    bd_milestone : BD_milestone()
        The BD milestone object whose information we are extracting 
        for.
        
    max_b_surface_trajs_to_extract : int
        The maximum number of trajectories from the b-surface whose
        encounter complex should be extracted. This value exists to
        prevent too many encounter structures from being extracted.
        
    force_overwrite : bool, default False
        Whether to overwrite existing files that have already been
        extracted.
        
    restart : bool, default False
        Whether this is extraction is being performed for a restart
        run. No files will be overwritten for a restart extraction.
    """
    lig_pqr_filenames = []
    rec_pqr_filenames = []
    b_surface_dir = os.path.join(model.anchor_rootdir, 
                                 model.k_on_info.b_surface_directory)
    b_surface_ligand_pqr = model.browndye_settings.ligand_pqr_filename
    b_surface_ligand_pqrxml = os.path.splitext(
        b_surface_ligand_pqr)[0] + ".xml"
    b_surface_ligand_pqrxml_full_path = os.path.join(b_surface_dir, 
                                                     b_surface_ligand_pqrxml)
    assert os.path.exists(b_surface_ligand_pqrxml_full_path), "PQRXML file %s "\
        "not found for b-surface." % b_surface_ligand_pqrxml_full_path
    b_surface_receptor_pqr = model.browndye_settings.receptor_pqr_filename
    b_surface_receptor_pqrxml = os.path.splitext(
        b_surface_receptor_pqr)[0] + ".xml"
    b_surface_receptor_pqrxml_full_path = os.path.join(
        b_surface_dir, b_surface_receptor_pqrxml)
    assert os.path.exists(b_surface_receptor_pqrxml_full_path), "PQRXML file "\
        "%s not found for b-surface." % b_surface_receptor_pqrxml_full_path
    assert os.path.exists(extract_directory)
    
    from_state = str(bd_milestone.outer_milestone.index)
    to_state = str(bd_milestone.inner_milestone.index)
    sitename = "{}_{}".format(from_state, to_state)
    
    empty_pqrxml_path = runner_browndye2.make_empty_pqrxml(extract_directory)
    process_trajectories = os.path.join(
        model.browndye_settings.browndye_bin_dir, "process_trajectories")
    vtf_trajectory = os.path.join(
        model.browndye_settings.browndye_bin_dir, "vtf_trajectory")
    
    quitting = False
    counter = 0
    
    results_file_glob = os.path.join(b_surface_dir, base.BROWNDYE_OUTPUT)
    results_file_list = glob.glob(results_file_glob)
    assert len(results_file_list) > 0, \
        "No b-surface simulation has yet been run: cannot extract to FHPD."
    num_restarts = len(results_file_list)
    
    for i in range(1,num_restarts+1):
        for j in range(model.browndye_settings.n_threads):
            if quitting: break
            if not silent:
                print("extracting trajectories from traj number:", i, 
                      "thread:", j)
            output_filename = os.path.join(extract_directory, 
                                          "rxn_output%d_%d.txt" % (i,j))
            if os.path.exists(output_filename):
                if force_overwrite:
                    if not silent:
                        print("force_overwrite set to True: existing files "\
                              "will be overwritten.")
                elif restart:
                    if not silent:
                        print("restarting extraction.")
                else:
                    print("This folder already has existing output files and "\
                          "the entered command would overwrite them. If you "\
                          "desire to overwrite the existing files, then use "\
                          "the --force_overwrite (-f) option, and all "\
                          "outputs will be deleted and replaced by a new run.")
                    raise Exception("Cannot overwrite existing Browndye "\
                                    "outputs.")
                
            traj_filename = os.path.join(b_surface_dir, "traj%d_%d.xml" % (i,j))
            trajindex_filename = os.path.join(
                b_surface_dir, "traj%d_%d.index.xml" % (i,j))
            #assert os.path.exists(traj_filename), "trajectory output file "\
            #    "%s not found for b-surface. Are you sure you ran b-surface "\
            #    "simulations?" % traj_filename
            #assert os.path.exists(trajindex_filename), "trajectory output "\
            #    "file %s not found for b-surface. Are you sure you ran "\
            #    "b-surface simulations?" % traj_filename
            if not os.path.exists(traj_filename) \
                    or not os.path.exists(trajindex_filename):
                if not silent:
                    print("CONTINUING because {} or {} don't exist".format(
                        traj_filename, trajindex_filename))
                continue
            
            command = "echo 'Browndye Trajectory number'; "\
                +process_trajectories+" -traj %s -index %s -srxn %s > %s" \
                % (traj_filename, trajindex_filename, sitename, 
                   output_filename)
            if not silent:
                print("running command:", command)
            std_out = subprocess.check_output(command, stderr=subprocess.STDOUT,
                                               shell=True)
            assert os.path.exists(output_filename) and \
                    os.stat(output_filename).st_size > 0.0, "Problem running "\
                "process_trajectories: reaction list file not generated."
            number_list = []
            subtraj_list = []
            with open(output_filename, "r") as f:
                for line in f.readlines():
                    if re.search("<number>",line):
                        number_list.append(int(line.strip().split()[1]))
                    elif re.search("<subtrajectory>",line):
                        subtraj_list.append(int(line.strip().split()[1]))
            
            if len(number_list) == 0 or len(subtraj_list) == 0:
                continue
            # sort both lists concurrently so that structures are in order
            number_list, subtraj_list = zip(*sorted(zip(
                number_list, subtraj_list)))
            
            for k, rxn_number in enumerate(number_list):
                if counter > max_b_surface_trajs_to_extract:
                    quitting = True
                    break
                rxn_subtraj = subtraj_list[k]
                proc_traj_basename = os.path.join(
                    extract_directory, "proc_traj%d_%d_%d" % (i, j, k))
                xml_traj_filename = proc_traj_basename + ".xml"
                command = process_trajectories+" -traj %s -index %s -n %d "\
                    "-sn %d -nstride 1 > %s" % (
                        traj_filename, trajindex_filename, rxn_number, 
                        rxn_subtraj, xml_traj_filename)
                if not silent:
                    print("running command:", command)
                std_out = subprocess.check_output(
                    command, stderr=subprocess.STDOUT, shell=True)
                assert os.path.exists(xml_traj_filename) and \
                    os.stat(xml_traj_filename).st_size > 0.0, \
                    "Problem running process_trajectories: trajectory XML "\
                    "file not generated."
                
                last_frame_name = proc_traj_basename + "_last.xml"
                make_proc_file_last_frame(xml_traj_filename, last_frame_name,
                                          empty_pqrxml_path, 
                                          b_surface_ligand_pqrxml_full_path)
                
                # write the last frame as a pqr file
                pqr_filename = os.path.join(extract_directory, 
                                            "lig%d_%d_%d.pqr" % (i,j,k))
                lig_pqr_filenames.append(pqr_filename)
                command = vtf_trajectory+" -traj %s -pqr > %s"\
                    % (last_frame_name, pqr_filename)
                if not silent:
                    print("running command:", command)
                std_out = subprocess.check_output(
                    command, stderr=subprocess.STDOUT, shell=True)
                assert os.path.exists(pqr_filename) and \
                    os.stat(pqr_filename).st_size > 0.0, "Problem running "\
                    "vtf_trajectory: ligand%d_%d_%d PQR file not generated." \
                    % (i, j, k)
                    
                make_proc_file_last_frame(xml_traj_filename, last_frame_name,
                                      b_surface_receptor_pqrxml_full_path, 
                                      empty_pqrxml_path)
                
                pqr_rec_filename = os.path.join(
                    extract_directory, "receptor%d_%d_%d.pqr" % (i,j,k))
                rec_pqr_filenames.append(pqr_rec_filename)
                command = vtf_trajectory+" -traj %s -pqr > "\
                    "%s" % (last_frame_name, pqr_rec_filename)
                if not silent:
                    print("running command:", command)
                std_out = subprocess.check_output(
                    command, stderr=subprocess.STDOUT, shell=True)
                assert os.path.exists(pqr_filename) and \
                    os.stat(pqr_filename).st_size > 0.0, "Problem running "\
                    "vtf_trajectory: receptor%d_%d_%d PQR file not generated." \
                    % (i,j,k)
                
                os.remove(xml_traj_filename)
                counter += 1
    
    assert len(lig_pqr_filenames) > 0, "No trajectories found in b_surface "\
            "simulations. Consider using larger outermost milestone or "\
            "simulating more b-surface trajectories."
    assert len(lig_pqr_filenames) == len(rec_pqr_filenames)
    return lig_pqr_filenames, rec_pqr_filenames

def make_big_fhpd_trajectory(fhpd_traj_filename, lig_pqr_filenames, 
                             rec_pqr_filenames):
    """
    Combine all extracted PQR files from the b-surface simulation 
    encounter complexes, and make one big PDB trajectory representing
    the first hitting point distribution (FHPD).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temporary_pdb_filenames = []
        for i, (lig_pqr_filename, rec_pqr_filename) in enumerate(zip(
                lig_pqr_filenames, rec_pqr_filenames)):
            lig_frame = parmed.load_file(lig_pqr_filename)
            rec_frame = parmed.load_file(rec_pqr_filename)
            combined_frame = rec_frame + lig_frame
            temp_pdb_filename = os.path.join(temp_dir, "fhpd_TEMP%d.pdb" % i)
            combined_frame.save(temp_pdb_filename, overwrite=True)
            temporary_pdb_filenames.append(temp_pdb_filename)
            
        traj = mdtraj.load(temporary_pdb_filenames)
        traj.save_pdb(fhpd_traj_filename)
        
    return


def write_bd_fhpd(model, max_b_surface_trajs_to_extract):
    """
    
    """
    if model.k_on_info is None:
        return
    
    for bd_milestone in model.k_on_info.bd_milestones:
        fhpd_traj_filename = "fhpd_{}.pdb".format(bd_milestone.index)
        fhpd_traj_path = os.path.join(
            model.anchor_rootdir, model.k_on_info.b_surface_directory,
            fhpd_traj_filename)
        print("Writing first hitting point distribution for BD milestone "\
              "{} to file {}.".format(bd_milestone.index, fhpd_traj_path))
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_directory = temp_dir
            lig_pqr_files, rec_pqr_files = extract_bd_surface(
                model, bd_milestone, extract_directory, 
                max_b_surface_trajs_to_extract, force_overwrite=True)
            if len(lig_pqr_files) == 0:
                continue
            make_big_fhpd_trajectory(fhpd_traj_path, lig_pqr_files, rec_pqr_files)
    
    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "model_file", metavar="MODEL_FILE", type=str, 
        help="name of model file for SEEKR2 calculation. This would be the "\
        "XML file generated in the prepare stage.")
    argparser.add_argument(
        "-m", "--max_fhpd_size", dest="max_fhpd_size", 
        default=MAX_B_SURFACE_FHPD_STRUCTURES, type=int, 
        help="The maximum number of frames that a FHPD trajectory should "\
        "have. Default: {}.".format(MAX_B_SURFACE_FHPD_STRUCTURES))
    
    args = argparser.parse_args() # parse the args into a dictionary
    args = vars(args)
    model_file = args["model_file"]
    max_fhpd_size = args["max_fhpd_size"]
    
    model = base.load_model(model_file)
    
    write_bd_fhpd(model, max_fhpd_size)