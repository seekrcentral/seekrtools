import os
import pytest

import seekrtools.extract_bd_fhpd as extract_bd_fhpd

TEST_DIRECTORY = os.path.dirname(__file__)

def test_make_proc_file_last_frame(tmp_path):
    """
    Test the function that extracts the last frame of the BD simulation
    trajectory.
    """
    input_filename = os.path.join(TEST_DIRECTORY, "data/proc_traj_test.xml")
    output_filename = os.path.join(tmp_path, "proc_last.xml")
    extract_bd_fhpd.make_proc_file_last_frame(
        input_filename, output_filename, "dummy1", "dummy2")
    assert os.path.exists(output_filename)
    return

def test_make_big_fhpd_trajectory(tmp_path):
    """
    Test the function that combines all FHPD structures into a single
    trajectory PDB.
    """
    lig_pqr_filename = os.path.join(TEST_DIRECTORY, 
                                    "data/tryp_ben_encounter_lig.pqr")
    lig_pqr_filenames = [lig_pqr_filename, lig_pqr_filename]
    rec_pqr_filename = os.path.join(TEST_DIRECTORY, 
                                    "data/tryp_ben_encounter_rec.pqr")
    rec_pqr_filenames = [rec_pqr_filename, rec_pqr_filename]
    output_filename = os.path.join(tmp_path, "big_fhpd.pqr")
    extract_bd_fhpd.make_big_fhpd_trajectory(
        output_filename, lig_pqr_filenames, rec_pqr_filenames)
    assert os.path.exists(output_filename)
    return