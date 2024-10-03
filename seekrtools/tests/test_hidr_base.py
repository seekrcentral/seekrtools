"""
test_hidr_base.py
"""

import os
import pytest

import seekr2.modules.common_base as base
import seekrtools.hidr.hidr_base as hidr_base

def test_find_anchors_with_starting_structure(host_guest_mmvt_model):
    """
    
    """
    result = hidr_base.find_anchors_with_starting_structure(host_guest_mmvt_model)
    assert result == [0]
    return

def test_find_destinations(host_guest_mmvt_model):
    """
    
    """
    model = host_guest_mmvt_model
    destination_str="any"
    anchors_with_starting_structures = \
        hidr_base.find_anchors_with_starting_structure(host_guest_mmvt_model)
    result1, complete_anchor_list = hidr_base.find_destinations(
        model, destination_str, anchors_with_starting_structures)
    assert result1 == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    destination_str="1"
    result2, complete_anchor_list = hidr_base.find_destinations(
        model, destination_str, anchors_with_starting_structures)
    assert result2 == [1]
    return

def test_check_destinations(host_guest_mmvt_model):
    
    """
    
    """
    model = host_guest_mmvt_model
    destination_str="any"
    anchors_with_starting_structures = \
        hidr_base.find_anchors_with_starting_structure(host_guest_mmvt_model)
    destination_list, complete_anchor_list = hidr_base.find_destinations(
        model, destination_str, anchors_with_starting_structures)
    relevant_starting_anchor_indices = hidr_base.check_destinations(
        model, anchors_with_starting_structures, destination_list)
    assert relevant_starting_anchor_indices == [0]
    
    destination_str="1"
    destination_list, complete_anchor_list = hidr_base.find_destinations(
        model, destination_str, anchors_with_starting_structures)
    relevant_starting_anchor_indices = hidr_base.check_destinations(
        model, anchors_with_starting_structures, destination_list)
    assert relevant_starting_anchor_indices == [0]
    
    return

def test_save_new_model(host_guest_mmvt_model):
    """
    
    """
    root = host_guest_mmvt_model.anchor_rootdir
    hidr_base.save_new_model(host_guest_mmvt_model)
    old_model_path = os.path.join(root, hidr_base.HIDR_MODEL_BASE.format(0))
    print("old_model_path:", old_model_path)
    assert(os.path.exists(old_model_path))
    new_model_path = os.path.join(root, "model.xml")
    assert(os.path.exists(new_model_path))
    
    old_model_path2 = os.path.join(root, hidr_base.HIDR_MODEL_BASE.format(1))
    hidr_base.save_new_model(host_guest_mmvt_model)
    assert(os.path.exists(old_model_path2))
    assert(os.path.exists(new_model_path))
    return

def test_change_anchor_pdb_filename(host_guest_mmvt_model):
    """
    
    """
    anchor = host_guest_mmvt_model.anchors[0]
    new_pdb_filename = "test1.pdb"
    hidr_base.change_anchor_pdb_filename(anchor, new_pdb_filename)
    assert anchor.amber_params.pdb_coordinates_filename == new_pdb_filename