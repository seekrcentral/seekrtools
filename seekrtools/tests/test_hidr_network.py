"""
test_hidr_network.py
"""

import os
import pytest

import numpy as np
import openmm.unit as unit

import seekrtools.hidr.hidr_network as hidr_network

def test_find_edge_distance_spherical(host_guest_mmvt_model):
    """
    Test find distance between two anchors in a typically spherical
    milestone calculation (hostguest).
    """
    src_anchor_index = host_guest_mmvt_model.anchors[0].index
    dest_anchor_index = host_guest_mmvt_model.anchors[1].index
    dist = hidr_network.find_edge_distance(
        host_guest_mmvt_model, src_anchor_index, dest_anchor_index)
    assert np.isclose(dist, 0.1)
    
def test_find_next_anchor_index(host_guest_mmvt_model):
    """
    
    """
    visited_anchor_dict = {0:0.0, 1:0.1}
    prev_anchor_index, next_anchor_index, next_anchor_distance \
        = hidr_network.find_next_anchor_index(
            host_guest_mmvt_model, visited_anchor_dict)
    assert prev_anchor_index == 1
    assert next_anchor_index == 2
    assert np.isclose(next_anchor_distance, 0.2)
    return

def test_get_procedure(host_guest_mmvt_model):
    """
    
    """
    procedure = hidr_network.get_procedure(host_guest_mmvt_model, [0], [2])
    assert procedure == [(0,1), (1,2)]
    
    procedure = hidr_network.get_procedure(host_guest_mmvt_model, [0], [4])
    assert procedure == [(0,1), (1,2), (2,3), (3,4)]
    return

def test_estimate_simulation_time(host_guest_mmvt_model):
    """
    
    """
    procedure = hidr_network.get_procedure(host_guest_mmvt_model, [0], [1])
    velocity = 345.0 * unit.nanometers / unit.nanoseconds
    est_time = hidr_network.estimate_simulation_time(
        host_guest_mmvt_model, procedure, velocity)
    expected_time = 0.1 * unit.nanometers / velocity
    assert np.isclose(est_time.value_in_unit(unit.nanoseconds), 
                      expected_time.value_in_unit(unit.nanoseconds))
    return