"""
Unit and regression test for the seekrtools package.
"""

# Import package, test suite, and other packages as needed

import pytest
import sys
import os
import random

import simtk

import seekrtools

def test_seekrtools_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "seekrtools" in sys.modules


def test_utilities_serialize_box_vectors():
    box_vector = simtk.unit.Quantity(
        [[64.91, 0.0, 0.0], 
        [-21.63, 61.20, 0.0], 
        [-21.63, -30.60, 53.00]], 
        unit=simtk.unit.angstrom)
    xmlstr = seekrtools.utilities.serialize_box_vectors(
        box_vector) #, to_file='/tmp/test.xml')
    expected_output = """<?xml version="1.0" ?>
<box_vectors>
   <A>
      <x>6.491</x>
      <y>0.0</y>
      <z>0.0</z>
   </A>
   <B>
      <x>-2.163</x>
      <y>6.120000000000001</y>
      <z>0.0</z>
   </B>
   <C>
      <x>-2.163</x>
      <y>-3.0600000000000005</y>
      <z>5.300000000000001</z>
   </C>
</box_vectors>
"""
    assert xmlstr == expected_output
    return
    
def test_utilities_deserialize_box_vectors():
    xml_input = """<?xml version="1.0" ?>
<box_vectors>
   <A>
      <x>6.491</x>
      <y>0.0</y>
      <z>0.0</z>
   </A>
   <B>
      <x>-2.163</x>
      <y>6.120000000000001</y>
      <z>0.0</z>
   </B>
   <C>
      <x>-2.163</x>
      <y>-3.0600000000000005</y>
      <z>5.300000000000001</z>
   </C>
</box_vectors>
"""
    box_vector = seekrtools.utilities.deserialize_box_vectors(
        xml_input, is_file=False)
    expected_vector = simtk.unit.Quantity(
        [[64.91, 0.0, 0.0], 
        [-21.63, 61.20, 0.0], 
        [-21.63, -30.60, 53.00]], 
        unit=simtk.unit.angstrom)
    for i in range(3):
        for j in range(3):
                assert expected_vector[i][j] == box_vector[i][j]

def test_utilities_serialize_deserialize_box_vectors(tmp_path):
    n = 10 # how many tests to run
    test_file = tmp_path / "test.xml"
    for i in range(n):
        box_vector1 = simtk.unit.Quantity(
            [[0.0, 0.0, 0.0], 
             [0.0, 0.0, 0.0], 
             [0.0, 0.0, 0.0]], 
            unit=simtk.unit.angstrom)
        box_vector2 = simtk.unit.Quantity(
            [[0.0, 0.0, 0.0], 
             [0.0, 0.0, 0.0], 
             [0.0, 0.0, 0.0]], 
            unit=simtk.unit.angstrom)
        for j in range(3):
            for k in range(3):
                val = simtk.unit.Quantity(random.uniform(-100.0,100.0), 
                                          unit=simtk.unit.angstrom)
                box_vector1[j][k] = val
                box_vector2[j][k] = val
        
        seekrtools.utilities.serialize_box_vectors(box_vector1, 
                                                   to_file=test_file)
        box_vector3 = seekrtools.utilities.deserialize_box_vectors(test_file)
        for j in range(3):
            for k in range(3):
                assert box_vector1[j][k] == box_vector3[j][k]
                assert box_vector2[j][k] == box_vector3[j][k]
                