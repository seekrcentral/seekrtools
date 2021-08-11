"""
utilities.py
Seekrtools is a python library layer that interfaces with SEEKR programs such as 
SEEKR2 and provides a number of useful utilities and extensions.

This script provides a number of potentially useful functions and objects when
running OpenMM and OpenMM simulations
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom

#from simtk.unit import nanometer, Quantity
import simtk.unit as unit

def serialize_box_vectors(box_vectors, to_file=''):
    """
    Takes a 3x3 simtk.unit.Quantity representing an openMM or Parmed system box 
    vectors and serializes it as xml format and write to a file or a string

    Parameters
    ----------
    box_vectors : simtk.unit.Quantity, Required
        The boxVector object from parmed or OpenMM
        
    to_file : str, Optional, default: ''
        If writing the XML to a file is desired, then enter a valid file path
        and name. If to_file is an empty string '', then no file is written,
        the XML is only returned as a string.

    Returns
    -------
    xmlstr : str
        The XML written to a string
    """
    assert box_vectors is not None
    xmlBox_vectors = ET.Element('box_vectors')
    box_vectors_unitless = box_vectors.value_in_unit(unit.nanometer)
    xmlA = ET.SubElement(xmlBox_vectors, 'A')
    xmlAx = ET.SubElement(xmlA, 'x')
    xmlAx.text = str(box_vectors_unitless[0][0])
    xmlAy = ET.SubElement(xmlA, 'y')
    xmlAy.text = str(box_vectors_unitless[0][1])
    xmlAz = ET.SubElement(xmlA, 'z')
    xmlAz.text = str(box_vectors_unitless[0][2])
    xmlB = ET.SubElement(xmlBox_vectors, 'B')
    xmlBx = ET.SubElement(xmlB, 'x')
    xmlBx.text = str(box_vectors_unitless[1][0])
    xmlBy = ET.SubElement(xmlB, 'y')
    xmlBy.text = str(box_vectors_unitless[1][1])
    xmlBz = ET.SubElement(xmlB, 'z')
    xmlBz.text = str(box_vectors_unitless[1][2])
    xmlC = ET.SubElement(xmlBox_vectors, 'C')
    xmlCx = ET.SubElement(xmlC, 'x')
    xmlCx.text = str(box_vectors_unitless[2][0])
    xmlCy = ET.SubElement(xmlC, 'y')
    xmlCy.text = str(box_vectors_unitless[2][1])
    xmlCz = ET.SubElement(xmlC, 'z')
    xmlCz.text = str(box_vectors_unitless[2][2])
    
    xmlstr = minidom.parseString(ET.tostring(xmlBox_vectors)).toprettyxml(
            indent="   ")
    
    if to_file:
        our_file=open(to_file, 'w')
        our_file.write(xmlstr)
        our_file.close()
        
    return xmlstr

def deserialize_box_vectors(xmlInput, is_file=True):
    """
    Takes an XML string or file and converts to a 3x3 simtk.unit.Quantity for
    representing parmed or OpenMM box vectors.

    Parameters
    ----------
    xmlInput : str, Required
        The name of the file to read for XML, or, if is_file is False, then 
        read the xmlInput string itself as the XML
        
    is_file : bool, Optional, default: True
        If reading the XML to a file is desired, then enter a valid file path
        and name. If is_file is false, then the xmlInput string is read as
        the XML itself.

    Returns
    -------
    result : simtk.unit.Quantity
        The box vectors in a 3x3 simtk.unit.Quantity object for easy input to
        parmed or OpenMM.
    """
    if is_file:
        tree = ET.parse(xmlInput)
        xmlBox_vectors = tree.getroot()
    else:
        xmlBox_vectors = ET.fromstring(xmlInput)
        
    assert xmlBox_vectors.text is not None
    xmlA = xmlBox_vectors.find('A')
    xmlAx = float(xmlA.find('x').text)
    xmlAy = float(xmlA.find('y').text)
    xmlAz = float(xmlA.find('z').text)
    xmlB = xmlBox_vectors.find('B')
    xmlBx = float(xmlB.find('x').text)
    xmlBy = float(xmlB.find('y').text)
    xmlBz = float(xmlB.find('z').text)
    xmlC = xmlBox_vectors.find('C')
    xmlCx = float(xmlC.find('x').text)
    xmlCy = float(xmlC.find('y').text)
    xmlCz = float(xmlC.find('z').text)
    box_vectors = unit.Quantity([[xmlAx, xmlAy, xmlAz], 
                                 [xmlBx, xmlBy, xmlBz],
                                 [xmlCx, xmlCy, xmlCz]], 
                                 unit=unit.nanometer)
    return box_vectors

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    pass
