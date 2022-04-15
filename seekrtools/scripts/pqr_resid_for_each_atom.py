"""
In Browndye, all the partial charges within a residue are lumped
into 'test charges'. In order to improve accuracy, one may give each
atom its own residue number. This script does this automatically.

Usage: python pqr_resid_for_each_atom.py INPUT_FILE OUTPUT_FILE
"""

import sys
import copy

import parmed

def pqr_resid_for_each_atom(old_pqr_filename, new_pqr_filename):
    old_pqr_struct = parmed.load_file(old_pqr_filename, skip_bonds=True)
    new_pqr_struct = parmed.Structure()
    counter = 0
    for i, old_residue in enumerate(old_pqr_struct.residues):
        resname = old_residue.name
        for j, old_atom in enumerate(old_residue.atoms):
            new_atom = copy.deepcopy(old_atom)
            new_pqr_struct.add_atom(new_atom, resname=resname, resnum=counter)
            counter += 1
    
    new_pqr_struct.save(new_pqr_filename, overwrite=True, renumber=True)
    return

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        exit()
        
    if sys.argv[1] in ["-h", "--help" "-help"]:
        print(__doc__)
        exit()
    
    assert len(sys.argv) == 3, "This script takes two arguments: INPUT_FILE and "\
        "OUTPUT_FILE."
    
    old_pqr_filename = sys.argv[1]
    new_pqr_filename = sys.argv[2]
    pqr_resid_for_each_atom(old_pqr_filename, new_pqr_filename)
    
