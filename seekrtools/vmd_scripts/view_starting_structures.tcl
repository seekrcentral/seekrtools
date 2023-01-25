# This script can be called from VMD and will nicely visualize all the
# starting structures in each of the anchors.

# How to use this function from the VMD tcl console:
# source view_starting_structures.tcl
# view_starting_structures [/path/to/model/directory]

proc view_starting_structures {{path_to_model_dir "."}} {
    cd $path_to_model_dir
    
    # Find the anchor directories sorted in numerical order
    set sorted_anchor_dirs {}
    set unsorted_anchor_dirs [ls anchor_*]
    set num_anchors [expr "[llength $unsorted_anchor_dirs] - 1"]
    for {set i 0} {$i < $num_anchors} {incr i} {
        lappend sorted_anchor_dirs "anchor_${i}"
    }
    
    # Load the starting structure of each anchor into a VMD trajectory
    for {set i 0} {$i < $num_anchors} {incr i} {
        set anchor_dir [lindex $sorted_anchor_dirs $i]
        cd "[lindex $anchor_dir]/building"
        set pdb_file [lindex [ls "*.pdb"] 1]
        puts "pdb_file: $pdb_file"
        if {$i == 0} {
            set my_molecule [mol new $pdb_file first 0 last 0]
        } else {
            mol addfile $pdb_file first 0 last 0
        }
        cd ../..
    }
    
    # Align all structures
    set all_atoms_frame_0 [atomselect $my_molecule all frame 0]
    set protein_atoms_frame_0 [atomselect $my_molecule protein frame 0]
    for {set i 0} {$i < $num_anchors} {incr i} {
        set all_atoms_frame_i [atomselect $my_molecule all frame $i]
        set protein_atoms_frame_i [atomselect $my_molecule protein frame $i]
        $all_atoms_frame_i move [measure fit $protein_atoms_frame_i $protein_atoms_frame_0]
    }
    
    # Draw the proteins and ligands nicely
    mol delrep 0 $my_molecule
    mol representation NewCartoon
    mol selection "protein"
    mol addrep $my_molecule
    
    mol representation Licorice
    mol selection "not protein and not water and not name 'Na+' 'Cl-' 'K+'"
    mol addrep $my_molecule
}
