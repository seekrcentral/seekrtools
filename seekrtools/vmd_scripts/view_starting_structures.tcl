# This script can be called from VMD and will nicely visualize all the
# starting structures in each of the anchors.

# How to use this function from the VMD tcl console:
# source view_starting_structures.tcl
# view_starting_structures [/path/to/model/directory]

proc extract_rec_list_from_model_file {model_file_name} {
    set infile [open $model_file_name r]
    set rec_index_list ""
    
    while { [gets $infile line] >= 0 } {
        if {[string first "group1_e" $line] != -1} {
            set expr_str1 [string first ">" $line]
            set num_start_index [expr "$expr_str1 + 1"]
            set expr_str2 [string first "/" $line]
            set num_end_index [expr "$expr_str2 - 2"]
            set atom_index [string range $line $num_start_index $num_end_index]
            lappend rec_index_list $atom_index
            
        }
    }
    close $infile
    return $rec_index_list
}

proc extract_lig_list_from_model_file {model_file_name} {
    set infile [open "model.xml" r]
    set lig_index_list ""
    
    while { [gets $infile line] >= 0 } {
        if {[string first "group2_e" $line] != -1} {
            set expr_str1 [string first ">" $line]
            set num_start_index [expr "$expr_str1 + 1"]
            set expr_str2 [string first "/" $line]
            set num_end_index [expr "$expr_str2 - 2"]
            set atom_index [string range $line $num_start_index $num_end_index]
            lappend lig_index_list $atom_index
        }
    }
    close $infile
    return $lig_index_list
}

proc view_starting_structures {{path_to_model_dir "."} {rec_sel "protein"}} {
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
        set file_list_raw [ls "*.pdb"]
        set file_list [string trim $file_list_raw "*" ]
        set pdb_file [lindex $file_list 1]
        puts "pdb_file: $pdb_file"
        if {$pdb_file == ""} {cd ../..; continue}
        if {$i == 0} {
            set my_molecule [mol new $pdb_file first 0 last 0]
        } else {
            mol addfile $pdb_file first 0 last 0
        }
        cd ../..
    }
    
    # Align all structures
    set all_atoms_frame_0 [atomselect $my_molecule all frame 0]
    set protein_atoms_frame_0 [atomselect $my_molecule $rec_sel frame 0]
    for {set i 0} {$i < $num_anchors} {incr i} {
        set all_atoms_frame_i [atomselect $my_molecule all frame $i]
        set protein_atoms_frame_i [atomselect $my_molecule $rec_sel frame $i]
        $all_atoms_frame_i move [measure fit $protein_atoms_frame_i $protein_atoms_frame_0]
    }
    
    # Draw the proteins and ligands nicely
    mol delrep 0 $my_molecule
    mol representation NewCartoon
    mol selection "$rec_sel"
    mol addrep $my_molecule
    
    set model_file_name "model.xml"
    set rec_index_list [extract_rec_list_from_model_file $model_file_name]
    set lig_index_list [extract_lig_list_from_model_file $model_file_name]
    
    mol representation Licorice
    #mol selection "not $rec_sel and not water and not name 'Na+' 'Cl-' 'K+'"
    mol selection "index $lig_index_list"
    mol addrep $my_molecule
    
    # Draw receptor site atoms
    mol representation VDW
    mol selection "index $rec_index_list"
    mol addrep $my_molecule
    
    # Draw lig and site COM spheres
    set site [atomselect top "index $rec_index_list"]
    $site frame 0
    draw color cyan
    draw sphere [measure center $site] radius 0.5 resolution 22
    
    set lig [atomselect top "index $lig_index_list"]
    $lig frame 0
    draw color green
    draw sphere [measure center $lig] radius 0.5 resolution 22
    
    puts "Cyan sphere: center of site"
    puts "Green sphere: center of ligand"
}
