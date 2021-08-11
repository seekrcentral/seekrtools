Tutorial: Generating Trypsin/Benzamidine Starting Structures
============================================================

In this tutorial, we will be performing steered molecular dynamics (SMD) on the
equilibrated trypsin/benzamidine system to pull the ligand slowly out of the
bound state, saving structures along the way, which we can then turn around and 
use as starting structures for a SEEKR2 calculation.

Gather the Necessary Files
--------------------------

If you completed the previous tutorial in this series, namely, the
:doc:`Trypsin/Benzamidine system equilibration<tutorial_tryp_ben_equilibrate>`
, then you should have all the necessary files, but in case you don't, you
can download them below:

:download:`tryp_ben.prmtop <media/tryp_ben.prmtop>`

:download:`tryp_ben.inpcrd <media/tryp_ben.inpcrd>`

:download:`tryp_ben.pdb <media/tryp_ben.pdb>`

:download:`tryp_ben_receptor.pqr <media/tryp_ben_receptor.pqr>`

:download:`tryp_ben_ligand.pqr <media/tryp_ben_ligand.pqr>`

:download:`equilibrated.pdb <media/equilibrated.pdb>`

If you lost or forgot the final ligand-site distance from the equilibration
calculation, you can (probably) safely assume that the final distance was
approximately 0.05 nm (0.5 Angstroms).

Modify the SMD Script
---------------------

A sample SMD script is provided in the Seekrtools git repository.
Instead of using the script directly, you should copy it over into the same
directory as your other files::

  cp /PATH/TO/seekrtools/seekrtools/pullOutLigand_SMD.py tryp_ben_smd.py
  
Obviously, change /PATH/TO/seekrtools to the actual path, and make sure that
you are copying into the correct directory as well (The one with the
.prmtop, .inpcrd, .pdb, and .pqr files.

Open up your script, named **tryp_ben_smd.py** in a text editor like vim,
emacs, or gedit. Alternatively, if you prefer, you could use an IDE for 
writing Python code.

View the section below the comment labeled **# MODIFY THESE VARIABLES**. Let us
consider each line.

The line defining **rec_indices** must be **changed** to the same values that
you had used in the **tryp_ben_equil.py** script, which should have been the 
following::

  rec_indices = [2478, 2489, 2499, 2535, 2718, 2745, 2769, 2787, 2794, 2867, 
      2926]
      
Similarly, the line defining **lig_indices** must also be **changed** to the 
same values that you had used in the **tryp_ben_equil.py** script. They were
equal to the following::

  lig_indices = [3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229]
  
**Change** the line ``prmtop_filename = "equilibrated.parm7"`` to 
``prmtop_filename = "tryp_ben.prmtop"``

**Change** the line ``inpcrd_filename = "equilibrated.rst7"`` to 
``inpcrd_filename = "tryp_ben.inpcrd"``

.. note::
  Did you know that *.parm7* files are equivalent to *.prmtop* files? Similarly,
  *.rst7* files are equivalent to *.inpcrd* files.

Leave the line ``input_pdb_file = "equilibrated.pdb"`` as it is. This was the
output structure from your **tryp_ben_equil.py** script in the last tutorial.

**Change** the line ``temperature = 298.15 * unit.kelvin``  to 
``temperature = 298.0 * unit.kelvin``, as you had done for the 
**tryp_ben_equil.py** script.

Leave the line ``spring_constant = 90000.0*unit.kilojoules_per_mole / 
unit.nanometer**2`` as is. This is
the spring constant for the harmonic force with a steadily increasing 
equilibrium length to draw the molecule out of the binding site.

The setting ``cuda_device_index = "0"`` needs to be chosen based on which
GPU you wish to use for simulation, and will probably be the same as what
you used for the **tryp_ben_equil.py** script.

The setting ``nonbonded_cutoff = 0.9*unit.nanometer`` should be the same as
was used in the **tryp_ben_equil.py** script.

The setting for ``time_step = 0.002 * unit.picoseconds`` should remain, since
we want a timestep of 2 fs.

You can optionally change ``trajectory_filename = "smd_trajectory.pdb"``
if you want the trajectory file from the SMD simulation to have a different 
name.

The setting ``trajectory_interval = 100000`` tells the interval of steps
between when the trajectory file should be updated.

Leave ``total_num_steps = 50000000`` as it is, since this will be an SMD 
simulation lasting 100 ns. This value can be changed if a simulation of a
different length is desired.

Leave ``num_windows = 100``. This quantity represents how many "windows" there
will be, or how many distinct locations along the unbinding path that where
the harmonic force equilibrium distance will be adjusted to.

Leave ``show_state_output = False`` as it is - we don't want the regular
state information output showing to the standard output as the SMD simulation 
is proceeding.

Finally, the variable **target_radii** should be set to the following values::

  target_radii = [0.15, 0.25, 0.35, 0.45, 0.75, 0.85, 1.15, 1.25, 1.55, 1.65, 
      1.95]
  
These will be the locations of the *anchors* (in nm), or the centers of the 
Voronoi cells in the SEEKR2 calculation. We want to save the structures when 
they come close to anchor points because we want them to be far away from 
*milestones*.

Since we start at a ligand-site distance of around 0.05 nm, the first target
point was chosen to be 0.1 nm (1 Angstrom) beyond that.

.. note::
  How do the locations of the anchors relate to the locations of the milestones?
  By default (although it can be overrided), the milestone surfaces are located
  exactly mid-way between adjacent anchor points.

Choosing Anchor Points
----------------------

For your own system(s), you will need to choose the locations of your *anchor*
points. This is not a trivial task, and there does not yet exist any systematic
or automated way to choose optimal anchor placement - though we at the SEEKR 
team are working tirelessly towards that goal. 

The good news is: a SEEKR2 calculation is relatively insensitive to anchor
locations - so no matter what anchors you choose, the calculation will 
probably work out alright (within limits).

Some things to keep in mind when choosing anchor points:

   * Anchors should be spaced approximately equally in **free energy**. 
     Therefore, for most systems, this means that anchors should be placed
     closely near the bottom of the binding site, but spaced further apart
     out near or in the bulk solvent.
     
   * Practice shows that anchors should probably not be spaced closer than 
     0.05 nm (0.5 Angstroms) apart, nor further than 0.2 nm (2 Angstroms) apart.
     Too close, and transitions will lose velocity decorrelation and violate the
     Markov property. Too far apart, and not enough transitions will be observed
     to construct meaningful statistics.

Run the SMD Script
------------------

Now that the SMD script is ready, go ahead and run in Python::

  python tryp_ben_smd.py
  
If the script ran successfully without errors, then a number of PDB files
will have been generated with names like "smd_at0.15.pdb", "smd_at0.25.pdb", 
etc.

These files will be used within the **SEEKR2 model input file**.

Constructing the SEEKR2 Model Input File
----------------------------------------

Now, you should have all the files necessary for a SEEKR2 calculation.

For reference, use a text editor to open the *input_tryp_ben_mmvt.xml* file 
located in *seekr2/seekr2/data/trypsin_benzamidine_files* (in the SEEKR2 
repostory, you won't find this file in the Seekrtools repository).

First, consider one of the **<input_anchor>** XML blocks.::

  <input_anchor class="Spherical_cv_anchor">
        <radius>0.05</radius>
        <lower_milestone_radius/>
        <upper_milestone_radius>0.1</upper_milestone_radius>
        <starting_amber_params class="Amber_params">
            <prmtop_filename>data/trypsin_benzamidine_files/tryp_ben.prmtop</prmtop_filename>
            <box_vectors/>
            <pdb_coordinates_filename>data/trypsin_benzamidine_files/mmvt/tryp_ben_at0.pdb</pdb_coordinates_filename>
        </starting_amber_params>
        <bound_state>True</bound_state>
        <bulk_anchor>False</bulk_anchor>
    </input_anchor>
    
Notice that the **<prmtop_filename>** tag contains the name of the *.prmtop* 
file. Also, the **<pdb_coordinates_filename>** tag contains a PDB file. In our
case, we would probably rename either **equilibrated.pdb** or one of the 
**smd_at#.##.pdb** files to place into this tag.

Next, consider the **<browndye_settings_input>** tag.::

  <browndye_settings_input class="Browndye_settings_input">
        <binary_directory></binary_directory>
        <receptor_pqr_filename>data/trypsin_benzamidine_files/trypsin.pqr</receptor_pqr_filename>
        <ligand_pqr_filename>data/trypsin_benzamidine_files/benzamidine.pqr</ligand_pqr_filename>
        <apbs_grid_spacing>0.5</apbs_grid_spacing>
    ...

Notice the **<receptor_pqr_filename>** and **<ligand_pqr_filename>** tags. These
fields take the PQR files we generated when we parametrized the 
trypsin/benzamidine system in a previous tutorial: 
:doc:`Trypsin/Benzamidine system parametrization<tutorial_tryp_ben_parametrize>`.

Where to do next? You are ready to perform a SEEKR2 calculation. So if you
haven't already, visit the SEEKR2 tutorials to learn how to run a SEEKR2
calculation: https://seekr2.readthedocs.io/en/latest/tutorial.html.