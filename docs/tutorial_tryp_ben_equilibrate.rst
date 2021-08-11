Tutorial: Equilibrating the Trypsin/Benzamidine System
======================================================

In this tutorial, we will be equilibrating the trypsin/benzamidine system that
was parametrized in the previous tutorial, 
:doc:`Trypsin/Benzamidine system parametrization<tutorial_tryp_ben_parametrize>`

OpenMM must be installed for this tutorial.

.. note::
  This tutorial assumes that you are using a computer equipped with one or
  more graphical processing units (GPUs). If your computer doesn't have a GPU,
  then you will need to transfer all files over to a computer equipped with
  a GPU (and also with OpenMM installed) in order to run equilibration 
  simulations.

Gather the Necessary Files
--------------------------

If you completed the previous tutorial in this series, namely, the
:doc:`Trypsin/Benzamidine system parametrization<tutorial_tryp_ben_parametrize>`
, then you should have all the necessary files, but in case you don't, you
can download them below:

:download:`tryp_ben.prmtop <media/tryp_ben.prmtop>`

:download:`tryp_ben.inpcrd <media/tryp_ben.inpcrd>`

:download:`tryp_ben.pdb <media/tryp_ben.pdb>`

:download:`tryp_ben_receptor.pqr <media/tryp_ben_receptor.pqr>`

:download:`tryp_ben_ligand.pqr <media/tryp_ben_ligand.pqr>`

Modify the Equilibration Script
-------------------------------

A sample equilibration script is provided in the Seekrtools git repository.
Instead of using the script directly, you should copy it over into the same
directory as your other files::

  cp /PATH/TO/seekrtools/seekrtools/sample_md_min_equil.py tryp_ben_equil.py
  
Obviously, change /PATH/TO/seekrtools to the actual path, and make sure that
you are copying into the correct directory as well (The one with the
.prmtop, .inpcrd, .pdb, and .pqr files.

Open up your script, named **tryp_ben_equil.py** in a text editor like vim,
emacs, or gedit. Alternatively, if you prefer, you could use an IDE for 
writing Python code.

View the section below the comment labeled **# MODIFY THESE VARIABLES**. Let us
consider each line.

**Change** the line ``prmtop_filename = "molecule.prmtop"`` to 
``prmtop_filename = "tryp_ben.prmtop"``

**Change** the line ``inpcrd_filename = "molecule.inpcrd"`` to 
``inpcrd_filename = "tryp_ben.inpcrd"``

**Change** the line ``input_pdb_file = "molecule.pdb"`` to 
``input_pdb_file = "tryp_ben.pdb"``

You can keep the line ``trajectory_filename = "equilibration_trajectory.pdb"``
as it is, unless you want the equilibration trajectory to have a different
file name.

Keep ``steps_per_trajectory_update = 300000`` as it is, unless you wish the
trajectory file to be updated with a different interval.

Keep ``output_pdb_file = "equilibrated.pdb"`` as it is, unless you want the
final structure to have a different file name.

Keep the line ``minimize = True`` as it is so that minimizations will be
performed.

Keep the line ``num_steps = 30000000`` as it is to perform 60 ns of 
equilibration. Change this value if you want to equilibrate for a different
length of time.

Keep the line ``steps_per_energy_update = 5000`` as it is, unless you want to
be updated on system information in standard output with a different interval.

Keep the line ``time_step = 0.002 * unit.picoseconds`` to use a 2 fs
timestep.

The line defining **rec_indices** must be **changed**. Change it to the 
following::

  rec_indices = [2478, 2489, 2499, 2535, 2718, 2745, 2769, 2787, 2794, 2867, 
      2926]
      
This list represents the *atom indices* whose center of mass defines the 
binding site.

.. note::
  For your own system of interest, you will need to decide which atoms best 
  define the binding site of your receptor. This is not a trivial task, and the
  choice of atoms must be made carefully. Typically, we choose a set of ten to 
  twenty alpha carbons in receptor residues that surround the bound ligand
  molecule within a certain distance to the ligand. Alternatively, one may 
  choose the alpha carbons of residues which make key interactions with the 
  ligand. There is no clearly defined right way to make this choice of atoms. 
  Nevertheless, it would be wise to take the time to make a reasonable 
  selection for your own systems.
  
The line defining **lig_indices** must also be **changed**. Change it to the
following::

  lig_indices = [3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229]
  
This list represents the *atom indices* whose center of mass defines the 
ligand's location in space. In our case, we have chosen the heavy atoms
(non-hydrogen atoms) in the ligand molecule. Notice that the numbering is for
the atoms in the tryp_ben.prmtop file.

.. note::
  For your own system of interest, you will need to also choose the selection
  of atom indices to define your ligand. Fortunately, this is a much clearer
  task than choosing the atoms which define the binding site. We typically 
  select all heavy atoms (non-hydrogens) in the ligand molecule.

The line ``spring_constant = 9000.0 * unit.kilojoules_per_mole * 
unit.nanometers**2`` is the strength
of the harmonic restraint that will hold the ligand within the binding site.
This is a relatively high value, and users may wish to weaken this spring
constant somewhat if the ligand already binds to the site pretty tightly. For
now, one can just leave this as it is. The equilibrium distance for the 
harmonic restraint is determined by the input structure defined by the
**input_pdb_file** variable, and the distance measured between the centers
of masses of the atoms defined in **lig_indices** and **lig_indices** is
what defines that equilibrium distance for the harmonic restraint.

The line ``temperature = 298.15 * unit.kelvin`` defines the temperature of
the equilibration. Technically, we ran all simulations at 298 Kelvin, so
**change** this line to ``temperature = 298.0 * unit.kelvin``.

The line ``constant_pressure = True`` allows the water box to relax in a 
constant-pressure environment. You can leave this line as it is.

The line ``target_pressure = 1.0 * unit.bar`` instructs the barostat to seek
a target temperature of 1 bar. You can leave this line as it is.

The line ``cuda_index = "0"`` defines which GPU on your computer to use. If 
your computer is equipped with one GPU, then you can leave this line as it is.
However, if you are fortunate enough to have two or more GPUs, you can
experiment to see which *cuda_index* provides the best performance by changing
from ``"0"`` to ``"1"`` or ``"2"``, etc. If your system has zero GPUs, then you
must gain access to a computer with GPUs to run this script.

Finally, the line ``nonbonded_cutoff = 0.9 * unit.nanometer`` defines the
nonbonded cutoff for the MD simulation. Scientists use different values for this
entry depending on forcefield type, preference, and a number of other factors.
This line should be fine to use as it is for this tutorial.

Run the Equilibration
---------------------

When the input file is all ready, you can run the Python script.::

  python tryp_ben_equil.py
  
Assuming that no errors are encountered, the equilibration will begin running.
Depending on the value entered for *num_steps* in the **tryp_ben_equil.py**
script, and the speed of your GPU, the simulation may run for minutes, hours,
or days. (You can always shorten the number of steps in the "num_steps"
variable if it runs too long, though you probably want at least several tens
of nanoseconds to allow your system to equilibrate).

At the end of the simulation, a helpful benchmark of the equilibration 
simulations should be printed, which will give you an idea for the performance
that SEEKR2 is likely to have. Also, the **equilibrated.pdb** structure
will be generated.

Also the final distance between the ligand and the site will be printed.
Be sure to take note of this value, it should be approximately 0.05 nm
(0.5 Angstroms), you will need this quantity for the next tutorial.

Once **equilibrated.pdb** is obtained, and the final ligand-site distance, 
you may proceed to the next step.

Download any Missing Files
--------------------------

If anything went wrong with any steps above, you can download the file below
to use for later tutorials. 

:download:`equilibrated.pdb <media/equilibrated.pdb>`

