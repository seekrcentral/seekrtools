Tutorial: Generating Trypsin/Benzamidine Starting Structures With HIDR
======================================================================

In this tutorial, we will be performing metadynamics (metaD) on the
equilibrated trypsin/benzamidine system to gently draw the ligand out of the
bound state, saving structures along the way, which we can then turn around and 
use as starting structures for a SEEKR2 calculation. This will be accomplished
using the HIDR (Holo Insertion by Directed Restraints) tool.

.. note::
  This tutorial assumes that you are using a computer equipped with one or
  more graphical processing units (GPUs). If your computer doesn't have a GPU,
  then you will need to transfer all files over to a computer equipped with
  a GPU (and also with OpenMM installed) in order to run MD simulations.

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

Constructing the SEEKR2 Model Input File
----------------------------------------

Just like SEEKR, the HIDR program needs a model.xml file in order to run, so
we will need to run the SEEKR2 prepare.py program on a Model Input file 
that contains no starting structures.

Download the following Model Input file:

:download:`input_tryp_ben_hidr.xml <media/input_tryp_ben_hidr.xml>`

Open up the file in a text viewer and take a look at the inputs. Notice that
every anchor has no <pdb_coordinates_filename> entries. This is because HIDR
is going to provide these for us. Notice however that the "tryp_ben.prmtop"
and PQR files were entered, since they do not vary by anchor.

First, we will run SEEKR2's prepare.py program to make us an empty model.::

  python ~/seekr2/seekr2/prepare.py input_tryp_ben_hidr.xml
  
Now the model XML file and the entire filetree has been generated at 
~/tryp_ben_hidr_tutorial/ (The <root_directory> tag in the Model input XML), 
but the model is empty - no PDB files define starting structures in any of
the anchor directories.

Running HIDR on the new Model using MetaD
-----------------------------------------

We will use HIDR's metadynamics (metaD) functions to slowly pull
the system into every anchor and save the structures for later SEEKR2
calculations.

.. note::
  HIDR can use other methods besides metaD to populate starting structures,
  including SMD, RAMD, and ratcheting. Consult the HIDR documentation as well as
  other Seekrtools tutorials to see how to use these other methods if so
  desired. (Tutorials are still be under construction at the time of writing).

Run HIDR with the following command.::

  python ~/seekrtools/seekrtools/hidr/hidr.py any ~/tryp_ben_hidr_tutorial/model.xml -M metaD -p tryp_ben.pdb

This command is likely to run for hours or possibly days, depending on the 
speed of your GPU.

Additional HIDR Settings for MetaD
----------------------------------

One can get a good overview of HIDR arguments by running HIDR with the "-h"
argument.::

  python ~/seekrtools/seekrtools/hidr/hidr.py -h
  
Some significant options include running metaD with lower Gaussian heights,
which can be an even more gentle way of drawing the ligand out of the site.
Care must be taken when choosing the optimal Gaussian height, as a value that
is too small will simply never exit, but a value that is too large might make
the ligand exit too harshly. The recommended procedure is to start at a 
relatively low value, and if the ligand doesn't escape in a reasonable amount 
of time (tens of ns), then use a progressively larger value until it escapes.::

  -H 0.2
  
Where to do next? You are ready to perform a SEEKR2 calculation. So if you
haven't already, visit the SEEKR2 tutorials to review how to run a SEEKR2
calculation, if needed.
