Installation
============

Dependencies
------------

 * SEEKR2 (https://seekr2.readthedocs.io/en/latest/installation.html) (Recommended)
 * OpenMM (http://docs.openmm.org/latest/userguide/library.html) (Recommended)
 * AmberTools (https://ambermd.org/AmberTools.php) (Optional - required for some tutorials)

Make sure that you have installed SEEKR2 before Seekrtools.
(Most Seekrtools programs have SEEKR2 as a dependency). You can find the SEEKR2 
Github repostory at https://github.com/seekrcentral/seekr2.git
and the SEEKR2 documentation at https://seekr2.readthedocs.io/en/latest.


Install Seekrtools
------------------
If you are using Conda (recommended) with SEEKR2, make sure that the environment
is activated before executing the following steps to install Seekrtools::

  git clone https://github.com/seekrcentral/seekrtools.git
  cd seekrtools
  python setup.py install
  
Test Seekrtools
---------------
One may also optionally run tests.::

  python setup.py test


  
  