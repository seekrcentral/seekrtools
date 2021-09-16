seekrtools
==============================

Simulation-Enabled Estimation of Kinetic Rates Tools

Seekrtools is a set of software tools that interfaces with SEEKR programs, 
most notably with SEEKR2, in order to prepare and facilitate multiscale
milestoning calculations.

This README is only a quickstart guide to get Seekrtools up and running as soon 
as possible. To see more detailed **instructions** and **tutorials**, please see 
https://seekrtools.readthedocs.io/en/latest or the docs/ subfolder.

## Quick Install

### Dependencies
- SEEKR2
- OpenMM (see SEEKR2 documentation to install OpenMM alongside SEEKR2)

Make sure that you have installed SEEKR2 before Seekrtools.
(Most Seekrtools programs have SEEKR2 as a dependency). You can find the SEEKR2 
Github repostory at https://github.com/seekrcentral/seekr2.git
and the SEEKR2 documentation at https://seekr2.readthedocs.io/en/latest.


### Install Seekrtools
If you are using Conda (recommended) with SEEKR, make sure that the environment
is activated before executing the following steps to install Seekrtools:

```
git clone https://github.com/seekrcentral/seekrtools.git
cd seekrtools
python setup.py install
```

## Authors and Contributors

The following people have contributed directly to the coding and validation
efforts of Seekrtools (listed an alphabetical order of last name). 
Thanks also to everyone who has helped or will help improve this project by 
providing feedback, bug reports, or other comments.

* Rommie Amaro (principal investigator)
* Anand Ojha (developer)
* Andy Stokely (developer)
* Lane Votapka (lead developer)

### Citing Seekrtools

If you use Seekrtools, please cite one or more of the following SEEKR papers:

* Votapka, L. W.; Stokely, A. M.; Ojha, A. A.; Amaro, R. E. SEEKR2: Versatile Multiscale Milestoning Utilizing the OpenMM 7.5 Molecular Dynamics Engine. J. Chem. Inf. Mod. In Review. https://doi.org/10.33774/chemrxiv-2021-pplfs

* Votapka, L. W.; Jagger, B. R.; Heyneman, A. L.; Amaro, R. E. SEEKR: Simulation Enabled Estimation of Kinetic Rates, A Computational Tool to Estimate Molecular Kinetics and Its Application to Trypsin–Benzamidine Binding. J. Phys. Chem. B 2017, 121 (15), 3597–3606. https://doi.org/10.1021/acs.jpcb.6b09388. 

* Jagger, B. R.; Ojha, A. A.; Amaro, R. E. Predicting Ligand Binding Kinetics Using a Markovian Milestoning with Voronoi Tessellations Multiscale Approach. J. Chem. Theory Comput. 2020. https://doi.org/10.1021/acs.jctc.0c00495. 

* Jagger, B. R.; Lee, C. T.; Amaro, R. E. Quantitative Ranking of Ligand Binding Kinetics with a Multiscale Milestoning Simulation Approach. J. Phys. Chem. Lett. 2018, 9 (17), 4941–4948. https://doi.org/10.1021/acs.jpclett.8b02047. 

* Votapka LW, Amaro RE (2015) Multiscale Estimation of Binding Kinetics Using Brownian Dynamics, Molecular Dynamics and Milestoning. PLOS Computational Biology 11(10): e1004381. https://doi.org/10.1371/journal.pcbi.1004381


### Copyright

Copyright (c) 2021, Lane Votapka


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
