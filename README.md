# Experimental Data for the active-phase mapping paper
This branch hosts the codebase used to control and manipulate experimental syntehsis, characterization, and analysis of Gold nanoparticles work used in the active phase mapping paper.
There are two folders:
1. Seed mediated growth of Gold nanoparticles in `aunp_seed_mediated_phasemap`.
2. Bio-mineralization of Gold nanoparticles with peptides in `peptide_phase_mapping`.

You should be able to run many of these notebooks if you have installed the packages from the [activephasemap](https://github.com/pozzo-research-group/activephasemap/tree/main) program. 

Some notes:
    0. This code is provided as a way to understand the pre-processing and revisit any plotting routiens used. The main intention is to have all the data collected, processed, and visuzliaed in a single place. You should play around with the notebooks and install any missing packages. I have provided a pip froze version of the environment in `requirements.txt`.
    1. TEM image data is missing as it would be too large to host on GitHub but they are available in the backup data in Benson Hall. Look for a orange colored sandisk SSD with my name sticker on it.
    2. Some of the opentrons scipts use custom labware. These can be found in the pozzo group's [github page](https://github.com/pozzo-research-group/OT2-DOE/tree/main/OT2_DOE/Custom%20Labware)
