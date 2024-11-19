# clasp-model

This repository contains the data and code for my (upcoming) paper about the mechanisms driving root zonation in *A. thaliana* CLASP mutants. The bulk of this project was done between May and November 2024, under the supervision of [Dr. Eric Cytrynbaum](https://personal.math.ubc.ca/~cytryn/index.shtml) (Primary Supervisor, UBC Mathematics) and [Dr. Geoffrey Wasteneys](https://wasteneyslab.wixsite.com/ubcwasteneys) (Experimental Collaborator, UBC Botany). Prior to the summer, I was awarded an [NSERC USRA](https://www.nserc-crsng.gc.ca/students-etudiants/ug-pc/usra-brpc_eng.asp) to carry out this project. 

A working environment for running the notebooks contained in this repository can be found in the `environment.yml` file. Not all of the packages in this file are strictly necessary, as I have added various Python packages to my environment over the past six months that did not end up in the final notebooks. If you wish to create your own environment, the libraries `numpy`, `scipy`, `sympy`, `matplotlib`, `pandas`, and `numba` at the versions specified in `environment.yml` are absolutely necessary.

Data for the project is contained in the `/data` directory. Images used in the paper can be found in the `/img` directory. The code for the project is divided into six notebooks and the `modules.py` file which contains various helper functions.

- `final-plots.ipynb` contains the script used to process the raw cell file data as well as some figures related to this data processing.
- `final-bes1.ipynb` contains the intracellular model used in the paper. `final-bes1-modified.ipynb` is a failed experiment where I attempted to differentiate the *brinCLASPpro* mutant from the wild type by modulating its elongation rate. 
- `final-column.ipynb` is the initial cell column model that fails to explain the *brinCLASPpro* mutant. `final-column-modified.ipynb` contains the modified cell column model with a biphasic division function that successfully explains all three mutants.
- `final-column-atrichbolast.ipynb` is the same model as in `final-column-modified.ipynb`, just fitted to atrichoblast data. Differences in the results between the two cell column types are discussed in the paper.

I tried my best to adequately document all the code I wrote for this project. However, please feel free to reach out to me with any questions. My email is `rileywheadon@gmail.com`. Thanks for taking a look at my work!

