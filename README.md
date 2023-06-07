# 2023_CoenSit

Code to replicate the figures from Coen, Sit, et al--2023.
Code comments and examples may be updated compared to the cited version on Zenodo.

The corresponding data is available here:

Once you have downloaded, unzip is such that you have a single directory containing the sub-directories for each mouse (PC011, PC012 etc.) and the two additional folders "XHistology" and "XSupData"

You should then change line 34 in the "helpers/+prc/pathFinder.m" function to match this directory

## Matlab instructions

Requires the "Optimizaton toolbox" in MATLAB

## Python instructions

First you need to install the python environment in `src/environment.yml`

```
conda env create -f environment.yml 
```

then you need to install `src` as a package, change your directory to `src` and do

```
pip install -e .
```

The code to generate figures are in `src/data/coen2023mouse`


