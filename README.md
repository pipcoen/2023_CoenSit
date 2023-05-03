# 2023_CoenSit

Code to replicate the figures from Coen, Sit, et al--2023.
Code comments and examples may be updated compared to the cited version on Zenodo. 

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


