"""
This script produces figure 4d of the paper.
This is the fitted kernels for a neuron during the passive condition.

"""
import pandas as pd
import pickle as pkl
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

# Plotting
import sciplotlib.polish as splpolish
import sciplotlib.style as splstyle


import src.models.predict_model as pmodel
# import src.models.jax_decision_model as jaxdmodel
import time

import src.data.analyse_spikes as anaspikes
import src.data.process_ephys_data as pephys

import src.visualization.vizmodel as vizmodel
import src.visualization.vizpikes as vizpikes
import src.visualization.vizstat as vizstat
import src.visualization.vizbehaviour as vizbehave
# import src.models.psychometric_model as psychmodel
import src.models.psth_regression as psth_regression

from collections import defaultdict

import xarray as xr

import scipy.stats as sstats
import src.data.stat as stat

import itertools


# Single neuron decoding
import sklearn.linear_model as sklinear
import sklearn.model_selection as sklselect
import sklearn

import pdb


fig_folder = '/Users/timothysit/coen2023mouse'
fig_name = 'fig-4d.pdf'


def main():
    exp = 15
    cell_idx = 23
    passive_regression_folder = '/Volumes/Partition 1/reports/figures/passive-kernel-regression/regression-psth-cv-include-unimodal-models-2-fold-cv-include-mse-w-kernels/'
    passive_regression_data = pd.read_pickle(
        os.path.join(passive_regression_folder, 'exp%.f_regression_results.pkl') % exp)
    passive_performance_df = passive_regression_data['X_set_results']['addition']['model_performance_df']
    passive_performance_cv_mean = passive_performance_df.groupby('Cell').agg('mean')
    passive_kernels = passive_regression_data['X_set_results']['addition']['kernels']
    passive_kernels = passive_kernels.mean('Cv')

    correlation_window = [-0.05, 0.4]

    time_subset_passive_kernels = passive_kernels.sel(Time=slice(correlation_window[0], correlation_window[1]))

    # Calculate correlations
    num_cell = len(time_subset_passive_kernels.Cell)

    neuron_corr_dict = defaultdict(list)

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        fig.set_size_inches(8, 3)
        axs[0].set_title('Stim on', size=12)
        axs[1].set_title('Aud left/right', size=12)
        axs[2].set_title('Vis left/right', size=12)

        for n_feat, feat in enumerate(['stimOn', 'audSign', 'visSign']):
            passive_kernel = time_subset_passive_kernels.sel(Feature=feat)
            peri_event_time = time_subset_passive_kernels.Time

            axs[n_feat].plot(peri_event_time, passive_kernel.isel(Cell=cell_idx))

        # fig.text(-0.01, 0.7, 'Passive', size=12, rotation=0, ha='center')
        # fig.text(-0.01, 0.3, 'Active', size=12, rotation=0, ha='center')

        fig.text(0.5, 0, 'Peri-stimulus time (s)', size=12, ha='center')
        axs[0].set_ylabel('Firing rate (spikes/s)', size=12)
        fig.tight_layout()

        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()