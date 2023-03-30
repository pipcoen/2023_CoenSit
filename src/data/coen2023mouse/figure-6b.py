"""
This scripts plots figure 6b from the paper.
This is the accumualtor model output for the trained mice.

Internal notes:

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

import jax
# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')

import src.models.predict_model as pmodel
import src.models.jax_decision_model as jaxdmodel
import time

import src.data.analyse_spikes as anaspikes
import src.data.process_ephys_data as pephys

import src.visualization.vizmodel as vizmodel
import src.visualization.vizpikes as vizpikes
import src.visualization.vizstat as vizstat
import src.visualization.vizbehaviour as vizbehave
import src.models.psychometric_model as psychmodel
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

import src.visualization.report_plot_model_vs_mouse_behaviour as plot_mouse_vs_model

fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
fig_name = 'fig-6b.pdf'


def main():
    alignment_ds_path = '/Volumes/Partition 1/data/interim/multispaceworld-rnn/new_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials.nc'
    model_number = 18
    # model_result_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/drift-model-%.f/'% model_number
    model_result_folder = '/Volumes/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f' % model_number
    target_vis_cond_list = [-0.8, -0.4, -0.2, 0.2, 0.4, 0.8, 0.0, 0.0, 0.8, 0.8, -0.8, -0.8, -0.1, 0.1]
    target_aud_cond_list = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                            -60, 60, 60, -60, 60, -60, np.inf, np.inf]
    all_stim_cond_pred_matrix_dict, pre_preprocessed_alignment_ds_dev = jaxdmodel.load_model_outputs(
        model_result_folder, alignment_ds_path, target_random_seed=None,
        target_vis_cond_list=target_vis_cond_list,
        target_aud_cond_list=target_aud_cond_list, drift_param_N=1)

    peri_stim_time = np.linspace(-0.1, 0.3, 143)
    # peri_stim_time = np.linspace(-0.1, 0.3, 200)
    ave_method = 'mean'

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = vizmodel.plot_multiple_model_stim_cond_output(all_stim_cond_pred_matrix_dict,
                                                                peri_stim_time=peri_stim_time,
                                                                include_decision_threshold_line=True,
                                                                ave_method=ave_method)
        fig.tight_layout()

        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()