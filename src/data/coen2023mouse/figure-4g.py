"""


Internal note:
This if from T7 recovery file of figure-5-for-pip
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
import time

import src.data.analyse_spikes as anaspikes
import src.data.process_ephys_data as pephys

import src.visualization.vizmodel as vizmodel
import src.visualization.vizpikes as vizpikes
import src.visualization.vizstat as vizstat
import src.visualization.vizbehaviour as vizbehave
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


# Settings for all the figures
fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
main_data_folder = '/Volumes/Partition 1/data/interim'
fig_name = 'fig-4g'
fig_ext = '.pdf'

def main():
    # model_results_folder = '/media/timsit/Partition 1/reports/figures/passive-kernel-regression/regression-psth-cv/'
    # model_results_folder = '/media/timsit/Partition 1/reports/figures/passive-kernel-regression/regression-psth-cv-include-unimodal-models/'

    # 2-fold cross validation
    # model_results_folder = '/media/timsit/Partition 1/reports/figures/passive-kernel-regression/regression-psth-cv-include-unimodal-models-2-fold-cv/'

    # 2021-03-16: Try this guy instead
    model_results_folder = '/Volumes/Partition 1/reports/figures/passive-kernel-regression/regression-psth-cv-include-unimodal-models-2-fold-cv-include-mse-w-kernels-2021-03-16/'

    behave_df_path = '/Volumes/Partition 1/data/interim/passive-m2-new-parent/subset/ephys_behaviour_df.pkl'
    all_models_df = psth_regression.load_combined_models_df(model_results_folder,
                                                            behave_df_path)

    addition_model_df = all_models_df.loc[
        all_models_df['model'] == 'addition'
        ]

    target_metric = ['audKernelMean', 'visKernelMean', 'varExplained', 'biasKernelMean',
                     'audKernelSignedMax', 'visKernelSignedMax', 'Subject', 'Exp', 'Cell']
    addition_model_cv_mean_metric = addition_model_df.groupby('neuron').agg('mean')[target_metric]

    passive_neuron_df_w_hemisphere = pd.read_pickle(
        '/Volumes/Partition 1/data/interim/passive-m2-new-parent/subset/neuron_df_with_hem.pkl'
    )

    MOs_passive_neuron_df_w_hemisphere = passive_neuron_df_w_hemisphere.loc[
        (passive_neuron_df_w_hemisphere['cellLoc'] == 'MOs') &
        (passive_neuron_df_w_hemisphere['subjectRef'] != 1)
        ]

    # sort by mouse and experiment, then give hemisphere information
    addition_model_cv_mean_metric = addition_model_cv_mean_metric.sort_values('Exp')
    mouseExpGrouped_mean_metric_df = addition_model_cv_mean_metric.reset_index().set_index(['Subject', 'Exp'])

    MOs_passive_neuron_df_w_hemisphere = MOs_passive_neuron_df_w_hemisphere.sort_values('expRef')

    # Move hemisphere information
    mouseExpGrouped_mean_metric_df['hemisphere'] = MOs_passive_neuron_df_w_hemisphere['hemisphere'].values

    # Scale auditory kernel and visual kernel by the bias kernel
    mouseExpGrouped_mean_metric_df['audKernelMeanOverBias'] = \
        mouseExpGrouped_mean_metric_df['audKernelMean'] / np.abs(mouseExpGrouped_mean_metric_df['biasKernelMean'])

    mouseExpGrouped_mean_metric_df['visKernelMeanOverBias'] = \
        mouseExpGrouped_mean_metric_df['visKernelMean'] / np.abs(mouseExpGrouped_mean_metric_df['biasKernelMean'])

    # fig_folder = '/media/timsit/Partition 1/reports/figures/figure-5-for-pip/'
    # fig_name = '4_aud_vs_vis_temporal_kernel_signed_w_exmaple_highlighted.pdf'

    vExpalined_to_alpha = True
    min_var_explained = 0.02
    max_var_explained = 0.99  # those at 1.0 very likely ones with zero firing rate.
    scale_color_range = [0.0, 1.0]
    scatter_size = 10
    pad_ratio = 0.015

    highlight_neurons = [{
        'Exp': 15, 'neuron': 23,
    }]

    highlight_scatter_size = 20

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = vizmodel.plot_regression_aud_lr_vs_vis_lr_weights(
            mouseExpGrouped_mean_metric_df,
            metrics_to_plot=['audKernelMean', 'visKernelMean'],
            title_name='Mean of temporal kernel',
            min_var_explained=min_var_explained,
            max_var_explained=max_var_explained,
            vExpalined_to_alpha=vExpalined_to_alpha,
            scale_color_range=scale_color_range,
            scatter_size=scatter_size,
            highlight_neurons=highlight_neurons,
            pad_ratio=pad_ratio, highlight_scatter_size=highlight_scatter_size,
            fig=None, ax=None)
        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
