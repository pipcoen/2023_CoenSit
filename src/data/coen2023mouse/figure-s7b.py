"""
This scripts generate figure S7b of the paper.
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

fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
fig_name = 'fig-s7b.pdf'

def main():

    # model_results_folder = '/Volumes/Partition 1/reports/figures/passive-kernel-regression/regression-psth-cv-include-unimodal-models-2-fold-cv-include-mse-w-kernels-2021-03-16/'
    model_results_folder = '/Volumes/Partition 1/reports/figures/passive-kernel-regression/regression-psth-cv-coherent-and-conflict-new-error-window'

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

    min_bias_kernel = 0  # 0.2
    min_var_explained = 0.02  # 0.02
    metrics_to_plot = ['audKernelMeanAbs', 'visKernelMeanAbs']

    plot_best_fit_line = True
    metric_mode = 'signed'
    mouseExpGrouped_mean_metric_df['audKernelMeanAbs'] = np.abs(mouseExpGrouped_mean_metric_df['audKernelMean'])
    mouseExpGrouped_mean_metric_df['visKernelMeanAbs'] = np.abs(mouseExpGrouped_mean_metric_df['visKernelMean'])

    # mouseExpGrouped_mean_metric_df['audKernelMeanOverBiasAbs'] = np.abs(mouseExpGrouped_mean_metric_df['audKernelMeanOverBias'])
    # mouseExpGrouped_mean_metric_df['visKernelMeanOverBiasAbs'] = np.abs(mouseExpGrouped_mean_metric_df['visKernelMeanOverBias'])

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = vizmodel.plot_regression_aud_lr_vs_vis_lr_weights(mouseExpGrouped_mean_metric_df,
                                                                    metrics_to_plot=metrics_to_plot,
                                                                    title_name='Mean of temporal kernel',
                                                                    metric_mode=metric_mode,
                                                                    min_var_explained=min_var_explained, cal_corr=True,
                                                                    plot_best_fit_line=plot_best_fit_line,
                                                                    min_bias_kernel=min_bias_kernel,
                                                                    fig=None, ax=None)
        ax.set_xlim([-0.05, 6])
        ax.set_ylim([-0.05, 6])

        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()