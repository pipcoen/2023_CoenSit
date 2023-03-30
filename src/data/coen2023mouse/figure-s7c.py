"""
This script generate figure S7c of the paper.

Internal
--------
This is from figure_for_paper_regression_model.py

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

import pdb



def plot_passive_kernel_lateralisation(all_models_df, passive_neuron_df_w_hemisphere,
                                       min_var_explained=0.02, min_kernel_mean_amp=None, include_cell_pos=False,
                                       fig_folder=None, fig_name=None):

    aud_only_model = all_models_df.loc[
        all_models_df['model'] == 'audOnly'
        ].groupby(['Exp', 'Cell']).agg('mean')

    vis_only_model = all_models_df.loc[
        all_models_df['model'] == 'visOnly'
        ].groupby(['Exp', 'Cell']).agg('mean')

    aud_plus_vis_model = all_models_df.loc[
        all_models_df['model'] == 'addition'
        ].groupby(['Exp', 'Cell']).agg('mean')

    MOs_passive_neuron_df_w_hemisphere = passive_neuron_df_w_hemisphere.loc[
        (passive_neuron_df_w_hemisphere['cellLoc'] == 'MOs') &
        (passive_neuron_df_w_hemisphere['subjectRef'] != 1)
        ]

    aud_plus_vis_model['hemisphere'] = MOs_passive_neuron_df_w_hemisphere['hemisphere'].values
    if include_cell_pos:
        aud_plus_vis_model['cellPos'] = MOs_passive_neuron_df_w_hemisphere['cellPos'].values

    # To look at Vis CPD, we compare full model (aud + vis) with model with vis removed (aud only)
    vis_cpd = (aud_only_model['varExplained'] - aud_plus_vis_model['varExplained']) / (aud_only_model['varExplained'])
    aud_cpd = (vis_only_model['varExplained'] - aud_plus_vis_model['varExplained']) / (vis_only_model['varExplained'])

    aud_plus_vis_model['aud_cpd'] = aud_cpd
    aud_plus_vis_model['vis_cpd'] = vis_cpd

    cpd_sig_a_and_v_neurons_df = aud_plus_vis_model.loc[
        (aud_plus_vis_model['aud_cpd'] > 0) &
        (aud_plus_vis_model['vis_cpd'] > 0) &
        (aud_plus_vis_model['varExplained'] >= 0.02)
        ]

    cpd_sig_a_neurons_df = aud_plus_vis_model.loc[
        (aud_plus_vis_model['aud_cpd'] > 0) &
        (aud_plus_vis_model['varExplained'] >= min_var_explained)
        ]

    cpd_sig_v_neurons_df = aud_plus_vis_model.loc[
        (aud_plus_vis_model['vis_cpd'] > 0) &
        (aud_plus_vis_model['varExplained'] >= min_var_explained)
        ]

    if min_kernel_mean_amp is not None:
        cpd_sig_a_neurons_df = cpd_sig_a_neurons_df.loc[
            np.abs(cpd_sig_a_neurons_df['audKernelMean']) >= min_kernel_mean_amp
        ]

        cpd_sig_v_neurons_df = cpd_sig_v_neurons_df.loc[
            np.abs(cpd_sig_v_neurons_df['visKernelMean']) >= min_kernel_mean_amp
        ]


    if fig_folder is not None:

        # Visual kernel, for significant visual neurons
        # fig_name = 'SupX_sig_cpd_vis_kernel_mean_left_right_hemisphere.pdf'

        print('Number of visual neurons %.f' % len(cpd_sig_v_neurons_df))

        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig, ax = vizmodel.plot_hemisphere_and_kernel_weights(cpd_sig_v_neurons_df,
                                                  metric_to_compare='visKernelMean', jitter_level=0.03,
                                                  min_var_explained=min_var_explained,
                                                  fig=None, ax=None)
            ax.set_ylabel('Visual kernel mean', size=12)

            fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
        """
        # Auditory kernel, for significant auditory neurons

        print('Number of auditory neurons %.f' % len(cpd_sig_a_neurons_df))

        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig, ax = vizmodel.plot_hemisphere_and_kernel_weights(cpd_sig_a_neurons_df,
                                                                  metric_to_compare='audKernelMean',
                                                                  jitter_level=0.03,
                                                                  min_var_explained=min_var_explained,
                                                                  fig=None, ax=None)
            #  ax.set_ylabel('Visual kernel mean', size=12)
            ax.set_ylabel('Auditory kernel mean', size=12)

            fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
        """



def main():

    fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
    fig_name = 'fig-s7c.pdf'

    process_params = dict(
        plot_passive_kernel_lateralisation=dict(
            neuron_types=['all', 'sig_kernel'],
            kernels_to_plot=['audSign', 'visSign', 'movLeft', 'movRight'],
            # neuron_df_w_hem_path='/media/timsit/Partition 1/data/interim/passive-m2-new-parent/subset/neuron_df_with_hem.pkl',
            # model_results_folder='/media/timsit/Partition 1/reports/figures/passive-kernel-regression/regression-psth-cv-include-unimodal-models-2-fold-cv-include-mse-w-kernels-2021-03-16/',
            # fig_folder='/media/timsit/Partition 1/reports/figures/supp-fig-for-pip/',
            neuron_df_w_hem_path='/Volumes/Partition 1/data/interim/passive-m2-new-parent/subset/neuron_df_with_hem.pkl',
            model_results_folder='/Volumes/Partition 1/reports/figures/passive-kernel-regression/regression-psth-cv-include-unimodal-models-2-fold-cv-include-mse-w-kernels-2021-03-16/',
            fig_folder=fig_folder,
            fig_name=fig_name,
            min_var_explained=0.02, min_kernel_mean_amp=0.05,
    ))


    param_dict = process_params['plot_passive_kernel_lateralisation']

    model_results_folder = param_dict['model_results_folder']

    # behave_df_path = '/media/timsit/Partition 1/data/interim/passive-m2-new-parent/subset/ephys_behaviour_df.pkl'
    behave_df_path = '/Volumes/Partition 1/data/interim/passive-m2-new-parent/subset/ephys_behaviour_df.pkl'
    all_models_df = psth_regression.load_combined_models_df(model_results_folder,
                                                            behave_df_path)

    passive_neuron_df_w_hemisphere = pd.read_pickle(
        param_dict['neuron_df_w_hem_path']
    )

    plot_passive_kernel_lateralisation(all_models_df, passive_neuron_df_w_hemisphere,
                                       min_var_explained=param_dict['min_var_explained'],
                                       fig_folder=param_dict['fig_folder'],
                                       min_kernel_mean_amp=param_dict['min_kernel_mean_amp'],
                                       fig_name=param_dict['fig_name'])

if __name__ == '__main__':
    main()