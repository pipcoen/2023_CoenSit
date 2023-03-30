"""
This plots figure 6f of the paper.
This is the psychometric fit of the behaviour output of the naive model.


Internal notes:
This is from notebook 21.22
Note that it currently requires xarray == 0.19.0
TODO: there are some pickle files that should be convereted to something more readable

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
fig_name = 'fig-6f.pdf'


def main():
    decision_threshold_val = 1
    model_number = 47
    drift_param_N = 1
    target_random_seed = 0

    # save_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn'
    # model_result_folder = '/media/timsit/T7/drift-model-%.f/'% model_number
    # save_name = 'naive_sig_124_neurons_samples_stim_subset.nc'
    # save_name = 'naive_sig_124_neurons_samples_stim_subset_train_test.nc'
    # save_name = 'naive_sig_141_neurons_samples_stim_subset_train_test_w_labels.nc' # before model 40
    # save_name = 'naive_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels.nc'  # for model 40 and 41

    # Model 42
    # samples_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/'
    # samples_save_name = 'naive_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_2.nc'

    # Model 47
    model_result_folder = '/Volumes/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f/' % model_number
    samples_save_folder = '/Volumes/Ultra Touch/multispaceworld-rnn-samples/'
    samples_save_name = 'naive_sorted_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_1.nc'

    alignment_ds = xr.open_dataset(os.path.join(samples_save_folder, samples_save_name))

    start_time = -0.1
    end_time = 0.3

    # re-index trial values
    alignment_ds = alignment_ds.assign_coords({'Trial': np.arange(0, len(alignment_ds.Trial.values))})

    subset_neuron_idx = [0, 1]
    subset_alignment_ds = alignment_ds.isel(Cell=subset_neuron_idx)

    pre_preprocessed_alignment_ds_dev = subset_alignment_ds.where(
        (subset_alignment_ds['PeriEventTime'] >= start_time) &
        (subset_alignment_ds['PeriEventTime'] <= end_time), drop=True
    )

    target_vis_cond_list = [-0.8, -0.4, -0.2, 0.2, 0.4, 0.8, 0.0, 0.0, 0.8, 0.8, -0.8, -0.8,
                            -0.1, 0.1]
    target_aud_cond_list = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                            -60, 60, 60, -60, 60, -60, np.inf, np.inf]

    if target_random_seed is not None:
        search_str = '*shuffle_%.f*.pkl' % target_random_seed

    else:
        search_str = '*%.f*.pkl' % drift_param_N

    model_results = [pd.read_pickle(x) for x in glob.glob(os.path.join(model_result_folder, search_str))]

    print('Number of model results found: %.f' % len(model_results))

    model_type = 'drift'

    y_test_pred_da_list = [m_result['y_test_pred_da'] for m_result in model_results]
    # y_test_pred_da_list = [m_result['y_dev_pred_da'] for m_result in model_results]

    all_stim_cond_pred_matrix_dict = jaxdmodel.get_stim_cond_response(
        alignment_ds=subset_alignment_ds, y_test_pred_da_list=y_test_pred_da_list,
        target_vis_cond_list=target_vis_cond_list, target_aud_cond_list=target_aud_cond_list
    )

    model_behaviour_df = jaxdmodel.all_stim_cond_pred_matrix_dict_to_model_behaviour_df(
        all_stim_cond_pred_matrix_dict=all_stim_cond_pred_matrix_dict,
        alignment_ds=pre_preprocessed_alignment_ds_dev,
        right_decision_threshold_val=decision_threshold_val,
        left_decision_threshold_val=-decision_threshold_val,
        model_type=model_type,
        left_choice_val=0, right_choice_val=1,
        target_vis_cond_list=target_vis_cond_list, target_aud_cond_list=target_aud_cond_list
    )

    # remove early and no response trials
    go_model_behaviour_df = model_behaviour_df.loc[
        model_behaviour_df['reactionTime'] >= 0
        ]

    # model_results[0].keys()

    no_aud_off_subset_active_behaviour_df = pd.read_pickle(
        '/Volumes/Partition 1/data/interim/multispaceworld-rnn/mouse_ephys_behaviour_compare_to_compare_w_drift_model_20.pkl')

    vis_exp_lower_bound = 0.59  # orignally 0.6
    vis_exp_init_guess = 0.595
    vis_exp_upper_bound = 0.6  # originally 3

    left_decision_threshold_val = -0.89
    right_decision_threshold_val = 1.06

    # For model 31
    # left_decision_threshold_val = -2
    # right_decision_threshold_val = 1.757

    # For model 30
    # left_decision_threshold_val = -1.20
    # right_decision_threshold_val = 1.44

    # For model 40

    left_decision_threshold_val = -2
    right_decision_threshold_val = 1.72

    # Model 42
    left_decision_threshold_val = -1.930612
    right_decision_threshold_val = 1.722449

    # model 47
    left_decision_threshold_val = -1.791837
    right_decision_threshold_val = 1.965306

    model_small_norm_term = 0.01
    fig, ax = plot_mouse_vs_model.plot_mouse_vs_model_psychometric(all_stim_cond_pred_matrix_dict,
                                                                   pre_preprocessed_alignment_ds_dev,
                                                                   no_aud_off_subset_active_behaviour_df,
                                                                   target_vis_cond_list=[-0.8, -0.4, -0.2, 0.2, 0.4,
                                                                                         0.8, 0.0, 0.0, 0.8, 0.8, -0.8,
                                                                                         -0.8, -0.1, 0.1],
                                                                   target_aud_cond_list=[np.inf, np.inf, np.inf, np.inf,
                                                                                         np.inf, np.inf,
                                                                                         -60, 60, 60, -60, 60, -60,
                                                                                         np.inf, np.inf],
                                                                   left_decision_threshold_val=left_decision_threshold_val,
                                                                   right_decision_threshold_val=right_decision_threshold_val,
                                                                   all_17_mice_popt=[-0.1268, -2.5418, 2.7152, 0.6510,
                                                                                     -1.4541, 1.7149],
                                                                   model_scatter_marker=['o', 'o'],
                                                                   custom_logodds=False,
                                                                   vis_exp_lower_bound=vis_exp_lower_bound,
                                                                   vis_exp_init_guess=vis_exp_init_guess,
                                                                   vis_exp_upper_bound=vis_exp_upper_bound,
                                                                   model_small_norm_term=model_small_norm_term)


    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()




