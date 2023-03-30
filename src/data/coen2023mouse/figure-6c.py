"""
This scripts plots figure 6c from the paper.
This is the accumualtor model output for the naive mice.

Internal notes:
This is from notebook 21.22
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

def main():
    # Load data
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
    samples_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/'
    samples_save_name = 'naive_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_2.nc'

    # Model 47
    model_result_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f/' % model_number
    samples_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/'
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

    model_results[0].keys()

    all_stim_cond_pred_matrix_dict = jaxdmodel.get_stim_cond_response(
        alignment_ds=subset_alignment_ds, y_test_pred_da_list=y_test_pred_da_list,
        target_vis_cond_list=target_vis_cond_list, target_aud_cond_list=target_aud_cond_list
    )


    # Make the plot

    fig_ext = '.pdf'
    fig_name = 'X_drift_model_output_naive_model_%.f' % model_number
    ave_method = 'mean'

    # model_output_file = '/media/timsit/T7/drift-model-28/model_output_and_behaviour/model_output_per_stim_cond_test_seed_1.pkl'
    # model_output_file = '/media/timsit/T7/drift-model-29/model_output_and_behaviour/model_output_per_stim_cond_test_seed_0.pkl'
    # model_output_file = '/media/timsit/T7/drift-model-30/model_output_and_behaviour/model_output_per_stim_cond_test_seed_0.pkl'
    # model_output_file = '/media/timsit/T7/drift-model-31/model_output_and_behaviour/model_output_per_stim_cond_test_seed_0.pkl'
    # model_output_file = '/media/timsit/T7/drift-model-%.f/model_output_and_behaviour/model_output_per_stim_cond_test_seed_0.pkl' % model_number
    # model_output_file = '/media/timsit/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f/model_output_and_behaviour/model_output_per_stim_cond_test_seed_%.f.pkl' % (model_number, target_random_seed)
    # with open(model_output_file, 'rb') as handle:
    #      all_stim_cond_pred_matrix_dict = pkl.load(handle)

    # peri_stim_time = np.linspace(-0.1, 0.3, 143)
    peri_stim_time = np.linspace(-0.1, 0.3, 200)

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = vizmodel.plot_multiple_model_stim_cond_output(all_stim_cond_pred_matrix_dict,
                                                                peri_stim_time=peri_stim_time,
                                                                include_decision_threshold_line=True,
                                                                ave_method=ave_method)

        fig_folder = '/media/timsit/Partition 1/reports/figures/revision-figs-for-pip/'