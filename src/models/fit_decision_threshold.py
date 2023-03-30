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
import re

import src.visualization.report_plot_model_vs_mouse_behaviour as plot_mouse_vs_model


def main():

    decision_threshold_val = 1
    model_number = 55
    drift_param_N = 1
    target_random_seed = None
    target_cv_idx = None
    model_type = 'drift'


    # Decision threshold search parameters
    small_norm_term = 0.01
    vis_exp_lower_bound = 0.59  # orignally 0.6
    vis_exp_init_guess = 0.595
    vis_exp_upper_bound = 0.6  # originally 3
    left_decision_threshold_search_vals = np.linspace(-0.3, -2, 50)
    right_decision_threshold_search_vals = np.linspace(0.3, 2, 50)
    # mouse behaviour to fit to
    no_aud_off_subset_active_behaviour_df = pd.read_pickle(
        '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/mouse_ephys_behaviour_compare_to_compare_w_drift_model_20.pkl')


    if model_number in [44, 45]:
        samples_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/'
        samples_save_name = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/max_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials.nc'
    elif model_number in [42, 46]:
        samples_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/'
        samples_save_name = 'naive_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_2.nc'
    elif model_number in [47, 49]:
        samples_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/'
        samples_save_name = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/naive_sorted_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_1.nc'
    elif model_number in [55]:
        samples_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/'
        samples_save_name = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/naive_sorted_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_2.nc'

    save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-models/'
    model_result_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f/' % model_number



    # GET pre_preprocessed_alignment_ds_dev, I think this is just used for getting the stimulus conditions
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
        if target_cv_idx is not None:
            search_str = '*shuffle_%.f_cv%.f.pkl' % (target_random_seed, target_cv_idx)
    else:
        search_str = '*%.f*.pkl' % drift_param_N


    # Loop through each model result separately
    model_result_fpaths = glob.glob(os.path.join(model_result_folder, search_str))
    model_result_fpaths = np.sort(model_result_fpaths)

    for fpath in tqdm(model_result_fpaths):

        random_seed = int(os.path.basename(fpath).split('_')[-2])
        cv_idx = int(re.findall(r'\d+', os.path.basename(fpath).split('_')[-1])[0])
        fit_results_save_path = os.path.join(
            model_result_folder, 'model_output_and_behaviour',
            'fit_results_df_fixed_vis_exp_seed_%.f_cv_%.f.csv' % (random_seed, cv_idx)
        )
        # skip already processed files
        if os.path.exists(fit_results_save_path):
            continue

        model_result = pd.read_pickle(fpath)
        y_test_pred_da_list = [model_result['y_test_pred_da']]
        all_stim_cond_pred_matrix_dict = jaxdmodel.get_stim_cond_response(
            alignment_ds=subset_alignment_ds, y_test_pred_da_list=y_test_pred_da_list,
            target_vis_cond_list=target_vis_cond_list, target_aud_cond_list=target_aud_cond_list
        )

        pd.options.mode.chained_assignment = None  # default='warn'
        aud_c_vis_cond_range = [-0.8, 0.8]
        aud_l_vis_cond_range = [-0.4, 0.8]
        aud_r_vis_cond_range = [-0.8, 0.4]
        fit_results_df = psychmodel.grid_search_decision_threshold(
            mouse_behaviour_df=no_aud_off_subset_active_behaviour_df,
            all_stim_cond_pred_matrix_dict=all_stim_cond_pred_matrix_dict,
            pre_preprocessed_alignment_ds_dev=pre_preprocessed_alignment_ds_dev,
            target_vis_cond_list=target_vis_cond_list, target_aud_cond_list=target_aud_cond_list,
            left_decision_threshold=left_decision_threshold_search_vals,
            right_decision_threshold=right_decision_threshold_search_vals,
            aud_c_vis_cond_range=aud_c_vis_cond_range,
            aud_l_vis_cond_range=aud_l_vis_cond_range,
            aud_r_vis_cond_range=aud_r_vis_cond_range,
            small_norm_term=small_norm_term,
            vis_exp_lower_bound=vis_exp_lower_bound,
            vis_exp_init_guess=vis_exp_init_guess,
            vis_exp_upper_bound=vis_exp_upper_bound,
            disable_progress_bar=True)

        fit_results_df.to_csv(fit_results_save_path)


if __name__ == '__main__':
    main()



