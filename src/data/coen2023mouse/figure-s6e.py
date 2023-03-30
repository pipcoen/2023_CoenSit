"""
This script generates figure S6e from the paper :
This is the projection of left/right choices for each stimulus condition

Internal notes:
This is from run_dPCA.py
"""
import pdb

import pandas as pd
import pickle as pkl
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

# Smoothing
import scipy.signal as spsignal
import scipy.ndimage as spimage

from tqdm import tqdm
import sciplotlib.style as splstyle
import sciplotlib.polish as splpolish


from pymer4.models import Lmer
import time

import src.data.analyse_spikes as anaspikes
import src.data.analyse_behaviour as anabehave
import src.data.process_ephys_data as pephys


import src.visualization.vizmodel as vizmodel
import src.visualization.vizpikes as vizpikes
import src.visualization.vizstat as vizstat

from collections import defaultdict

import xarray as xr

import scipy.stats as sstats
import src.data.stat as stat

import src.models.predict_model as pmodel
import sklearn.model_selection as skselect
import sklearn
import itertools


def cal_projection_to_choice(alignment_ds, activity_name ='firing_rate',
                             choice_var_name='choiceThreshDir',
                             train_test_split=False,
                             balance_stim_cond=False):
    """
    Calculates the projection of each trila's population vector to
    (1) the mean population vector for left choice across all trials
    (2) the mean population vector for right choice across all trials

    Parameters
    ----------
    alignment_ds : xarray dataset
    """

    # TODO: z-score firing rates?

    if train_test_split:

        # For each stimulus calculation, calculate the mean left and right population vector from all other conditions, then project the stimulus condition
        # of interest to that vector
        aud_levels = [-60, 60, 0]  # np.inf
        vis_levels = [[-0.8, -0.4, -0.2, 0.1], 0.0, [0.1, 0.2, 0.4, 0.8]]
        stim_levels = itertools.product(aud_levels, vis_levels)
        response_levels = [1, 2]

        condition_levels = itertools.product(stim_levels, response_levels)

        num_time = len(alignment_ds.Time)
        num_trial = len(alignment_ds.Trial)
        num_cell = len(alignment_ds.Cell)
        time_trial_projection_to_left = np.zeros((num_time, num_trial))
        time_trial_projection_to_right = np.zeros((num_time, num_trial))
        time_trial_projection_to_right_minus_left = np.zeros((num_time, num_trial))

        # Reset trial numbers to start at 0
        alignment_ds['Trial'] = ('Trial', np.arange(0, num_trial))

        for s_level, r_level in condition_levels:
            condition_subset_alignment_ds = alignment_ds.where(
                (alignment_ds['audDiff'] == s_level[0]) &
                (alignment_ds['visDiff'].isin(s_level[1])) &
                (alignment_ds[choice_var_name] == r_level),
                drop=True
            )

            stim_cond_activity = condition_subset_alignment_ds[activity_name].values

            # all_other_cond_alignment_ds = alignment_ds.where(
            #     ~alignment_ds['Trial'].isin(condition_subset_alignment_ds.Trial.values), drop=True
            #  )

            all_other_cond_alignment_ds = alignment_ds.where(
                (alignment_ds['audDiff'] != s_level[0]) &
                (~alignment_ds['visDiff'].isin(s_level[1])), drop=True
            )

            if balance_stim_cond:
                print('Subsetting trials from each stimulus condition to equalise the number of left/right trials')
                all_other_cond_alignment_ds = balance_choice_per_stim_cond(all_other_cond_alignment_ds,
                                                                           choice_var_name=choice_var_name)
                if all_other_cond_alignment_ds is None:
                    print('No stim cond meet balancing criteria, returning None')
                    return None

            mean_left_pop_vec = all_other_cond_alignment_ds.where(
                (all_other_cond_alignment_ds['PeriEventTime'] >= 0) &
                (all_other_cond_alignment_ds['PeriEventTime'] <= 0.1) &
                (all_other_cond_alignment_ds[choice_var_name] == 1), drop=True
            ).mean(['Time', 'Trial'])[activity_name].values

            mean_right_pop_vec = all_other_cond_alignment_ds.where(
                (all_other_cond_alignment_ds['PeriEventTime'] >= 0) &
                (all_other_cond_alignment_ds['PeriEventTime'] <= 0.1) &
                (all_other_cond_alignment_ds[choice_var_name] == 2), drop=True
            ).mean(['Time', 'Trial'])[activity_name].values


            # If one of the stimulus-choice pop vector is absent, then the projection is undefined
            if len(mean_left_pop_vec) == 0:
                mean_left_pop_vec = np.repeat(np.nan, (num_cell, ))
            if len(mean_right_pop_vec) == 0:
                mean_right_pop_vec = np.repeat(np.nan, (num_cell, ))

            mean_right_minus_left_pop_vec = mean_right_pop_vec - mean_left_pop_vec

            for time_idx in np.arange(num_time):
                for n_trial, trial_idx in enumerate(condition_subset_alignment_ds.Trial.values):

                    trial_vec = stim_cond_activity[:, time_idx, n_trial]
                    time_trial_projection_to_left[time_idx, trial_idx] = np.dot(mean_left_pop_vec, trial_vec) / (
                            np.linalg.norm(mean_left_pop_vec) * np.linalg.norm(trial_vec))
                    time_trial_projection_to_right[time_idx, trial_idx] = np.dot(mean_right_pop_vec, trial_vec) / (
                            np.linalg.norm(mean_right_pop_vec) * np.linalg.norm(trial_vec))
                    time_trial_projection_to_right_minus_left[time_idx, trial_idx] = np.dot(
                        mean_right_minus_left_pop_vec, trial_vec) / (np.linalg.norm(mean_right_minus_left_pop_vec) * np.linalg.norm(trial_vec))

        alignment_ds['projection_to_left'] = (['Time', 'Trial'], time_trial_projection_to_left)
        alignment_ds['projection_to_right'] = (['Time', 'Trial'], time_trial_projection_to_right)
        alignment_ds['projection_to_right_minus_left'] = (['Time', 'Trial'], time_trial_projection_to_right_minus_left)

    else:

        mean_left_pop_vec = alignment_ds.where(
            (alignment_ds['PeriEventTime'] >= 0) &
            (alignment_ds['PeriEventTime'] <= 0.1) &
            (alignment_ds[choice_var_name] == 1), drop=True
        ).mean(['Time', 'Trial'])[activity_name].values

        mean_right_pop_vec = alignment_ds.where(
            (alignment_ds['PeriEventTime'] >= 0) &
            (alignment_ds['PeriEventTime'] <= 0.1) &
            (alignment_ds[choice_var_name] == 2), drop=True
        ).mean(['Time', 'Trial'])[activity_name].values

        #if (np.sum(np.isnan(mean_left_pop_vec)) > 0) or (np.sum(np.isnan(mean_right_pop_vec)) > 0):
        #     pdb.set_trace()

        # TODO: need to deal with trial_vec being all zeros sometimes...

        cell_time_trial_vals = alignment_ds['firing_rate'].values
        num_time = len(alignment_ds.Time)
        num_trial = len(alignment_ds.Trial)
        time_trial_projection_to_left = np.zeros((num_time, num_trial))
        time_trial_projection_to_right = np.zeros((num_time, num_trial))

        mean_right_minus_left_pop_vec = mean_right_pop_vec - mean_left_pop_vec
        time_trial_projection_to_right_minus_left = np.zeros((num_time, num_trial))

        for time_idx in np.arange(num_time):
            for trial_idx in np.arange(num_trial):
                trial_vec = cell_time_trial_vals[:, time_idx, trial_idx]
                time_trial_projection_to_left[time_idx, trial_idx] = np.dot(mean_left_pop_vec, trial_vec) / (
                            np.linalg.norm(mean_left_pop_vec) * np.linalg.norm(trial_vec))
                time_trial_projection_to_right[time_idx, trial_idx] = np.dot(mean_right_pop_vec, trial_vec) / (
                            np.linalg.norm(mean_right_pop_vec) * np.linalg.norm(trial_vec))
                time_trial_projection_to_right_minus_left[time_idx, trial_idx] = np.dot(mean_right_minus_left_pop_vec, trial_vec) / (
                            np.linalg.norm(mean_right_minus_left_pop_vec) * np.linalg.norm(trial_vec))

        alignment_ds['projection_to_left'] = (['Time', 'Trial'], time_trial_projection_to_left)
        alignment_ds['projection_to_right'] = (['Time', 'Trial'], time_trial_projection_to_right)
        alignment_ds['projection_to_right_minus_left'] = (['Time', 'Trial'], time_trial_projection_to_right_minus_left)

    return alignment_ds


def group_projection_by_stim_cond_and_choice(alignment_ds, choice_var_name='choiceThreshDir'):
    """

    """
    # group projections

    aud_levels = [-60, 60, 0]  # np.inf

    vis_levels = [[-0.8, -0.4, -0.2, 0.1], 0.0, [0.1, 0.2, 0.4, 0.8]]
    stim_levels = itertools.product(aud_levels, vis_levels)
    response_levels = [1, 2]

    condition_levels = itertools.product(stim_levels, response_levels)

    aud_cond_dict = {-60: 'left', 60: 'right', 0: 'center'}
    response_cond_dict = {1: 'left', 2: 'right'}

    trial_mean_ds_list = list()
    trial_count_list = list()

    # choice_var_name = 'responseMade'
    # activity_name = 'projection_to_left'  # 'firing_rate'

    for s_level, r_level in condition_levels:
        condition_subset_alignment_ds = alignment_ds.where(
            (alignment_ds['audDiff'] == s_level[0]) &
            (alignment_ds['visDiff'].isin(s_level[1])) &
            (alignment_ds[choice_var_name] == r_level),
            drop=True
        )

        if type(s_level[1]) is not list:
            vis_cond = 'center'
        elif s_level[1][0] < 0:
            vis_cond = 'left'
        elif s_level[1][0] > 0:
            vis_cond = 'right'

        aud_cond = aud_cond_dict[s_level[0]]
        response_cond = response_cond_dict[r_level]

        trial_mean_activity_ds = condition_subset_alignment_ds.mean('Trial')

        n_dim_to_expand = 3

        if len(trial_mean_activity_ds.Cell) > 0:

            projection_to_left = trial_mean_activity_ds['projection_to_left'].isel(Cell=0).values
            projection_to_right = trial_mean_activity_ds['projection_to_right'].isel(Cell=0).values
            projection_to_right_minus_left = trial_mean_activity_ds['projection_to_right_minus_left'].isel(Cell=0).values

            trial_mean_ds = xr.Dataset(
                {'projection_to_left': (['Time', 'audCond', 'visCond', 'responseMade'],
                                        projection_to_left.reshape(projection_to_left.shape + (1,) * n_dim_to_expand)
                                        ),
                 'projection_to_right': (['Time', 'audCond', 'visCond', 'responseMade'],
                                         projection_to_right.reshape(projection_to_right.shape + (1,) * n_dim_to_expand)
                                         ),
                 'projection_to_right_minus_left': (['Time', 'audCond', 'visCond', 'responseMade'],
                                         projection_to_right_minus_left.reshape(projection_to_right_minus_left.shape + (1,) * n_dim_to_expand)
                                         ),
                 },
                coords={'Time': trial_mean_activity_ds['Time'],
                        'audCond': [aud_cond],
                        'visCond': [vis_cond],
                        'responseMade': [response_cond]}
            )

            trial_count_list.append(len(condition_subset_alignment_ds['Trial']))
            trial_mean_ds_list.append(trial_mean_ds)

    all_trial_cond_mean_ds = xr.combine_by_coords(trial_mean_ds_list)

    return all_trial_cond_mean_ds


def plot_projection(all_trial_cond_mean_ds, projection_name='projection_to_left',
                    projection_error_name=None, show_error_shade=False,
                    xlim=None, ylim=[0, 1], line_alpha=1, line_width=1, shade_alpha=0.3,
                    truncate_values=True,
                    fig=None, axs=None):
    """
    Parameters
    ----------
    all_trial_cond_mean_ds : xarray dataset
    projection_name : str
    xlim : list (optional)
    ylim : list (optional)
    projection_error_name : str
        name of the variable in the xarray dataset giving the error (standard deviation, or standard error)
        across trials or sessions
    fig : matplotlib figure object (optional)
    axs : matplotlib axes object (optional)
    line_alpha : float
        transparency of line (0 - 1)
    Returns
    -------
    """

    if truncate_values:
        print('Not plotting values beyond xlim')

    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)

    aud_conds = ['left', 'center', 'right']  # np.unique(all_trial_cond_mean_ds['audCond'].values)
    vis_conds = ['left', 'center', 'right']  # np.unique(all_trial_cond_mean_ds['visCond'].values)

    peri_event_time = all_trial_cond_mean_ds['PeriEventTime'].values

    for n_aud_cond, aud_cond in enumerate(aud_conds):

        for n_vis_cond, vis_cond in enumerate(vis_conds):

            if vis_cond not in all_trial_cond_mean_ds['visCond'].values:
                continue

            subset_ds = all_trial_cond_mean_ds.where(
                (all_trial_cond_mean_ds['audCond'] == aud_cond) &
                (all_trial_cond_mean_ds['visCond'] == vis_cond), drop=True
            )

            left_choice_projection_to_x = subset_ds.where(
                subset_ds['responseMade'] == 'left', drop=True
            )[projection_name].values.flatten()

            right_choice_projection_to_x = subset_ds.where(
                subset_ds['responseMade'] == 'right', drop=True
            )[projection_name].values.flatten()

            # right_choice_projection_to_left
            if len(left_choice_projection_to_x) > 0:
                if truncate_values:
                    subset_idx = np.where((peri_event_time >= xlim[0]) & (peri_event_time <= xlim[1]))[0]
                    axs[n_aud_cond, n_vis_cond].plot(peri_event_time[subset_idx],
                                                     left_choice_projection_to_x[subset_idx],
                                                     color='blue', alpha=line_alpha, lw=line_width)
                else:
                    axs[n_aud_cond, n_vis_cond].plot(peri_event_time,
                                                     left_choice_projection_to_x,
                                                 color='blue', alpha=line_alpha, lw=line_width)
                if show_error_shade:
                    left_choice_projection_to_x_error = subset_ds.where(
                        subset_ds['responseMade'] == 'left', drop=True
                    )[projection_error_name].values.flatten()
                    if truncate_values:
                        subset_idx = np.where((peri_event_time >= xlim[0]) & (peri_event_time <= xlim[1]))[0]
                        y_shade_lower = left_choice_projection_to_x - left_choice_projection_to_x_error
                        y_shade_upper = left_choice_projection_to_x + left_choice_projection_to_x_error
                        axs[n_aud_cond, n_vis_cond].fill_between(peri_event_time[subset_idx],
                                                                 y_shade_lower[subset_idx],
                                                                 y_shade_upper[subset_idx],
                                                                 color='blue', alpha=shade_alpha, lw=0)
                    else:
                        y_shade_lower = left_choice_projection_to_x - left_choice_projection_to_x_error
                        y_shade_upper = left_choice_projection_to_x + left_choice_projection_to_x_error
                        axs[n_aud_cond, n_vis_cond].fill_between(peri_event_time,
                                                         y_shade_lower, y_shade_upper,
                                                         color='blue', alpha=shade_alpha, lw=0)

            if len(right_choice_projection_to_x) > 0:
                if truncate_values:
                    subset_idx = np.where((peri_event_time >= xlim[0]) & (peri_event_time <= xlim[1]))[0]
                    axs[n_aud_cond, n_vis_cond].plot(peri_event_time[subset_idx], right_choice_projection_to_x[subset_idx],
                                                     color='red', alpha=line_alpha, lw=line_width)
                else:
                    axs[n_aud_cond, n_vis_cond].plot(peri_event_time, right_choice_projection_to_x,
                                                     color='red', alpha=line_alpha, lw=line_width)
                if show_error_shade:
                    right_choice_projection_to_x_error = subset_ds.where(
                        subset_ds['responseMade'] == 'right', drop=True
                    )[projection_error_name].values.flatten()
                    y_shade_lower = right_choice_projection_to_x - right_choice_projection_to_x_error
                    y_shade_upper = right_choice_projection_to_x + right_choice_projection_to_x_error

                    if truncate_values:
                        subset_idx = np.where((peri_event_time >= xlim[0]) & (peri_event_time <= xlim[1]))[0]
                        axs[n_aud_cond, n_vis_cond].fill_between(peri_event_time[subset_idx],
                                                                 y_shade_lower[subset_idx],
                                                                 y_shade_upper[subset_idx],
                                                                 color='red', alpha=shade_alpha, lw=0)
                    else:
                        axs[n_aud_cond, n_vis_cond].fill_between(peri_event_time,
                                                                 right_choice_projection_to_x - right_choice_projection_to_x_error,
                                                                 right_choice_projection_to_x + right_choice_projection_to_x_error,
                                                                 color='red', alpha=shade_alpha, lw=0)

            axs[n_aud_cond, n_vis_cond].set_title('aud %s vis %s' % (aud_cond, vis_cond), size=9)

    if xlim is not None:
        axs[n_aud_cond, n_vis_cond].set_xlim(xlim)

    axs[n_aud_cond, n_vis_cond].set_ylim(ylim)
    fig.text(0.5, 0, 'Peri-movement time (s)', size=11, ha='center')
    fig.tight_layout()

    return fig, axs



def main():
    process = 'cal_and_plot_projection_all_sessions'

    subject_exp = {
        1: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        2: [13, 15, 16, 17, 18, 19],
        3: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        4: [32, 33, 34, 35, 36, 37, 38, 39, 40],
        5: [42, 43, 44, 45, 46],
        6: [48, 49, 50, 51, 52, 53, 54],
    }

    process_params = {
        'cal_and_plot_projection_all_sessions': dict(
            custom_xlim=[-0.2, 0.1],
            custom_ylim=[-0.25, 0.25],
            # custom_ylim=[-0.3, 0.3],
            smooth=True,
            train_test_split=True,
            min_cell_count=0,
            fig_ext=['.png', '.pdf'],
            shade_mean_calculation_window=True,
            label_y_axis=True,
            combine_all_sessions_in_one_plot=True,
            output_data_folder='/Volumes/Partition 1/reports/figures/choice-vector-projection/calculated-projections',
            fig_folder='/Volumes/Partition 1/reports/figures/choice-vector-projection/',
            plot_individual_sessions=False,
            show_error_shade=True,
        )
    }

    min_cell_count = process_params[process]['min_cell_count']
    custom_xlim = process_params[process]['custom_xlim']
    custom_ylim = process_params[process]['custom_ylim']
    fig_folder = process_params[process]['fig_folder']
    output_data_folder = process_params[process]['output_data_folder']
    plot_individual_sessions = process_params[process]['plot_individual_sessions']
    show_error_shade = process_params[process]['show_error_shade']

    print('Saving figures to %s' % fig_folder)

    all_session_all_trial_cond_mean_ds = []

    ##### CALCULATION PART ######
    for subject in subject_exp.keys():

        for exp in subject_exp[subject]:
            alignment_ds = load_data(subject, exp, alignment_folder=alignment_folder, aligned_event=aligned_event)

            projection_save_path = os.path.join(output_data_folder, '%s_%s_projections.nc' % (subject, exp))
            if os.path.exists(projection_save_path):
                print('Projection already calculated, skipping')
                all_trial_cond_mean_ds = xr.open_dataset(projection_save_path)
                all_session_all_trial_cond_mean_ds.append(all_trial_cond_mean_ds)
            else:
                if len(alignment_ds.Cell) > min_cell_count:

                    alignment_ds = cal_projection_to_choice(alignment_ds,
                                                            train_test_split=process_params[process][
                                                                'train_test_split'])
                    all_trial_cond_mean_ds = group_projection_by_stim_cond_and_choice(alignment_ds)
                    failed_cal_projection = 0

                    # save data
                    all_trial_cond_mean_ds['numCell'] = len(alignment_ds.Cell)
                    all_trial_cond_mean_ds.to_netcdf(projection_save_path)

                else:
                    failed_cal_projection = 1

                if (1 - failed_cal_projection):
                    # TODO: instead of plotting here, save the result
                    all_session_all_trial_cond_mean_ds.append(all_trial_cond_mean_ds)

    #### PLOTTING PART
    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, axs = plt.subplots(3, 3)
        fig.set_size_inches(8, 8)
        subset_ds_list = []
        for ds in all_session_all_trial_cond_mean_ds:
            if ds.numCell > min_cell_count:
                ds = ds.drop('numCell')

                if plot_individual_sessions:
                    fig, axs = plot_projection(
                        ds, projection_name='projection_to_right_minus_left',
                        ylim=custom_ylim, xlim=custom_xlim, line_alpha=0.2, line_width=0.2,
                        fig=fig, axs=axs)

                subset_ds_list.append(ds)

        # all_projection_r_minus_left_array = [x['projection_to_right_minus_left'] for x in all_session_all_trial_cond_mean_ds]
        # all_projection_r_minus_left_array = xr.concat(all_projection_r_minus_left_array, dim='Exp')
        all_session_ds = xr.concat(subset_ds_list, dim='Exp')

        if process_params[process]['smooth']:
            print('Smoothing traces')
            window_width = 20
            sigma = 2
            gaussian_window = spsignal.windows.gaussian(M=window_width, std=sigma)

            # note that the mid-point is included (ie. the peak of the gaussian)
            half_gaussian_window = gaussian_window.copy()
            half_gaussian_window[:int((window_width - 1) / 2)] = 0

            # normalise so it sums to 1
            half_gaussian_window = half_gaussian_window / np.sum(half_gaussian_window)
            smoothed_values = spimage.filters.convolve1d(all_session_ds['projection_to_right_minus_left'].values,
                                                         weights=half_gaussian_window,
                                                         axis=1)
            all_session_ds['projection_to_right_minus_left'] = (
            ['Exp', 'Time', 'audCond', 'visCond', 'responseMade'], smoothed_values)

        all_session_mean_ds = all_session_ds.mean('Exp')
        all_session_std_ds = all_session_ds.std('Exp')
        all_session_mean_ds['projection_to_right_minus_left_std'] = all_session_std_ds['projection_to_right_minus_left']
        all_session_mean_ds['projection_to_right_minus_left_sem'] = all_session_std_ds[
                                                                        'projection_to_right_minus_left'] / np.sqrt(
            len(all_session_ds.Exp))

        fig, axs = plot_projection(all_session_mean_ds,
                                   projection_name='projection_to_right_minus_left',
                                   projection_error_name='projection_to_right_minus_left_sem',
                                   ylim=custom_ylim, xlim=custom_xlim, line_alpha=1, line_width=1, shade_alpha=0.2,
                                   show_error_shade=show_error_shade,
                                   fig=fig, axs=axs)

        # Double check the axes limits are set correctly
        [ax.set_xlim(custom_xlim) for ax in axs.flatten()]
        [ax.set_ylim(custom_ylim) for ax in axs.flatten()]

        # Fill in the time period to calculate the projection difference
        axs[0, 0].axvspan(-0.1, 0, color='gray', alpha=0.25)

    fig_name = 'all_sesisons_projection_to_right_minus_left'

    if process_params[process]['smooth']:
        fig_name += '_smoothed'

    fig.text(-0.02, 0.5, r'$S_C(\vec{x}_t, \vec{\mu}_R - \vec{\mu}_L)$',
             rotation=90, size=11, va='center')
    for ext in process_params[process]['fig_ext']:
        fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')