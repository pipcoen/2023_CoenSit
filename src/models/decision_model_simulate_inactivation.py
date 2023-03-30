# uncompyle6 version 3.7.5.dev0
# Python bytecode 3.7 (3394)
# Decompiled from: Python 2.7.17 (default, Feb 27 2021, 15:10:58) 
# [GCC 7.5.0]
# Warning: this version of Python has problems handling the Python 3 "byte" type in constants properly.

# Embedded file name: /home/timsit/multisensory-integration/src/models/decision_model_simulate_inactivation.py
# Compiled at: 2021-05-06 22:16:17
# Size of source mod 2**32: 51856 bytes
import numpy as np, xarray as xr, pandas as pd, os, glob
import matplotlib.pyplot as plt
import sciplotlib.style as splstyle
import src.data.analyse_spikes as anaspikes
import src.data.analyse_behaviour as anabehave
import src.data.process_ephys_data as pephys
import src.models.network_model as nmodel
import src.data.stat as stat
import scipy.stats as sstats
import pdb, jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import src.models.jax_decision_model as jax_dmodel
import src.models.decision_model_sample_and_fit as dmodel_sample_and_fit
print(jax.devices())
import os, functools
from jax.experimental import optimizers
import pickle as pkl
import itertools
import sklearn.model_selection as sklselection
import src.data.struct_to_dataframe as stdf


def load_data(combined_ds_filename, inactivation_trial_cond='coherent', start_time=-0.1, end_time=0.3):
    """

    Parameters
    ----------
    combined_ds_filename
    inactivation_trial_cond (str)

    Returns
    -------

    """
    alignment_ds = xr.open_dataset(combined_ds_filename)
    alignment_ds = alignment_ds.where(((alignment_ds['PeriEventTime'] >= start_time) & (alignment_ds['PeriEventTime'] <= end_time)),
      drop=True)
    if inactivation_trial_cond == 'coherent':
        target_cond_ds = alignment_ds.where(((alignment_ds['audDiff'] == 60) & (alignment_ds['visDiff'] == 0.8)),
          drop=True)
    else:
        if inactivation_trial_cond == 'conflict':
            target_cond_ds = alignment_ds.where(((alignment_ds['audDiff'] == -60) & (alignment_ds['visDiff'] == 0.8)),
              drop=True)
        else:
            if inactivation_trial_cond == 'all':
                target_cond_ds = alignment_ds
            else:
                print('No valid trial condition found')
    smooth_multiplier = 5
    alignment_ds_smoothed = target_cond_ds.stack(trialTime=['Trial', 'Time'])
    sigma = 3 * smooth_multiplier
    window_width = 20 * smooth_multiplier
    scale_data = False
    alignment_ds_smoothed['firing_rate'] = (
     [
      'Cell', 'trialTime'],
     anaspikes.smooth_spikes((alignment_ds_smoothed['firing_rate']),
       method='half_gaussian',
       sigma=sigma,
       window_width=window_width,
       custom_window=None))
    return (
     alignment_ds, alignment_ds_smoothed)


def load_model(model_result_folder, model_result_fname, alignment_ds, N=100, set_to_select_min_loss='test'):
    model_results = pd.read_pickle(os.path.join(model_result_folder, model_result_fname))
    model_params = np.stack(model_results['param_history'])
    epoch_min_loss = np.argmin(model_results['test_loss'])
    model_best_params = model_params[epoch_min_loss, :]
    if N == 999:
        c_param = 1.0
    else:
        c_param = 2 ** (-(1 / N))
    y_test_pred_da_list = [model_results['y_test_pred_da']]
    all_stim_cond_pred_matrix_dict = jax_dmodel.get_stim_cond_response(alignment_ds=alignment_ds,
      y_test_pred_da_list=y_test_pred_da_list)
    stim_cond_output = all_stim_cond_pred_matrix_dict[(60, 0.8)]
    return (
     model_best_params, c_param, stim_cond_output)


def do_inactivation(alignment_ds, alignment_ds_smoothed, vis_left_neuron_idx,
                    vis_right_neuron_idx, aud_neuron_idx, vis_left_inactivation_scaling=1.0,
                    vis_right_inactivation_scaling=3.0, cell_idx_method='isel'):
    """

    Parameters
    ----------
    alignment_ds (xarray dataset)
    alignment_ds_smoothed : (xarray dataset)
        xarray with dimesions (Cell, trialTime)
        where trialTime is a stacked dimension with trial and time
    vis_left_inactivation_scaling
    vis_right_inactivation_scaling

    Returns
    -------

    """
    if cell_idx_method == 'isel':
        vis_left_neuron_smoothed = alignment_ds_smoothed.isel(Cell=vis_left_neuron_idx)
        vis_left_neuron_inactivated_smoothed = alignment_ds_smoothed.isel(Cell=vis_left_neuron_idx)
        vis_right_neuron_smoothed = alignment_ds_smoothed.isel(Cell=vis_right_neuron_idx)
        vis_right_neuron_inactivated_smoothed = alignment_ds_smoothed.isel(Cell=vis_right_neuron_idx)
    else:
        if cell_idx_method == 'sel':
            vis_left_neuron_smoothed = alignment_ds_smoothed.sel(Cell=vis_left_neuron_idx)
            vis_left_neuron_inactivated_smoothed = alignment_ds_smoothed.sel(Cell=vis_left_neuron_idx)
            vis_right_neuron_smoothed = alignment_ds_smoothed.sel(Cell=vis_right_neuron_idx)
            vis_right_neuron_inactivated_smoothed = alignment_ds_smoothed.sel(Cell=vis_right_neuron_idx)
        else:
            print('Warning: no valid cell index method selected')

    vis_left_neuron_inactivated_smoothed['firing_rate'] = vis_left_neuron_inactivated_smoothed['firing_rate'] / vis_left_inactivation_scaling
    vis_right_neuron_inactivated_smoothed['firing_rate'] = vis_right_neuron_inactivated_smoothed['firing_rate'] / vis_right_inactivation_scaling
    vis_left_neuron_inactivated_smoothed_zscore = vis_left_neuron_inactivated_smoothed



    vis_left_neuron_inactivated_smoothed_zscore_val = sstats.zmap(scores=(vis_left_neuron_inactivated_smoothed['firing_rate']),
      compare=(vis_left_neuron_smoothed['firing_rate']),
      axis=1)



    if np.sum(np.isnan(vis_left_neuron_inactivated_smoothed_zscore_val)) > 0:
        print('Warning: vis left neuron have at least one nan, replacing with zeros for now...')
        vis_left_neuron_inactivated_smoothed_zscore_val[np.isnan(vis_left_neuron_inactivated_smoothed_zscore_val)] = 0
    vis_left_neuron_inactivated_smoothed_zscore = vis_left_neuron_inactivated_smoothed_zscore.assign({'firing_rate': (['Cell', 'trialTime'],
                     vis_left_neuron_inactivated_smoothed_zscore_val)})
    vis_left_neuron_smoothed_zscore = vis_left_neuron_smoothed
    vis_left_neuron_smoothed_zscore['firing_rate'] = (
     [
      'Cell', 'trialTime'],
     sstats.zscore((vis_left_neuron_smoothed_zscore['firing_rate']),
       axis=1))
    if np.sum(np.isnan(vis_left_neuron_smoothed_zscore['firing_rate'])) > 0:
        vis_left_neuron_smoothed_zscore_vals = vis_left_neuron_smoothed_zscore['firing_rate'].values
        vis_left_neuron_smoothed_zscore_vals[np.isnan(vis_left_neuron_smoothed_zscore_vals)] = 0
        vis_left_neuron_smoothed_zscore['firing_rate'] = (['Cell', 'trialTime'], vis_left_neuron_smoothed_zscore_vals)
    vis_right_neuron_inactivated_smoothed_zscore = vis_right_neuron_inactivated_smoothed
    vis_right_neuron_inactivated_smoothed_zscore_val = sstats.zmap(scores=(vis_right_neuron_inactivated_smoothed['firing_rate']),
      compare=(vis_right_neuron_smoothed['firing_rate']),
      axis=1)
    if np.sum(np.isnan(vis_right_neuron_inactivated_smoothed_zscore_val)) > 0:
        print('Warning: vis right neuron have at least one nan, replacing with zeros for now...')
        vis_right_neuron_inactivated_smoothed_zscore_val[np.isnan(vis_right_neuron_inactivated_smoothed_zscore_val)] = 0
    vis_right_neuron_inactivated_smoothed_zscore = vis_right_neuron_inactivated_smoothed_zscore.assign({'firing_rate': (['Cell', 'trialTime'],
                     vis_right_neuron_inactivated_smoothed_zscore_val)})
    vis_right_neuron_smoothed_zscore = vis_right_neuron_smoothed
    vis_right_neuron_smoothed_zscore['firing_rate'] = (
     [
      'Cell', 'trialTime'],
     sstats.zscore((vis_right_neuron_smoothed_zscore['firing_rate']),
       axis=1))
    if np.sum(np.isnan(vis_right_neuron_smoothed_zscore['firing_rate'])) > 0:
        vis_right_neuron_smoothed_zscore_vals = vis_right_neuron_smoothed_zscore['firing_rate'].values
        vis_right_neuron_smoothed_zscore_vals[np.isnan(vis_right_neuron_smoothed_zscore_vals)] = 0
        vis_right_neuron_smoothed_zscore['firing_rate'] = (['Cell', 'trialTime'], vis_right_neuron_smoothed_zscore_vals)
    if cell_idx_method == 'isel':
        aud_neuron_smoothed_zscore = alignment_ds_smoothed.isel(Cell=aud_neuron_idx)
    elif cell_idx_method == 'sel':
        aud_neuron_smoothed_zscore = alignment_ds_smoothed.sel(Cell=aud_neuron_idx)

    aud_neuron_smoothed_zscore['firing_rate'] = (
     [
      'Cell', 'trialTime'],
     sstats.zscore((aud_neuron_smoothed_zscore['firing_rate']),
       axis=1))
    if np.sum(np.isnan(aud_neuron_smoothed_zscore['firing_rate'])) > 0:
        print('Auditory neuron zscore has nans, setting to zero for now')
        aud_neuron_smoothed_zscore_vals = aud_neuron_smoothed_zscore['firing_rate'].values
        aud_neuron_smoothed_zscore_vals[np.isnan(aud_neuron_smoothed_zscore_vals)] = 0
        aud_neuron_smoothed_zscore['firing_rate'] = (['Cell', 'trialTime'], aud_neuron_smoothed_zscore_vals)

    alignment_ds_smoothed_zscore = alignment_ds_smoothed
    alignment_ds_smoothed_zscore = xr.concat([
     vis_left_neuron_smoothed_zscore,
     vis_right_neuron_smoothed_zscore,
     aud_neuron_smoothed_zscore],
      dim='Cell')
    alignment_ds_smoothed_zscore = alignment_ds_smoothed_zscore.unstack()
    alignment_ds_smoothed_vis_inactivated = xr.concat([
     vis_left_neuron_inactivated_smoothed_zscore,
     vis_right_neuron_inactivated_smoothed_zscore,
     aud_neuron_smoothed_zscore],
      dim='Cell')
    alignment_ds_smoothed_vis_inactivated = alignment_ds_smoothed_vis_inactivated.unstack()
    alignment_ds_smoothed_zscore = alignment_ds_smoothed_zscore.sel(Cell=(alignment_ds.Cell.values))
    alignment_ds_smoothed_vis_inactivated = alignment_ds_smoothed_vis_inactivated.sel(Cell=(alignment_ds.Cell.values))

    return alignment_ds_smoothed_zscore, alignment_ds_smoothed_vis_inactivated


def do_inactivation_no_smoothing(alignment_ds, alignment_ds_smoothed, vis_left_neuron_idx,
                    vis_right_neuron_idx, aud_neuron_idx, vis_left_inactivation_scaling=1.0,
                    vis_right_inactivation_scaling=3.0, cell_idx_method='isel', smooth_activity=False):

    if cell_idx_method == 'isel':
        vis_left_neuron_smoothed = alignment_ds_smoothed.isel(Cell=vis_left_neuron_idx)
        vis_left_neuron_inactivated_smoothed = alignment_ds_smoothed.isel(Cell=vis_left_neuron_idx)
        vis_right_neuron_smoothed = alignment_ds_smoothed.isel(Cell=vis_right_neuron_idx)
        vis_right_neuron_inactivated_smoothed = alignment_ds_smoothed.isel(Cell=vis_right_neuron_idx)
    else:
        if cell_idx_method == 'sel':
            vis_left_neuron_smoothed = alignment_ds_smoothed.sel(Cell=vis_left_neuron_idx)
            vis_left_neuron_inactivated_smoothed = alignment_ds_smoothed.sel(Cell=vis_left_neuron_idx)
            vis_right_neuron_smoothed = alignment_ds_smoothed.sel(Cell=vis_right_neuron_idx)
            vis_right_neuron_inactivated_smoothed = alignment_ds_smoothed.sel(Cell=vis_right_neuron_idx)
        else:
            print('Warning: no valid cell index method selected')


    vis_left_neuron_inactivated_smoothed['firing_rate'] = vis_left_neuron_inactivated_smoothed['firing_rate'] / vis_left_inactivation_scaling
    vis_right_neuron_inactivated_smoothed['firing_rate'] = vis_right_neuron_inactivated_smoothed['firing_rate'] / vis_right_inactivation_scaling
    vis_left_neuron_inactivated_smoothed_zscore = vis_left_neuron_inactivated_smoothed
    vis_left_neuron_inactivated_smoothed_zscore_val = sstats.zmap(scores=(vis_left_neuron_inactivated_smoothed['firing_rate']),
      compare=(vis_left_neuron_smoothed['firing_rate']),
      axis=1)


    if np.sum(np.isnan(vis_left_neuron_inactivated_smoothed_zscore_val)) > 0:
        print('Warning: vis left neuron have at least one nan, replacing with zeros for now...')
        vis_left_neuron_inactivated_smoothed_zscore_val[np.isnan(vis_left_neuron_inactivated_smoothed_zscore_val)] = 0

    vis_left_neuron_inactivated_smoothed_zscore = vis_left_neuron_inactivated_smoothed_zscore.assign({'firing_rate': (['Cell', 'trialTime'],
                     vis_left_neuron_inactivated_smoothed_zscore_val)})
    vis_left_neuron_smoothed_zscore = vis_left_neuron_smoothed
    vis_left_neuron_smoothed_zscore['firing_rate'] = (
     [
      'Cell', 'trialTime'],
     sstats.zscore((vis_left_neuron_smoothed_zscore['firing_rate']),
       axis=1))

    if np.sum(np.isnan(vis_left_neuron_smoothed_zscore['firing_rate'])) > 0:
        vis_left_neuron_smoothed_zscore_vals = vis_left_neuron_smoothed_zscore['firing_rate'].values
        vis_left_neuron_smoothed_zscore_vals[np.isnan(vis_left_neuron_smoothed_zscore_vals)] = 0
        vis_left_neuron_smoothed_zscore['firing_rate'] = (['Cell', 'trialTime'], vis_left_neuron_smoothed_zscore_vals)

    vis_right_neuron_inactivated_smoothed_zscore = vis_right_neuron_inactivated_smoothed
    vis_right_neuron_inactivated_smoothed_zscore_val = sstats.zmap(scores=(vis_right_neuron_inactivated_smoothed['firing_rate']),
      compare=(vis_right_neuron_smoothed['firing_rate']),
      axis=1)


    if np.sum(np.isnan(vis_right_neuron_inactivated_smoothed_zscore_val)) > 0:
        print('Warning: vis right neuron have at least one nan, replacing with zeros for now...')
        vis_right_neuron_inactivated_smoothed_zscore_val[np.isnan(vis_right_neuron_inactivated_smoothed_zscore_val)] = 0
    vis_right_neuron_inactivated_smoothed_zscore = vis_right_neuron_inactivated_smoothed_zscore.assign({'firing_rate': (['Cell', 'trialTime'],
                     vis_right_neuron_inactivated_smoothed_zscore_val)})
    vis_right_neuron_smoothed_zscore = vis_right_neuron_smoothed
    vis_right_neuron_smoothed_zscore['firing_rate'] = (
     [
      'Cell', 'trialTime'],
     sstats.zscore((vis_right_neuron_smoothed_zscore['firing_rate']),
       axis=1))

    if np.sum(np.isnan(vis_right_neuron_smoothed_zscore['firing_rate'])) > 0:
        vis_right_neuron_smoothed_zscore_vals = vis_right_neuron_smoothed_zscore['firing_rate'].values
        vis_right_neuron_smoothed_zscore_vals[np.isnan(vis_right_neuron_smoothed_zscore_vals)] = 0
        vis_right_neuron_smoothed_zscore['firing_rate'] = (['Cell', 'trialTime'], vis_right_neuron_smoothed_zscore_vals)
    if cell_idx_method == 'isel':
        aud_neuron_smoothed_zscore = alignment_ds_smoothed.isel(Cell=aud_neuron_idx)
    elif cell_idx_method == 'sel':
        aud_neuron_smoothed_zscore = alignment_ds_smoothed.sel(Cell=aud_neuron_idx)
        aud_neuron_smoothed_zscore['firing_rate'] = (
         [
          'Cell', 'trialTime'],
         sstats.zscore((aud_neuron_smoothed_zscore['firing_rate']),
           axis=1))

    if np.sum(np.isnan(aud_neuron_smoothed_zscore['firing_rate'])) > 0:
        print('Auditory neuron zscore has nans, setting to zero for now')
        aud_neuron_smoothed_zscore_vals = aud_neuron_smoothed_zscore['firing_rate'].values
        aud_neuron_smoothed_zscore_vals[np.isnan(aud_neuron_smoothed_zscore_vals)] = 0
        aud_neuron_smoothed_zscore['firing_rate'] = (['Cell', 'trialTime'], aud_neuron_smoothed_zscore_vals)


    alignment_ds_smoothed_zscore = alignment_ds_smoothed
    alignment_ds_smoothed_zscore = xr.concat([
     vis_left_neuron_smoothed_zscore,
     vis_right_neuron_smoothed_zscore,
     aud_neuron_smoothed_zscore],
      dim='Cell')
    alignment_ds_smoothed_zscore = alignment_ds_smoothed_zscore.unstack()
    alignment_ds_smoothed_vis_inactivated = xr.concat([
     vis_left_neuron_inactivated_smoothed_zscore,
     vis_right_neuron_inactivated_smoothed_zscore,
     aud_neuron_smoothed_zscore],
      dim='Cell')
    alignment_ds_smoothed_vis_inactivated = alignment_ds_smoothed_vis_inactivated.unstack()
    alignment_ds_smoothed_zscore = alignment_ds_smoothed_zscore.sel(Cell=(alignment_ds.Cell.values))
    alignment_ds_smoothed_vis_inactivated = alignment_ds_smoothed_vis_inactivated.sel(Cell=(alignment_ds.Cell.values))

    return alignment_ds_smoothed_zscore, alignment_ds_smoothed_vis_inactivated


def do_hemisphere_inactivation(alignment_ds, alignment_ds_smoothed, hemisphere_vec, left_hemisphere_scaling=1.0, right_hemisphere_scaling=0.5):
    left_neuron_idx = np.where(hemisphere_vec == 0)[0]
    right_neuron_idx = np.where(hemisphere_vec == 1)[0]
    alignment_ds = alignment_ds.assign_coords({'cellIdx': ('Cell', np.arange(len(alignment_ds.Cell.values)))})
    alignment_ds_smoothed = alignment_ds_smoothed.assign_coords({'cellIdx': ('Cell', np.arange(len(alignment_ds.Cell.values)))})
    left_neuron_smoothed = alignment_ds_smoothed.isel(Cell=left_neuron_idx).copy()
    left_neuron_inactivated_smoothed = alignment_ds_smoothed.isel(Cell=left_neuron_idx).copy()
    right_neuron_smoothed = alignment_ds_smoothed.isel(Cell=right_neuron_idx).copy()
    right_neuron_inactivated_smoothed = alignment_ds_smoothed.isel(Cell=right_neuron_idx)
    left_neuron_inactivated_smoothed['firing_rate'] = left_neuron_inactivated_smoothed['firing_rate'] / left_hemisphere_scaling
    right_neuron_inactivated_smoothed['firing_rate'] = right_neuron_inactivated_smoothed['firing_rate'] / right_hemisphere_scaling
    left_neuron_inactivated_smoothed_zscore = left_neuron_inactivated_smoothed
    left_neuron_inactivated_smoothed_zscore_val = sstats.zmap(scores=(left_neuron_inactivated_smoothed['firing_rate']),
      compare=(left_neuron_smoothed['firing_rate']),
      axis=1)
    right_neuron_inactivated_smoothed_zscore = right_neuron_inactivated_smoothed
    right_neuron_inactivated_smoothed_zscore_val = sstats.zmap(scores=(right_neuron_inactivated_smoothed['firing_rate']),
      compare=(right_neuron_smoothed['firing_rate']),
      axis=1)
    left_neuron_smoothed_zscore = left_neuron_smoothed
    left_neuron_smoothed_zscore['firing_rate'] = (
     [
      'Cell', 'trialTime'],
     sstats.zscore((left_neuron_smoothed_zscore['firing_rate']),
       axis=1))
    right_neuron_smoothed_zscore = right_neuron_smoothed
    right_neuron_smoothed_zscore['firing_rate'] = (
     [
      'Cell', 'trialTime'],
     sstats.zscore((right_neuron_smoothed_zscore['firing_rate']),
       axis=1))
    if np.sum(np.isnan(left_neuron_smoothed_zscore['firing_rate'])) > 0:
        left_neuron_smoothed_zscore_vals = left_neuron_smoothed_zscore['firing_rate'].values
        left_neuron_smoothed_zscore_vals[np.isnan(left_neuron_smoothed_zscore_vals)] = 0
        left_neuron_smoothed_zscore['firing_rate'] = (['Cell', 'trialTime'], left_neuron_smoothed_zscore_vals)
    if np.sum(np.isnan(right_neuron_smoothed_zscore['firing_rate'])) > 0:
        right_neuron_smoothed_zscore_vals = right_neuron_smoothed_zscore['firing_rate'].values
        right_neuron_smoothed_zscore_vals[np.isnan(right_neuron_smoothed_zscore_vals)] = 0
        right_neuron_smoothed_zscore['firing_rate'] = (['Cell', 'trialTime'], right_neuron_smoothed_zscore_vals)
    alignment_ds_smoothed_zscore = xr.concat([
     left_neuron_smoothed_zscore,
     right_neuron_smoothed_zscore],
      dim='Cell')
    alignment_ds_smoothed_zscore = alignment_ds_smoothed_zscore.unstack()
    alignment_ds_smoothed_inactivated = xr.concat([
     left_neuron_inactivated_smoothed_zscore,
     right_neuron_inactivated_smoothed_zscore],
      dim='Cell')
    alignment_ds_smoothed_inactivated = alignment_ds_smoothed_inactivated.unstack()
    alignment_ds_smoothed_zscore = alignment_ds_smoothed_zscore.sortby('cellIdx')
    alignment_ds_smoothed_inactivated = alignment_ds_smoothed_inactivated.sortby('cellIdx')
    return (
     alignment_ds_smoothed_zscore, alignment_ds_smoothed_inactivated)


def get_behaviour_from_model_output(peri_event_time, model_control_output, model_vis_inactivated_output,
                                    decision_threshold=1, left_decision_threshold=None, right_decision_threshold=None,
                                    alignment_ds=None, include_no_go=True, verbose=False, left_choice_val=0, right_choice_val=1):
    """

    Parameters
    ----------
    peri_event_time (numpy ndarray)
        vector with the time relative to stimulus onset (seconds)
    model_vis_inactivated_output (numpy ndarray)
        matrix with dimensions (trial, timeBins)
    model_control_output (numpy ndarray)
        matrix with dimensions (trial, timeBins)
    decision_threshold : (float)
        value for decision threshold
        assume symmetric (ie. +1 for going right, -1 for going left)
    alignment_ds : (xarray dataset)

    Returns
    -------

    """
    control_decision_list = list()
    control_decision_time_list = list()
    inactivation_decision_list = list()
    inactivation_decision_time_list = list()
    if left_decision_threshold is None and right_decision_threshold is None:
        for trial_output_vec in model_control_output:
            decision_frame = np.where(np.abs(trial_output_vec) >= decision_threshold)[0]
            if len(decision_frame) > 0:
                decision_time = peri_event_time[decision_frame[0]]
                control_decision_time_list.append(decision_time)
                decision_val = trial_output_vec[decision_frame[0]]
                control_decision_list.append(decision_val >= 1)
            elif include_no_go:
                control_decision_list.append(np.nan)
                control_decision_time_list.append(np.nan)

        for trial_output_vec in model_vis_inactivated_output:
            decision_frame = np.where(np.abs(trial_output_vec) >= decision_threshold)[0]
            if len(decision_frame) > 0:
                decision_time = peri_event_time[decision_frame[0]]
                inactivation_decision_time_list.append(decision_time)
                decision_val = trial_output_vec[decision_frame[0]]
                inactivation_decision_list.append(decision_val >= 1)
            elif include_no_go:
                inactivation_decision_list.append(np.nan)
                inactivation_decision_time_list.append(np.nan)

    else:
        for trial_output in model_control_output:
            left_choice_query = np.where(trial_output <= left_decision_threshold)[0]
            right_choice_query = np.where(trial_output >= right_decision_threshold)[0]
            if (len(left_choice_query) == 0) & (len(right_choice_query) == 0):
                control_decision_time_list.append(np.nan)
                control_decision_list.append(np.nan)
            else:
                if len(left_choice_query) > 0:
                    left_choice_frame = left_choice_query[0]
                else:
                    left_choice_frame = 999
                if len(right_choice_query) > 0:
                    right_choice_frame = right_choice_query[0]
                else:
                    right_choice_frame = 999
                if left_choice_frame < right_choice_frame:
                    control_decision_list.append(left_choice_val)
                    control_decision_time_list.append(peri_event_time[left_choice_frame])
                elif left_choice_frame > right_choice_frame:
                    control_decision_list.append(right_choice_val)
                    control_decision_time_list.append(peri_event_time[right_choice_frame])

        for trial_output in model_vis_inactivated_output:
            left_choice_query = np.where(trial_output <= left_decision_threshold)[0]
            right_choice_query = np.where(trial_output >= right_decision_threshold)[0]
            if (len(left_choice_query) == 0) & (len(right_choice_query) == 0):
                inactivation_decision_time_list.append(np.nan)
                inactivation_decision_list.append(np.nan)
            else:
                if len(left_choice_query) > 0:
                    left_choice_frame = left_choice_query[0]
                else:
                    left_choice_frame = 999
                if len(right_choice_query) > 0:
                    right_choice_frame = right_choice_query[0]
                else:
                    right_choice_frame = 999
                if left_choice_frame < right_choice_frame:
                    inactivation_decision_list.append(left_choice_val)
                    inactivation_decision_time_list.append(peri_event_time[left_choice_frame])
                elif left_choice_frame > right_choice_frame:
                    inactivation_decision_list.append(right_choice_val)
                    inactivation_decision_time_list.append(peri_event_time[right_choice_frame])

    control_decision_time_vec = np.array(control_decision_time_list)
    inactivation_decision_time_vec = np.array(inactivation_decision_time_list)
    control_decision_vec = np.array(control_decision_list)
    inactivation_decision_vec = np.array(inactivation_decision_list)
    if not include_no_go:
        control_decision_vec = control_decision_vec[(control_decision_time_vec >= 0)]
        inactivation_decision_vec = inactivation_decision_vec[(inactivation_decision_time_vec >= 0)]
        control_decision_time_vec = control_decision_time_vec[(control_decision_time_vec >= 0)]
        inactivation_decision_time_vec = inactivation_decision_time_vec[(inactivation_decision_time_vec >= 0)]
    if verbose:
        print('Number of control go trials: %.f' % len(control_decision_time_vec))
        print('Number of inactivation go trials: %.f' % len(inactivation_decision_vec))
    trial_cond_type_list = list()
    trial_cond_type_list.extend(np.repeat('Control', len(control_decision_time_vec)))
    trial_cond_type_list.extend(np.repeat('Inactivation', len(inactivation_decision_time_vec)))
    all_decision_time_vec = np.concatenate([control_decision_time_vec,
     inactivation_decision_time_vec])
    all_trial_choice = np.concatenate([control_decision_vec, inactivation_decision_vec])
    model_behaviour_df = pd.DataFrame.from_dict({'TrialCond':trial_cond_type_list,  'reactionTime':all_decision_time_vec, 
     'chooseRight':all_trial_choice})
    if alignment_ds is not None:
        vis_cond_vec = alignment_ds.isel(Time=0, Cell=0)['visDiff'].values
        vis_cond_list = np.concatenate([vis_cond_vec, vis_cond_vec])
        aud_cond_vec = alignment_ds.isel(Time=0, Cell=0)['audDiff'].values
        aud_cond_list = np.concatenate([aud_cond_vec, aud_cond_vec])
        model_behaviour_df['visCond'] = vis_cond_list
        model_behaviour_df['audCond'] = aud_cond_list
    return model_behaviour_df


def simulate_other_stim_conds(pre_preprocessed_alignment_ds, vis_cond_to_sim=[
 -0.1, -0.2, -0.4, 0.1, 0.2, 0.4, -0.1, -0.2, -0.4, 0.1, 0.2, 0.4], aud_cond_to_sim=[
 60, 60, 60, 60, 60, 60, -60, -60, -60, -60, -60, -60], reindex_cells=True):
    """
    Simulate stimulus conditions without passive activity data available.
    Parameters
    ----------
    pre_preprocessed_alignment_ds : (xarray dataset)
    model_best_params : (numpy ndarray)
    stim_on_bin : (int)
    c_param : (float)
        accumulation parameter (ie. time constant of decay) of the model
    vis_cond_to_sim : (list)
        list of visual condition to simulate
    aud_cond_to_sim : (list)
        list of auditory condition to simulate
    Returns
    -------

    """
    if reindex_cells:
        pre_preprocessed_alignment_ds = pre_preprocessed_alignment_ds.assign_coords({'Cell': np.arange(len(pre_preprocessed_alignment_ds.Cell))})
    vis_neuron_ds = pre_preprocessed_alignment_ds.where((pre_preprocessed_alignment_ds['modality'].isin([
     'visLeft', 'visRight'])),
      drop=True)
    aud_neuron_ds = pre_preprocessed_alignment_ds.where((pre_preprocessed_alignment_ds['modality'].isin([
     'audLeft', 'audRight'])),
      drop=True)
    simulated_stim_cond_ds_list = list()
    for vis_sim, aud_sim in zip(vis_cond_to_sim, aud_cond_to_sim):
        vis_sim_cond_neuron_ds = vis_neuron_ds.where(((vis_neuron_ds['audDiff'] == np.inf) & (vis_neuron_ds['visDiff'] == vis_sim)),
          drop=True)
        aud_sim_cond_neuron_ds = aud_neuron_ds.where(((aud_neuron_ds['audDiff'] == aud_sim) & (aud_neuron_ds['visDiff'] == 0)),
          drop=True)
        vis_sim_cond_neuron_ds = vis_sim_cond_neuron_ds.assign_coords({'Trial': np.arange(len(vis_sim_cond_neuron_ds.Trial.values))})
        aud_sim_cond_neuron_ds = aud_sim_cond_neuron_ds.assign_coords({'Trial': np.arange(len(aud_sim_cond_neuron_ds.Trial.values))})
        combined_neuron_ds = xr.concat([vis_sim_cond_neuron_ds, aud_sim_cond_neuron_ds], dim='Cell')
        combined_neuron_ds['visDiff'] = vis_sim_cond_neuron_ds.isel(Cell=0, Time=0)['visDiff']
        combined_neuron_ds['audDiff'] = aud_sim_cond_neuron_ds.isel(Cell=0, Time=0)['audDiff']
        combined_neuron_ds = combined_neuron_ds.sel(Cell=(pre_preprocessed_alignment_ds.Cell))
        simulated_stim_cond_ds_list.append(combined_neuron_ds)

    simulated_stim_cond_ds = xr.concat(simulated_stim_cond_ds_list, dim='Trial')
    simulated_stim_cond_ds['PeriEventTime'] = simulated_stim_cond_ds['PeriEventTime'].isel(Trial=0)
    return simulated_stim_cond_ds


def single_cv_file_inactivation(input_activity_path, model_result_path, vis_left_scaling=0.5, vis_right_scaling=1.0, inactivate_after_stim=False, dataset='test'):
    """
    Load simulated model data and run inactivation.
    Parameters
    ----------
    input_activity_path
    model_result_path
    vis_left_scaling : (float)
        how much to scale the activity of visual left neurons by
    vis_right_scaling : (float)
        how much to scale the activity of visual right neurons by
    inactivate_after_stim : (bool)
        whether to do the inactivation only after the stimulus onset
    Returns
    -------

    """
    alignment_ds = xr.open_dataset(input_activity_path)
    N = 999
    model_results = pd.read_pickle(model_result_path)
    model_params = np.stack(model_results['param_history'])
    if N == 999:
        c_param = 1.0
    else:
        c_param = 2 ** (-(1 / N))
    y_test_pred_da_list = [model_results['y_test_pred_da']]
    if dataset == 'test':
        pre_preprocessed_alignment_ds = model_results['pre_preprocessed_alignment_ds_test']
        epoch_min_loss = np.argmin(model_results['test_loss'])
        model_best_params = model_params[epoch_min_loss, :]
    else:
        if dataset == 'dev':
            pre_preprocessed_alignment_ds = model_results['pre_preprocessed_alignment_ds_dev']
            epoch_min_loss = np.argmin(model_results['dev_loss'])
            model_best_params = model_params[epoch_min_loss, :]
        else:
            pre_preprocessed_alignment_ds['Cell'] = np.arange(len(pre_preprocessed_alignment_ds.Cell))
            alignment_ds['Cell'] = np.arange(len(alignment_ds.Cell))
            pre_preprocessed_alignment_ds['modality'] = alignment_ds['modality']
            vis_left_neuron_idx = np.where(pre_preprocessed_alignment_ds['modality'] == 'visLeft')[0]
            vis_right_neuron_idx = np.where(pre_preprocessed_alignment_ds['modality'] == 'visRight')[0]
            pre_preprocessed_alignment_ds_stacked = pre_preprocessed_alignment_ds.stack(trialTime=['Trial', 'Time'])
            activity_name = 'scaled_firing_rate'
            inactivation_pre_preprocessed_alignment_ds_stacked = pre_preprocessed_alignment_ds_stacked.copy()
            inactivation_activity_da = inactivation_pre_preprocessed_alignment_ds_stacked.where((inactivation_pre_preprocessed_alignment_ds_stacked['Cell'].isin(vis_right_neuron_idx)),
              drop=True)[activity_name]
            if inactivate_after_stim:
                inactivation_activity_da_pre_stim = inactivation_activity_da.where((inactivation_activity_da['PeriEventTime'] < 0),
                  drop=True)
                inactivation_activity_da_post_stim = inactivation_activity_da.where((inactivation_activity_da['PeriEventTime'] >= 0),
                  drop=True)
                inactivation_activity_da_post_stim = inactivation_activity_da_post_stim * vis_left_scaling
                inactivation_activity_da = xr.concat([inactivation_activity_da_pre_stim,
                 inactivation_activity_da_post_stim],
                  dim='trialTime')
            else:
                inactivation_activity_da = inactivation_activity_da * vis_left_scaling
        other_activity_da = inactivation_pre_preprocessed_alignment_ds_stacked.where((~inactivation_pre_preprocessed_alignment_ds_stacked['Cell'].isin(vis_right_neuron_idx)),
          drop=True)[activity_name]
        other_activity_da = other_activity_da * vis_right_scaling
        recombined_activity_da = xr.concat([inactivation_activity_da, other_activity_da], dim='Cell')
        inactivation_pre_preprocessed_alignment_ds_stacked[activity_name] = recombined_activity_da
        inactivation_pre_preprocessed_alignment_ds = inactivation_pre_preprocessed_alignment_ds_stacked.unstack()
        stim_on_bin = 50
        input_tensor_vis_inactivated = inactivation_pre_preprocessed_alignment_ds[activity_name].transpose('Trial', 'Cell', 'Time').values
        model_vis_inactivated_output = jax_dmodel.accumulate_after_stim_w_neuron_bias_fixed_drift_model(input_tensor_vis_inactivated,
          params=model_best_params,
          stim_on_bin=stim_on_bin,
          update_weight=c_param)
        inactivation_pre_preprocessed_alignment_ds['PeriEventTime'] = inactivation_pre_preprocessed_alignment_ds['PeriEventTime'].isel(Trial=0)
        decision_threshold_val = 1
        model_type = 'drift'
        target_vis_cond_list = [
         -0.8, -0.4, -0.2, 0.2, 0.4, 0.8,
         0.0, 0.0, 0.8, 0.8, -0.8, -0.8,
         -0.1, 0.1]
        target_aud_cond_list = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         -60, 60, 60, -60, 60, -60, np.inf, np.inf]
        model_vis_inactivated_output_da = xr.DataArray(model_vis_inactivated_output,
          dims=['Trial', 'Time'], coords={'Trial':inactivation_pre_preprocessed_alignment_ds.Trial, 
         'Time':inactivation_pre_preprocessed_alignment_ds.Time})
        og_model_inactivation_stim_cond_pred_matrix_dict = jax_dmodel.get_stim_cond_response(alignment_ds=inactivation_pre_preprocessed_alignment_ds,
          y_test_pred_da_list=[
         model_vis_inactivated_output_da],
          target_vis_cond_list=target_vis_cond_list,
          target_aud_cond_list=target_aud_cond_list)
        og_model_inactivation_behaviour_df = jax_dmodel.all_stim_cond_pred_matrix_dict_to_model_behaviour_df(all_stim_cond_pred_matrix_dict=og_model_inactivation_stim_cond_pred_matrix_dict,
          alignment_ds=inactivation_pre_preprocessed_alignment_ds,
          left_decision_threshold_val=(-decision_threshold_val),
          right_decision_threshold_val=decision_threshold_val,
          model_type=model_type,
          left_choice_val=0,
          right_choice_val=1,
          target_vis_cond_list=target_vis_cond_list,
          target_aud_cond_list=target_aud_cond_list)
        return (
         og_model_inactivation_behaviour_df, og_model_inactivation_stim_cond_pred_matrix_dict, inactivation_pre_preprocessed_alignment_ds)


def single_cv_file_inactivation_hemisphere(input_activity_path, model_result_path, left_scaling=1.0, right_scaling=1.0, inactivate_after_stim=False, dataset='test', model_name='hem-weight-constraint'):
    alignment_ds = xr.open_dataset(input_activity_path)
    N = 999
    model_results = pd.read_pickle(model_result_path)
    model_params = np.stack(model_results['param_history'])
    if N == 999:
        c_param = 1.0
    else:
        c_param = 2 ** (-(1 / N))
    y_test_pred_da_list = [model_results['y_test_pred_da']]
    if dataset == 'test':
        pre_preprocessed_alignment_ds = model_results['pre_preprocessed_alignment_ds_test']
        epoch_min_loss = np.argmin(model_results['test_loss'])
        model_best_params = model_params[epoch_min_loss, :]
    elif dataset == 'dev':
        pre_preprocessed_alignment_ds = model_results['pre_preprocessed_alignment_ds_dev']
        epoch_min_loss = np.argmin(model_results['dev_loss'])
        model_best_params = model_params[epoch_min_loss, :]
    else:
        pre_preprocessed_alignment_ds['Cell'] = np.arange(len(pre_preprocessed_alignment_ds.Cell))

    alignment_ds['Cell'] = np.arange(len(alignment_ds.Cell))
    neuron_hemisphere_info_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/new_window_permutation_test_passive_140_neuron_hem_info.pkl'
    neuron_hemisphere_info = pd.read_pickle(neuron_hemisphere_info_path)
    hemisphere_vec = (neuron_hemisphere_info['hemisphere'] == 'R').astype(float).values

    if len(hemisphere_vec) != len(pre_preprocessed_alignment_ds.Cell):
        hemisphere_vec = np.tile(hemisphere_vec, 2)
    else:
        pre_preprocessed_alignment_ds = pre_preprocessed_alignment_ds.assign_coords({'hemisphere': ('Cell', hemisphere_vec)})
        left_neuron_idx = np.where(pre_preprocessed_alignment_ds['hemisphere'] == 0)[0]
        right_neuron_idx = np.where(pre_preprocessed_alignment_ds['hemisphere'] == 1)[0]
        pre_preprocessed_alignment_ds_stacked = pre_preprocessed_alignment_ds.stack(trialTime=['Trial', 'Time'])
        activity_name = 'scaled_firing_rate'
        inactivation_pre_preprocessed_alignment_ds_stacked = pre_preprocessed_alignment_ds_stacked.copy()
        inactivation_activity_da = inactivation_pre_preprocessed_alignment_ds_stacked.where(
            (inactivation_pre_preprocessed_alignment_ds_stacked['Cell'].isin(right_neuron_idx)),
          drop=True)[activity_name]
    if inactivate_after_stim:
        inactivation_activity_da_pre_stim = inactivation_activity_da.where((inactivation_activity_da['PeriEventTime'] < 0),
          drop=True)
        inactivation_activity_da_post_stim = inactivation_activity_da.where((inactivation_activity_da['PeriEventTime'] >= 0),
          drop=True)
        inactivation_activity_da_post_stim = inactivation_activity_da_post_stim * right_scaling
        inactivation_activity_da = xr.concat([inactivation_activity_da_pre_stim,
         inactivation_activity_da_post_stim],
          dim='trialTime')
    else:
        inactivation_activity_da = inactivation_activity_da * right_scaling

    """
    other_activity_da = inactivation_pre_preprocessed_alignment_ds_stacked.where(
        (~inactivation_pre_preprocessed_alignment_ds_stacked['Cell'].isin(right_neuron_idx)),
              drop=True)[activity_name]
    other_activity_da = other_activity_da * left_scaling
    """
    # 2021-12-06 temp change
    other_activity_da = inactivation_pre_preprocessed_alignment_ds_stacked.where(
        (~inactivation_pre_preprocessed_alignment_ds_stacked['Cell'].isin(right_neuron_idx)),
        drop=True)[activity_name]
    if inactivate_after_stim:
        other_activity_da_pre_stim = other_activity_da.where((other_activity_da['PeriEventTime'] < 0),
          drop=True)
        other_activity_da_post_stim = other_activity_da.where((other_activity_da['PeriEventTime'] >= 0),
          drop=True)
        # other_activity_da_post_stim = other_activity_da_post_stim * right_scaling
        other_activity_da_post_stim = other_activity_da_post_stim * left_scaling
        other_activity_da = xr.concat([other_activity_da_pre_stim,
         other_activity_da_post_stim],
          dim='trialTime')
    else:
        other_activity_da = other_activity_da * left_scaling




    recombined_activity_da = xr.concat([inactivation_activity_da, other_activity_da], dim='Cell')
    inactivation_pre_preprocessed_alignment_ds_stacked[activity_name] = recombined_activity_da
    inactivation_pre_preprocessed_alignment_ds = inactivation_pre_preprocessed_alignment_ds_stacked.unstack()
    stim_on_bin = 50
    input_tensor_inactivated = inactivation_pre_preprocessed_alignment_ds[activity_name].transpose('Trial', 'Cell', 'Time').values

    if model_name == 'hem-weight-constraint':
        model_inactivated_output = jax_dmodel.accumulate_after_stim_w_neuron_bias_fixed_drift_model_w_hem(input_tensor_inactivated,
          params=model_best_params,
          stim_on_bin=stim_on_bin,
          update_weight=c_param,
          hemisphere_vec=hemisphere_vec)
    else:
        model_inactivated_output = jax_dmodel.accumulate_after_stim_w_neuron_bias_fixed_drift_model(input_tensor_inactivated,
          params=model_best_params,
          stim_on_bin=stim_on_bin,
          update_weight=c_param)

    inactivation_pre_preprocessed_alignment_ds['PeriEventTime'] = inactivation_pre_preprocessed_alignment_ds['PeriEventTime'].isel(Trial=0)
    decision_threshold_val = 1
    model_type = 'drift'
    target_vis_cond_list = [
     -0.8, -0.4, -0.2, 0.2, 0.4, 0.8,
     0.0, 0.0, 0.8, 0.8, -0.8, -0.8,
     -0.1, 0.1]
    target_aud_cond_list = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
     -60, 60, 60, -60, 60, -60, np.inf, np.inf]
    model_inactivated_output_da = xr.DataArray(model_inactivated_output,
      dims=['Trial', 'Time'], coords={'Trial':inactivation_pre_preprocessed_alignment_ds.Trial,
     'Time':inactivation_pre_preprocessed_alignment_ds.Time})
    og_model_inactivation_stim_cond_pred_matrix_dict = jax_dmodel.get_stim_cond_response(alignment_ds=inactivation_pre_preprocessed_alignment_ds,
      y_test_pred_da_list=[
     model_inactivated_output_da],
      target_vis_cond_list=target_vis_cond_list,
      target_aud_cond_list=target_aud_cond_list)
    og_model_inactivation_behaviour_df = jax_dmodel.all_stim_cond_pred_matrix_dict_to_model_behaviour_df(all_stim_cond_pred_matrix_dict=og_model_inactivation_stim_cond_pred_matrix_dict,
      alignment_ds=inactivation_pre_preprocessed_alignment_ds,
      left_decision_threshold_val=(-decision_threshold_val),
      right_decision_threshold_val=decision_threshold_val,
      model_type=model_type,
      left_choice_val=0,
      right_choice_val=1,
      target_vis_cond_list=target_vis_cond_list,
      target_aud_cond_list=target_aud_cond_list)
    return og_model_inactivation_behaviour_df, og_model_inactivation_stim_cond_pred_matrix_dict, inactivation_pre_preprocessed_alignment_ds


def single_cv_file_inactivation_based_on_weight(input_activity_path,
                                                model_result_path, left_scaling=0.5, right_scaling=1.0,
                                                inactivate_after_stim=False, dataset='test'):
    """
    Performs inactivation based on weight
    """
    alignment_ds = xr.open_dataset(input_activity_path)
    N = 999
    model_results = pd.read_pickle(model_result_path)
    model_params = np.stack(model_results['param_history'])
    if N == 999:
        c_param = 1.0
    else:
        c_param = 2 ** (-(1 / N))
    y_test_pred_da_list = [model_results['y_test_pred_da']]


    if dataset == 'test':
        pre_preprocessed_alignment_ds = model_results['pre_preprocessed_alignment_ds_test']
        epoch_min_loss = np.argmin(model_results['test_loss'])
        model_best_params = model_params[epoch_min_loss, :]
    elif dataset == 'dev':
        pre_preprocessed_alignment_ds = model_results['pre_preprocessed_alignment_ds_dev']
        epoch_min_loss = np.argmin(model_results['dev_loss'])
        model_best_params = model_params[epoch_min_loss, :]
    else:
        pre_preprocessed_alignment_ds['Cell'] = np.arange(len(pre_preprocessed_alignment_ds.Cell))
        alignment_ds['Cell'] = np.arange(len(alignment_ds.Cell))

    num_cell = len(alignment_ds['Cell'])
    neuron_weights = model_best_params[num_cell:]

    hemisphere_vec = (neuron_weights < 0).astype(float)
    pre_preprocessed_alignment_ds = pre_preprocessed_alignment_ds.assign_coords({'hemisphere': ('Cell', hemisphere_vec)})
    left_neuron_idx = np.where(pre_preprocessed_alignment_ds['hemisphere'] == 0)[0]
    right_neuron_idx = np.where(pre_preprocessed_alignment_ds['hemisphere'] == 1)[0]
    pre_preprocessed_alignment_ds_stacked = pre_preprocessed_alignment_ds.stack(trialTime=['Trial', 'Time'])
    activity_name = 'scaled_firing_rate'
    inactivation_pre_preprocessed_alignment_ds_stacked = pre_preprocessed_alignment_ds_stacked.copy()
    inactivation_activity_da = inactivation_pre_preprocessed_alignment_ds_stacked.where(
        (inactivation_pre_preprocessed_alignment_ds_stacked['Cell'].isin(right_neuron_idx)),
      drop=True)[activity_name]

    if inactivate_after_stim:
        inactivation_activity_da_pre_stim = inactivation_activity_da.where((inactivation_activity_da['PeriEventTime'] < 0),
          drop=True)
        inactivation_activity_da_post_stim = inactivation_activity_da.where((inactivation_activity_da['PeriEventTime'] >= 0),
          drop=True)
        inactivation_activity_da_post_stim = inactivation_activity_da_post_stim * right_scaling
        inactivation_activity_da = xr.concat([inactivation_activity_da_pre_stim,
         inactivation_activity_da_post_stim],
          dim='trialTime')
    else:
        inactivation_activity_da = inactivation_activity_da * right_scaling

    other_activity_da = inactivation_pre_preprocessed_alignment_ds_stacked.where(
        (~inactivation_pre_preprocessed_alignment_ds_stacked['Cell'].isin(right_neuron_idx)),
      drop=True)[activity_name]

    if inactivate_after_stim:
        other_activity_da_pre_stim = other_activity_da.where((other_activity_da['PeriEventTime'] < 0),
          drop=True)
        other_activity_da_post_stim = other_activity_da.where((other_activity_da['PeriEventTime'] >= 0),
          drop=True)
        # other_activity_da_post_stim = other_activity_da_post_stim * right_scaling
        other_activity_da_post_stim = other_activity_da_post_stim * left_scaling
        other_activity_da = xr.concat([other_activity_da_pre_stim,
         other_activity_da_post_stim],
          dim='trialTime')
    else:
        other_activity_da = other_activity_da * left_scaling


    recombined_activity_da = xr.concat([inactivation_activity_da, other_activity_da], dim='Cell')
    inactivation_pre_preprocessed_alignment_ds_stacked[activity_name] = recombined_activity_da
    inactivation_pre_preprocessed_alignment_ds = inactivation_pre_preprocessed_alignment_ds_stacked.unstack()
    stim_on_bin = 50
    input_tensor_inactivated = inactivation_pre_preprocessed_alignment_ds[activity_name].transpose('Trial', 'Cell', 'Time').values
    model_inactivated_output = jax_dmodel.accumulate_after_stim_w_neuron_bias_fixed_drift_model(input_tensor_inactivated,
      params=model_best_params,
      stim_on_bin=stim_on_bin,
      update_weight=c_param)

    inactivation_pre_preprocessed_alignment_ds['PeriEventTime'] = inactivation_pre_preprocessed_alignment_ds['PeriEventTime'].isel(Trial=0)
    decision_threshold_val = 1
    model_type = 'drift'
    target_vis_cond_list = [
     -0.8, -0.4, -0.2, 0.2, 0.4, 0.8,
     0.0, 0.0, 0.8, 0.8, -0.8, -0.8,
     -0.1, 0.1]
    target_aud_cond_list = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
     -60, 60, 60, -60, 60, -60, np.inf, np.inf]
    model_inactivated_output_da = xr.DataArray(model_inactivated_output,
      dims=['Trial', 'Time'], coords={'Trial':inactivation_pre_preprocessed_alignment_ds.Trial,
     'Time':inactivation_pre_preprocessed_alignment_ds.Time})

    og_model_inactivation_stim_cond_pred_matrix_dict = jax_dmodel.get_stim_cond_response(alignment_ds=inactivation_pre_preprocessed_alignment_ds,
      y_test_pred_da_list=[
     model_inactivated_output_da],
      target_vis_cond_list=target_vis_cond_list,
      target_aud_cond_list=target_aud_cond_list)

    og_model_inactivation_behaviour_df = jax_dmodel.all_stim_cond_pred_matrix_dict_to_model_behaviour_df(all_stim_cond_pred_matrix_dict=og_model_inactivation_stim_cond_pred_matrix_dict,
      alignment_ds=inactivation_pre_preprocessed_alignment_ds,
      left_decision_threshold_val=(-decision_threshold_val),
      right_decision_threshold_val=decision_threshold_val,
      model_type=model_type,
      left_choice_val=0,
      right_choice_val=1,
      target_vis_cond_list=target_vis_cond_list,
      target_aud_cond_list=target_aud_cond_list)


    return og_model_inactivation_behaviour_df, og_model_inactivation_stim_cond_pred_matrix_dict, inactivation_pre_preprocessed_alignment_ds


def multiple_cv_inactivation(model_folder, input_activity_path, aud_LR_selective_cells,
                             vis_LR_selective_cells,
                             target_random_seed=0,
                             target_cv_index=[0, 1, 2, 3, 4], num_cv_fold=5,
                             vis_left_inactivation_scaling=1,
                             vis_right_inactivation_scaling=1,
                             mouse_origin='trained', subsample_random_seed=None,
                             shuffle_aud_vis_labels_random_seed=None, just_get_input_tensor=True):

    random_seed = target_random_seed
    stim_on_bin = 50
    c_param = 1
    start_time = -0.1
    end_time = 0.3

    alignment_ds = xr.open_dataset(input_activity_path)

    if subsample_random_seed is not None:
        train_test_group = alignment_ds.isel(Cell=0).trainTestGroup.values
        alignment_ds = alignment_ds.drop(['trainTestGroup', 'TrainTestGroup'])
    else:
        train_test_group = None

    # Get train test splits
    alignment_ds_list = list()

    for n_cond, (aud_diff, vis_diff) in enumerate(itertools.product(
            np.unique(alignment_ds['audDiff']), np.unique(alignment_ds['visDiff']))):

        try:
            stim_cond_alignment_ds = alignment_ds.where(
                (alignment_ds['visDiff'] == vis_diff) &
                (alignment_ds['audDiff'] == aud_diff), drop=True
            )
        except:
            print('No trials for visDiff: %.2f and audDiff: %.2f' % (vis_diff, aud_diff))
            continue

        stim_cond_alignment_ds = stim_cond_alignment_ds.assign({'stimCondID': ('Trial',
                                                                               np.repeat(n_cond, len(
                                                                                   stim_cond_alignment_ds.Trial.values)))})

        alignment_ds_list.append(stim_cond_alignment_ds)

    alignment_ds = xr.concat(alignment_ds_list, dim='Trial')

    stim_cond_label = alignment_ds['stimCondID'].values
    train_test_splitter = sklselection.StratifiedKFold(num_cv_fold, shuffle=True,
                                                       random_state=random_seed)
    cv_dev_test_idx = list(train_test_splitter.split(alignment_ds.Trial.values,
                                                     stim_cond_label))


    # Get cells to do inactivation
    cell_idx_method = 'isel'

    if mouse_origin == 'trained':
        selective_cells_df = pd.concat([aud_LR_selective_cells, vis_LR_selective_cells])
        selective_cells_df = dmodel_sample_and_fit.compile_alignment_ds_list_get_index_only(selective_cells_df)
        selective_cells_df['sorted_index'] = np.arange(len(selective_cells_df))

        sig_percentile = 95
        min_aud_abs_fr = 1
        min_vis_abs_fr = 1
        min_aud_on_off_abs_fr = 1
        diff_max_time = 0.6

        aud_LR_selective_cells = selective_cells_df.loc[
            (selective_cells_df['audLRabsPercentile'] >= sig_percentile) &
            (selective_cells_df['visLRabsPercentile'] < sig_percentile) &
            (selective_cells_df['audLRabsDiff'] >= min_aud_abs_fr) &
            (selective_cells_df['audLRmaxDiffTime'] > 0) &
            (selective_cells_df['audLRmaxDiffTime'] < diff_max_time) &
            (selective_cells_df['subjectRef'] != 1)
            ]

        vis_LR_selective_cells = selective_cells_df.loc[
            (selective_cells_df['audLRabsPercentile'] < sig_percentile) &
            (selective_cells_df['visLRabsPercentile'] >= sig_percentile) &
            (selective_cells_df['vis_lr_max_diff_time'] > 0) &
            (selective_cells_df['vis_lr_max_diff_time'] < diff_max_time) &
            (selective_cells_df['visLRabsDiff'] >= min_vis_abs_fr) &
            (selective_cells_df['subjectRef'] != 1)
            ]
    elif mouse_origin == 'naive':
        selective_cells_df = pd.concat([aud_LR_selective_cells, vis_LR_selective_cells])
        selective_cells_df = dmodel_sample_and_fit.compile_alignment_ds_list_get_index_only(selective_cells_df)
        selective_cells_df['sorted_index'] = np.arange(len(selective_cells_df))

        # For random seed 1 sorted
        if subsample_random_seed == 1:
            aud_sig_percentile = 91.313
            vis_sig_percentile = 73.636

        # For random seed 2 sorted
        if subsample_random_seed == 2:
            aud_sig_percentile = 91.515
            vis_sig_percentile = 71.364

        # For random seed 3 sorted
        if subsample_random_seed == 3:
            aud_sig_percentile = 91.75
            vis_sig_percentile = 59.009

        # Random seed 4 sorted
        if subsample_random_seed == 4:
            aud_sig_percentile = 91.712
            vis_sig_percentile = 57.608

        # Random seed 5 sorted
        if subsample_random_seed == 5:
            aud_sig_percentile = 91.912
            vis_sig_percentile = 55.405

        min_aud_abs_fr = 2.5
        min_vis_abs_fr = 2.5
        min_aud_on_off_abs_fr = 2.5
        diff_max_time = 0.6
        vis_diff_min_time = 0
        aud_diff_min_time = 0

        vis_LR_selective_cells = selective_cells_df.loc[
            (selective_cells_df['audLRabsPercentile'] < vis_sig_percentile) &
            (selective_cells_df['visLRabsPercentile'] >= vis_sig_percentile) &
            (selective_cells_df['vis_lr_max_diff_time'] >= vis_diff_min_time) &
            (selective_cells_df['vis_lr_max_diff_time'] < diff_max_time) &
            (selective_cells_df['visLRabsDiff'] >= min_vis_abs_fr)
            ]

        aud_LR_selective_cells = selective_cells_df.loc[
            (selective_cells_df['audLRabsPercentile'] >= aud_sig_percentile) &
            (selective_cells_df['visLRabsPercentile'] < aud_sig_percentile) &
            (selective_cells_df['audLRabsDiff'] >= min_aud_abs_fr) &
            (selective_cells_df['audLRmaxDiffTime'] >= aud_diff_min_time) &
            (selective_cells_df['audLRmaxDiffTime'] < diff_max_time)
            ]

    aud_neuron_idx = aud_LR_selective_cells['sorted_index'].values
    vis_left_neuron_idx = vis_LR_selective_cells.loc[
        (vis_LR_selective_cells['visLRsign'] == -1)]['sorted_index'].values
    vis_right_neuron_idx = vis_LR_selective_cells.loc[
        (vis_LR_selective_cells['visLRsign'] == 1)]['sorted_index'].values

    new_cell_idx = np.concatenate([aud_neuron_idx, vis_left_neuron_idx, vis_right_neuron_idx])
    print('Shape of new cell idx')
    print(np.shape(new_cell_idx))

    alignment_ds_stacked = alignment_ds.stack(trialTime=['Trial', 'Time'])

    alignment_ds_smoothed = alignment_ds_stacked

    # This part is surprinsigly memory intensive...
    # Maybe can extract the firing rate first and assign everyting later
    if cell_idx_method == 'isel':
        vis_left_neuron_smoothed = alignment_ds_smoothed.isel(Cell=vis_left_neuron_idx)
        vis_left_neuron_inactivated_smoothed = alignment_ds_smoothed.isel(Cell=vis_left_neuron_idx)
        vis_right_neuron_smoothed = alignment_ds_smoothed.isel(Cell=vis_right_neuron_idx)
        vis_right_neuron_inactivated_smoothed = alignment_ds_smoothed.isel(Cell=vis_right_neuron_idx)

    vis_left_neuron_inactivated_smoothed['firing_rate'] = vis_left_neuron_inactivated_smoothed[
                                                              'firing_rate'] / vis_left_inactivation_scaling
    vis_right_neuron_inactivated_smoothed['firing_rate'] = vis_right_neuron_inactivated_smoothed[
                                                               'firing_rate'] / vis_right_inactivation_scaling

    vis_left_neuron_inactivated_smoothed_zscore = vis_left_neuron_inactivated_smoothed

    vis_left_neuron_inactivated_smoothed_zscore_val = sstats.zmap(
        scores=(vis_left_neuron_inactivated_smoothed['firing_rate']),
        compare=(vis_left_neuron_smoothed['firing_rate']),
        axis=1)


    # Z-scoring
    if np.sum(np.isnan(vis_left_neuron_inactivated_smoothed_zscore_val)) > 0:
        print('Warning: vis left neuron have at least one nan, replacing with zeros for now...')
        vis_left_neuron_inactivated_smoothed_zscore_val[np.isnan(vis_left_neuron_inactivated_smoothed_zscore_val)] = 0

    vis_left_neuron_inactivated_smoothed_zscore = vis_left_neuron_inactivated_smoothed_zscore.assign(
        {'firing_rate': (['Cell', 'trialTime'],
                         vis_left_neuron_inactivated_smoothed_zscore_val)})
    vis_left_neuron_smoothed_zscore = vis_left_neuron_smoothed
    vis_left_neuron_smoothed_zscore['firing_rate'] = (
        [
            'Cell', 'trialTime'],
        sstats.zscore((vis_left_neuron_smoothed_zscore['firing_rate']),
                      axis=1))

    if np.sum(np.isnan(vis_left_neuron_smoothed_zscore['firing_rate'])) > 0:
        vis_left_neuron_smoothed_zscore_vals = vis_left_neuron_smoothed_zscore['firing_rate'].values
        vis_left_neuron_smoothed_zscore_vals[np.isnan(vis_left_neuron_smoothed_zscore_vals)] = 0
        vis_left_neuron_smoothed_zscore['firing_rate'] = (['Cell', 'trialTime'], vis_left_neuron_smoothed_zscore_vals)

    vis_right_neuron_inactivated_smoothed_zscore = vis_right_neuron_inactivated_smoothed
    vis_right_neuron_inactivated_smoothed_zscore_val = sstats.zmap(
        scores=(vis_right_neuron_inactivated_smoothed['firing_rate']),
        compare=(vis_right_neuron_smoothed['firing_rate']),
        axis=1)

    if np.sum(np.isnan(vis_right_neuron_inactivated_smoothed_zscore_val)) > 0:
        print('Warning: vis right neuron have at least one nan, replacing with zeros for now...')
        vis_right_neuron_inactivated_smoothed_zscore_val[np.isnan(vis_right_neuron_inactivated_smoothed_zscore_val)] = 0
    vis_right_neuron_inactivated_smoothed_zscore = vis_right_neuron_inactivated_smoothed_zscore.assign(
        {'firing_rate': (['Cell', 'trialTime'],
                         vis_right_neuron_inactivated_smoothed_zscore_val)})
    vis_right_neuron_smoothed_zscore = vis_right_neuron_smoothed
    vis_right_neuron_smoothed_zscore['firing_rate'] = (
        [
            'Cell', 'trialTime'],
        sstats.zscore((vis_right_neuron_smoothed_zscore['firing_rate']),
                      axis=1))

    if np.sum(np.isnan(vis_right_neuron_smoothed_zscore['firing_rate'])) > 0:
        vis_right_neuron_smoothed_zscore_vals = vis_right_neuron_smoothed_zscore['firing_rate'].values
        vis_right_neuron_smoothed_zscore_vals[np.isnan(vis_right_neuron_smoothed_zscore_vals)] = 0
        vis_right_neuron_smoothed_zscore['firing_rate'] = (['Cell', 'trialTime'], vis_right_neuron_smoothed_zscore_vals)
    if cell_idx_method == 'isel':
        aud_neuron_smoothed_zscore = alignment_ds_smoothed.isel(Cell=aud_neuron_idx)
    elif cell_idx_method == 'sel':
        aud_neuron_smoothed_zscore = alignment_ds_smoothed.sel(Cell=aud_neuron_idx)

    aud_neuron_smoothed_zscore['firing_rate'] = (
        [
            'Cell', 'trialTime'],
        sstats.zscore((aud_neuron_smoothed_zscore['firing_rate']),
                      axis=1))

    if np.sum(np.isnan(aud_neuron_smoothed_zscore['firing_rate'])) > 0:
        print('Auditory neuron zscore has nans, setting to zero for now')
        aud_neuron_smoothed_zscore_vals = aud_neuron_smoothed_zscore['firing_rate'].values
        aud_neuron_smoothed_zscore_vals[np.isnan(aud_neuron_smoothed_zscore_vals)] = 0
        aud_neuron_smoothed_zscore['firing_rate'] = (['Cell', 'trialTime'], aud_neuron_smoothed_zscore_vals)

    # Creating input tensor
    vis_left_neuron_inactivated_smoothed_zscore_vals = vis_left_neuron_inactivated_smoothed_zscore[
        'firing_rate'].unstack().values
    vis_right_neuron_inactivated_smoothed_zscore_vals = vis_right_neuron_inactivated_smoothed_zscore[
        'firing_rate'].unstack().values
    aud_neuron_smoothed_zscore_vals = aud_neuron_smoothed_zscore['firing_rate'].unstack().values

    input_tensor = np.concatenate([
        aud_neuron_smoothed_zscore_vals,
        vis_left_neuron_inactivated_smoothed_zscore_vals,
        vis_right_neuron_inactivated_smoothed_zscore_vals,
    ])

    # Re-arrange cells to the original order
    reorder_cell_idx = np.argsort(new_cell_idx)
    input_tensor = input_tensor[reorder_cell_idx, :, :]

    # Then reshape to Trial Cell Time
    input_tensor = np.swapaxes(input_tensor, 0, 1)

    # Subset time
    peri_event_time = alignment_ds.Time.values
    subset_time_index = np.where(
        (peri_event_time >= -0.1) &
        (peri_event_time <= 0.3)
    )[0]

    input_tensor = input_tensor[:, :, subset_time_index]
    print('Shape of input tensor')
    print(np.shape(input_tensor))

    if just_get_input_tensor:
        return input_tensor

    all_stim_cond_pred_matrix_dict_list = []
    for cv_fold_idx in target_cv_index:
        model_result_fname = 'driftParam_c_1_shuffle_%.f_cv%.f.pkl' % (target_random_seed, cv_fold_idx)
        N = 999
        model_best_params, c_param, stim_cond_output = load_model(model_folder,
                                                                        model_result_fname,
                                                                            alignment_ds=alignment_ds,
                                                                            N=N)

        if train_test_group is not None:
            test_trials = np.where(train_test_group == 1)[0]
            # dev_trials = np.where(train_test_group == 0)[0]
        else:

            dev_trials, test_trials = cv_dev_test_idx[cv_fold_idx]
        input_tensor_test = input_tensor[test_trials, :]

        if shuffle_aud_vis_labels_random_seed is not None:
            np.random.seed(shuffle_aud_vis_labels_random_seed)
            random_trials = np.random.permutation(np.arange(0, len(test_trials)))
            input_tensor_test = input_tensor_test[random_trials, :]

        model_output = jax_dmodel.accumulate_after_stim_w_neuron_bias_fixed_drift_model(
            input_tensor_test, params=model_best_params, stim_on_bin=stim_on_bin, update_weight=c_param)
        print('Shape of model output')
        print(np.shape(model_output))

        target_vis_cond_list = [-0.8, -0.4, -0.2, 0.2, 0.4, 0.8, 0.0, 0.0, 0.8, 0.8, -0.8, -0.8,
                                -0.1, 0.1]
        target_aud_cond_list = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                -60, 60, 60, -60, 60, -60, np.inf, np.inf]
        alignment_ds_test = alignment_ds.isel(Trial=test_trials)
        aud_cond_per_trial = alignment_ds_test.isel(Cell=0)['audDiff'].values
        vis_cond_per_trial = alignment_ds_test.isel(Cell=0)['visDiff'].values
        all_stim_cond_pred_matrix_dict = dict()

        for target_vis_cond, target_aud_cond in zip(target_vis_cond_list, target_aud_cond_list):
            stim_cond_trial_indices = np.where(
                (aud_cond_per_trial == target_aud_cond) &
                (vis_cond_per_trial == target_vis_cond))[0]

            all_stim_cond_pred_matrix_dict[(target_aud_cond, target_vis_cond)] = \
                model_output[stim_cond_trial_indices, :]

        all_stim_cond_pred_matrix_dict_list.append(all_stim_cond_pred_matrix_dict)

    # Combined all the cross-validation set into one big set
    all_stim_cond_pred_matrix_dict_combined = {}

    for key in all_stim_cond_pred_matrix_dict_list[0].keys():
        all_stim_cond_pred_matrix_dict_combined[key] = \
            np.concatenate([x[key] for x in all_stim_cond_pred_matrix_dict_list])

    return all_stim_cond_pred_matrix_dict_combined


def preload_cv_model_params(model_folder, target_random_seed, alignment_ds, target_cv_index=[0, 1, 2, 3, 4]):
    model_best_params_list = list()
    c_param_list = list()
    for cv_fold_idx in target_cv_index:
        model_result_fname = 'driftParam_c_1_shuffle_%.f_cv%.f.pkl' % (target_random_seed, cv_fold_idx)
        N = 999
        model_best_params, c_param, stim_cond_output = load_model(model_folder,
                                                                        model_result_fname,
                                                                            alignment_ds=alignment_ds,
                                                                            N=N)
        model_best_params_list.append(model_best_params)
        c_param_list.append(c_param)

    return model_best_params_list, c_param_list


def run_input_through_model_params(input_tensor, alignment_ds, model_best_params_list, c_param_list, train_test_group,
                                   shuffle_aud_vis_labels_random_seed, cv_dev_test_idx=None,
                                   stim_on_bin=50, target_cv_index=[0, 1, 3, 4]):

    all_stim_cond_pred_matrix_dict_list = []

    for cv_fold_idx in target_cv_index:

        model_best_params = model_best_params_list[cv_fold_idx]
        c_param = c_param_list[cv_fold_idx]

        if train_test_group is not None:
            test_trials = np.where(train_test_group == 1)[0]
            # dev_trials = np.where(train_test_group == 0)[0]
        else:
            dev_trials, test_trials = cv_dev_test_idx[cv_fold_idx]


        input_tensor_test = input_tensor[test_trials, :]

        if shuffle_aud_vis_labels_random_seed is not None:
            np.random.seed(shuffle_aud_vis_labels_random_seed)
            random_trials = np.random.permutation(np.arange(0, len(test_trials)))
            input_tensor_test = input_tensor_test[random_trials, :]

        model_output = jax_dmodel.accumulate_after_stim_w_neuron_bias_fixed_drift_model(
            input_tensor_test, params=model_best_params, stim_on_bin=stim_on_bin, update_weight=c_param)
        print('Shape of model output')
        print(np.shape(model_output))

        target_vis_cond_list = [-0.8, -0.4, -0.2, 0.2, 0.4, 0.8, 0.0, 0.0, 0.8, 0.8, -0.8, -0.8,
                                -0.1, 0.1]
        target_aud_cond_list = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                -60, 60, 60, -60, 60, -60, np.inf, np.inf]
        alignment_ds_test = alignment_ds.isel(Trial=test_trials)
        aud_cond_per_trial = alignment_ds_test.isel(Cell=0)['audDiff'].values
        vis_cond_per_trial = alignment_ds_test.isel(Cell=0)['visDiff'].values
        all_stim_cond_pred_matrix_dict = dict()

        for target_vis_cond, target_aud_cond in zip(target_vis_cond_list, target_aud_cond_list):
            stim_cond_trial_indices = np.where(
                (aud_cond_per_trial == target_aud_cond) &
                (vis_cond_per_trial == target_vis_cond))[0]

            all_stim_cond_pred_matrix_dict[(target_aud_cond, target_vis_cond)] = \
                model_output[stim_cond_trial_indices, :]

        all_stim_cond_pred_matrix_dict_list.append(all_stim_cond_pred_matrix_dict)

    # Combined all the cross-validation set into one big set
    all_stim_cond_pred_matrix_dict_combined = {}

    for key in all_stim_cond_pred_matrix_dict_list[0].keys():
        all_stim_cond_pred_matrix_dict_combined[key] = \
            np.concatenate([x[key] for x in all_stim_cond_pred_matrix_dict_list])

    return all_stim_cond_pred_matrix_dict_combined


def load_mouse_inactivation_data():
    # April 2021: new numbers from Pip
    mouse_inactivation_fits = stdf.loadmat(
        '/media/timsit/Partition 1/data/interim/multispaceworld-mice-fits/miceFitsInactivationNew_april2021.mat')
    mouse_inactivation_fits = mouse_inactivation_fits['miceFitsInactivationNew']

    mouse_inactivation_fits_vis_vals = np.array(mouse_inactivation_fits['visValues']).flatten()
    mouse_inactivation_fits_aud_vals = np.array(mouse_inactivation_fits['audValues']).flatten()
    mouse_inactivation_fits_pRight = np.array(mouse_inactivation_fits['fracRightTurnsLog']).flatten()

    # Mouse first
    # mouse_popt_inact = np.array([-0.76063857,  0.03611851,  3.35999308,  0.85019176, -1.2402565 ,
    #     1.86516393])

    # mouse_popt_inact = np.array([0.5691, 2.6661, 0.7523, 0.5879, 1.7806, 2.2095])

    # New values (bias, visScaleR, visScaleL, N, audScaleR, audScaleL)
    mouse_popt_inact = np.array([0.5691, 2.6661, 0.7523, 0.5879, 1.7806, 2.2095])

    # Changed to the ordering I use : beta, s_vl, s_vr, y, s_al, s_ar
    mouse_popt_inact = np.array([0.5691, -0.7523, 2.6661, 0.5879, -2.2095,
                                 1.7806])

    unscale_data_points = True

    if unscale_data_points:
        # Unscale the data points
        max_v_val = 0.8 ** mouse_popt_inact[3]
        min_v_val = -0.8 ** mouse_popt_inact[3]
        mouse_inactivation_fits_vis_vals = (mouse_inactivation_fits_vis_vals + 1) / 2 * (
                    max_v_val - min_v_val) + min_v_val

    # mouse_inactivation_aud_left_loc = np.where(mouse_inactivation_fits_aud_vals == -1)[0]
    # mouse_inactivation_aud_right_loc = np.where(mouse_inactivation_fits_aud_vals == 1)[0]

    # April 2021 new coding of aud
    mouse_inactivation_aud_left_loc = np.where(mouse_inactivation_fits_aud_vals == -60)[0]
    mouse_inactivation_aud_right_loc = np.where(mouse_inactivation_fits_aud_vals == 60)[0]

    mouse_inactivation_aud_left_vis_vals = mouse_inactivation_fits_vis_vals[mouse_inactivation_aud_left_loc]
    mouse_inactivation_aud_left_logPright = mouse_inactivation_fits_pRight[mouse_inactivation_aud_left_loc]

    mouse_inactivation_aud_right_vis_vals = mouse_inactivation_fits_vis_vals[mouse_inactivation_aud_right_loc]
    mouse_inactivation_aud_right_logPright = mouse_inactivation_fits_pRight[mouse_inactivation_aud_right_loc]

    mouse_inactivation_aud_center_loc = np.where(mouse_inactivation_fits_aud_vals == 0)[0]
    mouse_inactivation_aud_center_vis_vals = mouse_inactivation_fits_vis_vals[mouse_inactivation_aud_center_loc]
    mouse_inactivation_aud_center_logPright = mouse_inactivation_fits_pRight[mouse_inactivation_aud_center_loc]

    all_17_mice_popt = [-0.1268, -2.5418, 2.7152, 0.6510, -1.4541, 1.7149]

    return mouse_inactivation_aud_left_logPright, mouse_inactivation_aud_center_logPright, mouse_inactivation_aud_right_logPright



def main():
    """

    model_control_output : (numpy ndarray)
        matrix with dimensions (numTrial, numTimePoints)
        in most use cases expect (360, 200)

    Returns
    -------

    """
    combined_ds_filename = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/new_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials_w_modality.nc'
    model_result_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/drift-model-16/'
    model_result_fname = 'driftParam_c_1_shuffle_0_cv0.pkl'
    model_inactivation_result_save_folder = os.path.join(model_result_folder, 'inactivation')
    if not os.path.exists(model_inactivation_result_save_folder):
        os.mkdir(model_inactivation_result_save_folder)
    save_fname = 'shuffle_0_cv0_coherent_vis_right_inactivation_scale_2_include_simulated_stim.pkl'
    N = 999
    stim_on_bin = 50
    inactivation_trial_cond = 'all'
    inactivation_neurons = 'vis'
    cell_idx_method = 'isel'
    activity_name = 'firing_rate'
    reindex_cells = True
    vis_left_inactivation_scaling = 1.0
    vis_right_inactivation_scaling = 2.0
    smooth_spikes = True
    simulate_stim_conds = True
    print('Loading data...')
    alignment_ds, alignment_ds_smoothed = load_data(combined_ds_filename, inactivation_trial_cond)
    if reindex_cells:
        alignment_ds = alignment_ds.assign_coords({'Cell': np.arange(len(alignment_ds.Cell.values))})
        alignment_ds_smoothed = alignment_ds_smoothed.assign_coords({'Cell': np.arange(len(alignment_ds.Cell.values))})
    vis_left_neuron_idx = np.where(alignment_ds['modality'] == 'visLeft')[0]
    vis_right_neuron_idx = np.where(alignment_ds['modality'] == 'visRight')[0]
    aud_neuron_idx = np.where(alignment_ds['modality'].isin(['audLeft', 'audRight']))[0]
    if simulate_stim_conds:
        print('Simulating unobserved stimulus conditions')
        simulated_stim_conds_ds = simulate_other_stim_conds(pre_preprocessed_alignment_ds=(alignment_ds_smoothed.unstack()))
        alignment_ds_smoothed = alignment_ds_smoothed.unstack()
        alignment_ds_smoothed = xr.concat([alignment_ds_smoothed, simulated_stim_conds_ds], dim='Trial')
        alignment_ds_smoothed = alignment_ds_smoothed.assign_coords({'Trial': np.arange(len(alignment_ds_smoothed.Trial.values))})
        alignment_ds_smoothed = alignment_ds_smoothed.stack(trialTime=['Trial', 'Time'])
    print('Extracting original model results...')
    model_best_params, c_param, stim_cond_output = load_model(model_result_folder, model_result_fname, alignment_ds=alignment_ds,
      N=N)
    print('Performing inactivation...')
    print('Selected trial cond type to inactive: %s' % inactivation_trial_cond)
    if inactivation_neurons == 'vis':
        alignment_ds_smoothed_zscore, alignment_ds_smoothed_vis_inactivated = do_inactivation(alignment_ds, alignment_ds_smoothed,
          vis_left_neuron_idx=vis_left_neuron_idx, vis_right_neuron_idx=vis_right_neuron_idx,
          aud_neuron_idx=aud_neuron_idx,
          cell_idx_method=cell_idx_method,
          vis_left_inactivation_scaling=vis_left_inactivation_scaling,
          vis_right_inactivation_scaling=vis_right_inactivation_scaling)
    elif inactivation_neurons == 'all-left':
            alignment_ds_smoothed_zscore, alignment_ds_smoothed_vis_inactivated = do_inactivation(alignment_ds, alignment_ds_smoothed,
              vis_left_neuron_idx=vis_left_neuron_idx, vis_right_neuron_idx=vis_right_neuron_idx,
              aud_neuron_idx=aud_neuron_idx,
              cell_idx_method=cell_idx_method,
              vis_left_inactivation_scaling=vis_left_inactivation_scaling,
              vis_right_inactivation_scaling=vis_right_inactivation_scaling)


    print('Compiling inactivation and original model output')
    input_tensor_control = alignment_ds_smoothed_zscore['firing_rate'].unstack().transpose('Trial', 'Cell', 'Time').values
    input_tensor_vis_inactivated = alignment_ds_smoothed_vis_inactivated['firing_rate'].transpose('Trial', 'Cell', 'Time').values
    model_control_output = jax_dmodel.accumulate_after_stim_w_neuron_bias_fixed_drift_model(input_tensor_control,
      params=model_best_params, stim_on_bin=stim_on_bin,
      update_weight=c_param)
    model_vis_inactivated_output = jax_dmodel.accumulate_after_stim_w_neuron_bias_fixed_drift_model(input_tensor_vis_inactivated,
      params=model_best_params,
      stim_on_bin=stim_on_bin,
      update_weight=c_param)

    print('Extracting behaviour from model output')
    model_behaviour_df = get_behaviour_from_model_output(peri_event_time=(alignment_ds['PeriEventTime'].values),
      model_control_output=model_control_output,
      model_vis_inactivated_output=model_vis_inactivated_output,
      decision_threshold=1,
      verbose=False,
      alignment_ds=(alignment_ds_smoothed.unstack()),
      include_no_go=True)
    model_output_dict = {'control':model_control_output,
     'inactivation':model_vis_inactivated_output,
     'behaviourDF':model_behaviour_df,
     'inactivation_trial_cond':inactivation_trial_cond}
    print('Saving data')
    with open(os.path.join(model_inactivation_result_save_folder, save_fname), 'wb') as (handle):
        pkl.dump(model_output_dict, handle)
    print('All done!')


if __name__ == '__main__':
    main()