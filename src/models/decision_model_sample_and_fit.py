import numpy as np

import xarray as xr
import pandas as pd

import os
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import sciplotlib.style as splstyle

import src.data.analyse_spikes as anaspikes
import src.data.process_ephys_data as pephys
import src.models.network_model as nmodel
import src.data.stat as stat
import scipy.stats as sstats

import src.visualization.vizmodel as vizmodel

import sklearn.model_selection as sklselection
import sklearn.linear_model as sklinear

from tqdm import tqdm

import itertools
import pickle as pkl
import pdb

# Model fitting dependencies
import src.models.jax_decision_model as jax_dmodel
import functools
from jax.experimental import optimizers
import jax


def compile_alignment_ds_list(selected_cells_df, stim_alignment_folder, include_modality=False,
                              cell_idx_method='idx', include_reverse_map=True):
    """

    Parameters
    ----------
    selected_cells_df : (pandas dataframe)
    stim_alignment_folder : (str)
    include_modality : (bool)
    Returns
    -------

    """

    alignment_ds_list = list()

    if include_reverse_map:
        brain_region_df_list = []

    for subject in np.unique([selected_cells_df['subjectRef']]):
        subject_df = selected_cells_df.loc[
            selected_cells_df['subjectRef'] == subject
            ]

        for exp in np.unique(subject_df['expRef']):

            exp_df = subject_df.loc[
                subject_df['expRef'] == exp
                ]

            for brain_region in np.unique(exp_df['brainRegion']):
                brain_region_df = exp_df.loc[
                    exp_df['brainRegion'] == brain_region
                    ]

                alignment_ds = pephys.load_subject_exp_alignment_ds(
                    alignment_folder=stim_alignment_folder,
                    subject_num=subject, exp_num=exp,
                    target_brain_region=brain_region,
                    aligned_event='stimOnTime',
                    alignment_file_ext='.nc')

                if include_reverse_map:
                    brain_region_df_list.append(brain_region_df)

                # subset cells
                if cell_idx_method == 'index':
                    subest_cell_idx = brain_region_df.index
                    subset_cell_ds = alignment_ds.sel(Cell=subest_cell_idx)
                elif cell_idx_method == 'Cell':
                    subest_cell_idx = brain_region_df['Cell'].values
                    subset_cell_ds = alignment_ds.sel(Cell=subest_cell_idx)
                elif cell_idx_method == 'ALL':
                    subset_cell_ds = alignment_ds

                if include_modality:
                    subset_cell_ds.assign_coords({'modality': ('Cell', brain_region_df['modality'].values)})

                alignment_ds_list.append(subset_cell_ds)
    if include_reverse_map:
        return brain_region_df_list, alignment_ds_list
    else:
        return alignment_ds_list




def compile_alignment_ds_list_get_index_only(selected_cells_df):
    """

    Parameters
    ----------
    selected_cells_df : (pandas dataframe)
    stim_alignment_folder : (str)
    include_modality : (bool)
    Returns
    -------

    """

    brain_region_df_list = []

    for subject in np.unique([selected_cells_df['subjectRef']]):
        subject_df = selected_cells_df.loc[
            selected_cells_df['subjectRef'] == subject
            ]

        for exp in np.unique(subject_df['expRef']):

            exp_df = subject_df.loc[
                subject_df['expRef'] == exp
                ]

            for brain_region in np.unique(exp_df['brainRegion']):
                brain_region_df = exp_df.loc[
                    exp_df['brainRegion'] == brain_region
                    ]

                brain_region_df_list.append(brain_region_df)

    return pd.concat(brain_region_df_list)



def get_stim_cond_trial_count(alignment_ds_list, aud_conditions=[-60, 60, np.inf],
                              visual_conditions=[-0.8, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.8]):
    """

    Parameters
    ----------
    alignment_ds_list
    aud_conditions
    visual_conditions

    Returns
    -------

    """
    stim_cond_trial_count_list = list()
    stim_cond_trial_count_df_list = list()
    for aud_cond, vis_cond in itertools.product(aud_conditions, visual_conditions):

        stim_cond_ds_list = list()
        stim_cond_trial_count_list = list()

        for alignment_ds in alignment_ds_list:
            stim_cond_ds = anaspikes.get_target_condition_ds(
                alignment_ds, aud=aud_cond, vis=vis_cond)
            stim_cond_ds_list.append(stim_cond_ds)
            stim_cond_trial_count_list.append(len(stim_cond_ds.Trial.values))

        min_num_trial = np.min(stim_cond_trial_count_list)

        stim_cond_trial_count_dict = dict()
        stim_cond_trial_count_dict['aud_cond'] = [aud_cond]
        stim_cond_trial_count_dict['vis_cond'] = [vis_cond]
        stim_cond_trial_count_dict['trial_count'] = [min_num_trial]
        stim_cond_trial_count_df_list.append(pd.DataFrame.from_dict(stim_cond_trial_count_dict))

    stim_cond_trial_count_df = pd.concat(stim_cond_trial_count_df_list)

    return stim_cond_trial_count_df


def get_cell_psth_for_sampling(alignment_ds_list, stim_cond_trial_count_df,
                               min_trial_count=36, aud_conditions=[-60, 60, np.inf],
                               visual_conditions=[-0.8, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.8],
                               train_test_split=False, load_to_memory=False):
    """

    Parameters
    ----------
    alignment_ds_list
    stim_cond_trial_count_df
    min_trial_count
    aud_conditions
    visual_conditions : (list)
    include_modality : (bool)
    Returns
    -------

    """
    cell_combined_stim_cond_ds_list = list()

    if train_test_split:
        cell_combined_stim_cond_ds_test_list = list()


    for aud_cond, vis_cond in itertools.product(aud_conditions, visual_conditions):

        # check stim cond exists (eg. vis -0.4 aud 60 does not exist)
        stim_cond_trial_count = stim_cond_trial_count_df.loc[
            (stim_cond_trial_count_df['aud_cond'] == aud_cond) &
            (stim_cond_trial_count_df['vis_cond'] == vis_cond)
            ]['trial_count'].values

        if stim_cond_trial_count < min_trial_count:
            continue

        stim_cond_ds_list = list()

        if train_test_split:
            stim_cond_ds_test_list = list()

        for alignment_ds in alignment_ds_list:

            stim_cond_ds = anaspikes.get_target_condition_ds(alignment_ds,
                                                                 aud=aud_cond, vis=vis_cond)

            # Take the mean across trials
            if not train_test_split:
                stim_cond_ds_mean_across_trials = stim_cond_ds.mean('Trial')
            else:
                trial_numbers_shuffled = np.random.permutation(np.arange(len(stim_cond_ds.Trial)))
                train_trials = trial_numbers_shuffled[0:int(len(trial_numbers_shuffled) / 2)]
                test_trials = trial_numbers_shuffled[int(len(trial_numbers_shuffled) / 2):]
                stim_cond_ds_mean_across_trials = stim_cond_ds.isel(Trial=train_trials).mean('Trial')
                stim_cond_ds_mean_across_trials_test = stim_cond_ds.isel(Trial=test_trials).mean('Trial')

            # stim_cond_ds_subset = stim_cond_ds.isel(Trial=slice(0, min_trial_count))

            # re-index trials so we can combine them across experiments later
            # stim_cond_ds_subset = stim_cond_ds_subset.assign_coords({'Trial': np.arange(min_trial_count)})
            if train_test_split:
                if load_to_memory:
                    stim_cond_ds_mean_across_trials_test = stim_cond_ds_mean_across_trials_test.load()
                stim_cond_ds_test_list.append(stim_cond_ds_mean_across_trials_test)

            stim_cond_ds_list.append(stim_cond_ds_mean_across_trials)

        if len(stim_cond_ds.Trial.values) > 0:
            all_cell_combined_stim_cond_ds = xr.concat(stim_cond_ds_list, dim='Cell')
            if train_test_split:
                all_cell_combined_stim_cond_ds_test = xr.concat(stim_cond_ds_test_list, dim='Cell')
                cell_combined_stim_cond_ds_test_list.append(all_cell_combined_stim_cond_ds_test)

        if load_to_memory:
            all_cell_combined_stim_cond_ds = all_cell_combined_stim_cond_ds.load()
        cell_combined_stim_cond_ds_list.append(all_cell_combined_stim_cond_ds)

    if train_test_split:
        return cell_combined_stim_cond_ds_list, cell_combined_stim_cond_ds_test_list
    else:
        return cell_combined_stim_cond_ds_list


def get_poisson_samples(subset_cell_combined_stim_cond_ds_list, num_samples_to_get=360,
                        time_bin_width=2/1000):
    """

    Parameters
    ----------
    subset_cell_combined_stim_cond_ds_list
    num_samples_to_get
    time_bin_width : (float)
        time bin width in seconds (fixed by the alignment file)
    include_modality : (bool)
        whether to include a Cell dimension describing the selectivity of the neuron: (aud, vis or audvis)
    Returns
    -------

    """


    all_stim_cond_cell_ds_samples_list = list()

    for subset_stim_cond_ds in tqdm(subset_cell_combined_stim_cond_ds_list):

        cell_by_time_mean_fr = subset_stim_cond_ds['firing_rate'].values
        cell_by_time_mean_spike_count = cell_by_time_mean_fr * time_bin_width

        num_cell = len(subset_stim_cond_ds.Cell)
        num_time_bin = len(subset_stim_cond_ds.Time)

        stim_cond_all_cell_ds_samples_list = list()

        for cell_idx in np.arange(num_cell):

            # cell_psth = cell_by_time_mean_fr[cell_idx, :]
            cell_psth_spike_count = cell_by_time_mean_spike_count[cell_idx, :]
            spike_rate_samples = np.zeros((num_samples_to_get, num_time_bin))  # pre-allocate

            for time_bin in np.arange(num_time_bin):
                # mean_fr = cell_psth[time_bin]
                mean_spike_count = cell_psth_spike_count[time_bin]
                # spike_samples = sstats.poisson.rvs(mu=mean_fr, size=num_samples_to_get)
                spike_samples = sstats.poisson.rvs(mu=mean_spike_count, size=num_samples_to_get)

                spike_rate_samples[:, time_bin] = spike_samples / time_bin_width
                # spike_rate_samples[:, time_bin] = spike_samples

            # make cell ds
            visDiffRounded = np.round(subset_stim_cond_ds['visDiff'].values[0], 2)
            stim_cond_cell_ds_samples = xr.Dataset({'firing_rate': (['Trial', 'Time'],
                                                                    spike_rate_samples),
                                                    'audDiff': ('Trial', np.repeat(
                                                        subset_stim_cond_ds['audDiff'].values[0],
                                                        num_samples_to_get)),
                                                    'visDiff': ('Trial', np.repeat(
                                                        visDiffRounded, num_samples_to_get))})

            stim_cond_all_cell_ds_samples_list.append(stim_cond_cell_ds_samples)

        stim_cond_all_cell_ds_samples = xr.concat(stim_cond_all_cell_ds_samples_list, dim='Cell')
        stim_cond_all_cell_ds_samples = stim_cond_all_cell_ds_samples.assign_coords({
            'Cell': subset_stim_cond_ds.Cell.values,
            'Trial': np.arange(num_samples_to_get),
            'Time': subset_stim_cond_ds.PeriEventTime
        })

        all_stim_cond_cell_ds_samples_list.append(stim_cond_all_cell_ds_samples)

    all_stim_cond_cell_ds_samples = xr.concat(all_stim_cond_cell_ds_samples_list, dim='Trial')
    all_stim_cond_cell_ds_samples = all_stim_cond_cell_ds_samples.assign_coords(
        {'Trial': np.arange(0, len(all_stim_cond_cell_ds_samples.Trial.values))})



    return all_stim_cond_cell_ds_samples


def sample_neurons(stim_alignment_folder, selected_cells_df, save_path, include_modality=False,
                   include_audio_center=False, num_samples_to_get=360, cell_idx_method='idx',
                   alignment_ds_list=None, train_test_splits=False, test_size=0.5, min_trial_count=36):
    """
    Sample selected neurons (using Poisson distribution)
    Parameters
    ----------
    stim_alignment_folder : (str)
    selected_cells_df : (pandas dataframe)
    save_path : (str)
    include_modality : (bool)
        whether to include modality information.
    include_audio_center : (bool)
        whether to include audio center in the sampling
    num_samples_to_get : (int)
        number of samples per stimulus condition to sample
    Returns
    -------

    """

    if include_audio_center:
        aud_conditions = [-60, 60, np.inf, 0]
    else:
        aud_conditions = [-60, 60, np.inf]

    if alignment_ds_list is None:
        alignment_ds_list = compile_alignment_ds_list(selected_cells_df=selected_cells_df,
                                                      stim_alignment_folder=stim_alignment_folder,
                                                      include_modality=include_modality,
                                                      cell_idx_method=cell_idx_method)

    stim_cond_trial_count_df = get_stim_cond_trial_count(alignment_ds_list,
                             aud_conditions=aud_conditions,
                              visual_conditions=[-0.8, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.8])


    if not train_test_splits:
        cell_combined_stim_cond_ds_list = get_cell_psth_for_sampling(alignment_ds_list,
                                          stim_cond_trial_count_df,
                                          min_trial_count=min_trial_count, aud_conditions=aud_conditions,
                                            visual_conditions=[-0.8, -0.4, -0.2, -0.1, 0.0, 0.1,
                                                                0.2, 0.4, 0.8],
                                                load_to_memory=False)

        all_stim_cond_cell_ds_samples = get_poisson_samples(
            subset_cell_combined_stim_cond_ds_list=cell_combined_stim_cond_ds_list,
            num_samples_to_get=num_samples_to_get)
    else:
        print('Splitting data to train and test splits, then sampling from the two PSTHs')
        cell_combined_stim_cond_ds_list, cell_combined_stim_cond_ds_test_list = get_cell_psth_for_sampling(
            alignment_ds_list, stim_cond_trial_count_df,
            min_trial_count=min_trial_count,
            aud_conditions=aud_conditions,
            visual_conditions=[
                -0.8, -0.4, -0.2, -0.1, 0.0,
                0.1, 0.2, 0.4, 0.8],
            train_test_split=train_test_splits, load_to_memory=True)
        all_stim_cond_cell_ds_samples_train = get_poisson_samples(
            subset_cell_combined_stim_cond_ds_list=cell_combined_stim_cond_ds_list,
            num_samples_to_get=(int(num_samples_to_get / 2)))
        all_stim_cond_cell_ds_samples_test = get_poisson_samples(
            subset_cell_combined_stim_cond_ds_list=cell_combined_stim_cond_ds_test_list,
            num_samples_to_get=(int(num_samples_to_get / 2)))
        all_stim_cond_cell_ds_samples_train['trainTestGroup'] = ('Trial',
                                                                 np.repeat(0,
                                                                           len(all_stim_cond_cell_ds_samples_train.Trial)))
        all_stim_cond_cell_ds_samples_test['trainTestGroup'] = ('Trial',
                                                                np.repeat(1,
                                                                          len(all_stim_cond_cell_ds_samples_train.Trial)))
        all_stim_cond_cell_ds_samples = xr.concat([all_stim_cond_cell_ds_samples_train,
                                                   all_stim_cond_cell_ds_samples_test],
                                                  dim='Trial')



    all_stim_cond_cell_ds_samples.to_netcdf(save_path)


def simulate_other_stim_conds(pre_preprocessed_alignment_ds, model_best_params, stim_on_bin=50,
                              c_param=1,
                    vis_cond_to_sim=[-0.1, -0.2, -0.4, 0.1, 0.2, 0.4, -0.1, -0.2, -0.4, 0.1, 0.2, 0.4],
                    aud_cond_to_sim=[60, 60, 60, 60, 60, 60, -60, -60, -60, -60, -60, -60]):
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

    pre_preprocessed_alignment_ds = pre_preprocessed_alignment_ds.assign_coords(
        {'Cell': np.arange(len(pre_preprocessed_alignment_ds.Cell))}
    )

    vis_neuron_ds = pre_preprocessed_alignment_ds.where(
        pre_preprocessed_alignment_ds['modality'].isin(
            ['visLeft', 'visRight']), drop=True
    )

    aud_neuron_ds = pre_preprocessed_alignment_ds.where(
        pre_preprocessed_alignment_ds['modality'].isin(
            ['audLeft', 'audRight']), drop=True
    )

    simulated_pred_matrix_dict = dict()

    for vis_sim, aud_sim in zip(vis_cond_to_sim, aud_cond_to_sim):
        vis_sim_cond_neuron_ds = vis_neuron_ds.where(
            (vis_neuron_ds['audDiff'] == np.inf) &
            (vis_neuron_ds['visDiff'] == vis_sim), drop=True
        )

        aud_sim_cond_neuron_ds = aud_neuron_ds.where(
            (aud_neuron_ds['audDiff'] == aud_sim) &
            (aud_neuron_ds['visDiff'] == 0), drop=True
        )

        # reindex trial numbers
        vis_sim_cond_neuron_ds = vis_sim_cond_neuron_ds.assign_coords(
            {'Trial': np.arange(len(vis_sim_cond_neuron_ds.Trial.values))})
        aud_sim_cond_neuron_ds = aud_sim_cond_neuron_ds.assign_coords(
            {'Trial': np.arange(len(aud_sim_cond_neuron_ds.Trial.values))})

        combined_neuron_ds = xr.concat([vis_sim_cond_neuron_ds, aud_sim_cond_neuron_ds], dim='Cell')
        combined_neuron_ds['visDiff'] = vis_sim_cond_neuron_ds.isel(Cell=0, Time=0)['visDiff']
        combined_neuron_ds['audDiff'] = aud_sim_cond_neuron_ds.isel(Cell=0, Time=0)['audDiff']

        # Re-order the cells to the original order (otherwise weights will be wrong)
        combined_neuron_ds = combined_neuron_ds.sel(Cell=pre_preprocessed_alignment_ds.Cell)

        input_tensor_simulated_conds = combined_neuron_ds['scaled_firing_rate'].transpose(
            'Trial', 'Cell', 'Time').values

        simulated_aud_diff = np.unique(combined_neuron_ds['audDiff'])[0]
        simulated_vis_diff = np.unique(combined_neuron_ds['visDiff'])[0]

        simulated_cond_model_output = jax_dmodel.accumulate_after_stim_w_neuron_bias_fixed_drift_model(
            input_tensor_simulated_conds, params=model_best_params,
            stim_on_bin=stim_on_bin,
            update_weight=c_param)

        simulated_pred_matrix_dict[(simulated_aud_diff, simulated_vis_diff)] = simulated_cond_model_output

    return simulated_pred_matrix_dict


def main():

    # Sampling parameters
    run_sampling = False
    num_samples_to_get = 360
    stim_alignment_folder = '/media/timsit/Partition 1/data/interim/passive-m2-new-parent-alignment-2ms/'
    save_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn'
    # save_name = 'max_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials.nc'

    # 2020-12-16: Include audio center
    save_name = 'max_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_%.f_trials_include_audio_center.nc' % num_samples_to_get

    # Model fitting parameters
    # input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/max_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials.nc'
    # model_save_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/drift-model-12-max-window-permutation-test-sig-cells-aud-vis-balanced/'
    # Mean window permutation test (140 neurons)
    input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/new_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials.nc'

    # Model 20
    # Same as model 12 with modifications but defined
    # smoothing, scaling, 5 fold cross validation split per stim cond
    # weight parameter and bias parameter
    # number of auditory and visual trials balanced
    # auditory center included I think?
    # input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/max_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials.nc'


    # 2020-01-23 More trials (but may cause memory issue)
    # input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/max_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_720_trials_include_audio_center.nc'

    # 2021-02-13: neurons selected using regression method
    # input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/kernel_sig_133_neurons_samples.nc'

    # model_save_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/drift-model-12-mean-window-permutation-test-sig-cells-aud-vis-balanced/'
    # model_number = 23
    # model_save_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/drift-model-%.f/' % model_number

    # Model 24: 140 neurons with contraint on weights: left hemisphere neurons: positive right, right hemisphere neuron; negative weight
    # model_number = 24
    # model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number

    # Model 25: 140 neurons with contraint on weights: left hemisphere neurons: positive right, right hemisphere neuron; negative weight
    # Same as model 24 but turned smoothing off
    # Model 26: same as 24, but train only on highest visual
    # model_number = 26
    # model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number

    # Model 27: reflected neurons (double the amount of neurons)
    # input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/new_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials_reflected.nc'
    # model_number = 27
    # model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number

    # Model 28: naive mouse neurons
    # input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/naive_sig_129_neurons_samples.nc'
    # model_number = 28
    # model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    
    # Model 29 : naive mouse neurons 
    # input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/naive_sig_124_neurons_samples_stim_subset_train_test.nc'
    # model_number = 29
    # model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number

    # Model 30
    
    # Model 31: naive mouse neurons v2 (match for number of neurons in trained) with train test split
    input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/naive_sig_141_neurons_samples_stim_subset_train_test_w_labels.nc'
    model_number = 32
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = False  # normally false

    # Model 32: old model (trained mice, 140 neurons) without train test group (just to verify results look the same as before with slightly modified code)
    #input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/new_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials_v2.nc'
    #model_number = 32
    #model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    #split_using_trainTestGroup = False  # normally false

    # Model 33: trained 140 neurons with two PSTH samples per neuron
    input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/trained_sig_140_neurons_samples_w_train_test_labels_v2.nc'
    model_number = 33
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = False  # normally false
    
    
    # Model 34: trained 140 neurons with two PSTH samples per neuron
    input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/trained_sig_140_neurons_samples_w_train_test_labels_v2.nc'
    model_number = 34
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = True  # normally false

    # Model 35: naive mouse neurons v2 (match for number of neurons in trained) with train test split
    input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/naive_sig_141_neurons_samples_stim_subset_train_test_w_labels.nc'
    model_number = 35
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = True

    # Model 36: trained mouse neurons with single PSTH control
    input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/trained_sig_140_neurons_samples_w_train_test_labels_control.nc'
    model_number = 36
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = False

    # model 37 : no smoothing, also no splitting
    # input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/trained_sig_140_neurons_samples_w_train_test_labels_control.nc'
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/trained_sig_140_neurons_samples_w_train_test_labels_control.nc'
    model_number = 37
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = False
    shuffle_test_set_aud_vis_labels = False

    """
    # model 38: model 20 again (smoothing)
    input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/max_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials.nc'
    model_number = 38
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = False

    # model 39: model 20 with no smoothing
    input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/max_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials.nc'
    model_number = 39
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = False

    # Model 40: (not smoothed) naive mouse neurons with train test PSTH
    input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/naive_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels.nc'
    model_number = 40
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = True

    # Model 41: (not smoothed) same as model 40 but shuffle the test set auditory and visual labels randomly to act as a control
    input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/naive_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels.nc'
    model_number = 41
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = True
    shuffle_test_set_aud_vis_labels = True

    # Model 42: (not smoothed) naive mouse neurons with train test PSTH, random seed 2 for subsampling
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/naive_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_2.nc'
    model_number = 42
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = True

    # Model 43: (not smoothed) naive mouse neurons with train test PSTH, random seed 3 for subsampling
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/naive_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_3.nc'
    model_number = 43
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = True


    # Model 44: Trained mouse neurons again
    input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/max_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials.nc'
    model_number = 44
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = False
    shuffle_test_set_aud_vis_labels = False

    # Model 45: Trained mouse neurons with shuffling of activity matrix in test set
    input_activity_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/max_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials.nc'
    model_number = 45
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = False
    shuffle_test_set_aud_vis_labels = True


    # Model 46: naive mouse neurons with shuffling of activity matrix in test set
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/naive_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_2.nc'
    model_number = 46
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = True
    shuffle_test_set_aud_vis_labels = True

    # Model 47: naive mouse neurons spike sorted, random seed 1 for subsampling (no smoothing)
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/naive_sorted_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_1.nc'
    model_number = 47
    model_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f' % model_number
    split_using_trainTestGroup = True
    shuffle_test_set_aud_vis_labels = False

    # Model 48: naive mouse neurons spike sorted, random seed 2 for subsampling (no smoothing)
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/naive_sorted_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_2.nc'
    model_number = 48
    model_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f' % model_number
    split_using_trainTestGroup = True
    shuffle_test_set_aud_vis_labels = False

    # Model 49: same as model 47, but shuffle test set aud vis labels
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/naive_sorted_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_1.nc'
    model_number = 49
    model_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f' % model_number
    split_using_trainTestGroup = True
    shuffle_test_set_aud_vis_labels = True

    # Model 50: naive mouse neurons spike sorted, random seed 3 for subsampling (no smoothing)
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/naive_sorted_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_3.nc'
    model_number = 50
    model_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f' % model_number
    split_using_trainTestGroup = True
    shuffle_test_set_aud_vis_labels = False

    # Model 51: naive mouse neurons spike sorted, random seed 4 for subsampling (no smoothing)
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/naive_sorted_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_4.nc'
    model_number = 51
    model_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f' % model_number
    split_using_trainTestGroup = True
    shuffle_test_set_aud_vis_labels = False

    # Model 52: naive mouse neurons spike sorted, random seed 5 for subsampling (no smoothing)
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/naive_sorted_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_5.nc'
    model_number = 52
    model_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f' % model_number
    split_using_trainTestGroup = True
    shuffle_test_set_aud_vis_labels = False
    

    # Model 53: trained mouse neurons, 5 ms bins and no smoothing
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/trained_sig_140_neurons_5ms.nc'
    model_number = 53
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = False
    shuffle_test_set_aud_vis_labels = False

    # Model 54: trained mouse neurons (140), 2 ms bins and no smoothing
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/trained_sig_140_neurons_time_subset.nc'
    model_number = 54
    model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    split_using_trainTestGroup = False
    shuffle_test_set_aud_vis_labels = False

    # Model 55: same as model 48, but shuffle test set aud vis labels
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/naive_sorted_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_2.nc'
    model_number = 55
    model_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f' % model_number
    split_using_trainTestGroup = True
    shuffle_test_set_aud_vis_labels = True

    # Model 56: same as model 50, but shuffle test set aud vis labels
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/naive_sorted_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_3.nc'
    model_number = 56
    model_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f' % model_number
    split_using_trainTestGroup = True
    shuffle_test_set_aud_vis_labels = True
    """

    # TODO: run more trained model 140 neurons with shuffle


    # load neuron hemisphere info
    """
    neuron_hemisphere_info_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/new_window_permutation_test_passive_140_neuron_hem_info.pkl'
    neuron_hemisphere_info = pd.read_pickle(neuron_hemisphere_info_path)
    hemisphere_vec = (neuron_hemisphere_info['hemisphere'] == 'R').astype(float).values

    # repeat the hemisphere vec (double the neurons)
    hemisphere_vec = np.tile(hemisphere_vec, 2)
    """

    # Drift model 54
    """
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/trained_sig_140_neurons_time_subset.nc'
    model_number = 54
    # model_save_folder = '/media/timsit/T7/drift-model-%.f' % model_number
    model_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f/' % model_number
    split_using_trainTestGroup = False
    shuffle_test_set_aud_vis_labels = False
    """

    # Model 57: same as model 51, but shuffle test set aud vis labels
    input_activity_path = '/media/timsit/Ultra Touch/multispaceworld-rnn-samples/naive_sorted_sig_140_neurons_sub_samples_stim_subset_train_test_w_labels_random_seed_4.nc'
    model_number = 57
    model_save_folder = '/media/timsit/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f' % model_number
    split_using_trainTestGroup = True
    shuffle_test_set_aud_vis_labels = True



    fit_model = True

    if run_sampling:
        # TODO: this is currently 198 neurons in selected_cells_df?
        print('Sampling neurons...')
        # Get neurons to samples
        sig_cell_save_folder = '/media/timsit/Partition 1/data/interim/two-way-ANOVA/MOs-windowed-find-peak-permutation-test/'
        sig_cell_save_name = 'allStimCondPermutationTestNewParentSearchPostStimOnlyPassiveMaxDiff.pkl'
        # save_name = ''
        all_exp_test_results_df = pd.read_pickle(os.path.join(sig_cell_save_folder, sig_cell_save_name))
        include_audio_center = True

        if include_audio_center:
            print('Including audio center condition in the sampling')

        sig_percentile = 95
        min_aud_abs_fr = 2.5
        min_vis_abs_fr = 2.5
        min_aud_on_off_abs_fr = 2.5
        diff_max_time = 0.6
        # diff_max_time = 3
        aud_diff_min_time = 0.0
        vis_diff_min_time = 0.0

        aud_LR_selective_cells = all_exp_test_results_df.loc[
            (all_exp_test_results_df['audLRabsPercentile'] >= sig_percentile) &
            (all_exp_test_results_df['visLRabsPercentile'] < sig_percentile) &
            (all_exp_test_results_df['audLRabsDiff'] >= min_aud_abs_fr) &
            (all_exp_test_results_df['audLRmaxDiffTime'] >= aud_diff_min_time) &
            (all_exp_test_results_df['audLRmaxDiffTime'] < diff_max_time) &
            (all_exp_test_results_df['subjectRef'] != 1)
            ]

        vis_LR_selective_cells = all_exp_test_results_df.loc[
            (all_exp_test_results_df['audLRabsPercentile'] < sig_percentile) &
            (all_exp_test_results_df['visLRabsPercentile'] >= sig_percentile) &
            (all_exp_test_results_df['vis_lr_max_diff_time'] >= vis_diff_min_time) &
            (all_exp_test_results_df['vis_lr_max_diff_time'] < diff_max_time) &
            (all_exp_test_results_df['visLRabsDiff'] >= min_vis_abs_fr) &
            (all_exp_test_results_df['subjectRef'] != 1)
            ]

        aud_and_vis_LR_selective_cells = all_exp_test_results_df.loc[
            (all_exp_test_results_df['audLRabsPercentile'] >= sig_percentile) &
            (all_exp_test_results_df['visLRabsPercentile'] >= sig_percentile) &
            (all_exp_test_results_df['vis_lr_max_diff_time'] > 0) &
            (all_exp_test_results_df['audLRmaxDiffTime'] > 0) &
            (all_exp_test_results_df['subjectRef'] != 1)
            ]

        aud_on_off_selective_cells = all_exp_test_results_df.loc[
            (all_exp_test_results_df['audOnOffAbsDiffPercentile'] >= sig_percentile) &
            (all_exp_test_results_df['audOnoffMaxDiffTime'] > 0) &
            (all_exp_test_results_df['audOnOffAbsDiff'] > min_aud_on_off_abs_fr) &
            (all_exp_test_results_df['subjectRef'] != 1)
            ]

        print('Number of aud LR cells: %.f' % len(aud_LR_selective_cells))
        print('Number of vis LR cells: %.f' % len(vis_LR_selective_cells))
        print('Number of aud LR and vis LR cells: %.f' % len(aud_and_vis_LR_selective_cells))
        print('Number of aud on LR cells: %.f' % len(aud_on_off_selective_cells))

        aud_LR_selective_cells['modality'] = 'aud'
        vis_LR_selective_cells['modality'] = 'vis'
        aud_and_vis_LR_selective_cells['modality'] = 'audvis'

        selected_cells_df = pd.concat([aud_LR_selective_cells, vis_LR_selective_cells,
                                       aud_and_vis_LR_selective_cells])

        sample_neurons(stim_alignment_folder, selected_cells_df=selected_cells_df,
                       include_modality=True, include_audio_center=include_audio_center,
                       save_path=os.path.join(save_folder, save_name), num_samples_to_get=num_samples_to_get)

        print('Done!')

    if fit_model:
        print('Fitting model and saving it to %s' % model_save_folder)

        if not os.path.exists(model_save_folder):
            os.mkdir(model_save_folder)

        alignment_ds = xr.open_dataset(input_activity_path)

        # re-index cells
        alignment_ds = alignment_ds.assign_coords(Cell=np.arange(len(alignment_ds.Cell.values)))
        peri_event_time = alignment_ds.PeriEventTime.values

        if not os.path.exists(model_save_folder):
            os.mkdir(model_save_folder)

        num_shuffle = 1
        random_seed_list = [0, 1, 2, 3, 4]  # original: [0] or [0, 1, 2]
        num_cv_fold = 5
        split_per_stim_cond = True
        y_test_pred_da_list = list()
        smooth_spikes = False  # original: True
        only_train_on_highest_visual_level = False  # original: False
        remove_conflict_trials = False  # Original: False
        remove_coherent_trials = False  # Original: False
        balance_aud_only_and_vis_only_trials = True  # original: True
        evenly_balance_aud_only_and_vis_only_trials = False  # original: False
        scale_data = True  # Original: True
        hemisphere_weight_constraint = False  # only true for newer models

        # N = np.array([100, 200, 300, 400, 500])
        # c = 2 ** - (1 / N)
        c = 1

        # just set the update weight to 1
        N = [9999]
        update_weight_to_search_over = [c]

        test_size = 0.3
        start_time = -0.1
        end_time = 0.3
        stim_on_bin = 50
        step_size = 0.01
        model_type = 'drift'

        num_epochs = 300  # 500 without smoothing, 300 with smoothing
        loss_update_interval = 5  # originally 5

        for update_N, update_weight in zip(N, update_weight_to_search_over):

            for random_seed in random_seed_list:

                # for cv_fold_idx in np.arange(num_cv_fold):
                for cv_fold_idx in [0, 1, 2, 3, 4]:  # orignally: [0, 1, 2, 3, 4]

                    input_tensor_dev, input_tensor_test, Y_dev, Y_test, \
                    pre_preprocessed_alignment_ds_dev, pre_preprocessed_alignment_ds_test = \
                        jax_dmodel.process_and_train_test_split(
                            alignment_ds=alignment_ds, test_size=test_size,
                            smooth_spikes=smooth_spikes, scale_data=scale_data, smooth_multiplier=5,
                            start_time=start_time, end_time=end_time, random_seed=random_seed,
                            model_type=model_type,
                            only_train_on_highest_visual_level=only_train_on_highest_visual_level,
                            remove_conflict_trials=remove_conflict_trials, remove_coherent_trials=remove_coherent_trials,
                            balance_aud_only_and_vis_only_trials=balance_aud_only_and_vis_only_trials,
                            evenly_balance_aud_only_and_vis_only_trials=evenly_balance_aud_only_and_vis_only_trials,
                            num_cv_fold=num_cv_fold, split_per_stim_cond=split_per_stim_cond,
                            cv_fold_idx=cv_fold_idx, split_using_trainTestGroup=split_using_trainTestGroup,
                            shuffle_test_set_aud_vis_labels=shuffle_test_set_aud_vis_labels)

                    num_neuron = np.shape(input_tensor_dev)[1]
                    neuron_weights = np.random.normal(0, 0.3, size=num_neuron)
                    neuron_bias = np.random.normal(0, 0.3, size=num_neuron)

                    # drift_rate = np.random.uniform(low=1, high=500)
                    model_params = np.concatenate([neuron_bias.flatten(), neuron_weights.flatten()])

                    # 2021-10-12 Back to original model
                    decision_model = functools.partial(
                         jax_dmodel.accumulate_after_stim_w_neuron_bias_fixed_drift_model,
                         update_weight=update_weight, stim_on_bin=stim_on_bin)

                    # 2021-04-29: Trying out hemisphere model
                    #  = functools.partial(
                    #           jax_dmodel.accumulate_after_stim_w_neuron_bias_fixed_drift_model_w_hem,
                    #          update_weight=update_weight, stim_on_bin=stim_on_bin,
                    #           hemisphere_vec=hemisphere_vec)

                    loss_function = functools.partial(
                        jax_dmodel.hinge_and_mse_loss_function,
                        decision_model=decision_model, stim_on_bin=stim_on_bin
                    )

                    update_function = functools.partial(jax_dmodel.simple_update, loss_function=loss_function,
                                                        step_size=step_size)
                    batch_size = None
                    optimizer_func = optimizers.adam(step_size=0.01, b1=0.9, b2=0.999, eps=1e-8)

                    # pdb.set_trace()
                    param_history, epoch_list, dev_loss_list, test_loss_list = jax_dmodel.fit_model(
                        loss_function=loss_function, update_function=update_function, params=model_params,
                        x_dev=input_tensor_dev, y_dev=Y_dev,
                        x_test=input_tensor_test, y_test=Y_test,
                        num_epochs=num_epochs, loss_update_interval=loss_update_interval,
                        optimizer_func=optimizer_func, batch_size=batch_size
                    )

                    # find epoch with minimum test loss
                    # TODO: try by minimising just dev loss
                    min_test_loss_idx = np.where(test_loss_list == np.nanmin(test_loss_list))[0][0]

                    if np.all(np.isnan(test_loss_list)):
                        print('Warning: all NaNs in test loss')

                    model_output_dev = decision_model(input_tensor_dev, param_history[min_test_loss_idx])
                    model_output_test = decision_model(input_tensor_test, param_history[min_test_loss_idx])

                    y_test_pred_da = xr.DataArray(
                        model_output_test, dims=['Trial', 'Time'],
                        coords={'Trial': pre_preprocessed_alignment_ds_test.Trial,
                                'Time': pre_preprocessed_alignment_ds_test.Time}
                    )

                    # Also save training set prediction

                    y_dev_pred_da = xr.DataArray(
                        model_output_dev, dims=['Trial', 'Time'],
                        coords={'Trial': pre_preprocessed_alignment_ds_dev.Trial,
                                'Time': pre_preprocessed_alignment_ds_dev.Time, }
                    )

                    # Save everything into one dictionary
                    model_result = dict()

                    model_result['name'] = 'drift_model_all_cond'
                    model_result['random_seed'] = random_seed
                    model_result['y_test_pred_da'] = y_test_pred_da
                    model_result['y_dev_pred_da'] = y_dev_pred_da
                    # model_result['input_tensor_dev'] = input_tensor_dev
                    # model_result['input_tensor_test'] = input_tensor_test
                    model_result['pre_preprocessed_alignment_ds_dev'] = pre_preprocessed_alignment_ds_dev
                    model_result['pre_preprocessed_alignment_ds_test'] = pre_preprocessed_alignment_ds_test
                    model_result['param_history'] = param_history
                    model_result['dev_loss'] = dev_loss_list
                    model_result['test_loss'] = test_loss_list
                    model_result['epoch'] = epoch_list
                    model_result['cv_fold_idx'] = cv_fold_idx
                    model_result['update_N'] = update_N
                    model_result['peri_event_time'] = peri_event_time
                    model_result['model_type'] = 'accumulate_after_stim_w_neuron_bias_fixed_drift_model'
                    model_result['balance_aud_only_and_vis_only_trials'] = balance_aud_only_and_vis_only_trials
                    model_result['smooth_spikes'] = smooth_spikes
                    model_result['scale_data'] = scale_data
                    model_result['remove_conflict_trials'] = remove_conflict_trials
                    model_result['remove_coherent_trials'] = remove_coherent_trials
                    model_result['only_train_on_highest_visual_level'] = only_train_on_highest_visual_level
                    model_result['evenly_balance_aud_only_and_vis_only_trials'] = evenly_balance_aud_only_and_vis_only_trials
                    model_result['split_per_stim_cond'] = split_per_stim_cond
                    model_result['input_activity_path'] = input_activity_path
                    model_result['hemisphere_weight_constraint'] = hemisphere_weight_constraint
                    update_weight_string = str(update_N).replace('.', 'p')

                    # model_save_name = 'driftParam_N_%s_shuffle_%.f_cv%.f.pkl' % (update_weight_string, random_seed, cv_fold_idx)
                    model_save_name = 'driftParam_c_1_shuffle_%.f_cv%.f.pkl' % (random_seed, cv_fold_idx)
                    with open(os.path.join(model_save_folder, model_save_name), 'wb') as handle:
                        pkl.dump(model_result, handle)


if __name__ == '__main__':
    # jax.config.update('jax_platform_name', 'cpu')  # uncomment if using cpu only JAX
    main()

