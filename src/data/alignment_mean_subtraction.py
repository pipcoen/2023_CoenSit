import pdb

import xarray as xr
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict


# import src.data.analyse_spikes as anaspikes
import sklearn.metrics as sklmetrics
import sklearn.linear_model as sklinear



def align_single_event(binned_spike_ds, behave_df, event_name, time_before_alignment=0.7, time_after_alignment=0.7,
                       include_trial_coord=False, include_response=False):
    """

    Parameters
    ----------
    binned_spike_ds
    behave_df
    event_name
    time_before_alignment
    time_after_alignment
    include_trial_coord
    include_response : (bool)
        whether to include response made

    Returns
    -------

    """

    event_time = behave_df[event_name]
    window_start = event_time - time_before_alignment
    window_end = event_time + time_after_alignment

    event_bins = list(zip(window_start,
                          window_end))

    event_interval_bins = pd.IntervalIndex.from_tuples(event_bins, closed='both')
    aligned_xarray_tuple = binned_spike_ds.groupby_bins('Time', event_interval_bins)
    aligned_xarray_list = [i[1] for i in list(aligned_xarray_tuple)]

    new_aligned_xarray_list = list()
    for n_xarray, aligned_xarray in enumerate(aligned_xarray_list):
        peri_event_time = aligned_xarray['Time'].values - behave_df[event_name].iloc[n_xarray]
        overall_time = aligned_xarray['Time'].values
        aligned_xarray = aligned_xarray.assign({'BinTime': ('Time', overall_time)})
        aligned_xarray = aligned_xarray.assign({'PeriEventTime': ('Time', peri_event_time)})
        aligned_xarray = aligned_xarray.assign_coords({'Time': np.arange(len(overall_time))})
        new_aligned_xarray_list.append(aligned_xarray)

    # Concatenate alignment ds
    aligned_ds = xr.concat(new_aligned_xarray_list, dim='Trial')

    # Add Trial dimension information
    aligned_ds = aligned_ds.assign({'timeToFirstMove': ('Trial', behave_df['firstTimeToWheelMove'].values)})
    aligned_ds = aligned_ds.assign({'visDiff': ('Trial', behave_df['visDiff'].values)})
    aligned_ds = aligned_ds.assign({'audDiff': ('Trial', behave_df['audDiff'].values)})

    if include_response:
        aligned_ds = aligned_ds.assign({'responseMade': ('Trial', behave_df['responseMade'].values)})

    if include_trial_coord:
        aligned_ds = aligned_ds.assign_coords({'Trial': ('Trial', np.arange(len(behave_df)))})

    return aligned_ds


def align_stim_and_movement(binned_spike_ds, behave_df, time_before_alignment=0.7, time_after_alignment=0.7):
    """
    Align binned spikes to some time before stimulus and some time
    :param binned_spike_ds:
    :param behave_df:
    :param time_before_alignment:
    :param time_after_alignment:
    :return:
    """

    stim_on_time_rounded = np.round(behave_df['stimOnTime'], 3)
    first_move_times_rounded = np.round(behave_df['firstMoveTimes'], 3)
    before_stim_window = stim_on_time_rounded - time_before_alignment
    after_stim_window = first_move_times_rounded + time_after_alignment

    event_bins = list(zip(before_stim_window,
                          after_stim_window))

    event_interval_bins = pd.IntervalIndex.from_tuples(event_bins, closed='both')
    aligned_xarray_tuple = binned_spike_ds.groupby_bins('Time', event_interval_bins)
    aligned_xarray_list = [i[1] for i in list(aligned_xarray_tuple)]

    new_aligned_xarray_list = list()
    for n_xarray, aligned_xarray in enumerate(aligned_xarray_list):
        peri_stim_time = aligned_xarray['Time'].values - behave_df['stimOnTime'].iloc[n_xarray]
        peri_movement_time = aligned_xarray['Time'].values - behave_df['firstMoveTimes'].iloc[n_xarray]
        overall_time = aligned_xarray['Time'].values
        aligned_xarray = aligned_xarray.assign({'PeriStimTime': ('Time', peri_stim_time)})
        aligned_xarray = aligned_xarray.assign({'BinTime': ('Time', overall_time)})
        aligned_xarray = aligned_xarray.assign({'PeriMovementTime': ('Time', peri_movement_time)})
        aligned_xarray = aligned_xarray.assign_coords({'Time': np.arange(len(overall_time))})

        new_aligned_xarray_list.append(aligned_xarray)

    # Concatenate alignment ds
    aligned_ds = xr.concat(new_aligned_xarray_list, dim='Trial')

    # Add Trial dimension information
    aligned_ds = aligned_ds.assign({'timeToFirstMove': ('Trial', behave_df['firstTimeToWheelMove'].values)})
    aligned_ds = aligned_ds.assign({'visDiff': ('Trial', behave_df['visDiff'].values)})
    aligned_ds = aligned_ds.assign({'audDiff': ('Trial', behave_df['audDiff'].values)})

    return aligned_ds


def aligned_stim_mean_subtraction(aligned_ds, pre_stim_time=0.1, post_stim_time=0.09,
                                  subset_stim_cond=None):
    """
    This relies on Xarray clever way of aligning xarray to make sure subtraction is broadcasted to the
    corresponding coordinates of the two xarrays.
    :param aligned_ds:
    :param pre_stim_time:
    :param post_stim_time:
    :param subset_stim_cond:
    :return:
    """

    # make a copy of the original alignment ds to be used for subtraction
    mean_subtracted_stim_time_subset = aligned_ds.copy()

    if subset_stim_cond is None:
        aligned_ds_stim_time_subset = aligned_ds.where(
            (aligned_ds['PeriStimTime'] >= -pre_stim_time) &
            (aligned_ds['PeriStimTime'] < post_stim_time), drop=True
        )

        aligned_ds_stim_time_subset_mean = aligned_ds_stim_time_subset['SpikeRate'].mean('Trial')

        aligned_ds_stim_time_subset_mean, _ = xr.align(aligned_ds_stim_time_subset_mean,
                                                       aligned_ds['SpikeRate'],
                                                       join='right', fill_value=0)

        mean_subtracted_stim_time_subset['SpikeRate'] = mean_subtracted_stim_time_subset['SpikeRate'] - \
                                                        aligned_ds_stim_time_subset_mean
    elif subset_stim_cond == 'direction':
        # subset based on stimulus direction (group all the visual contrast levels)
        aud_cond_list = ['left', 'right', 'center', 'off']
        vis_cond_list = ['left', 'right', 'off']

        for aud_cond, vis_cond in itertools.product(aud_cond_list, vis_cond_list):
            stim_cond_ds = anaspikes.get_target_condition_ds(multi_condition_ds=aligned_ds,
                                                             aud=aud_cond, vis=vis_cond)
            aligned_ds_stim_time_subset = stim_cond_ds.where(
            (stim_cond_ds['PeriStimTime'] >= -pre_stim_time) &
            (stim_cond_ds['PeriStimTime'] < post_stim_time), drop=True)

            aligned_ds_stim_time_subset_mean = aligned_ds_stim_time_subset['SpikeRate'].mean('Trial')

            aligned_ds_stim_time_subset_mean, _ = xr.align(aligned_ds_stim_time_subset_mean,
                                                           aligned_ds['SpikeRate'],
                                                           join='right', fill_value=0)

            mean_subtracted_stim_time_subset['SpikeRate'] = mean_subtracted_stim_time_subset['SpikeRate'] - \
                                                            aligned_ds_stim_time_subset_mean

    return mean_subtracted_stim_time_subset


def realign_ds_to_movement(stim_and_movement_aligned_ds, pre_movement_time=0.7,
                           post_movement_time=0.7):
    """
    Realign data to be movement aligned
    :param stim_and_movement_aligned_ds:
    :param pre_movement_time:
    :param post_movement_time:
    :return:
    """

    movement_aligned_ds = stim_and_movement_aligned_ds.where(
            (stim_and_movement_aligned_ds['PeriMovementTime'] >= -pre_movement_time) &
            (stim_and_movement_aligned_ds['PeriMovementTime'] <= post_movement_time), drop=True
        )

    # not sure if this is a bug or what
    movement_aligned_ds['timeToFirstMove'] = stim_and_movement_aligned_ds['timeToFirstMove']

    # Re-align everything (since the 'Time' (index) dimension will be different for each trial
    # depending on the movement time)
    realigned_ds_list = list()
    for trial in np.unique(movement_aligned_ds['Trial']):
        trials_ds = movement_aligned_ds.isel(Trial=trial).dropna('Time')
        trials_ds = trials_ds.assign_coords({'Time': np.arange(len(trials_ds['Time']))})
        realigned_ds_list.append(trials_ds)

    movement_aligned_ds = xr.concat(realigned_ds_list, dim='Trial')

    return movement_aligned_ds


def add_passive_psth_to_predict_active_psth(choice_stim_aligned_ds, passive_aligned_ds, subset_time_window=[-0.1, 1.0],
                                            stim_cond_list=['arvr', 'arvl', 'alvr', 'alvl'],
                                            train_test_split=None, test_size=0.5, include_components=False,
                                            multiple_cells='infer', include_num_trial=True, include_mean_stim_fr=False,
                                            zero_passive_response=False, error_metric=['mse'],
                                            fit_multiple_movement_template=False, realign_movement=False,
                                            vis_contrast_levels=np.array([0.8]), coherent_vis_contrast_levels=np.array([0.4, 0.8]),
                                            conflict_vis_contrast_levels=np.array([0.8]), include_single_trials=False,
                                            add_multimodal_conditions=False, movement_realignment_window=[-0.1, 0.4],
                                            peri_movement_time_to_insert_movement=0, bin_width=0.002,
                                            window_to_get_baseline_for_subtraction=None, random_seed=None, verbose=False):
    """

    Parameters
    ----------
    choice_stim_aligned_ds : (xarray dataset)
        xarray dataset
        either aligned to stimulus or aligned to movement onset
    passive_aligned_ds : (xarray dataset)
        xarray dataset containing dimensions (Cell, Trial, Time)
        Time needs to have coordinates 'PeriEventTime'
        time is aligned to stimulus onset
    subset_time_window : (list)
        list containing 2 elements
        the first element is the time in seconds of the start of the window relative to the stimulus to
        the second element is the time in seconds of the end of the window relative to the stimulus
    stim_cond_list : (list)
        list of stimulus conditions to loop over
    train_test_split : (str)
        if string, then does some type of train/test set split
        if None, then uses all of the data
    test_size : (float)
        test size in proportion, must be greater than 0 and less than 1
    include_components : (bool)
        whether to include the extracted movement only component
        and passive PSTH.
    include_num_trial : (bool)
        whether to include the number of trials for each stimulus condition. (to access whether there are enough
        trials for a non-noisy estimate)
    include_mean_stim_fr : (bool)
        whether to include the mean stimulus response firing rate (for use later to subset neurons with
        no response in passive condition, in which case the additive model works trivially)
    fit_multiple_movement_template : (bool)
        whether to fit a separate movement PSTH for each stimulus condition
        requires the chocie_stim_aligned_ds to be a dictionary
    realign_movement : (bool)
        whether to use the movement component in terms of a movement-aligned activity
        the prediction then becomes dependent on the reaction time for each trial.
        this then assumes that the provided xarray dataset choice_stim_aligned_ds is movement aligned
    include_single_trials : (bool)
        whether to include single trial train, test and prediction (eg. to be used for
        statistical test)
    verbose : (bool)
        whether to print out silent errors / details about procedure being performed.
    add_multimodal_conditions : (bool)
        whether to add unimodal conditions to make multimodal conditions
        eg. ALVR will be made by adding audio left only and visual right only
    peri_movement_time_to_insert_movement : (float)
        time relative to movement to insert movement component (in seconds)
        normally 0: ie. add the movement PSTH starting from the movement
        but may be set to before movement to capture preparatory movement activity
    window_to_get_baseline_for_subtraction : (list or None)
        list containing two element
        if None, then baseline subtraction is not performed
    bin_width : (float)
        window to bin neural activity (in seconds)
    random_seed : (int)
        random seed to use for random train/test split
    Returns
    -------

    """
    if multiple_cells == 'infer':
        if len(choice_stim_aligned_ds.Cell) > 1:
            multiple_cells = True
        else:
            multiple_cells = False
    if multiple_cells:
        num_cells = len(choice_stim_aligned_ds.Cell)
    else:
        num_cells = 1

    train_size = 1 - test_size
    choice_stim_aligned_ds_time_subset = choice_stim_aligned_ds.where(
        ((choice_stim_aligned_ds['PeriEventTime'] >= subset_time_window[0]) &
         (choice_stim_aligned_ds['PeriEventTime'] <= subset_time_window[1])),
      drop=True)

    prediction_and_actual_psth = defaultdict(dict)
    if train_test_split == 'per-stim-cond':
        stim_cond_and_choice_ds_train_list = list()
        stim_cond_and_choice_ds_test_list = list()
        choice_mean_train_dict = dict()
        for stim_cond in stim_cond_list:
            if stim_cond == 'arvr':
                passive_vis_cond = coherent_vis_contrast_levels
                passive_aud_cond = [60]
            elif stim_cond == 'arvl':
                passive_vis_cond = conflict_vis_contrast_levels * -1
                passive_aud_cond = [60]
            elif stim_cond == 'alvr':
                passive_vis_cond = conflict_vis_contrast_levels
                passive_aud_cond = [-60]
            elif stim_cond == 'alvl':
                passive_vis_cond = coherent_vis_contrast_levels * -1
                passive_aud_cond = [-60]
            elif stim_cond == 'arv0':
                passive_vis_cond = [0]
                passive_aud_cond = [60]
            elif stim_cond == 'alv0':
                passive_vis_cond = [0]
                passive_aud_cond = [-60]
            elif stim_cond == 'acv0':
                passive_vis_cond = [0]
                passive_aud_cond = [0]
            elif stim_cond == 'acvl':
                passive_vis_cond = vis_contrast_levels * -1
                passive_aud_cond = [0]
            elif stim_cond == 'acvr':
                passive_vis_cond = vis_contrast_levels
                passive_aud_cond = [0]

            stim_cond_and_choice_ds = choice_stim_aligned_ds_time_subset.where(
                (choice_stim_aligned_ds_time_subset['visDiff'].isin(passive_vis_cond) &
                 choice_stim_aligned_ds_time_subset['audDiff'].isin(passive_aud_cond)),
                drop=True)

            stim_cond_trials = stim_cond_and_choice_ds.Trial
            num_stim_cond_trial = len(stim_cond_trials)
            test_num_trial = int(train_size * num_stim_cond_trial)
            if random_seed is not None:
                np.random.seed(random_seed)
            train_trials = np.random.choice(stim_cond_trials, test_num_trial, replace=False)
            test_trials = stim_cond_trials[~stim_cond_trials.isin(train_trials)]
            try:
                stim_cond_and_choice_ds_train = stim_cond_and_choice_ds.sel(Trial=train_trials)
            except:
                pdb.set_trace()

            stim_cond_and_choice_ds_test = stim_cond_and_choice_ds.sel(Trial=test_trials)
            stim_cond_and_choice_ds_train_list.append(stim_cond_and_choice_ds_train)
            stim_cond_and_choice_ds_test_list.append(stim_cond_and_choice_ds_test)
            if fit_multiple_movement_template:
                if realign_movement:
                    try:
                        train_stim_cond_and_choice_ds = stim_cond_and_choice_ds.sel(Trial=train_trials)
                        if len(train_stim_cond_and_choice_ds.Trial) != 0:
                            train_stim_cond_and_choice_ds_movement_realigned = stim_align_to_move_align(train_stim_cond_and_choice_ds,
                              activity_name=['smoothed_fr', 'stimSubtractedActivity'], multiple_cells=multiple_cells,
                              time_start=(movement_realignment_window[0]),
                              time_end=(movement_realignment_window[1]),
                              bin_width=bin_width)
                            choice_mean_train_dict[stim_cond] = train_stim_cond_and_choice_ds_movement_realigned['stimSubtractedActivity'].mean('Trial')
                        else:
                            choice_mean_train_dict[stim_cond] = train_stim_cond_and_choice_ds['stimSubtractedActivity'].mean('Trial')
                    except:
                        pdb.set_trace()

                else:
                    choice_mean_train_dict[stim_cond] = stim_cond_and_choice_ds.sel(Trial=train_trials)['stimSubtractedActivity'].mean('Trial')
            if include_num_trial:
                prediction_and_actual_psth[stim_cond]['num_trial'] = num_stim_cond_trial

        all_cond_choice_ds_train = xr.concat(stim_cond_and_choice_ds_train_list, dim='Trial')
        all_cond_choice_ds_test = xr.concat(stim_cond_and_choice_ds_test_list, dim='Trial')
        if realign_movement:
            if len(all_cond_choice_ds_train.Trial) > 0:
                choice_mean_train_ds = stim_align_to_move_align(all_cond_choice_ds_train, activity_name=[
                 'smoothed_fr', 'stimSubtractedActivity'],
                  multiple_cells=multiple_cells,
                  time_start=(movement_realignment_window[0]),
                  time_end=(movement_realignment_window[1]),
                  bin_width=bin_width)
                choice_mean_train = choice_mean_train_ds['stimSubtractedActivity'].mean('Trial')
                choice_mean_test_ds = stim_align_to_move_align(all_cond_choice_ds_test, activity_name=[
                 'smoothed_fr', 'stimSubtractedActivity'],
                  multiple_cells=multiple_cells,
                  time_start=(movement_realignment_window[0]),
                  time_end=(movement_realignment_window[1]),
                  bin_width=bin_width)
                choice_mean_test = choice_mean_test_ds['stimSubtractedActivity'].mean('Trial')
            else:
                if verbose:
                    print('No choice condition, realignment failed.')
                choice_mean_train = all_cond_choice_ds_train['stimSubtractedActivity'].mean('Trial')
        else:
            choice_mean_train = all_cond_choice_ds_train['stimSubtractedActivity'].mean('Trial')
            choice_mean_train_all = all_cond_choice_ds_train['stimSubtractedActivity']
            choice_mean_test_ds = stim_align_to_move_align(all_cond_choice_ds_test, activity_name=[
             'smoothed_fr', 'stimSubtractedActivity'],
              multiple_cells=multiple_cells,
              time_start=(movement_realignment_window[0]),
              time_end=(movement_realignment_window[1]),
              bin_width=bin_width)
            choice_mean_test_all = choice_mean_test_ds['stimSubtractedActivity']
    else:
        if realign_movement:
            choice_mean = stim_align_to_move_align(choice_stim_aligned_ds_time_subset, activity_name=[
             'smoothed_fr', 'stimSubtractedActivity'],
              multiple_cells=multiple_cells,
              time_start=(movement_realignment_window[0]),
              time_end=(movement_realignment_window[1]),
              bin_width=bin_width)['stimSubtractedActivity'].mean('Trial')
        else:
            choice_mean = choice_stim_aligned_ds_time_subset['stimSubtractedActivity'].mean('Trial')

    for stim_cond in stim_cond_list:
        if stim_cond == 'arvr':
            passive_vis_cond = coherent_vis_contrast_levels
            passive_aud_cond = [60]
        else:
            if stim_cond == 'arvl':
                passive_vis_cond = conflict_vis_contrast_levels * -1
                passive_aud_cond = [60]
            elif stim_cond == 'alvr':
                passive_vis_cond = conflict_vis_contrast_levels
                passive_aud_cond = [-60]
            elif stim_cond == 'alvl':
                passive_vis_cond = coherent_vis_contrast_levels * -1
                passive_aud_cond = [-60]
            elif stim_cond == 'arv0':
                passive_vis_cond = [0]
                passive_aud_cond = [60]
            elif stim_cond == 'alv0':
                passive_vis_cond = [0]
                passive_aud_cond = [-60]
            elif stim_cond == 'acv0':
                passive_vis_cond = [0]
                passive_aud_cond = [0]
            elif stim_cond == 'acvl':
                passive_vis_cond = vis_contrast_levels * -1
                passive_aud_cond = [0]
            elif stim_cond == 'acvr':
                passive_vis_cond = vis_contrast_levels
                passive_aud_cond = [0]

        if train_test_split == 'per-stim-cond':

            stim_cond_and_choice_ds_train = all_cond_choice_ds_train.where(
                (all_cond_choice_ds_train['visDiff'].isin(passive_vis_cond) &
                 all_cond_choice_ds_train['audDiff'].isin(passive_aud_cond)),
              drop=True).mean('Trial')['smoothed_fr']

            stim_cond_and_choice_ds_train_single_trials = all_cond_choice_ds_train.where(
                (all_cond_choice_ds_train['visDiff'].isin(passive_vis_cond) &
                 all_cond_choice_ds_train['audDiff'].isin(passive_aud_cond)),
              drop=True)['smoothed_fr']

            stim_cond_and_choice_ds_test = all_cond_choice_ds_test.where(
                (all_cond_choice_ds_test['visDiff'].isin(passive_vis_cond) &
                 all_cond_choice_ds_test['audDiff'].isin(passive_aud_cond)),
              drop=True).mean('Trial')['smoothed_fr']

            stim_cond_and_choice_ds_test_single_trials = all_cond_choice_ds_test.where(
                (all_cond_choice_ds_test['visDiff'].isin(passive_vis_cond) &
                 all_cond_choice_ds_test['audDiff'].isin(passive_aud_cond)),
              drop=True)['smoothed_fr']

            passive_aligned_ds_time_subset = passive_aligned_ds.where(
                ((passive_aligned_ds['PeriEventTime'] >= subset_time_window[0]) &
                 (passive_aligned_ds['PeriEventTime'] <= subset_time_window[1])),
              drop=True)

            if add_multimodal_conditions:
                multimodal_adding_conditions = [
                 'alvl','arvr','alvr','arvl',
                 'acvl','acvr']
            else:
                multimodal_adding_conditions = [
                 'acvl', 'acvr']

            if stim_cond not in multimodal_adding_conditions:
                stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                    (passive_aligned_ds_time_subset['visDiff'].isin(passive_vis_cond) &
                     passive_aligned_ds_time_subset['audDiff'].isin(passive_aud_cond)),
                  drop=True).mean('Trial')

                if 'Trial' in passive_aligned_ds_time_subset.PeriEventTime.dims:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime.isel(Trial=0)
                else:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime

                stim_cond_passive = stim_cond_passive_stim_aligned_ds['smoothed_fr']

            elif stim_cond == 'acvl':

                aud_center_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                    (passive_aligned_ds_time_subset['visDiff'].isin([0]) &
                     passive_aligned_ds_time_subset['audDiff'].isin([0])),
                      drop=True).mean('Trial')
                vis_left_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                    (passive_aligned_ds_time_subset['visDiff'].isin(vis_contrast_levels * -1) &
                     passive_aligned_ds_time_subset['audDiff'].isin([np.inf])),
                  drop=True).mean('Trial')

                stim_cond_passive_stim_aligned_ds = aud_center_stim_cond_passive_stim_aligned_ds + vis_left_stim_cond_passive_stim_aligned_ds
                if 'Trial' in passive_aligned_ds_time_subset.PeriEventTime.dims:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime.isel(Trial=0)
                else:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime
                stim_cond_passive = stim_cond_passive_stim_aligned_ds['smoothed_fr']
            elif stim_cond == 'acvr':

                aud_center_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                    (passive_aligned_ds_time_subset['visDiff'].isin([0]) &
                     passive_aligned_ds_time_subset['audDiff'].isin([0])),
                  drop=True).mean('Trial')

                vis_right_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                    (passive_aligned_ds_time_subset['visDiff'].isin(vis_contrast_levels) &
                     passive_aligned_ds_time_subset['audDiff'].isin([np.inf])),
                  drop=True).mean('Trial')
                stim_cond_passive_stim_aligned_ds = aud_center_stim_cond_passive_stim_aligned_ds + vis_right_stim_cond_passive_stim_aligned_ds

                if 'Trial' in passive_aligned_ds_time_subset.PeriEventTime.dims:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime.isel(Trial=0)
                else:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime
                stim_cond_passive = stim_cond_passive_stim_aligned_ds['smoothed_fr']

            elif  stim_cond == 'arvr':

                aud_right_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                    (passive_aligned_ds_time_subset['visDiff'].isin([0]) &
                     passive_aligned_ds_time_subset['audDiff'].isin([60])),
                  drop=True).mean('Trial')

                vis_right_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                    (passive_aligned_ds_time_subset['visDiff'].isin(coherent_vis_contrast_levels) &
                     passive_aligned_ds_time_subset['audDiff'].isin([np.inf])),
                  drop=True).mean('Trial')
                stim_cond_passive_stim_aligned_ds = aud_right_stim_cond_passive_stim_aligned_ds + vis_right_stim_cond_passive_stim_aligned_ds
                if 'Trial' in passive_aligned_ds_time_subset.PeriEventTime.dims:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime.isel(Trial=0)
                else:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime
                stim_cond_passive = stim_cond_passive_stim_aligned_ds['smoothed_fr']
            elif stim_cond == 'alvl':
                aud_left_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                    (passive_aligned_ds_time_subset['visDiff'].isin([0]) &
                     passive_aligned_ds_time_subset['audDiff'].isin([-60])),
                  drop=True).mean('Trial')

                vis_left_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                    (passive_aligned_ds_time_subset['visDiff'].isin(coherent_vis_contrast_levels * -1) &
                     passive_aligned_ds_time_subset['audDiff'].isin([np.inf])),
                  drop=True).mean('Trial')

                stim_cond_passive_stim_aligned_ds = aud_left_stim_cond_passive_stim_aligned_ds + vis_left_stim_cond_passive_stim_aligned_ds

                if 'Trial' in passive_aligned_ds_time_subset.PeriEventTime.dims:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime.isel(Trial=0)
                else:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime
                stim_cond_passive = stim_cond_passive_stim_aligned_ds['smoothed_fr']
            elif stim_cond == 'alvr':

                aud_left_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                    (passive_aligned_ds_time_subset['visDiff'].isin([0]) &
                     passive_aligned_ds_time_subset['audDiff'].isin([-60])),
                  drop=True).mean('Trial')

                vis_right_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                    (passive_aligned_ds_time_subset['visDiff'].isin(coherent_vis_contrast_levels) &
                     passive_aligned_ds_time_subset['audDiff'].isin([np.inf])),
                  drop=True).mean('Trial')

                stim_cond_passive_stim_aligned_ds = aud_left_stim_cond_passive_stim_aligned_ds + vis_right_stim_cond_passive_stim_aligned_ds

                if 'Trial' in passive_aligned_ds_time_subset.PeriEventTime.dims:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime.isel(Trial=0)
                else:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime
                stim_cond_passive = stim_cond_passive_stim_aligned_ds['smoothed_fr']

            elif stim_cond == 'arvl':
                aud_right_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                    (passive_aligned_ds_time_subset['visDiff'].isin([0]) &
                     passive_aligned_ds_time_subset['audDiff'].isin([-60])),
                  drop=True).mean('Trial')

                vis_left_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                    (passive_aligned_ds_time_subset['visDiff'].isin(coherent_vis_contrast_levels) &
                     passive_aligned_ds_time_subset['audDiff'].isin([np.inf])),
                  drop=True).mean('Trial')

                stim_cond_passive_stim_aligned_ds = aud_right_stim_cond_passive_stim_aligned_ds + vis_left_stim_cond_passive_stim_aligned_ds
                if 'Trial' in passive_aligned_ds_time_subset.PeriEventTime.dims:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime.isel(Trial=0)
                else:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime
                stim_cond_passive = stim_cond_passive_stim_aligned_ds['smoothed_fr']

            if fit_multiple_movement_template:
                choice_mean_train_stim_cond = choice_mean_train_dict[stim_cond]
                if len(choice_mean_train_stim_cond.Time) != 0:
                    if zero_passive_response:
                        choice_plus_stim_passive = choice_mean_train_stim_cond.values
                    else:
                        if realign_movement:
                            stim_cond_test_ds = all_cond_choice_ds_test.where(
                                (all_cond_choice_ds_test['visDiff'].isin(passive_vis_cond) &
                                 all_cond_choice_ds_test['audDiff'].isin(passive_aud_cond)),
                              drop=True)

                            if (len(stim_cond_test_ds.Time) != 0) & (len(choice_mean_train_stim_cond.Time) != 0):
                                trial_rts = stim_cond_test_ds.isel(Time=0)['choiceInitTimeRelStim']
                                choice_mean_train_post_movement = choice_mean_train_stim_cond.sel(Time=(slice(peri_movement_time_to_insert_movement, None)))
                                num_trial = len(trial_rts)
                                if multiple_cells:
                                    total_stim_bins = np.shape(stim_cond_passive)[1]
                                else:
                                    total_stim_bins = len(stim_cond_passive)
                                choice_plus_stim_passive_all = np.zeros((num_trial, num_cells, total_stim_bins))
                                for n_trial, trial_rt in enumerate(trial_rts):
                                    time_idx_to_insert_movement_component = np.argmin(np.abs(trial_rt + peri_movement_time_to_insert_movement - passive_stim_align_peri_event_time)).values
                                    time_idx_to_insert_movement_component = int(time_idx_to_insert_movement_component)
                                    trial_choice_plus_stim_passive = stim_cond_passive.values
                                    num_stim_time_bins = np.shape(trial_choice_plus_stim_passive[:, time_idx_to_insert_movement_component:])[1]
                                    choice_mean_train_post_movement = choice_mean_train_post_movement.transpose('Time', 'Cell')
                                    num_movement_bins = len(choice_mean_train_post_movement.Time)
                                    idx_end = time_idx_to_insert_movement_component + num_movement_bins
                                    trial_choice_plus_stim_passive[:, time_idx_to_insert_movement_component:idx_end] = trial_choice_plus_stim_passive[:, time_idx_to_insert_movement_component:idx_end] + choice_mean_train_post_movement.values.T[:, 0:num_stim_time_bins]
                                    choice_plus_stim_passive_all[n_trial, :, :] = trial_choice_plus_stim_passive

                                choice_plus_stim_passive = np.mean(choice_plus_stim_passive_all, axis=0)
                            else:
                                choice_plus_stim_passive = np.nan
                                choice_plus_stim_passive_all = np.nan
                        else:
                            choice_plus_stim_passive = choice_mean_train_stim_cond.values + stim_cond_passive.values
                            choice_plus_stim_passive_all = choice_plus_stim_passive
                else:
                    choice_plus_stim_passive = np.nan
            else:
                if len(choice_mean_train.Time) != 0:
                    if zero_passive_response:
                        choice_plus_stim_passive = choice_mean_train.values
                    else:
                        if realign_movement:
                            stim_cond_test_ds = all_cond_choice_ds_test.where(
                                (all_cond_choice_ds_test['visDiff'].isin(passive_vis_cond) &
                                 all_cond_choice_ds_test['audDiff'].isin(passive_aud_cond)),
                              drop=True)
                            if len(stim_cond_test_ds.Time) != 0:
                                trial_rts = stim_cond_test_ds.isel(Time=0)['choiceInitTimeRelStim'].values
                                choice_mean_train_post_movement = choice_mean_train.sel(Time=(slice(peri_movement_time_to_insert_movement, None)))
                                num_trial = len(trial_rts)
                                if multiple_cells:
                                    total_stim_bins = np.shape(stim_cond_passive)[1]
                                else:
                                    total_stim_bins = len(stim_cond_passive)
                                choice_plus_stim_passive_all = np.zeros((num_trial, num_cells, total_stim_bins))
                                for n_trial, trial_rt in enumerate(trial_rts):
                                    add_shift = 0.1
                                    time_idx_to_insert_movement_component = np.argmin(np.abs(add_shift + trial_rt + peri_movement_time_to_insert_movement - passive_stim_align_peri_event_time.values))
                                    time_idx_to_insert_movement_component = int(time_idx_to_insert_movement_component)
                                    trial_choice_plus_stim_passive = stim_cond_passive.values
                                    if multiple_cells:
                                        num_stim_time_bins = np.shape(trial_choice_plus_stim_passive[:, time_idx_to_insert_movement_component:])[1]
                                        choice_mean_train_post_movement = choice_mean_train_post_movement.transpose('Time', 'Cell')
                                    else:
                                        num_stim_time_bins = len(trial_choice_plus_stim_passive[time_idx_to_insert_movement_component:])
                                    num_movement_bins = len(choice_mean_train_post_movement.Time)
                                    idx_end = time_idx_to_insert_movement_component + num_movement_bins
                                    if multiple_cells:
                                        trial_choice_plus_stim_passive[:, time_idx_to_insert_movement_component:idx_end] = \
                                            trial_choice_plus_stim_passive[:, time_idx_to_insert_movement_component:idx_end] + choice_mean_train_post_movement.values.T[:, 0:num_stim_time_bins]
                                    else:
                                        trial_choice_plus_stim_passive[time_idx_to_insert_movement_component:idx_end] = \
                                            trial_choice_plus_stim_passive[time_idx_to_insert_movement_component:idx_end] + choice_mean_train_post_movement.values[0:num_stim_time_bins]
                                    choice_plus_stim_passive_all[n_trial, :, :] = trial_choice_plus_stim_passive

                                choice_plus_stim_passive = np.mean(choice_plus_stim_passive_all, axis=0)  # cell by Time
                                # choice_plus_stim_passive = (multiple_cells or np.squeeze)(choice_plus_stim_passive)  # not quite sure what this was meant to b
                            else:
                                choice_plus_stim_passive = np.nan
                                choice_plus_stim_passive_all = np.nan
                                trial_rts = np.nan
                                stim_cond_and_choice_ds_train_single_trials = np.nan
                                stim_cond_and_choice_ds_test_single_trials = np.nan
                        else:
                            choice_plus_stim_passive = choice_mean_train.values + stim_cond_passive.values
                            stim_cond_test_ds = all_cond_choice_ds_test.where((all_cond_choice_ds_test['visDiff'].isin(passive_vis_cond) & all_cond_choice_ds_test['audDiff'].isin(passive_aud_cond)),
                              drop=True)
                            choice_plus_stim_passive_all = stim_cond_test_ds['stimSubtractedActivity'] + stim_cond_passive
                            choice_plus_stim_passive_all = choice_plus_stim_passive_all.transpose('Trial', 'Cell', 'Time')
                            choice_plus_stim_passive_all = choice_plus_stim_passive_all.values
                            if len(stim_cond_test_ds.Trial) > 0:
                                trial_rts = stim_cond_test_ds.isel(Time=0)['choiceInitTimeRelStim']
                            else:
                                trial_rts = np.nan
                else:
                    choice_plus_stim_passive = np.nan
                    choice_plus_stim_passive_all = np.nan
                    trial_rts = np.nan
                    stim_cond_and_choice_ds_train_single_trials = np.nan
                    stim_cond_and_choice_ds_test_single_trials = np.nan
            try:
                prediction_and_actual_psth[stim_cond]['prediction_single_trials'] = choice_plus_stim_passive_all
            except:
                pdb.set_trace()

            prediction_and_actual_psth[stim_cond]['rt_per_trial'] = trial_rts
            prediction_and_actual_psth[stim_cond]['actual_train_single_trials'] = stim_cond_and_choice_ds_train_single_trials
            prediction_and_actual_psth[stim_cond]['actual_test_single_trials'] = stim_cond_and_choice_ds_test_single_trials
            prediction_and_actual_psth[stim_cond]['prediction'] = choice_plus_stim_passive
            prediction_and_actual_psth[stim_cond]['actual_train'] = stim_cond_and_choice_ds_train
            prediction_and_actual_psth[stim_cond]['actual_test'] = stim_cond_and_choice_ds_test
            if include_mean_stim_fr:
                mean_passsive_stim_fr = stim_cond_passive_stim_aligned_ds['smoothed_fr'].mean('Time').values
                active_stim_cond_train_and_test_ds = xr.concat([stim_cond_and_choice_ds_train,
                 stim_cond_and_choice_ds_test],
                  dim='Trial')
                mean_active_stim_fr = active_stim_cond_train_and_test_ds.mean(['Trial', 'Time'])
                prediction_and_actual_psth[stim_cond]['passive_mean_fr'] = mean_passsive_stim_fr
                prediction_and_actual_psth[stim_cond]['active_mean_fr'] = mean_active_stim_fr

            if include_components:
                prediction_and_actual_psth[stim_cond]['stimComponent'] = stim_cond_passive.values
            if len(stim_cond_and_choice_ds_test) == 0 or len(stim_cond_and_choice_ds_train) == 0:
                prediction_and_actual_psth[stim_cond]['full_model_error'] = np.nan
                prediction_and_actual_psth[stim_cond]['additive_model_error'] = np.nan

            if multiple_cells:
                if len(stim_cond_and_choice_ds_test.Time) == 0 or len(stim_cond_and_choice_ds_train.Time) == 0:
                    prediction_and_actual_psth[stim_cond]['full_model_error'] = np.repeat(np.nan, num_cells)
                    prediction_and_actual_psth[stim_cond]['additive_model_error'] = np.repeat(np.nan, num_cells)

                assert error_metric in ('mse', 'cv_rmsd', 'rrse', 'explained_variance')
                if error_metric == 'mse':
                    full_model_error_matrix = stim_cond_and_choice_ds_train - stim_cond_and_choice_ds_test
                    full_model_error = np.linalg.norm(full_model_error_matrix, axis=1)
                    additive_model_error_matrix = choice_plus_stim_passive - stim_cond_and_choice_ds_test
                    additive_model_error = np.linalg.norm(additive_model_error_matrix, axis=1)
                elif error_metric == 'cv_rmsd':
                    full_model_error_matrix = stim_cond_and_choice_ds_train - stim_cond_and_choice_ds_test
                    full_model_error = np.linalg.norm(full_model_error_matrix, axis=1)
                    stim_cond_and_choice_ds_train_mean_over_time = np.mean((stim_cond_and_choice_ds_train ** 2), axis=1)
                    stim_cond_and_choice_ds_test_mean_over_time = np.mean((stim_cond_and_choice_ds_test ** 2), axis=1)
                    choice_plus_stim_passive_mean_over_time = np.mean((choice_plus_stim_passive ** 2), axis=1)
                    full_model_train_test_mean = (stim_cond_and_choice_ds_train_mean_over_time + stim_cond_and_choice_ds_test_mean_over_time) / 2
                    full_model_error = full_model_error / full_model_train_test_mean
                    additive_model_error_matrix = choice_plus_stim_passive - stim_cond_and_choice_ds_test
                    additive_model_error = np.linalg.norm(additive_model_error_matrix, axis=1)
                    additive_model_train_test_mean = (choice_plus_stim_passive_mean_over_time + stim_cond_and_choice_ds_test_mean_over_time) / 2
                    additive_model_error = additive_model_error / additive_model_train_test_mean
                elif error_metric == 'rrse':
                    full_model_error_matrix = stim_cond_and_choice_ds_train - stim_cond_and_choice_ds_test
                    full_model_pred_and_actual_diff = np.linalg.norm(full_model_error_matrix, axis=1) ** 2
                    test_set_diff_from_mean = np.linalg.norm((stim_cond_and_choice_ds_test - np.mean(stim_cond_and_choice_ds_test, axis=1)),
                      axis=1) ** 2
                    full_model_error = np.sqrt(full_model_pred_and_actual_diff / test_set_diff_from_mean)
                    additive_model_error_matrix = choice_plus_stim_passive - stim_cond_and_choice_ds_test
                    additive_model_pred_and_actual_diff = np.linalg.norm(additive_model_error_matrix, axis=1) ** 2
                    additive_model_error = np.sqrt(additive_model_pred_and_actual_diff / test_set_diff_from_mean)
                elif error_metric == 'explained_variance':
                    full_model_error_matrix = stim_cond_and_choice_ds_test - stim_cond_and_choice_ds_train
                    full_model_diff_var = np.var(full_model_error_matrix, axis=1)
                    test_var = np.var(stim_cond_and_choice_ds_test, axis=1)
                    full_model_error = 1 - full_model_diff_var / test_var
                    additive_model_error_matrix = stim_cond_and_choice_ds_test - choice_plus_stim_passive
                    additive_model_diff_var = np.var(additive_model_error_matrix)
                    additive_model_error = 1 - additive_model_diff_var / test_var
                    full_model_test_fit_error_matrix = stim_cond_and_choice_ds_test - stim_cond_and_choice_ds_test
                    full_model_test_fit_residual_var = np.var(full_model_test_fit_error_matrix)
                    full_model_test_fit_error = 1 - full_model_test_fit_residual_var / test_var

                if error_metric in ('mse', 'rrse'):
                    prediction_and_actual_psth[stim_cond]['full_model_error'] = full_model_error
                    prediction_and_actual_psth[stim_cond]['additive_model_error'] = additive_model_error
                else:
                    prediction_and_actual_psth[stim_cond]['full_model_error'] = full_model_error
                    prediction_and_actual_psth[stim_cond]['additive_model_error'] = additive_model_error
                    prediction_and_actual_psth[stim_cond]['full_model_test_fit_error'] = full_model_test_fit_error
                    prediction_and_actual_psth[stim_cond]['additive_model_test_fit_error'] = additive_model_test_fit_error
            else:
                prediction_and_actual_psth[stim_cond]['full_model_error'] = sspatial.distance.pdist(
                    np.stack([stim_cond_and_choice_ds_train, stim_cond_and_choice_ds_test]))[0]
                prediction_and_actual_psth[stim_cond]['additive_model_error'] = sspatial.distance.pdist(
                    np.stack([choice_plus_stim_passive, stim_cond_and_choice_ds_test]))[0]
                prediction_and_actual_psth[stim_cond]['passive_peri_event_time'] = passive_stim_align_peri_event_time.values
        else:
            passive_aligned_ds_time_subset = passive_aligned_ds.where(
                ((passive_aligned_ds['PeriEventTime'] >= subset_time_window[0]) &
                 (passive_aligned_ds['PeriEventTime'] <= subset_time_window[1])),
              drop=True)
            if window_to_get_baseline_for_subtraction is not None:

                passive_baseline_time_subset = passive_aligned_ds.where(
                    ((passive_aligned_ds['PeriEventTime'] >= window_to_get_baseline_for_subtraction[0]) &
                     (passive_aligned_ds['PeriEventTime'] <= window_to_get_baseline_for_subtraction[1])),
                  drop=True)

                global_baseline = passive_baseline_time_subset.mean(['Trial', 'Time'])['smoothed_fr'].values
            if add_multimodal_conditions:
                multimodal_adding_conditions = [
                 'alvl','arvr','alvr','arvl',
                 'acvl','acvr']
            else:
                multimodal_adding_conditions = ['acvl', 'acvr']

            if stim_cond not in multimodal_adding_conditions:

                stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                    (passive_aligned_ds_time_subset['visDiff'].isin(passive_vis_cond) &
                     passive_aligned_ds_time_subset['audDiff'].isin(passive_aud_cond)),
                  drop=True).mean('Trial')

                if 'Trial' in passive_aligned_ds_time_subset.PeriEventTime.dims:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime.isel(Trial=0)
                else:
                    passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime
                stim_cond_passive = stim_cond_passive_stim_aligned_ds['smoothed_fr']
            else:
                if stim_cond == 'acvl':
                    aud_center_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                        (passive_aligned_ds_time_subset['visDiff'].isin([0]) &
                         passive_aligned_ds_time_subset['audDiff'].isin([0])),
                      drop=True).mean('Trial')

                    vis_left_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                        (passive_aligned_ds_time_subset['visDiff'].isin(vis_contrast_levels * -1) &
                         passive_aligned_ds_time_subset['audDiff'].isin([np.inf])),
                      drop=True).mean('Trial')
                    stim_cond_passive_stim_aligned_ds = aud_center_stim_cond_passive_stim_aligned_ds + vis_left_stim_cond_passive_stim_aligned_ds
                    if 'Trial' in passive_aligned_ds_time_subset.PeriEventTime.dims:
                        passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime.isel(Trial=0)
                    else:
                        passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime
                    stim_cond_passive = stim_cond_passive_stim_aligned_ds['smoothed_fr']
                elif stim_cond == 'acvr':
                    aud_center_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                        (passive_aligned_ds_time_subset['visDiff'].isin([0]) &
                         passive_aligned_ds_time_subset['audDiff'].isin([0])),
                      drop=True).mean('Trial')
                    vis_right_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                        (passive_aligned_ds_time_subset['visDiff'].isin(vis_contrast_levels) &
                         passive_aligned_ds_time_subset['audDiff'].isin([np.inf])),
                      drop=True).mean('Trial')
                    stim_cond_passive_stim_aligned_ds = aud_center_stim_cond_passive_stim_aligned_ds + vis_right_stim_cond_passive_stim_aligned_ds
                    if 'Trial' in passive_aligned_ds_time_subset.PeriEventTime.dims:
                        passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime.isel(Trial=0)
                    else:
                        passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime
                    stim_cond_passive = stim_cond_passive_stim_aligned_ds['smoothed_fr']
                elif stim_cond == 'arvr':

                    aud_right_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                        (passive_aligned_ds_time_subset['visDiff'].isin([0]) &
                         passive_aligned_ds_time_subset['audDiff'].isin([60])),
                      drop=True).mean('Trial')
                    vis_right_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                        (passive_aligned_ds_time_subset['visDiff'].isin(coherent_vis_contrast_levels) &
                         passive_aligned_ds_time_subset['audDiff'].isin([np.inf])),
                      drop=True).mean('Trial')

                    stim_cond_passive_stim_aligned_ds = aud_right_stim_cond_passive_stim_aligned_ds + vis_right_stim_cond_passive_stim_aligned_ds
                    if 'Trial' in passive_aligned_ds_time_subset.PeriEventTime.dims:
                        passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime.isel(Trial=0)
                    else:
                        passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime
                    stim_cond_passive = stim_cond_passive_stim_aligned_ds['smoothed_fr']
                elif stim_cond == 'alvl':

                    aud_left_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                        (passive_aligned_ds_time_subset['visDiff'].isin([0]) &
                         passive_aligned_ds_time_subset['audDiff'].isin([-60])),
                      drop=True).mean('Trial')

                    vis_left_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                        (passive_aligned_ds_time_subset['visDiff'].isin(coherent_vis_contrast_levels * -1) &
                         passive_aligned_ds_time_subset['audDiff'].isin([np.inf])),
                      drop=True).mean('Trial')

                    stim_cond_passive_stim_aligned_ds = aud_left_stim_cond_passive_stim_aligned_ds + vis_left_stim_cond_passive_stim_aligned_ds

                    if 'Trial' in passive_aligned_ds_time_subset.PeriEventTime.dims:
                        passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime.isel(Trial=0)
                    else:
                        passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime
                    stim_cond_passive = stim_cond_passive_stim_aligned_ds['smoothed_fr']
                elif stim_cond == 'alvr':
                    aud_left_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                        (passive_aligned_ds_time_subset['visDiff'].isin([0]) &
                         passive_aligned_ds_time_subset['audDiff'].isin([-60])),
                      drop=True).mean('Trial')
                    vis_right_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where(
                        (passive_aligned_ds_time_subset['visDiff'].isin(coherent_vis_contrast_levels) &
                         passive_aligned_ds_time_subset['audDiff'].isin([np.inf])),
                      drop=True).mean('Trial')
                    stim_cond_passive_stim_aligned_ds = aud_left_stim_cond_passive_stim_aligned_ds + vis_right_stim_cond_passive_stim_aligned_ds
                    if 'Trial' in passive_aligned_ds_time_subset.PeriEventTime.dims:
                        passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime.isel(Trial=0)
                    else:
                        passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime
                    stim_cond_passive = stim_cond_passive_stim_aligned_ds['smoothed_fr']
                elif stim_cond == 'arvl':
                    aud_right_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where((passive_aligned_ds_time_subset['visDiff'].isin([0]) & passive_aligned_ds_time_subset['audDiff'].isin([-60])),
                      drop=True).mean('Trial')
                    vis_left_stim_cond_passive_stim_aligned_ds = passive_aligned_ds_time_subset.where((passive_aligned_ds_time_subset['visDiff'].isin(coherent_vis_contrast_levels) & passive_aligned_ds_time_subset['audDiff'].isin([np.inf])),
                      drop=True).mean('Trial')
                    stim_cond_passive_stim_aligned_ds = aud_right_stim_cond_passive_stim_aligned_ds + vis_left_stim_cond_passive_stim_aligned_ds
                    if 'Trial' in passive_aligned_ds_time_subset.PeriEventTime.dims:
                        passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime.isel(Trial=0)
                    else:
                        passive_stim_align_peri_event_time = passive_aligned_ds_time_subset.PeriEventTime
                    stim_cond_passive = stim_cond_passive_stim_aligned_ds['smoothed_fr']

            if window_to_get_baseline_for_subtraction is not None:
                if stim_cond in ('arvr', 'arvl', 'alvl', 'alvr', 'acvl', 'acvr'):
                    stim_cond_passive = stim_cond_passive - global_baseline
            stim_cond_and_choice_ds = choice_stim_aligned_ds_time_subset.where(
                (choice_stim_aligned_ds_time_subset['visDiff'].isin(passive_vis_cond) &
                 choice_stim_aligned_ds_time_subset['audDiff'].isin(passive_aud_cond)),
              drop=True)
            stim_cond_and_choice = stim_cond_and_choice_ds.mean('Trial')['smoothed_fr']
            if include_components:
                prediction_and_actual_psth[stim_cond]['stimComponent'] = stim_cond_passive.values
            if realign_movement:
                if len(stim_cond_and_choice_ds.Time) != 0:
                    trial_rts = stim_cond_and_choice_ds.isel(Time=0)['choiceInitTimeRelStim']
                    choice_mean_post_movement = choice_mean.sel(Time=(slice(peri_movement_time_to_insert_movement, None)))
                    num_trial = len(trial_rts)
                    if multiple_cells:
                        total_stim_bins = np.shape(stim_cond_passive)[1]
                    else:
                        total_stim_bins = len(stim_cond_passive)
                    choice_plus_stim_passive_all = np.zeros((num_trial, num_cells, total_stim_bins))
                    for n_trial, trial_rt in enumerate(trial_rts):
                        time_idx_to_insert_movement_component = np.argmin(np.abs(trial_rt - peri_movement_time_to_insert_movement - passive_stim_align_peri_event_time)).values
                        time_idx_to_insert_movement_component = int(time_idx_to_insert_movement_component)
                        trial_choice_plus_stim_passive = stim_cond_passive.values
                        if multiple_cells:
                            num_stim_time_bins = np.shape(trial_choice_plus_stim_passive[:, time_idx_to_insert_movement_component:])[1]
                            choice_mean_post_movement = choice_mean_post_movement.transpose('Time', 'Cell')
                        else:
                            num_stim_time_bins = len(trial_choice_plus_stim_passive[time_idx_to_insert_movement_component:])
                        num_movement_bins = len(choice_mean_post_movement.Time)
                        idx_end = time_idx_to_insert_movement_component + num_movement_bins
                        if multiple_cells:
                            trial_choice_plus_stim_passive[:, time_idx_to_insert_movement_component:idx_end] = \
                                trial_choice_plus_stim_passive[:, time_idx_to_insert_movement_component:idx_end] + choice_mean_post_movement.values.T[:, 0:num_stim_time_bins]
                        else:
                            trial_choice_plus_stim_passive[time_idx_to_insert_movement_component:idx_end] = \
                                trial_choice_plus_stim_passive[time_idx_to_insert_movement_component:idx_end] + choice_mean_post_movement.values[0:num_stim_time_bins]
                        choice_plus_stim_passive_all[n_trial, :, :] = trial_choice_plus_stim_passive

                    choice_plus_stim_passive = np.mean(choice_plus_stim_passive_all, axis=0)
                    choice_plus_stim_passive = (multiple_cells or np.squeeze)(choice_plus_stim_passive)
                else:
                    choice_plus_stim_passive = np.nan
                    trial_rts = np.nan
            else:
                choice_plus_stim_passive = choice_mean.values + stim_cond_passive.values
            prediction_and_actual_psth[stim_cond]['prediction'] = choice_plus_stim_passive
            prediction_and_actual_psth[stim_cond]['actual'] = stim_cond_and_choice
            prediction_and_actual_psth[stim_cond]['actual_full'] = choice_stim_aligned_ds.where(
                choice_stim_aligned_ds['audDiff'].isin(passive_aud_cond) &
                choice_stim_aligned_ds['visDiff'].isin(passive_vis_cond))['smoothed_fr'].mean('Trial')

            prediction_and_actual_psth[stim_cond]['passive_peri_event_time'] = passive_stim_align_peri_event_time.values

    if include_components:
        if train_test_split == 'per-stim-cond':
            prediction_and_actual_psth['choice_mean_train'] = choice_mean_train.values
        else:
            prediction_and_actual_psth['choice_mean_train'] = choice_mean.values
    return prediction_and_actual_psth


def make_X_from_stim_aligned_ds(stim_aligned_ds, feature_set=['baseline','audSign','visSign','moveLeft','moveRight'],
                                event_start_ends={'baseline':[-0.2, 0.7], 'stimOn':[-0.2, 0.7], 'audSign':[-0.2, 0.7],
                                                  'visSign':[-0.2, 0.7], 'moveLeft':[-0.2, 0.7], 'moveRight':[-0.2, 0.7],
                                                  'moveOnset':[-0.2, 0.7]},
                                rt_variable_name='choiceInitTimeRelStim', choice_var_name='choiceThreshDir',
                                return_feat_indices=False):
    """"
    Parameters
    ----------
    stim_aligned_ds : xarray dataset
    feature_set : list
    event_start_ends : dict
    rt_variable_name : str
    choice_var_name : str
    return_feat_indices : bool
    Returns
    --------
    all_trial_X_concat : numpy ndarray
    """

    num_features = len(feature_set)
    alignment_times = stim_aligned_ds.isel(Trial=0).PeriEventTime.values
    num_alignment_time_bins = len(alignment_times)
    if return_feat_indices:
        feat_index_dict = dict()
        weight_bin_start = 0
        for n_feat, feat in enumerate(feature_set):
            if feat == 'baseline':
                feat_index_dict[feat] = np.array([0])
                weight_bin_start = weight_bin_start + 1
            else:
                feat_index_dict[feat] = np.arange(weight_bin_start, weight_bin_start + num_alignment_time_bins)
                weight_bin_start = weight_bin_start + num_alignment_time_bins

    all_trial_X = list()
    for trial_idx in np.arange(len(stim_aligned_ds.Trial)):
        trial_ds = stim_aligned_ds.isel(Trial=trial_idx)
        trial_X = list()
        for feat in feature_set:
            alvl = (np.sign(trial_ds['audDiff']) == -1) * (np.sign(trial_ds['visDiff']) == -1)
            arvr = (np.sign(trial_ds['audDiff']) == 1) * (np.sign(trial_ds['visDiff']) == 1)
            alvr = (np.sign(trial_ds['audDiff']) == -1) * (np.sign(trial_ds['visDiff']) == 1)
            arvl = (np.sign(trial_ds['audDiff']) == 1) * (np.sign(trial_ds['visDiff']) == -1)
            if feat == 'audSign':
                feature_value = np.sign(trial_ds['audDiff'])
            else:
                if feat == 'visSign':
                    feature_value = np.sign(trial_ds['visDiff'])
                else:
                    if feat == 'moveLeft':
                        if 'Time' in trial_ds[choice_var_name].dims:
                            feature_value = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                        else:
                            feature_value = (trial_ds[choice_var_name] == 1).astype(float)
                    elif feat == 'moveRight':
                        if 'Time' in trial_ds[choice_var_name].dims:
                            feature_value = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                        else:
                            feature_value = (trial_ds[choice_var_name] == 2).astype(float)
                    elif feat == 'moveOnset':
                        if 'Time' in trial_ds[choice_var_name].dims:
                            move_left = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                            move_right = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                            feature_value = move_left + move_right
                        else:
                            move_left = (trial_ds[choice_var_name] == 1).astype(float)
                            move_right = (trial_ds[choice_var_name] == 2).astype(float)
                            feature_value = move_left + move_right
                    elif feat == 'moveDiff':
                        if 'Time' in trial_ds[choice_var_name].dims:
                            move_left = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                            move_right = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                            feature_value = move_right - move_left
                        else:
                            move_left = (trial_ds[choice_var_name] == 1).astype(float)
                            move_right = (trial_ds[choice_var_name] == 2).astype(float)
                            feature_value = move_right - move_left

                    elif feat == 'audSignVisSign':
                        feature_value = np.sign(trial_ds['audDiff']) * np.sign(trial_ds['visDiff'])
                    elif feat == 'moveLeft-alvl':
                        feature_value = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                        feature_value = feature_value * alvl
                    elif feat == 'moveLeft-arvr':
                        feature_value = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                        feature_value = feature_value * arvr
                    elif feat == 'moveLeft-alvr':
                        feature_value = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                        feature_value = feature_value * alvr
                    elif feat == 'moveLeft-arvl':
                        feature_value = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                        feature_value = feature_value * arvl
                    elif feat == 'moveRight-alvl':
                        feature_value = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                        feature_value = feature_value * alvl
                    elif feat == 'moveRight-arvr':
                        feature_value = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                        feature_value = feature_value * arvr
                    elif feat == 'moveRight-alvr':
                        feature_value = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                        feature_value = feature_value * alvr
                    elif feat == 'moveRight-arvl':
                        feature_value = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                        feature_value = feature_value * arvl
                    elif feat == 'moveOn-alvl':
                        if 'Time' in trial_ds[choice_var_name].dims:
                            move_left = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                            move_right = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                            move_on = move_left + move_right
                        else:
                            move_left = (trial_ds[choice_var_name] == 1).astype(float)
                            move_right = (trial_ds[choice_var_name] == 2).astype(float)
                            move_on = move_left + move_right
                        feature_value = move_on * alvl
                    elif feat == 'moveOn-arvr':
                        if 'Time' in trial_ds[choice_var_name].dims:
                            move_left = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                            move_right = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                            move_on = move_left + move_right
                        else:
                            move_left = (trial_ds[choice_var_name] == 1).astype(float)
                            move_right = (trial_ds[choice_var_name] == 2).astype(float)
                            move_on = move_left + move_right
                        feature_value = move_on * arvr
                    elif feat == 'moveOn-alvr':
                        if 'Time' in trial_ds[choice_var_name].dims:
                            move_left = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                            move_right = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                            move_on = move_left + move_right
                        else:
                            move_left = (trial_ds[choice_var_name] == 1).astype(float)
                            move_right = (trial_ds[choice_var_name] == 2).astype(float)
                            move_on = move_left + move_right
                        feature_value = move_on * alvr
                    elif feat == 'moveOn-arvl':
                        if 'Time' in trial_ds[choice_var_name].dims:
                            move_left = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                            move_right = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                            move_on = move_left + move_right
                        else:
                            move_left = (trial_ds[choice_var_name] == 1).astype(float)
                            move_right = (trial_ds[choice_var_name] == 2).astype(float)
                            move_on = move_left + move_right
                        feature_value = move_on * arvl
                    elif feat == 'moveDiff-alvl':
                        if 'Time' in trial_ds[choice_var_name].dims:
                            move_left = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                            move_right = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                            move_diff = move_right - move_left
                        else:
                            move_left = (trial_ds[choice_var_name] == 1).astype(float)
                            move_right = (trial_ds[choice_var_name] == 2).astype(float)
                            move_diff = move_right - move_left
                        feature_value = move_diff * alvl
                    elif feat == 'moveDiff-arvr':
                        if 'Time' in trial_ds[choice_var_name].dims:
                            move_left = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                            move_right = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                            move_diff = move_right - move_left
                        else:
                            move_left = (trial_ds[choice_var_name] == 1).astype(float)
                            move_right = (trial_ds[choice_var_name] == 2).astype(float)
                            move_diff = move_right - move_left
                        feature_value = move_diff * arvr
                    elif feat == 'moveDiff-alvr':
                        if 'Time' in trial_ds[choice_var_name].dims:
                            move_left = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                            move_right = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                            move_diff = move_right - move_left
                        else:
                            move_left = (trial_ds[choice_var_name] == 1).astype(float)
                            move_right = (trial_ds[choice_var_name] == 2).astype(float)
                            move_diff = move_right - move_left
                        feature_value = move_diff * alvr
                    elif feat == 'moveDiff-arvl':
                        if 'Time' in trial_ds[choice_var_name].dims:
                            move_left = (trial_ds[choice_var_name].isel(Time=0) == 1).astype(float)
                            move_right = (trial_ds[choice_var_name].isel(Time=0) == 2).astype(float)
                            move_diff = move_right - move_left
                        else:
                            move_left = (trial_ds[choice_var_name] == 1).astype(float)
                            move_right = (trial_ds[choice_var_name] == 2).astype(float)
                            move_diff = move_right - move_left
                        feature_value = move_diff * arvl
                    elif feat == 'stimOn':
                        feature_value = 1

            if feat != 'baseline':
                feature_X = np.zeros((num_alignment_time_bins, num_alignment_time_bins))
                if feat in ('audSign', 'visSign', 'audSignVisSign', 'stimOn'):
                    feature_vec = np.repeat(0, num_alignment_time_bins)
                    event_start = event_start_ends[feat][0]
                    event_end = event_start_ends[feat][1]
                    time_idx_to_include = np.where((alignment_times >= event_start) & (alignment_times <= event_end))[0]
                    feature_vec.flat[time_idx_to_include] = feature_value
                elif feat in ('moveLeft', 'moveRight', 'moveOnset', 'moveDiff', 'moveOn-alvl',
                                'moveOn-alvr', 'moveOn-arvl', 'moveOn-arvr', 'moveDiff-alvl',
                                'moveDiff-alvr', 'moveDiff-arvl', 'moveDiff-arvr'):
                    feature_vec = np.repeat(0, num_alignment_time_bins)
                    if 'Time' in trial_ds[rt_variable_name].dims:
                        trial_rt = trial_ds[rt_variable_name].isel(Time=0).values
                    else:
                        trial_rt = trial_ds[rt_variable_name].values
                    event_start = event_start_ends[feat][0]
                    event_end = event_start_ends[feat][1]
                    movement_alignment_times = alignment_times - trial_rt
                    time_idx_to_include = np.where((movement_alignment_times >= event_start) & (movement_alignment_times <= event_end))[0]
                    feature_vec.flat[time_idx_to_include] = feature_value
                np.fill_diagonal(feature_X, feature_vec)
                trial_X.append(feature_X)

        trial_X_concat = np.concatenate(trial_X, axis=1)
        all_trial_X.append(trial_X_concat)

    all_trial_X_concat = np.concatenate(all_trial_X, axis=0)
    if 'baseline' in feature_set:
        baseline_vec = np.repeat(1, np.shape(all_trial_X_concat)[0]).reshape(-1, 1)
        all_trial_X_concat = np.concatenate([baseline_vec, all_trial_X_concat], axis=1)
    if return_feat_indices:
        return (all_trial_X_concat, feat_index_dict)

    return all_trial_X_concat


def subtract_passive_stim_psth(cell_active_stim_aligned_ds, cell_passive_stim_aligned_ds, subset_response=1,
                               response_var_name='choiceThreshDir', active_vis_cond=[-0.8, -0.4],
                               active_aud_cond=[60], passive_vis_cond=[-0.8], passive_aud_cond=[60],
                               active_activity_name='smoothed_fr', passive_activity_name='smoothed_fr',
                               multiple_cells=False, zero_passive_response=False):
    """
    Subtract the passive stimulus-aligned activity from the active stimulus-aligned activity.

    Parameters
    ----------
    cell_active_stim_aligned_ds : xarray dataset
        active condition cell dataset aligned to stimulus onset
    cell_passive_stim_aligned_ds : xarray dataset
        passive ocndition cell dataset aligned to stimulus onset
    active_vis_cond : list
        visual conditions to include in the active condition
    active_aud_cond : list
    passive_vis_cond : list
    passive_aud_cond : list
    activity_name : str
    multiple_cells : bool
        whether multiple cells are contained in the dataset
    Returns
    -------

    """
    stim_cond_active_stim_aligned_ds = cell_active_stim_aligned_ds.where(
        (cell_active_stim_aligned_ds['visDiff'].isin(active_vis_cond) &
         cell_active_stim_aligned_ds['audDiff'].isin(active_aud_cond)), drop=True)

    if subset_response is not None:
        stim_cond_active_stim_aligned_ds = stim_cond_active_stim_aligned_ds.where(
            (stim_cond_active_stim_aligned_ds[response_var_name] == subset_response),
          drop=True)
    if len(stim_cond_active_stim_aligned_ds.Trial.values) == 0:
        print('No trials in active condition, returning None')
        return None

    stim_cond_passive_stim_aligned_ds = cell_passive_stim_aligned_ds.where(
        (cell_passive_stim_aligned_ds['visDiff'].isin(passive_vis_cond) &
         cell_passive_stim_aligned_ds['audDiff'].isin(passive_aud_cond)),
      drop=True).mean('Trial')

    passive_stim_cond_fr = stim_cond_passive_stim_aligned_ds[passive_activity_name].values

    if zero_passive_response:
        passive_stim_cond_fr[:] = 0
    if 'Trial' in stim_cond_active_stim_aligned_ds.PeriEventTime.dims:
        active_psth_peri_event_time = stim_cond_active_stim_aligned_ds.PeriEventTime.isel(Trial=0).values
        passive_psth_peri_event_time = cell_passive_stim_aligned_ds.PeriEventTime.isel(Trial=0).values
    else:
        active_psth_peri_event_time = stim_cond_active_stim_aligned_ds.PeriEventTime.values
        passive_psth_peri_event_time = cell_passive_stim_aligned_ds.PeriEventTime.values

    passive_psth_start_time = passive_psth_peri_event_time[0]
    passive_psth_end_time = passive_psth_peri_event_time[-1]
    active_psth_start_matching_idx = np.argmin(np.abs(active_psth_peri_event_time - passive_psth_start_time))
    active_psth_end_matching_idx = np.argmin(np.abs(active_psth_peri_event_time - passive_psth_end_time)) + 1

    try:
        if not multiple_cells:
            stim_cond_active_stim_aligned_ds[active_activity_name] = stim_cond_active_stim_aligned_ds[active_activity_name].transpose('Trial', 'Time')
        active_stim_cond_fr = stim_cond_active_stim_aligned_ds[active_activity_name].values
        active_stim_cond_fr_mean_subtracted = active_stim_cond_fr.copy()
    except:
        pdb.set_trace()

    if multiple_cells:
        active_stim_cond_fr_mean_subtracted[:, active_psth_start_matching_idx:active_psth_end_matching_idx] = \
            active_stim_cond_fr_mean_subtracted[:, active_psth_start_matching_idx:active_psth_end_matching_idx] - passive_stim_cond_fr[:, None, :]
    else:
        active_stim_cond_fr_mean_subtracted[:, active_psth_start_matching_idx:active_psth_end_matching_idx] = \
            active_stim_cond_fr_mean_subtracted[:, active_psth_start_matching_idx:active_psth_end_matching_idx] - passive_stim_cond_fr

    if multiple_cells:
        stim_cond_active_stim_aligned_ds['stimSubtractedActivity'] = (['Cell', 'Trial', 'Time'],
                                                                      active_stim_cond_fr_mean_subtracted)
    else:
        stim_cond_active_stim_aligned_ds['stimSubtractedActivity'] = (
         ['Trial', 'Time'], active_stim_cond_fr_mean_subtracted)

    return stim_cond_active_stim_aligned_ds


def stim_align_to_move_align(stim_aligned_ds, rt_time_rel_stim_varname='choiceInitTimeRelStim',
                             activity_name='smoothed_fr', time_start=-0.3, time_end=0.4,
                             bin_width=0.005, multiple_cells=False, round_based_on_bin_width=True, verbose=False):
    """
    Takes in stimulus aligned dataset and re-align to movement onset.
    Parameters
    ----------
    stim_aligned_ds : xarray dataset
        dataset aligned to stimulus onset
    rt_time_rel_stim_varname : str
        name of variable that gives the time of the reaction/movement relative to the stimulus onset
        on a trial-by-trial basis.
    activity_name : str or list
    time_start : float
        time to realign the movement
    time_end : float

    bin_width : float
        bin width of spikes (seconds)
    round_based_on_bin_width : bool
        whether to round the reaction time based on the bin width used
        eg. if using a bin-width of 5 ms, then reaction time will be rounded to the nearest 5 ms
        this makes alignment easier across trials.
    Returns
    -------
    movement_aligned_ds : xarray dataset
        xarray dataset with neural activity aligned to movement
    """
    time_bins = np.arange(time_start, time_end, bin_width)

    if 'Time' in stim_aligned_ds[rt_time_rel_stim_varname].dims:
        stim_aligned_ds[rt_time_rel_stim_varname] = stim_aligned_ds[rt_time_rel_stim_varname].isel(Time=0)

    movement_aligned_ds_list = list()

    for trial in stim_aligned_ds.Trial.values:
        trial_ds = stim_aligned_ds.sel(Trial=trial)
        rt_time_rel_stim = trial_ds[rt_time_rel_stim_varname]
        if round_based_on_bin_width:
            rt_time_rel_stim = np.round(rt_time_rel_stim / bin_width) * bin_width

        periMovementTime = trial_ds['PeriEventTime'] - rt_time_rel_stim

        if type(activity_name) is str:
            if multiple_cells:
                trial_movement_aligned_ds = xr.Dataset({'smoothed_fr':
                                        (['Time', 'Trial'], trial_ds[activity_name].values.reshape(-1, 1))},
                  coords={'Trial': [trial], 'Time': periMovementTime})
            else:
                trial_movement_aligned_ds = xr.Dataset({'smoothed_fr':
                                        (['Time', 'Trial'], trial_ds[activity_name].values.reshape(-1, 1))},
                  coords={'Trial':[trial], 'Time':periMovementTime})
        elif type(activity_name) is list:
            if multiple_cells:
                trial_movement_aligned_ds = xr.Dataset({'smoothed_fr':(
                  ['Cell', 'Time', 'Trial'], trial_ds['smoothed_fr'].values[:, :, np.newaxis]),
                 'stimSubtractedActivity':(
                  ['Cell', 'Time', 'Trial'], trial_ds['stimSubtractedActivity'].values[:, :, np.newaxis])},
                  coords={'Cell':trial_ds.Cell.values, 'Trial':[trial],  'Time':periMovementTime})
            else:
                trial_movement_aligned_ds = xr.Dataset({'smoothed_fr':(
                  ['Time', 'Trial'], trial_ds['smoothed_fr'].values.reshape(-1, 1)),
                 'stimSubtractedActivity':(
                  ['Time', 'Trial'], trial_ds['stimSubtractedActivity'].values.reshape(-1, 1))},
                  coords={'Trial':[trial], 'Time':periMovementTime})

        aligned_xarray_tuple = trial_movement_aligned_ds.groupby_bins('Time', time_bins, include_lowest=True)
        aligned_xarray_list = [i[1] for i in list(aligned_xarray_tuple)]
        aligned_xarray_list = [x.mean('Time') for x in aligned_xarray_list]
        trial_movement_aligned_ds = xr.concat(aligned_xarray_list, dim='Time')
        time_bins_temp = time_bins[:len(trial_movement_aligned_ds.Time)]
        trial_movement_aligned_ds = trial_movement_aligned_ds.assign_coords({'Time': time_bins_temp})
        movement_aligned_ds_list.append(trial_movement_aligned_ds)

    if len(movement_aligned_ds_list) == 1:
        movement_aligned_ds = movement_aligned_ds_list[0]
    else:
        movement_aligned_ds = xr.concat(movement_aligned_ds_list, dim='Trial')
    movement_aligned_ds = movement_aligned_ds.dropna('Time')

    return movement_aligned_ds

def get_realignment_movement_ds_list(cell_active_stim_aligned_ds, passive_cell_aligned_ds,
                                     subset_response=2, response_var_name='choiceThreshDir',
                                     marginalize_modality=None, stim_conds_to_include=['alvl', 'arvr', 'alvr', 'arvl'],
                                     alignment_event='movement', active_activity_name='smoothed_fr',
                                     conflict_contrast_levels=np.array([0.4, 0.8]),
                                     vis_contrast_levels=np.array([0.4, 0.8]),
                                     passive_contrast_levels=np.array([0.8]),
                                     coherent_contrast_levels=np.array([0.4, 0.8]),
                                     multiple_cells='infer', zero_passive_response=False,
                                     return_data_type='list', add_multimodal_conditions=False,
                                     window_to_get_baseline_for_subtraction=None, bin_width=0.002, verbose=False):
    """
    For each stimulus condition during the active condition, perform the following
    1. subtract the corresponding passive stimulus activity
    2. re-align the data from stimulus-aligned to movement-aligned

    Parameters
    ----------
    cell_active_stim_aligned_ds : xarray dataset
    passive_cell_aligned_ds : xarray dataset
    subset_response : int
    marginalize_modality : str or None
    alignment_event : str
        which event to align the activity to
        'stimulus' : keep the original alignment to stimulus
        'movement' : realign the activity to movement onset
    active_activity_name : str
        name of variable containing the activity of the neuron to use.
    multiple_cells : str or bool
        if bool, then specify whether the dataset provided contains multiple cells.
        if you use the special keyword 'infer', then I will use the Cell dimension to calculate whether there
        are multiple cells.
        if multiple cells are used, then I will vectorise the addition
    zero_passive_response : bool
        if True, then set stimulus response to zero
        this is used as a control.
    bin_width : float
        bin width of spikes in seconds. Need to make sure that movement and passive stimulus
        are aligned using the same bin width
    Returns
    -------

    """
    if multiple_cells == 'infer':
        if len(cell_active_stim_aligned_ds.Cell) > 1:
            if verbose:
                print('Multiple cells detected in dataset, will use vectorised version of code')
            multiple_cells = True
        else:
            multiple_cells = False
    passive_left_contrast_levels = passive_contrast_levels * -1
    left_vis_contrast_levels = vis_contrast_levels * -1
    if not add_multimodal_conditions:
        if verbose:
            print('Using original multimodal conditions')

        alvl_c_stim_subtracted_ds = subtract_passive_stim_psth(cell_active_stim_aligned_ds, passive_cell_aligned_ds,
          subset_response=subset_response,
          response_var_name=response_var_name,
          active_vis_cond=(coherent_contrast_levels * -1), active_aud_cond=[-60],
          passive_vis_cond=passive_left_contrast_levels, passive_aud_cond=[-60],
          active_activity_name=active_activity_name,
          multiple_cells=multiple_cells,
          zero_passive_response=zero_passive_response)

        if alvl_c_stim_subtracted_ds is not None:
            alvl_c_realign_movement_ds = stim_align_to_move_align(alvl_c_stim_subtracted_ds, activity_name=[
             active_activity_name, 'stimSubtractedActivity'],
              multiple_cells=multiple_cells,
              bin_width=bin_width)
        else:
            alvl_c_realign_movement_ds = None

        alvr_c_stim_subtracted_ds = subtract_passive_stim_psth(cell_active_stim_aligned_ds, passive_cell_aligned_ds,
          subset_response=subset_response, response_var_name=response_var_name,
          active_vis_cond=conflict_contrast_levels, active_aud_cond=[-60],
          passive_vis_cond=passive_contrast_levels, passive_aud_cond=[-60],
          multiple_cells=multiple_cells)

        if alvr_c_stim_subtracted_ds is not None:
            alvr_c_realign_movement_ds = stim_align_to_move_align(alvr_c_stim_subtracted_ds, activity_name=[
             active_activity_name, 'stimSubtractedActivity'],
              multiple_cells=multiple_cells,
              bin_width=bin_width)
        else:
            alvr_c_realign_movement_ds = None

        arvl_c_stim_subtracted_ds = subtract_passive_stim_psth(cell_active_stim_aligned_ds, passive_cell_aligned_ds,
          subset_response=subset_response,
          response_var_name=response_var_name,
          active_vis_cond=(-conflict_contrast_levels), active_aud_cond=[60],
          passive_vis_cond=passive_left_contrast_levels, passive_aud_cond=[60],
          multiple_cells=multiple_cells)

        if arvl_c_stim_subtracted_ds is not None:
            arvl_c_realign_movement_ds = stim_align_to_move_align(arvl_c_stim_subtracted_ds, activity_name=[
             'smoothed_fr', 'stimSubtractedActivity'],
              multiple_cells=multiple_cells,
              bin_width=bin_width)
        else:
            arvl_c_realign_movement_ds = None

        arvr_c_stim_subtracted_ds = subtract_passive_stim_psth(cell_active_stim_aligned_ds, passive_cell_aligned_ds,
          subset_response=subset_response,
          response_var_name=response_var_name,
          active_vis_cond=coherent_contrast_levels, active_aud_cond=[60],
          passive_vis_cond=passive_contrast_levels, passive_aud_cond=[60],
          multiple_cells=multiple_cells)

        if arvr_c_stim_subtracted_ds is not None:
            arvr_c_realign_movement_ds = stim_align_to_move_align(arvr_c_stim_subtracted_ds,
              activity_name=['smoothed_fr', 'stimSubtractedActivity'],
              multiple_cells=multiple_cells,
              bin_width=bin_width)
        else:
            arvr_c_realign_movement_ds = None

    additional_stim_cond_list = list()
    for stim_cond in stim_conds_to_include:
        if add_multimodal_conditions:
            if verbose:
                print('Subtracting audio and visual separately to get multimodal conditions')

            if stim_cond == 'arvr':
                active_vis_cond = vis_contrast_levels
                active_aud_cond = [60]
                passive_vis_cond = [0]
                passive_aud_cond = [60]

                aud_right_subtracted_ds = subtract_passive_stim_psth(cell_active_stim_aligned_ds, passive_cell_aligned_ds,
                      subset_response=subset_response,
                      response_var_name=response_var_name,
                      active_vis_cond=active_vis_cond,
                      active_aud_cond=active_aud_cond,
                      passive_vis_cond=passive_vis_cond,
                      passive_aud_cond=passive_aud_cond,
                      multiple_cells=multiple_cells)
                if aud_right_subtracted_ds is not None:
                    passive_vis_cond = vis_contrast_levels
                    passive_aud_cond = [np.inf]
                    stim_subtraced_ds = subtract_passive_stim_psth(aud_right_subtracted_ds, passive_cell_aligned_ds,
                          subset_response=subset_response,
                          response_var_name=response_var_name,
                          active_vis_cond=active_vis_cond,
                          active_aud_cond=active_aud_cond,
                          passive_vis_cond=passive_vis_cond,
                          passive_aud_cond=passive_aud_cond,
                          multiple_cells=multiple_cells,
                          active_activity_name='stimSubtractedActivity')
                else:
                    stim_subtraced_ds = None
            elif stim_cond == 'alvl':

                active_vis_cond = vis_contrast_levels * -1
                active_aud_cond = [-60]
                passive_vis_cond = [0]
                passive_aud_cond = [-60]

                aud_left_subtracted_ds = subtract_passive_stim_psth(cell_active_stim_aligned_ds, passive_cell_aligned_ds,
                          subset_response=subset_response,
                          response_var_name=response_var_name,
                          active_vis_cond=active_vis_cond,
                          active_aud_cond=active_aud_cond,
                          passive_vis_cond=passive_vis_cond,
                          passive_aud_cond=passive_aud_cond,
                          multiple_cells=multiple_cells)

                if aud_left_subtracted_ds is not None:
                    passive_vis_cond = vis_contrast_levels * -1
                    passive_aud_cond = [np.inf]
                    stim_subtraced_ds = subtract_passive_stim_psth(aud_left_subtracted_ds, passive_cell_aligned_ds,
                              subset_response=subset_response,
                              response_var_name=response_var_name,
                              active_vis_cond=active_vis_cond,
                              active_aud_cond=active_aud_cond,
                              passive_vis_cond=passive_vis_cond,
                              passive_aud_cond=passive_aud_cond,
                              multiple_cells=multiple_cells,
                              active_activity_name='stimSubtractedActivity')
                else:
                    stim_subtraced_ds = None

            elif stim_cond == 'arvl':
                 active_vis_cond = vis_contrast_levels * -1
                 active_aud_cond = [60]
                 passive_vis_cond = [0]
                 passive_aud_cond = [60]

                 aud_right_subtracted_ds = subtract_passive_stim_psth(cell_active_stim_aligned_ds, passive_cell_aligned_ds,
                              subset_response=subset_response,
                              response_var_name=response_var_name,
                              active_vis_cond=active_vis_cond,
                              active_aud_cond=active_aud_cond,
                              passive_vis_cond=passive_vis_cond,
                              passive_aud_cond=passive_aud_cond,
                              multiple_cells=multiple_cells)

                 if aud_right_subtracted_ds is not None:
                     passive_vis_cond = vis_contrast_levels * -1
                     passive_aud_cond = [np.inf]

                     stim_subtraced_ds = subtract_passive_stim_psth(aud_right_subtracted_ds, passive_cell_aligned_ds,
                                  subset_response=subset_response,
                                  response_var_name=response_var_name,
                                  active_vis_cond=active_vis_cond,
                                  active_aud_cond=active_aud_cond,
                                  passive_vis_cond=passive_vis_cond,
                                  passive_aud_cond=passive_aud_cond,
                                  multiple_cells=multiple_cells,
                                  active_activity_name='stimSubtractedActivity')
                 else:
                     stim_subtraced_ds = None

            elif stim_cond == 'alvr':
                active_vis_cond = vis_contrast_levels
                active_aud_cond = [-60]
                passive_vis_cond = [0]
                passive_aud_cond = [-60]

                aud_left_subtracted_ds = subtract_passive_stim_psth(cell_active_stim_aligned_ds, passive_cell_aligned_ds,
                                  subset_response=subset_response,
                                  response_var_name=response_var_name,
                                  active_vis_cond=active_vis_cond,
                                  active_aud_cond=active_aud_cond,
                                  passive_vis_cond=passive_vis_cond,
                                  passive_aud_cond=passive_aud_cond,
                                  multiple_cells=multiple_cells)

                if aud_left_subtracted_ds is not None:
                    passive_vis_cond = vis_contrast_levels
                    passive_aud_cond = [np.inf]
                    stim_subtraced_ds = subtract_passive_stim_psth(aud_left_subtracted_ds, passive_cell_aligned_ds,
                      subset_response=subset_response,
                      response_var_name=response_var_name,
                      active_vis_cond=active_vis_cond,
                      active_aud_cond=active_aud_cond,
                      passive_vis_cond=passive_vis_cond,
                      passive_aud_cond=passive_aud_cond,
                      multiple_cells=multiple_cells,
                      active_activity_name='stimSubtractedActivity')
                else:
                    stim_subtraced_ds = None

                if (stim_subtraced_ds is not None) & (alignment_event == 'movement'):
                    stim_subtraced_ds = stim_align_to_move_align(stim_subtraced_ds, activity_name=[
                     'smoothed_fr', 'stimSubtractedActivity'],
                      multiple_cells=multiple_cells,
                      bin_width=bin_width)
                if stim_subtraced_ds is not None:
                    if verbose:
                        print('Stim cond: %s' % stim_cond)
                        print('Trials' + str(stim_subtraced_ds.Trial.values))
                additional_stim_cond_list.append(stim_subtraced_ds)

            elif stim_cond == 'acvl':
                active_vis_cond = vis_contrast_levels * -1
                active_aud_cond = [0]
                passive_vis_cond = [0]
                passive_aud_cond = [0]
                audio_c_subtracted_ds = subtract_passive_stim_psth(cell_active_stim_aligned_ds, passive_cell_aligned_ds,
                  subset_response=subset_response,
                  response_var_name=response_var_name,
                  active_vis_cond=active_vis_cond,
                  active_aud_cond=active_aud_cond,
                  passive_vis_cond=passive_vis_cond,
                  passive_aud_cond=passive_aud_cond,
                  multiple_cells=multiple_cells)
                if audio_c_subtracted_ds is not None:
                    passive_vis_cond = vis_contrast_levels * -1
                    passive_aud_cond = [np.inf]
                    stim_subtraced_ds = subtract_passive_stim_psth(audio_c_subtracted_ds, passive_cell_aligned_ds,
                      subset_response=subset_response,
                      response_var_name=response_var_name,
                      active_vis_cond=active_vis_cond,
                      active_aud_cond=active_aud_cond,
                      passive_vis_cond=passive_vis_cond,
                      passive_aud_cond=passive_aud_cond,
                      multiple_cells=multiple_cells,
                      active_activity_name='stimSubtractedActivity')
                else:
                    stim_subtraced_ds = None

            elif stim_cond == 'acvr':
                active_vis_cond = vis_contrast_levels
                active_aud_cond = [0]
                passive_vis_cond = [0]
                passive_aud_cond = [0]
                audio_c_subtracted_ds = subtract_passive_stim_psth(cell_active_stim_aligned_ds, passive_cell_aligned_ds,
                  subset_response=subset_response,
                  response_var_name=response_var_name,
                  active_vis_cond=active_vis_cond,
                  active_aud_cond=active_aud_cond,
                  passive_vis_cond=passive_vis_cond,
                  passive_aud_cond=passive_aud_cond,
                  multiple_cells=multiple_cells)
                passive_vis_cond = vis_contrast_levels
                passive_aud_cond = [np.inf]
                if audio_c_subtracted_ds is not None:
                    stim_subtraced_ds = subtract_passive_stim_psth(audio_c_subtracted_ds, passive_cell_aligned_ds,
                      subset_response=subset_response,
                      response_var_name=response_var_name,
                      active_vis_cond=active_vis_cond,
                      active_aud_cond=active_aud_cond,
                      passive_vis_cond=passive_vis_cond,
                      passive_aud_cond=passive_aud_cond,
                      multiple_cells=multiple_cells,
                      active_activity_name='stimSubtractedActivity')
                else:
                    stim_subtraced_ds = None

            elif stim_cond in ('arv0', 'alv0', 'acv0'):

                if stim_cond == 'arv0':
                    active_vis_cond = [0]
                    active_aud_cond = [60]
                    passive_vis_cond = [0]
                    passive_aud_cond = [60]
                elif stim_cond == 'alv0':
                    active_vis_cond = [0]
                    active_aud_cond = [-60]
                    passive_vis_cond = [0]
                    passive_aud_cond = [-60]
                elif stim_cond == 'acv0':
                    active_vis_cond = [0]
                    active_aud_cond = [0]
                    passive_vis_cond = [0]
                    passive_aud_cond = [0]

                stim_subtraced_ds = subtract_passive_stim_psth(cell_active_stim_aligned_ds, passive_cell_aligned_ds,
                  subset_response=subset_response,
                  response_var_name=response_var_name,
                  active_vis_cond=active_vis_cond,
                  active_aud_cond=active_aud_cond,
                  passive_vis_cond=passive_vis_cond,
                  passive_aud_cond=passive_aud_cond,
                  multiple_cells=multiple_cells)

            else:
                stim_subtraced_ds = None

            if (stim_subtraced_ds is not None) & (alignment_event == 'movement'):
                stim_subtraced_ds = stim_align_to_move_align(stim_subtraced_ds, activity_name=[
                 'smoothed_fr', 'stimSubtractedActivity'],
                  multiple_cells=multiple_cells,
                  bin_width=bin_width)
            if stim_subtraced_ds is not None:
                if verbose:
                    print('Stim cond: %s' % stim_cond)
                    print('Trials' + str(stim_subtraced_ds.Trial.values))
            additional_stim_cond_list.append(stim_subtraced_ds)
            additional_stim_cond_list = [i for i in additional_stim_cond_list if i]

    if marginalize_modality == 'audio':
        vl_c_combined_ds_list = [alvl_c_realign_movement_ds, arvl_c_realign_movement_ds]
        vr_c_combined_ds_list = [alvr_c_realign_movement_ds, arvr_c_realign_movement_ds]
        vl_c_combined_ds_list = [i for i in vl_c_combined_ds_list if i]
        vr_c_combined_ds_list = [i for i in vr_c_combined_ds_list if i]
        if len(vl_c_combined_ds_list) == 0:
            vl_c_realign_movement_ds = None
        else:
            vl_c_realign_movement_ds = xr.concat(vl_c_combined_ds_list, dim='Trial')
        if len(vr_c_combined_ds_list) == 0:
            vr_c_realign_movement_ds = None
        else:
            vr_c_realign_movement_ds = xr.concat(vr_c_combined_ds_list, dim='Trial')
        realignment_movement_ds_list = [vl_c_realign_movement_ds, vr_c_realign_movement_ds]
    elif marginalize_modality == 'visual':
        al_c_combined_ds_list = [alvl_c_realign_movement_ds, alvr_c_realign_movement_ds]
        ar_c_combined_ds_list = [arvl_c_realign_movement_ds, arvr_c_realign_movement_ds]
        al_c_combined_ds_list = [i for i in al_c_combined_ds_list if i]
        ar_c_combined_ds_list = [i for i in ar_c_combined_ds_list if i]
        if len(al_c_combined_ds_list) == 0:
            al_c_realign_movement_ds = None
        else:
            al_c_realign_movement_ds = xr.concat(al_c_combined_ds_list, dim='Trial')
        if len(ar_c_combined_ds_list) == 0:
            ar_c_realign_movement_ds = None
        else:
            ar_c_realign_movement_ds = xr.concat(ar_c_combined_ds_list, dim='Trial')

    if alignment_event == 'movement':
        realignment_movement_ds_list = [al_c_realign_movement_ds, ar_c_realign_movement_ds]
    elif alignment_event == 'stimulus':
        realignment_movement_ds_list = []

    if return_data_type == 'list':
        if alignment_event == 'movement':
            if not add_multimodal_conditions:
                realignment_movement_ds_list = [
                 alvl_c_realign_movement_ds,
                 alvr_c_realign_movement_ds,
                 arvl_c_realign_movement_ds,
                 arvr_c_realign_movement_ds]
            else:
                realignment_movement_ds_list = list()
            realignment_movement_ds_list.extend(additional_stim_cond_list)
        elif alignment_event == 'stimulus':
            print('Note: returning stimulus-aligned activity')
            if not add_multimodal_conditions:
                realignment_movement_ds_list = [
                 alvl_c_stim_subtracted_ds,
                 alvr_c_stim_subtracted_ds,
                 arvl_c_stim_subtracted_ds,
                 arvr_c_stim_subtracted_ds]
            else:
                realignment_movement_ds_list = list()
            realignment_movement_ds_list.extend(additional_stim_cond_list)

    elif return_data_type == 'dict':
        if alignment_event == 'movement':
            realignment_movement_ds_list = dict()
            realignment_movement_ds_list['alvl'] = alvl_c_realign_movement_ds
            realignment_movement_ds_list['alvr'] = alvr_c_realign_movement_ds
            realignment_movement_ds_list['arvl'] = arvl_c_realign_movement_ds
            realignment_movement_ds_list['arvr'] = arvr_c_realign_movement_ds
        elif alignment_event == 'stimulus':
            print('Note: returning stimulus-aligned activity')
            realignment_movement_ds_list = dict()
            realignment_movement_ds_list['alvl'] = alvl_c_stim_subtracted_ds
            realignment_movement_ds_list['alvr'] = alvr_c_stim_subtracted_ds
            realignment_movement_ds_list['arvl'] = arvl_c_stim_subtracted_ds
            realignment_movement_ds_list['arvr'] = arvr_c_stim_subtracted_ds

    return realignment_movement_ds_list


def get_stim_cond_vis_and_aud_targets(stim_cond, coherent_vis_contrast_levels,
                                      conflict_vis_contrast_levels, vis_contrast_levels):
    """
    Converts the string variable 'stim_cond' to visual and auditory conditions
    Parameters
    ----------
    stim_cond : str
    coherent_vis_contrast_levels : list
    conflict_vis_contrast_levels : list
    vis_contrast_levels : list

    Returns
    -------
    passive_aud_cond : list
    """

    if stim_cond == 'arvr':
        passive_vis_cond = coherent_vis_contrast_levels
        passive_aud_cond = [60]
    elif stim_cond == 'arvl':
        passive_vis_cond = conflict_vis_contrast_levels * -1
        passive_aud_cond = [60]
    elif stim_cond == 'alvr':
        passive_vis_cond = conflict_vis_contrast_levels
        passive_aud_cond = [-60]
    elif stim_cond == 'alvl':
        passive_vis_cond = coherent_vis_contrast_levels * -1
        passive_aud_cond = [-60]
    elif stim_cond == 'arv0':
        passive_vis_cond = [0]
        passive_aud_cond = [60]
    elif stim_cond == 'alv0':
        passive_vis_cond = [0]
        passive_aud_cond = [-60]
    elif stim_cond == 'acv0':
        passive_vis_cond = [0]
        passive_aud_cond = [0]
    elif stim_cond == 'acvl':
        passive_vis_cond = vis_contrast_levels * -1
        passive_aud_cond = [0]
    elif stim_cond == 'acvr':
        passive_vis_cond = vis_contrast_levels
        passive_aud_cond = [0]

    return passive_aud_cond, passive_vis_cond


def fit_active_only_model(active_stim_aligned_ds, stim_cond_list, coherent_vis_contrast_levels,
                          conflict_vis_contrast_levels, vis_contrast_levels,
                          subset_time_window, feature_set=['baseline', 'audSign', 'visSign', 'moveLeft', 'moveRight'],
                          event_start_ends={'baseline':[-0.2, 0.7], 'audSign':[0.0, 0.7], 'visSign':[0.0, 0.7],
                                            'audSignVisSign':[0.0, 0.7], 'moveLeft':[-0.2, 0.7], 'moveRight':[-0.2, 0.7]},
                          test_size=0.5, rt_variable_name='choiceInitTimeRelStim',
                          choice_var_name='choiceThreshDir', train_test_split='per-stim-cond',
                          activity_name='firing_rate', random_seed=None, return_fits=False,
                          error_metric_method='all', return_model_data=False, custom_model=None,
                          error_metric=['mse'], error_window=None, min_trial_count=1):
    """

    Parameters
    ----------
    active_stim_aligned_ds
    stim_cond_list
    coherent_vis_contrast_levels
    conflict_vis_contrast_levels
    vis_contrast_levels
    subset_time_window
    event_start_ends : (dict)
        start and end times of each of the kernel
        note that movement kernels are aligned to
        NOTE: baseline time range does not matter, baseline should just be oen flat line...

    feature_set
    test_size
    rt_variable_name
    train_test_split
    activity_name
    random_seed
    return_fits
    error_metric_method
    return_model_data
    error_window : (list)
        if not None, then error is evaluated in the provided start and end times
        the start and end times are relative to stimulus onset
    Returns
    -------

    """

    active_stim_aligned_ds_time_subset = active_stim_aligned_ds.where(
        ((active_stim_aligned_ds['PeriEventTime'] >= subset_time_window[0]) &
         (active_stim_aligned_ds['PeriEventTime'] <= subset_time_window[1])),
      drop=True)

    stim_cond_and_choice_ds_train_list = list()
    stim_cond_and_choice_ds_test_list = list()

    if train_test_split == 'per-stim-cond':
        for stim_cond in stim_cond_list:
            if stim_cond == 'arvr':
                passive_vis_cond = coherent_vis_contrast_levels
                passive_aud_cond = [60]
            elif stim_cond == 'arvl':
                passive_vis_cond = conflict_vis_contrast_levels * -1
                passive_aud_cond = [60]
            elif stim_cond == 'alvr':
                passive_vis_cond = conflict_vis_contrast_levels
                passive_aud_cond = [-60]
            elif stim_cond == 'alvl':
                passive_vis_cond = coherent_vis_contrast_levels * -1
                passive_aud_cond = [-60]
            elif stim_cond == 'arv0':
                passive_vis_cond = [0]
                passive_aud_cond = [60]
            elif stim_cond == 'alv0':
                passive_vis_cond = [0]
                passive_aud_cond = [-60]
            elif stim_cond == 'acv0':
                passive_vis_cond = [0]
                passive_aud_cond = [0]
            elif stim_cond == 'acvl':
                passive_vis_cond = vis_contrast_levels * -1
                passive_aud_cond = [0]
            elif stim_cond == 'acvr':
                passive_vis_cond = vis_contrast_levels
                passive_aud_cond = [0]

            stim_cond_and_choice_ds = active_stim_aligned_ds_time_subset.where(
                (active_stim_aligned_ds_time_subset['visDiff'].isin(passive_vis_cond) &
                 active_stim_aligned_ds_time_subset['audDiff'].isin(passive_aud_cond)),
              drop=True)

            stim_cond_trials = stim_cond_and_choice_ds.Trial
            num_stim_cond_trial = len(stim_cond_trials)
            test_num_trial = int(test_size * num_stim_cond_trial)
            if random_seed is not None:
                np.random.seed(random_seed)
            train_trials = np.random.choice(stim_cond_trials, test_num_trial, replace=False)
            test_trials = stim_cond_trials[(~stim_cond_trials.isin(train_trials))]

            if min_trial_count is not None and len(test_trials) < min_trial_count:
                print('test_trials < min_trial_count, skipping...')
                model_error_test = None
                model_fits = None
                model_data = None

                if return_model_data:
                    return model_error_test, model_fits, model_data
                else:
                    return model_error_test, model_fits

            stim_cond_and_choice_ds_train = stim_cond_and_choice_ds.sel(Trial=train_trials)
            stim_cond_and_choice_ds_test = stim_cond_and_choice_ds.sel(Trial=test_trials)
            stim_cond_and_choice_ds_train_list.append(stim_cond_and_choice_ds_train)
            stim_cond_and_choice_ds_test_list.append(stim_cond_and_choice_ds_test)

        all_cond_choice_ds_train = xr.concat(stim_cond_and_choice_ds_train_list, dim='Trial')
        all_cond_choice_ds_test = xr.concat(stim_cond_and_choice_ds_test_list, dim='Trial')

        X_train, feat_idx_dict = make_X_from_stim_aligned_ds(
         stim_aligned_ds=all_cond_choice_ds_train, feature_set=feature_set,
          event_start_ends=event_start_ends,
          rt_variable_name=rt_variable_name,
          return_feat_indices=True,
          choice_var_name=choice_var_name)

        X_test, feat_idx_dict = make_X_from_stim_aligned_ds(stim_aligned_ds=all_cond_choice_ds_test, feature_set=feature_set,
          event_start_ends=event_start_ends,
          rt_variable_name=rt_variable_name,
          return_feat_indices=True,
          choice_var_name=choice_var_name)

        Y_train = all_cond_choice_ds_train.stack(trialTime=['Trial', 'Time'])[activity_name].T
        Y_test = all_cond_choice_ds_test.stack(trialTime=['Trial', 'Time'])[activity_name].T
        if custom_model is not None:
            model = custom_model
        else:
            model = sklinear.LinearRegression(fit_intercept=False)
        model = model.fit(X_train, Y_train)
        Y_test_predict = model.predict(X_test)
        model_error_matrix = Y_test_predict - Y_test
        model_error_test = np.linalg.norm(model_error_matrix, axis=0)

        if return_fits:
            model_fits = xr.Dataset({'Y_test': Y_test}, {'Y_test_predict': (['trialTime', 'Cell'], Y_test_predict)})
            model_fits = model_fits.unstack()
            model_fits['audDiff'] = all_cond_choice_ds_test.isel(Time=0)['audDiff']
            model_fits['visDiff'] = all_cond_choice_ds_test.isel(Time=0)['visDiff']
            model_fits[choice_var_name] = all_cond_choice_ds_test.isel(Time=0)[choice_var_name]
            model_error_test = dict()
            model_error_test_per_stim_cond = defaultdict(list)
            if error_metric_method == 'ave-per-stimulus':
                for target_choice_cond_val in (1, 2):
                    for target_stim_cond in stim_cond_list:

                        aud_cond, vis_cond = get_stim_cond_vis_and_aud_targets(target_stim_cond,
                          coherent_vis_contrast_levels=coherent_vis_contrast_levels,
                          conflict_vis_contrast_levels=conflict_vis_contrast_levels,
                          vis_contrast_levels=vis_contrast_levels)

                        choice_and_stim_conds_ds = model_fits.where(
                            ((model_fits[choice_var_name] == target_choice_cond_val) &
                             (model_fits['audDiff'] == aud_cond) &
                             model_fits['visDiff'].isin(vis_cond)),
                          drop=True)

                        if len(choice_and_stim_conds_ds.Trial) > 0:
                            test_mean = choice_and_stim_conds_ds['Y_test'].mean('Trial')
                            prediction_mean = choice_and_stim_conds_ds['Y_test_predict'].mean('Trial')
                            if error_window is not None:
                                peri_event_time = all_cond_choice_ds_test.PeriEventTime.isel(Trial=0).values
                                assert len(peri_event_time) == len(test_mean.Time)
                                subset_window_idx = np.where((peri_event_time >= error_window[0]) &
                                                             (peri_event_time <= error_window[1]))[0]
                                test_mean = test_mean.isel(Time=subset_window_idx)
                                prediction_mean = prediction_mean.isel(Time=subset_window_idx)
                            if 'mse' in error_metric:
                                stim_cond_error = np.linalg.norm((test_mean - prediction_mean), axis=1)
                                model_error_test_per_stim_cond['mse'].append(stim_cond_error)
                            if 'explained_variance' in error_metric:
                                num_cell = len(choice_and_stim_conds_ds.Cell)
                                stim_cond_error = np.zeros((num_cell,))
                                for cell in np.arange(num_cell):
                                    y_true = choice_and_stim_conds_ds['Y_test'].isel(Cell=cell).values
                                    stim_cond_error[cell] = sklmetrics.explained_variance_score(y_true=y_true,
                                      y_pred=(choice_and_stim_conds_ds['Y_test_predict'].isel(Cell=cell).values))

                                model_error_test_per_stim_cond['explained_variance'].append(stim_cond_error)
                            if 'ss_residual' in error_metric:
                                stim_cond_error = np.sum(((test_mean - prediction_mean) ** 2), axis=1)
                                model_error_test_per_stim_cond['ss_residual'].append(stim_cond_error)

                for e_metric in error_metric:
                    try:
                        all_stim_error_per_cell_matrix = np.stack((model_error_test_per_stim_cond[e_metric]), axis=0)
                    except:
                        pdb.set_trace()

                    model_metric_mean_across_stim_cond = np.nanmean(all_stim_error_per_cell_matrix, axis=0)
                    model_error_test[e_metric] = model_metric_mean_across_stim_cond

            elif error_metric_method == 'ave-per-trial':
                if error_window is not None:
                    peri_event_time = model_fits.PeriEventTime.isel(Trial=0).values
                    subset_window_idx = np.where((peri_event_time >= error_window[0]) & (peri_event_time <= error_window[1]))[0]
                    model_fits_subset = model_fits.isel(Time=subset_window_idx)
                    test_matrix = model_fits_subset['Y_test'].values
                    prediction_matrix = model_fits_subset['Y_test_predict'].values
                    if 'mse' in error_metric:
                        model_error_test['mse'] = np.mean(np.mean(((test_matrix - prediction_matrix) ** 2), axis=1), axis=1)
                    if 'ss_residual' in error_metric:
                        model_error_test['ss_residual'] = np.sum(np.sum(((test_matrix - prediction_matrix) ** 2), axis=1), axis=1)
                    if 'explained_variance' in error_metric:
                        num_cell = len(model_fits.Cell)
                        explained_var_per_cell = np.zeros((num_cell,))
                        for cell in np.arange(num_cell):
                            y_true = model_fits_subset['Y_test'].isel(Cell=cell).values
                            y_pred = model_fits_subset['Y_test_predict'].isel(Cell=cell).values
                            explained_var_per_cell[cell] = sklmetrics.explained_variance_score(y_true=(y_true.T),
                              y_pred=(y_pred.T))

                        model_error_test['explained_variance'] = explained_var_per_cell
            if return_model_data:
                model_data = dict()
                model_data['model'] = model
                model_data['feat_idx_dict'] = feat_idx_dict
                peri_event_time = active_stim_aligned_ds_time_subset.isel(Trial=0)['PeriEventTime']
                model_data['peri_event_time'] = peri_event_time
                model_weights = model.coef_
                n_cv = 1
                stim_kernel_matrix = list()
                stim_kernel_names = list()
                mov_kernel_matrix = list()
                mov_kernel_names = list()
                for n_feat, (feature, feat_idx) in enumerate(model_data['feat_idx_dict'].items()):
                    if feature in ('moveLeft', 'moveRight', 'moveOnset', 'moveDiff'):
                        num_idx_to_start = np.argmin(np.abs(peri_event_time.values))
                        peri_movement_time = (peri_event_time[num_idx_to_start:] - 0.2,)
                        movement_kernel = model_weights[:, feat_idx][:, num_idx_to_start:]
                        mov_kernel_matrix.append(movement_kernel)
                        mov_kernel_names.append(feature)
                    else:
                        stim_kernel_matrix.append(model_weights[:, feat_idx])
                        stim_kernel_names.append(feature)

                try:
                    stim_kernel_matrix = np.stack(stim_kernel_matrix)
                    mov_kernel_matrix = np.stack(mov_kernel_matrix)
                    num_cell = np.shape(model_weights)[0]
                    peri_event_time_vals = np.array(peri_event_time)
                    stim_kernel_ds = xr.DataArray(stim_kernel_matrix, dims=['Feature', 'Cell', 'Time'],
                                                  coords={'Cell':np.arange(num_cell),
                     'Feature':stim_kernel_names,
                     'Time':peri_event_time_vals})
                    model_data['stim_kernels'] = stim_kernel_ds
                    peri_movement_time_vals = np.array(peri_movement_time).flatten()
                    mov_kernel_ds = xr.DataArray(mov_kernel_matrix, dims=['Feature', 'Cell', 'Time'],
                                                 coords={'Cell':np.arange(num_cell),
                     'Feature':mov_kernel_names,
                     'Time':peri_movement_time_vals})
                    model_data['mov_kernels'] = mov_kernel_ds
                except:
                    pdb.set_trace()

            if return_model_data:
                return model_error_test, model_fits, model_data
            return model_error_test, model_fits
    else:
        return model_error_test
