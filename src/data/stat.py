import numpy as np
import scipy.stats as sstats
import pandas as pd
import xarray as xr
import joblib
import multiprocessing
# from tqdm import tqdm
from tqdm.notebook import tqdm
from collections import defaultdict
import itertools
import pdb

import src.data.process_ephys_data as pephys
import src.data.analyse_spikes as anaspikes

import scipy.special as sspecial

# parallel processing
import ray

# ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import sklearn.model_selection as sklselect
import sklearn.metrics as sklmetrics
import sklearn.linear_model as sklinear

import string

def get_sig_neurons(num_neuron):
    
    if neuron_selection == 'sorted':
        sig_neuron_id_list = all_test_df.sort_values(by=test_stat).iloc[0:num_neuron]['Cell']
    elif neuron_selection == 'random':
        None

    return sig_neuron_id_list


def correlate_time_series(x, y, method='pearson', remove_nan=False, check_variance=True):
    """
    Calculates the correlation of two univariate time series

    Parameters
    ----------
    x : (numpy ndarray)
        numpy ndarray of size (num_samples, )
    y: (numpy ndarray)
        numpy ndarray of size (num_samples, )
    method: (str)
        method to calculate the correlation
    remove_nan: (bool)
        whether to remove nan (and corresponding entries in the other vectors) before calculating correlation.
    check_variance: (bool)
        whether to check if the variance is 0 (ie. if one of the input is constant) before calculating correlation
        numpy.nan is returned in cases where one of the input has zero variance

    Output
    ---------
    time_series_correlation : (float)
        some correlation metric, if pearson, then value is a real number from 0 to 1
    """

    if check_variance:
        # correlation is not defined if the input x or y is constant
        if (np.var(x) == 0) or (np.var(y) == 0):
            time_series_correlation = np.nan

    if remove_nan:
        nan_location = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[~nan_location]
        y = y[~nan_location]

    if method == 'numpy':
        # This assumes univariate time series
        time_series_correlation = np.corrcoef(x=x, y=y)[0, 1]
    elif method == 'pearson':
        r, p = sstats.pearsonr(x, y)
        time_series_correlation = dict()
        time_series_correlation['r'] = r
        time_series_correlation['p'] = p
    else:
        print('Warning: no valid correlation method specified')

    return time_series_correlation


def single_cell_test_two_cond_diff(cell_ds, cell_idx, cond_1=None, cond_2=None,
                                   peri_alignment_time_range=[0, None],
                                   activity_name='firing_rate',
                                   unimodal_trials_only=False,
                                   test_type='ranksum', mean_rate_threshold=None,
                                   baseline_subtraction=False, exclude_nan=True,
                                   test_pre_post=False, pre_time_range=[None, 0],
                                   post_time_range=[0, None], include_diff=False,
                                   scaled_mean_diff_epsilon=0.1):
    """

    # TODO: allow specifying more than one test type at once.
    # TODO: need to test using apply ufunc / groupby to speed this up
    # TODO: include_diff need to also work for unequal trial numbers (just take the mean or median difference)
    Parameters
    --------------
    cell_ds : (xarray dataset)

    :param cell_idx: (int)
    :param cond_1:
    :param cond_2:
    :param peri_alignment_time_range:
    :param activity_name:
    :param unimodal_trials_only:
    :param test_type:
    :param mean_rate_threshold:
    :param baseline_subtraction:
    :param exclude_nan:
    :return:
    """
    test_dict = dict()

    cell_ds_time_sliced = cell_ds

    if (cond_1 == 'beforeEvent') & (cond_2 == 'afterEvent'):

        cond_1_cell_ds = cell_ds_time_sliced.where(
            cell_ds['PeriEventTime'] < 0, drop=True
        ).mean(dim='Time')

        cond_2_cell_ds = cell_ds_time_sliced.where(
            cell_ds['PeriEventTime'] >= 0, drop=True
        ).mean(dim='Time')

    else:
        """
        if (peri_alignment_time_range[0] is not None) and (peri_alignment_time_range[1] is not None):

            cell_ds_time_sliced = cell_ds_time_sliced.where(
                (cell_ds['PeriEventTime'] > peri_alignment_time_range[0]) |
                (cell_ds['PeriEventTime'] < peri_alignment_time_range[1]),
                drop=True
            )

            # TODO: there seems to be an issue with slicing twice in the xarray 0.15 that was not in
            # version 0.13... may need to test that and raise an issue
        """

        if peri_alignment_time_range[0] is not None:
            cell_ds_time_sliced = cell_ds_time_sliced.where(
                cell_ds_time_sliced['PeriEventTime'] > peri_alignment_time_range[0],
                drop=True
            )

        if peri_alignment_time_range[1] is not None:
            cell_ds_time_sliced = cell_ds_time_sliced.where(
                cell_ds_time_sliced['PeriEventTime'] < peri_alignment_time_range[1],
                drop=True
            )

        cell_ds_time_mean = cell_ds_time_sliced.mean(dim='Time')

        cond_1_cell_ds, cond_2_cell_ds = anaspikes.get_two_cond_ds(alignment_ds=cell_ds_time_mean,
                                                                   cond_1=cond_1, cond_2=cond_2,
                                                                   unimodal_trials_only=unimodal_trials_only)

        # pdb.set_trace()
        # cond_1_cell_ds only has the Trial dimension (the mean activity for the selected condition for each trial)

    # Check whether the dataset set is empty
    # TOOD: I think the below approach is wrong, should check Trial dim instead right?
    # import pdb
    # pdb.set_trace()

    if (len(cond_1_cell_ds.Trial.values) == 0) or (len(cond_2_cell_ds.Trial.values) == 0):
        print('One of the conditions specified has no trials, returning None')
        return None


    cond_1_cell_activity = cond_1_cell_ds[activity_name]
    cond_2_cell_activity = cond_2_cell_ds[activity_name]

    if exclude_nan:
        cond_1_cell_activity = cond_1_cell_activity[~np.isnan(cond_1_cell_activity)]
        cond_2_cell_activity = cond_2_cell_activity[~np.isnan(cond_2_cell_activity)]

    if test_type == 'ranksum':

        cond_1_cell_activity = cond_1_cell_ds[activity_name]
        cond_2_cell_activity = cond_2_cell_ds[activity_name]

        if exclude_nan:
            cond_1_cell_activity = cond_1_cell_activity[~np.isnan(cond_1_cell_activity)]
            cond_2_cell_activity = cond_2_cell_activity[~np.isnan(cond_2_cell_activity)]

        test_stat, p_val = sstats.ranksums(cond_1_cell_activity,
                                           cond_2_cell_activity)

        test_dict['cellIdx'] = [cell_idx]
        test_dict['cellLoc'] = [cell_ds['CellLoc'].values]
        test_dict['testStat'] = [test_stat]
        test_dict['pVal'] = [p_val]
        test_dict['test_name'] = [test_type]

    elif test_type == 'paired_ttest':

        cond_1_cell_activity = cond_1_cell_ds[activity_name]
        cond_2_cell_activity = cond_2_cell_ds[activity_name]

        if exclude_nan:
            cond_1_cell_activity = cond_1_cell_activity[~np.isnan(cond_1_cell_activity)]
            cond_2_cell_activity = cond_2_cell_activity[~np.isnan(cond_2_cell_activity)]

        test_stat, p_val = sstats.ttest_rel(cond_1_cell_activity,
                                           cond_2_cell_activity)

        test_dict['cellIdx'] = [cell_idx]
        test_dict['cellLoc'] = [cell_ds['CellLoc'].values]
        test_dict['testStat'] = [test_stat]
        test_dict['pVal'] = [p_val]
        test_dict['test_name'] = [test_type]

    elif test_type == 'bhattacharyya_coefficient':

        print('TODO: implement bhattacharyya coefficient')

    elif test_type == 'ks_2samp':

        cond_1_cell_activity = cond_1_cell_ds[activity_name]
        cond_2_cell_activity = cond_2_cell_ds[activity_name]

        if exclude_nan:
            cond_1_cell_activity = cond_1_cell_activity[~np.isnan(cond_1_cell_activity)]
            cond_2_cell_activity = cond_2_cell_activity[~np.isnan(cond_2_cell_activity)]

        ks_stat, p_val = sstats.ks_2samp(cond_1_cell_activity, cond_2_cell_activity)

        test_dict['cellIdx'] = [cell_idx]
        test_dict['cellLoc'] = [cell_ds['CellLoc'].values]
        test_dict['testStat'] = [ks_stat]
        test_dict['pVal'] = [p_val]
        test_dict['test_name'] = [test_type]

    elif test_type == 'fisher_discriminant':

        print('Implement fisher discriminant')

    elif test_type == 'ssmd':

        # Also related:
        # Stricly Standardised mean difference:
        # https://en.wikipedia.org/wiki/Strictly_standardized_mean_difference
        # Note that we assume that the two groups are independent

        # check whether we can run the vectorised methods (if there is more than one cell in cond_1_cell_activity
        if len(cond_1_cell_activity.Cell.values) > 1:

            cond_1_mean = cond_1_cell_activity.mean('Trial').values
            cond_2_mean = cond_2_cell_activity.mean('Trial').values
            cond_1_var = cond_1_cell_activity.var('Trial').values
            cond_2_var = cond_2_cell_activity.var('Trial').values

            ssmd = (cond_1_mean - cond_2_mean) / np.sqrt((cond_1_var + cond_2_var))
            cell_idx = cond_1_cell_activity.Cell.values
            test_dict['cellIdx'] = cell_idx
            test_dict['cellLoc'] = cell_ds['CellLoc'].values
            test_dict['testStat'] = ssmd
            test_dict['pVal'] = np.repeat(np.nan, len(cell_idx))
            test_dict['test_name'] = np.repeat(['ssmd'], len(cell_idx))


        else:

            cond_1_mean = np.mean(cond_1_cell_activity.values)
            cond_2_mean = np.mean(cond_2_cell_activity.values)
            cond_1_var = np.var(cond_1_cell_activity.values)
            cond_2_var = np.var(cond_2_cell_activity.values)

            ssmd = (cond_1_mean - cond_2_mean) / np.sqrt((cond_1_var + cond_2_var))

            test_dict['cellIdx'] = [cell_idx]
            test_dict['cellLoc'] = [cell_ds['CellLoc'].values]
            test_dict['testStat'] = [ssmd]
            test_dict['pVal'] = [np.nan]
            test_dict['test_name'] = ['ssmd']

    elif test_type == 'scaledmd':

        # Scaled mean difference with some small reguralisation term to prevent small noise
        # (mu_1 - mu_2) / (mu_1 + mu_2 + epsilon)

        # check whether we can run the vectorised methods (if there is more than one cell in cond_1_cell_activity
        if cond_1_cell_activity.Cell.values.size > 1:

            cond_1_mean = cond_1_cell_activity.mean('Trial').values
            cond_2_mean = cond_2_cell_activity.mean('Trial').values

            scaledmd = (cond_1_mean - cond_2_mean) / (cond_1_mean + cond_2_mean + scaled_mean_diff_epsilon)
            cell_idx = cond_1_cell_activity.Cell.values
            test_dict['cellIdx'] = cell_idx
            test_dict['cellLoc'] = cell_ds['CellLoc'].values
            test_dict['testStat'] = scaledmd
            test_dict['pVal'] = np.repeat(np.nan, len(cell_idx))
            test_dict['test_name'] = np.repeat(['scaledmd'], len(cell_idx))


        else:

            cond_1_mean = np.mean(cond_1_cell_activity.values)
            cond_2_mean = np.mean(cond_2_cell_activity.values)

            scaledmd = (cond_1_mean - cond_2_mean) / (cond_1_mean + cond_2_mean + scaled_mean_diff_epsilon)
            test_dict['cellIdx'] = [cell_idx]
            test_dict['cellLoc'] = [cell_ds['CellLoc'].values]
            test_dict['testStat'] = [scaledmd]
            test_dict['pVal'] = [np.nan]
            test_dict['test_name'] = ['scaledmd']

    elif test_type == 'meanDiff':

        # Just the difference in means

        # check whether we can run the vectorised methods (if there is more than one cell in cond_1_cell_activity
        if len(cond_1_cell_activity.Cell.values) > 1:

            cond_1_mean = cond_1_cell_activity.mean('Trial').values
            cond_2_mean = cond_2_cell_activity.mean('Trial').values

            meanDiff = (cond_1_mean - cond_2_mean)
            cell_idx = cond_1_cell_activity.Cell.values
            test_dict['cellIdx'] = cell_idx
            test_dict['cellLoc'] = cell_ds['CellLoc'].values
            test_dict['testStat'] = meanDiff
            test_dict['pVal'] = np.repeat(np.nan, len(cell_idx))
            test_dict['test_name'] = np.repeat(['meanDiff'], len(cell_idx))

        else:

            cond_1_mean = np.mean(cond_1_cell_activity.values)
            cond_2_mean = np.mean(cond_2_cell_activity.values)

            meanDiff = (cond_1_mean - cond_2_mean)
            test_dict['cellIdx'] = [cell_idx]
            test_dict['cellLoc'] = [cell_ds['CellLoc'].values]
            test_dict['testStat'] = [meanDiff]
            test_dict['pVal'] = [np.nan]
            test_dict['test_name'] = ['meanDiff']

    test_df = pd.DataFrame.from_dict(test_dict)

    if include_diff:
        test_df['meanDiff'] = np.mean(cond_2_cell_activity.values - cond_1_cell_activity.values)

    return test_df


def two_cond_test_stat(cond_1_activity, cond_2_activity, test_type='ranksum', exclude_nan=True,
                       cell_idx=None, cell_loc=None):

    test_dict = dict()

    if exclude_nan:
        cond_1_activity = cond_1_activity[~np.isnan(cond_1_activity)]
        cond_2_activity = cond_2_activity[~np.isnan(cond_2_activity)]

    test_stat, p_val = sstats.ranksums(cond_1_activity,
                                       cond_2_activity)

    test_dict['cellIdx'] = [cell_idx]
    test_dict['cellLoc'] = [cell_loc]
    test_dict['testStat'] = [test_stat]
    test_dict['pVal'] = [p_val]
    test_dict['test_name'] = [test_type]

    test_df = pd.DataFrame.from_dict(test_dict)

    return test_df


def test_two_cond_diff(alignment_ds, cond_1=None, cond_2=None,
                       unimodal_trials_only=False, per='cell',
                       peri_alignment_time_range=[0, None],
                       activity_name='firing_rate',
                       test_type='ranksum', mean_rate_threshold=None,
                       baseline_subtraction=False, run_parallel=False,
                       n_jobs=-2, include_diff=False, verbose=False,
                       exclude_nan=True, scaled_mean_diff_epsilon=0.1):
    """
    Statistical test of distribution of neural activity across two conditions.
    Parameters
    --------------
    alignment_ds: (xarray dataset)
        xarray dataset containing activity of cells for each trial. It should have dimensions
        (Cell, Time, Trial), with Time having coordinates 'PeriEventTime'. The ordering of the
        dimension should not matter (but this is not tested).
    cond_1: (str)
        first trial condition (eg. auditory on the left is coded as 'audLeft')
    cond_2: : (str)
        second trial condition (eg. auditory on the right is coded as 'audRight')
    per: (str)
        at what level to perform the statistitical test over
        'cell' : perform the statistical cell for each cell (independently)
    peri_alignment_time_range: (list)
        list containing 2 elements
        1st element: the start point in seconds (PeriEventTime) to test perform the statistical test
        2nd element: the end point in seconds
    activity_name: (str)
        name of variable in alignment_ds containing the activity of your cells
    test_type: (str)
        what statistical test to perform
    mean_rate_threshold: (float)
        minimum firing rate (spikes / second) for a cell to be included in the statistical test
    baseline_subtraction: (bool)
        whether to pre-process the data by subtracting the activity by the mean activity over time
    run_parallel: (bool)
        whether to run the statistical test in parallel
    include_diff (bool)
        whether to include the mean activity difference for each trial (for each cell)
        note this does not work if cond_1 and cond_2 have different trial numbers

    Output
    -------------

    """
    if per == 'cell':

        if verbose:
            print('Running statistical test cell by cell.')

        if run_parallel:

            if verbose:
                print('Running parallelized version.')

            cell_idx_list = np.arange(len(alignment_ds['Cell'])).astype(list)

            all_cell_two_cond_test_list = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(single_cell_test_two_cond_diff)(
                    cell_ds=alignment_ds.isel(Cell=cell_idx),
                    cell_idx=cell_idx,
                    cond_1=cond_1, cond_2=cond_2,
                    unimodal_trials_only=unimodal_trials_only,
                    peri_alignment_time_range=peri_alignment_time_range,
                    activity_name=activity_name,
                    test_type=test_type, mean_rate_threshold=mean_rate_threshold,
                    baseline_subtraction=baseline_subtraction,
                    scaled_mean_diff_epsilon=scaled_mean_diff_epsilon
                ) for cell_idx in tqdm(cell_idx_list))

            test_df = pd.concat(all_cell_two_cond_test_list)


        else:

            if verbose:
                print('Running looped version.')

            test_df_list = list()

            vectorised_methods = ['ssmd', 'scaledmd']

            if test_type in vectorised_methods:

                test_df = single_cell_test_two_cond_diff(
                    cell_ds=alignment_ds,
                    cell_idx=None,
                    cond_1=cond_1, cond_2=cond_2,
                    unimodal_trials_only=unimodal_trials_only,
                    peri_alignment_time_range=peri_alignment_time_range,
                    activity_name=activity_name,
                    test_type=test_type, mean_rate_threshold=mean_rate_threshold,
                    baseline_subtraction=baseline_subtraction,
                    include_diff=include_diff, exclude_nan=exclude_nan,
                    scaled_mean_diff_epsilon=scaled_mean_diff_epsilon
                )

            else:

                for cell_idx in tqdm(np.arange(len(alignment_ds['Cell']))):
                    cell_ds = alignment_ds.isel(Cell=cell_idx)

                    if verbose:
                        if cell_idx == 0:
                            print('Example cell dataset provided:')
                            print(cell_ds)

                    test_df = single_cell_test_two_cond_diff(
                        cell_ds=cell_ds,
                        cell_idx=cell_idx,
                        cond_1=cond_1, cond_2=cond_2,
                        unimodal_trials_only=unimodal_trials_only,
                        peri_alignment_time_range=peri_alignment_time_range,
                        activity_name=activity_name,
                        test_type=test_type, mean_rate_threshold=mean_rate_threshold,
                        baseline_subtraction=baseline_subtraction,
                        include_diff=include_diff, exclude_nan=exclude_nan
                    )

                    if test_df is None:
                        return None

                    test_df_list.append(test_df)

                test_df = pd.concat(test_df_list)

    elif per == 'allCell':

        alignment_ds_time_sliced = alignment_ds

        if peri_alignment_time_range[0] is not None:
            alignment_ds_time_sliced = alignment_ds_time_sliced.where(
                alignment_ds_time_sliced['PeriEventTime'] > peri_alignment_time_range[0],
                drop=True
            )

        if peri_alignment_time_range[1] is not None:
            alignment_ds_time_sliced = alignment_ds_time_sliced.where(
                alignment_ds_time_sliced['PeriEventTime'] < peri_alignment_time_range[1],
                drop=True
            )

        if cond_1 != 'beforeEvent':
            alignment_ds_time_mean = alignment_ds_time_sliced.mean(dim='Time')

            cond_1_alignment_ds, cond_2_alignment_ds = anaspikes.get_two_cond_ds(
                alignment_ds_time_mean, cond_1=cond_1, cond_2=cond_2, unimodal_trials_only=unimodal_trials_only)
        else:
            cond_1_alignment_ds, cond_2_alignment_ds = anaspikes.get_two_cond_ds(
                alignment_ds, cond_1=cond_1, cond_2=cond_2, unimodal_trials_only=unimodal_trials_only)

        if run_parallel:

            cell_idx_list = np.arange(len(alignment_ds_time_sliced['Cell'])).astype(list)

            all_cell_two_cond_test_list = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(two_cond_test_stat)(
                    cond_1_activity=cond_1_alignment_ds.isel(Cell=cell_idx)[activity_name],
                    cond_2_activity=cond_2_alignment_ds.isel(Cell=cell_idx)[activity_name],
                    cell_idx=cell_idx, cell_loc=cond_1_alignment_ds.isel(Cell=cell_idx)['CellLoc'].values,
                    test_type=test_type) for cell_idx in cell_idx_list)

            print(type(all_cell_two_cond_test_list[0]))
            # test_df = all_cell_two_cond_test_list
            test_df = pd.concat(all_cell_two_cond_test_list)

        else:
            two_cond_test_result_list = list()
            for cell_idx in tqdm(alignment_ds_time_sliced.Cell.values):
                two_cond_test_result = two_cond_test_stat(
                    cond_1_activity=cond_1_alignment_ds.isel(Cell=cell_idx)[activity_name],
                    cond_2_activity=cond_2_alignment_ds.isel(Cell=cell_idx)[activity_name],
                    cell_idx=cell_idx, cell_loc=cond_1_alignment_ds.isel(Cell=cell_idx)['CellLoc'].values,
                    test_type=test_type)
                two_cond_test_result_list.append(two_cond_test_result)

            test_df = pd.concat(two_cond_test_result_list)

    if verbose:
        print('Evaluated a total of %.f cells' % len(np.unique(test_df['cellIdx'])))
        print('Using a p-value of 0.05, %.f cells found to be significant' % (len(test_df.loc[test_df['pVal'] < 0.05])))
        print('Using a p-value of 0.01, %.f cells found to be significant' % (len(test_df.loc[test_df['pVal'] < 0.01])))

    return test_df


def windowed_test_two_cond_diff(alignment_ds, window_start_sec_list, window_end_sec_list,
                                cond_1='audLeft', cond_2='audRight', run_parallel=True,
                                per='cell', unimodal_trials_only=False, test_type='ranksum'):

    assert len(window_start_sec_list) == len(window_end_sec_list)
    test_df_window_list = list()

    for window_start_sec, window_end_sec in tqdm(zip(window_start_sec_list, window_end_sec_list)):
        test_df_window = test_two_cond_diff(alignment_ds,
                                                 cond_1=cond_1, cond_2=cond_2, per=per,
                                                 peri_alignment_time_range=[window_start_sec, window_end_sec],
                                                 activity_name='firing_rate',
                                                 test_type=test_type, mean_rate_threshold=None,
                                                 run_parallel=run_parallel,
                                                 unimodal_trials_only=unimodal_trials_only)

        test_df_window['window_start_sec'] = window_start_sec
        test_df_window['window_end_sec'] = window_end_sec

        test_df_window_list.append(test_df_window)

    test_df_window_combined = pd.concat(test_df_window_list)

    # also add the center of the window
    test_df_window_combined['window_center_sec'] = \
        (test_df_window_combined['window_start_sec'] + test_df_window_combined['window_end_sec']) / 2

    return test_df_window_combined


def alignment_ds_to_aud_vis_df(alignment_ds, audio_levels=[-60, np.inf, 60],
                               visual_levels=[-0.8, 0, 0.8],
                               peri_stim_time_range=[0.03, 0.23],
                               ):
    """
    Converts alignment dataset to dataframe with mean activity in a selected time window
    for each combination of audio and visual conditions.
    Parameters
    ----------
    :param alignment_ds:
    :param audio_levels:
    :param visual_levels:
    :param peri_stim_time_range: (list of floats)
        time to get the mean activity
        mean activity will be between the first value and second value (in seconds)
        inclusive on both sides
    :return:
    """

    # time to get the mean activity
    all_cell_aud_vis_cond_df_list = list()

    for aud_level, vis_level in itertools.product(audio_levels, visual_levels):

        if (aud_level == np.inf) and (vis_level == 0):
            # use pre-stimulus time as no stimulus condition
            aud_vis_ds_time_sliced = alignment_ds.where(
                alignment_ds['PeriEventTime'] <= 0, drop=True
            )

            aud_vis_ds_ave_fr = aud_vis_ds_time_sliced['firing_rate'].mean('Time')
            all_cell_aud_vis_cond_df = aud_vis_ds_ave_fr.to_dataframe()
        else:

            aud_vis_ds = alignment_ds.where(
                (alignment_ds['visDiff'] == vis_level) &
                (alignment_ds['audDiff'] == aud_level), drop=True
            )

            aud_vis_ds_time_sliced = aud_vis_ds.where(
                (aud_vis_ds['PeriEventTime'] >= peri_stim_time_range[0]) &
                (aud_vis_ds['PeriEventTime'] <= peri_stim_time_range[1]),
                drop=True
            )

            aud_vis_ds_ave_fr = aud_vis_ds_time_sliced['firing_rate'].mean('Time')

            all_cell_aud_vis_cond_df = aud_vis_ds_ave_fr.to_dataframe()

        all_cell_aud_vis_cond_df['visCond'] = vis_level
        all_cell_aud_vis_cond_df['audCond'] = aud_level

        all_cell_aud_vis_cond_df_list.append(all_cell_aud_vis_cond_df)

    all_cell_all_aud_vis_cond_df = pd.concat(all_cell_aud_vis_cond_df_list)

    # Recode the values to be used for categorical two-way ANOVA
    all_cell_all_aud_vis_cond_df['visCond'].replace({
        -0.8: -1, 0.8: 1
    }, inplace=True)

    all_cell_all_aud_vis_cond_df['audCond'].replace({
        -60: -1, 60: 1, np.inf: 0
    }, inplace=True)

    all_cell_all_aud_vis_cond_df = all_cell_all_aud_vis_cond_df.reset_index()

    return all_cell_all_aud_vis_cond_df


def aud_vis_two_way_anova(all_cell_all_aud_vis_cond_df):
    """
    Test for main effect of audio, visual, and interaction effects using two-way ANOVA
    :param all_cell_all_aud_vis_cond_df:
    :return:
    """
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    formula = 'firing_rate ~ C(visCond) + C(audCond) + C(visCond):C(audCond)'
    cell_list = list()
    vis_effect_list = list()
    aud_effect_list = list()
    aud_vis_interaction_effect_list = list()

    for cell_idx in np.unique(all_cell_all_aud_vis_cond_df['Cell']):
        cell_df = all_cell_all_aud_vis_cond_df.loc[
            all_cell_all_aud_vis_cond_df['Cell'] == cell_idx
            ]

        cell_model = ols(formula, cell_df).fit()
        aov_table = anova_lm(cell_model, typ=3)

        cell_list.append(cell_idx)
        vis_effect_list.append(aov_table['PR(>F)']['C(visCond)'])
        aud_effect_list.append(aov_table['PR(>F)']['C(audCond)'])
        aud_vis_interaction_effect_list.append(
            aov_table['PR(>F)']['C(visCond):C(audCond)'])

    per_cell_tw_anova_df = pd.DataFrame.from_dict(
        {'Cell': cell_list,
         'visPval': vis_effect_list,
         'audPval': aud_effect_list,
         'interactionPval': aud_vis_interaction_effect_list}
    )

    return per_cell_tw_anova_df



def subset_anova_neuron_types(per_cell_tw_anova_df, sig_threshold=0.01):

    vis_only_neurons = per_cell_tw_anova_df.loc[
        (per_cell_tw_anova_df['visPval'] <= sig_threshold) &
        (per_cell_tw_anova_df['audPval'] > sig_threshold) &
        (per_cell_tw_anova_df['interactionPval'] > sig_threshold)
        ]

    aud_only_neurons = per_cell_tw_anova_df.loc[
        (per_cell_tw_anova_df['visPval'] > sig_threshold) &
        (per_cell_tw_anova_df['audPval'] <= sig_threshold) &
        (per_cell_tw_anova_df['interactionPval'] >= sig_threshold)
        ]

    aud_and_vis_additive_neurons = per_cell_tw_anova_df.loc[
        (per_cell_tw_anova_df['visPval'] <= sig_threshold) &
        (per_cell_tw_anova_df['audPval'] <= sig_threshold) &
        (per_cell_tw_anova_df['interactionPval'] > sig_threshold)
        ]

    """
    interaction_neurons = per_cell_tw_anova_df.loc[
        ((per_cell_tw_anova_df['visPval'] > sig_threshold) |
         (per_cell_tw_anova_df['audPval'] > sig_threshold)) &
        (per_cell_tw_anova_df['interactionPval'] <= sig_threshold)
        ]
    """

    # require interaction neurons to have at least significant response to one modality
    interaction_neurons = per_cell_tw_anova_df.loc[
        ((per_cell_tw_anova_df['visPval'] <= sig_threshold) |
         (per_cell_tw_anova_df['audPval'] <= sig_threshold)) &
        (per_cell_tw_anova_df['interactionPval'] <= sig_threshold)
        ]

    # print('Using updated calculations')

    no_selectivity_neurons = per_cell_tw_anova_df.loc[
        (per_cell_tw_anova_df['visPval'] > sig_threshold) &
        (per_cell_tw_anova_df['audPval'] > sig_threshold) &
        (per_cell_tw_anova_df['interactionPval'] > sig_threshold)
        ]

    return vis_only_neurons, aud_only_neurons, aud_and_vis_additive_neurons, interaction_neurons, \
           no_selectivity_neurons


def compute_two_way_mean_rate(alignment_ds, audio_levels=[-60, np.inf, 60],
                              visual_levels=[-0.8, 0, 0.8], peri_stim_time_range=[0.03, 0.23]):
    """

    Parameters
    ----------
    alignment_ds
    audio_levels
    visual_levels
    peri_stim_time_range (list)
        start and end time in seconds to get the mean activity
    Returns
    -------

    """

    # time to get the mean activity
    all_cell_aud_vis_cond_df_list = list()

    for aud_level, vis_level in itertools.product(audio_levels, visual_levels):

        if (aud_level == np.inf) and (vis_level == 0):
            # use pre-stimulus time as no stimulus condition
            aud_vis_ds_time_sliced = alignment_ds.where(
                aud_vis_ds['PeriEventTime'] <= 0, drop=True
            )

            aud_vis_ds_ave_fr = aud_vis_ds_time_sliced['firing_rate'].mean('Time')
            all_cell_aud_vis_cond_df = aud_vis_ds_ave_fr.to_dataframe()
        else:

            aud_vis_ds = alignment_ds.where(
                (alignment_ds['visDiff'] == vis_level) &
                (alignment_ds['audDiff'] == aud_level), drop=True
            )

            aud_vis_ds_time_sliced = aud_vis_ds.where(
                (aud_vis_ds['PeriEventTime'] >= peri_stim_time_range[0]) &
                (aud_vis_ds['PeriEventTime'] <= peri_stim_time_range[1]),
                drop=True
            )

            aud_vis_ds_ave_fr = aud_vis_ds_time_sliced['firing_rate'].mean('Time')

            all_cell_aud_vis_cond_df = aud_vis_ds_ave_fr.to_dataframe()

        all_cell_aud_vis_cond_df['visCond'] = vis_level
        all_cell_aud_vis_cond_df['audCond'] = aud_level

        all_cell_aud_vis_cond_df_list.append(all_cell_aud_vis_cond_df)

    all_cell_all_aud_vis_cond_df = pd.concat(all_cell_aud_vis_cond_df_list)

    all_cell_all_aud_vis_cond_df = all_cell_all_aud_vis_cond_df.reset_index()

    return all_cell_all_aud_vis_cond_df


def windowed_test_pre_post_diff():



    return test_df_window_combined



# Combined Conditions Choice Probability


def tiecorrect(rankvals):
    """
    Tie correction factor for Mann-Whitney U and Kruskal-Wallis H tests.
    Parameters
    ----------
    rankvals : array_like
        A 1-D sequence of ranks.  Typically this will be the array
        returned by `~scipy.stats.rankdata`.
    Returns
    -------
    factor : float
        Correction factor for U or H.
    See Also
    --------
    rankdata : Assign ranks to the data
    mannwhitneyu : Mann-Whitney rank test
    kruskal : Kruskal-Wallis H test
    References
    ----------
    .. [1] Siegel, S. (1956) Nonparametric Statistics for the Behavioral
           Sciences.  New York: McGraw-Hill.
    Examples
    --------
    >>> from scipy.stats import tiecorrect, rankdata
    >>> tiecorrect([1, 2.5, 2.5, 4])
    0.9
    >>> ranks = rankdata([1, 3, 2, 4, 5, 7, 2, 8, 4])
    >>> ranks
    array([ 1. ,  4. ,  2.5,  5.5,  7. ,  8. ,  2.5,  9. ,  5.5])
    >>> tiecorrect(ranks)
    0.9833333333333333
    """
    arr = np.sort(rankvals)
    idx = np.nonzero(np.r_[True, arr[1:] != arr[:-1], True])[0]
    cnt = np.diff(idx).astype(np.float64)

    size = np.float64(arr.size)
    return 1.0 if size < 2 else 1.0 - (cnt**3 - cnt).sum() / (size**3 - size)


def shuffle_each_column(a):
    # shuffle each column independently
    # from: https://stackoverflow.com/questions/49426584/shuffle-independently-within-column-of-numpy-array
    idx = np.random.rand(*a.shape).argsort(0)
    out = a[idx, np.arange(a.shape[1])]

    return out


def cal_mann_whit_numerator(x, y, shuffle_matrix=None):
    """
    Calculates the numerator of the Mann-Whitney U test
    This is one sided and calculates numerator corresponding to U_1,
    corresponding to vector x
    Based on code in:
    https://github.com/scipy/scipy/blob/v1.4.1/scipy/stats/stats.py#L6334-L6429
    https://github.com/nsteinme/steinmetz-et-al-2019/blob/master/ccCP/mannWhitneyUshuf.m
    Note this is one sided (we don't take the smaller value)

    Parameters
    -------------

    """

    x = np.asarray(x)
    y = np.asarray(y)

    num_x = len(x)

    # ranked = sstats.rankdata(np.concatenate((x, y)))
    # T = tiecorrect(ranked)

    # this is equivalent to tiedrank in matlab
    ranked = sstats.rankdata(np.concatenate((x, y)))

    # print(np.shape(ranked))

    if shuffle_matrix is None:
        ranked_x = ranked[0:num_x]
        ranked_x_sum = np.sum(ranked_x)
    else:
        ranked_x = ranked[shuffle_matrix]
        ranked_x_sum = np.sum(ranked_x, axis=0)

    # if ranked_x == 0:
    #     print('All values are equal, cannot do test')
    #     mann_whit_numerator_x = None
    # else:
    mann_whit_numerator_x = ranked_x_sum - num_x * (num_x + 1) / 2

    return mann_whit_numerator_x


def cal_combined_conditions_choice_prob(cell_ds_time_sliced,
                                        unique_vis_cond,
                                        unique_aud_cond,
                                        num_shuffle=2000,
                                        verbose=True,
                                        cond_1_val=1,
                                        cond_2_val=2,
                                        choice_cond_dict={'left': 1, 'right': 2}):
    """
    Calculates the combined conditions choice probability for a single neuron
    Parameters
    ----------
    cell_ds_time_sliced
    unique_vis_cond
    unique_aud_cond
    num_shuffle
    verbose
    cond_1_val
    cond_2_val
    choice_cond_dict

    Returns
    -------

    """

    numerator_total = 0  # Note that this will become a vector of num_shuffle > 1
    denominator_total = 0

    for vis_cond, aud_cond in itertools.product(unique_vis_cond, unique_aud_cond):

        stim_cond_ds = cell_ds_time_sliced.where(
            (cell_ds_time_sliced['visDiff'] == vis_cond) &
            (cell_ds_time_sliced['audDiff'] == aud_cond), drop=True
        )

        if len(stim_cond_ds.Trial.values) == 0:
            # stim cond does not exist
            if verbose:
                print('''Stim cond vis %.1f and aud %.1f does not exist, 
                      skipping''' % (vis_cond, aud_cond))
            continue

        choice_cond_1_ds = stim_cond_ds.where(
            stim_cond_ds['responseMade'] == cond_1_val, drop=True
        )

        choice_cond_2_ds = stim_cond_ds.where(
            stim_cond_ds['responseMade'] == cond_2_val, drop=True
        )

        num_choice_cond_1_ds_trials = len(choice_cond_1_ds.Trial.values)
        num_choice_cond_2_ds_trials = len(choice_cond_2_ds.Trial.values)

        if (num_choice_cond_1_ds_trials == 0) or (num_choice_cond_2_ds_trials == 0):
            if verbose:
                print('No trials for one of the choices, skipping.')
            continue

        n_x_and_y = num_choice_cond_1_ds_trials + num_choice_cond_2_ds_trials
        n_x = num_choice_cond_1_ds_trials

        # generate shuffle matrix
        ordered_matrix = np.tile(np.arange(n_x_and_y), (num_shuffle, 1)).T
        # ordered_matrix_shuffled = np.random.permutation(ordered_matrix) # shuffle rows
        ordered_matrix_shuffled = shuffle_each_column(ordered_matrix)
        shuffle_matrix = ordered_matrix_shuffled.copy()[0:n_x, :]
        # check shuffle and original matrix are different


        # make the first column the original order
        # TODO: somehow assert fails if the line belowis placed
        # before the assert, which is weird...
        shuffle_matrix[:, 0] = np.arange(n_x)

        # check sum of each column is the same (ie. shuffling is within each column)
        column_sum = np.sum(ordered_matrix_shuffled, axis=0)
        assert np.all(column_sum == column_sum[0])

        # u_stat is equivalent to the 'n' in the matlab code
        u_stat_numerator = cal_mann_whit_numerator(
            x=choice_cond_1_ds['firing_rate'].values,
            y=choice_cond_2_ds['firing_rate'].values,
            shuffle_matrix=shuffle_matrix
        )

        total_possible_comparisons = num_choice_cond_1_ds_trials * num_choice_cond_2_ds_trials

        numerator_total += u_stat_numerator
        denominator_total += total_possible_comparisons

    if denominator_total > 0:
        choice_probability = numerator_total / denominator_total
        # This is basically equivalent to calculating the percentile
        # numerator_p_val = sstats.percentileofscore(numerator[1:], numerator[0]) / 100
        choice_probability_rel_to_shuffle = sstats.rankdata(choice_probability)[0]
        p_of_choice_probability = choice_probability_rel_to_shuffle / (num_shuffle + 1)
    else:
        if verbose:
            print('Warning: all stimulus conditions have only a single choice, '
                  'please double check this experiment')
        choice_probability = np.nan
        p_of_choice_probability = np.nan

    # choice_probability = numerator_total / denominator_total

    # This is basically equivalent to calculating the percentile
    # numerator_p_val = sstats.percentileofscore(numerator[1:], numerator[0]) / 100
    # choice_probability_rel_to_shuffle = sstats.rankdata(choice_probability)[0]
    # p_of_choice_probability = choice_probability_rel_to_shuffle / (num_shuffle + 1)

    return choice_probability, p_of_choice_probability


def cal_combined_conditions_choice_prob_numpy(cell_fr,
                                        stim_cond_id_per_trial,
                                        response_per_trial,
                                        num_shuffle=2000,
                                        verbose=True,
                                        cond_1_val=1,
                                        cond_2_val=2,
                                        choice_cond_dict={'left': 1, 'right': 2}):
    """
    Calculates the combined conditions choice probability for a single neuron
    Parameters
    ----------
    cell_ds_time_sliced
    unique_vis_cond
    unique_aud_cond
    num_shuffle
    verbose
    cond_1_val
    cond_2_val
    choice_cond_dict

    Returns
    -------

    """

    numerator_total = 0  # Note that this will become a vector of num_shuffle > 1
    denominator_total = 0

    for stim_cond_idx in np.unique(stim_cond_id_per_trial):

        cond_1_trials_to_get = np.where(
            (stim_cond_id_per_trial == stim_cond_idx) &
            (response_per_trial == cond_1_val)
        )[0]

        cond_2_trials_to_get = np.where(
            (stim_cond_id_per_trial == stim_cond_idx) &
            (response_per_trial == cond_2_val)
        )[0]

        choid_cond_1_data = cell_fr[cond_1_trials_to_get]
        choid_cond_2_data = cell_fr[cond_2_trials_to_get]

        num_choice_cond_1_ds_trials = len(choid_cond_1_data)
        num_choice_cond_2_ds_trials = len(choid_cond_2_data)

        if (num_choice_cond_1_ds_trials == 0) or (num_choice_cond_2_ds_trials == 0):
            if verbose:
                print('No trials for one of the choices, skipping.')
            continue

        n_x_and_y = num_choice_cond_1_ds_trials + num_choice_cond_2_ds_trials
        n_x = num_choice_cond_1_ds_trials

        # generate shuffle matrix
        ordered_matrix = np.tile(np.arange(n_x_and_y), (num_shuffle, 1)).T
        # ordered_matrix_shuffled = np.random.permutation(ordered_matrix) # shuffle rows
        ordered_matrix_shuffled = shuffle_each_column(ordered_matrix)
        shuffle_matrix = ordered_matrix_shuffled.copy()[0:n_x, :]
        # check shuffle and original matrix are different


        # make the first column the original order
        # TODO: somehow assert fails if the line belowis placed
        # before the assert, which is weird...
        shuffle_matrix[:, 0] = np.arange(n_x)

        # check sum of each column is the same (ie. shuffling is within each column)
        column_sum = np.sum(ordered_matrix_shuffled, axis=0)
        assert np.all(column_sum == column_sum[0])

        # u_stat is equivalent to the 'n' in the matlab code
        u_stat_numerator = cal_mann_whit_numerator(
            x=choid_cond_1_data,
            y=choid_cond_2_data,
            shuffle_matrix=shuffle_matrix
        )

        total_possible_comparisons = num_choice_cond_1_ds_trials * num_choice_cond_2_ds_trials

        numerator_total += u_stat_numerator
        denominator_total += total_possible_comparisons

    if denominator_total > 0:
        choice_probability = numerator_total / denominator_total
        # This is basically equivalent to calculating the percentile
        # numerator_p_val = sstats.percentileofscore(numerator[1:], numerator[0]) / 100
        choice_probability_rel_to_shuffle = sstats.rankdata(choice_probability)[0]
        p_of_choice_probability = choice_probability_rel_to_shuffle / (num_shuffle + 1)
    else:
        if verbose:
            print('Warning: all stimulus conditions have only a single choice, '
                  'please double check this experiment')
        choice_probability = np.nan
        p_of_choice_probability = np.nan

    # choice_probability = numerator_total / denominator_total

    # This is basically equivalent to calculating the percentile
    # numerator_p_val = sstats.percentileofscore(numerator[1:], numerator[0]) / 100
    # choice_probability_rel_to_shuffle = sstats.rankdata(choice_probability)[0]
    # p_of_choice_probability = choice_probability_rel_to_shuffle / (num_shuffle + 1)

    return choice_probability, p_of_choice_probability




@ray.remote
def ray_cal_combined_conditions_choice_prob(cell_ds_time_sliced,
                                        unique_vis_cond,
                                        unique_aud_cond,
                                        num_shuffle=2000,
                                        verbose=True,
                                        cond_1_val=1,
                                        cond_2_val=2,
                                        choice_cond_dict={'left': 1, 'right': 2},
                                        choice_var_name='responseMade'):
    """
    Calculates the combined conditions choice probability for a single neuron
    Parameters
    ----------
    cell_ds_time_sliced
    unique_vis_cond
    unique_aud_cond
    num_shuffle
    verbose
    cond_1_val
    cond_2_val
    choice_cond_dict

    Returns
    -------

    """

    numerator_total = 0  # Note that this will become a vector of num_shuffle > 1
    denominator_total = 0

    for vis_cond, aud_cond in itertools.product(unique_vis_cond, unique_aud_cond):

        stim_cond_ds = cell_ds_time_sliced.where(
            (cell_ds_time_sliced['visDiff'] == vis_cond) &
            (cell_ds_time_sliced['audDiff'] == aud_cond), drop=True
        )

        if len(stim_cond_ds.Trial.values) == 0:
            # stim cond does not exist
            if verbose:
                print('''Stim cond vis %.1f and aud %.1f does not exist, 
                      skipping''' % (vis_cond, aud_cond))
            continue

        choice_cond_1_ds = stim_cond_ds.where(
            stim_cond_ds[choice_var_name] == cond_1_val, drop=True
        )

        choice_cond_2_ds = stim_cond_ds.where(
            stim_cond_ds[choice_var_name] == cond_2_val, drop=True
        )

        num_choice_cond_1_ds_trials = len(choice_cond_1_ds.Trial.values)
        num_choice_cond_2_ds_trials = len(choice_cond_2_ds.Trial.values)

        if (num_choice_cond_1_ds_trials == 0) or (num_choice_cond_2_ds_trials == 0):
            if verbose:
                print('No trials for one of the choices, skipping.')
            continue

        n_x_and_y = num_choice_cond_1_ds_trials + num_choice_cond_2_ds_trials
        n_x = num_choice_cond_1_ds_trials

        # generate shuffle matrix
        ordered_matrix = np.tile(np.arange(n_x_and_y), (num_shuffle, 1)).T
        # ordered_matrix_shuffled = np.random.permutation(ordered_matrix) # shuffle rows
        ordered_matrix_shuffled = shuffle_each_column(ordered_matrix)
        shuffle_matrix = ordered_matrix_shuffled.copy()[0:n_x, :]
        # check shuffle and original matrix are different


        # make the first column the original order
        # TODO: somehow assert fails if the line belowis placed
        # before the assert, which is weird...
        shuffle_matrix[:, 0] = np.arange(n_x)

        # check sum of each column is the same (ie. shuffling is within each column)
        column_sum = np.sum(ordered_matrix_shuffled, axis=0)
        assert np.all(column_sum == column_sum[0])

        # u_stat is equivalent to the 'n' in the matlab code
        u_stat_numerator = cal_mann_whit_numerator(
            x=choice_cond_1_ds['firing_rate'].values,
            y=choice_cond_2_ds['firing_rate'].values,
            shuffle_matrix=shuffle_matrix
        )

        total_possible_comparisons = num_choice_cond_1_ds_trials * num_choice_cond_2_ds_trials

        numerator_total += u_stat_numerator
        denominator_total += total_possible_comparisons

    if denominator_total > 0:
        choice_probability = numerator_total / denominator_total
        # This is basically equivalent to calculating the percentile
        # numerator_p_val = sstats.percentileofscore(numerator[1:], numerator[0]) / 100
        choice_probability_rel_to_shuffle = sstats.rankdata(choice_probability)[0]
        p_of_choice_probability = choice_probability_rel_to_shuffle / (num_shuffle + 1)
    else:
        if verbose:
            print('Warning: all stimulus conditions have only a single choice, '
                  'please double check this experiment')
        choice_probability = np.nan
        p_of_choice_probability = np.nan

    return choice_probability, p_of_choice_probability


@ray.remote
def ray_cal_combined_conditions_stimulus_prob(cell_ds_time_sliced,
                                        unique_vis_cond,
                                        unique_aud_cond,
                                        unique_choice_cond,
                                        num_shuffle=2000,
                                        verbose=True,
                                        test_type='visLeftRight'):

    """
    Calculates the combined conditions choice probability for a single neuron
    Parameters
    ----------
    cell_ds_time_sliced
    unique_vis_cond
    unique_aud_cond
    num_shuffle
    verbose
    cond_1_val
    cond_2_val
    choice_cond_dict

    Returns
    -------

    """

    numerator_total = 0  # Note that this will become a vector of num_shuffle > 1
    denominator_total = 0

    # Select which stimulus condition to "marganlise out" by comparing the firing rate
    # by controlling for the other stimulus condition.
    if test_type == 'visLeftRight':
        marginal_stim_cond = unique_aud_cond
        cond_1_val = unique_vis_cond[0]
        cond_2_val = unique_vis_cond[1]
    elif test_type == 'audLeftRight':
        marginal_stim_cond = unique_vis_cond
        cond_1_val = unique_aud_cond[0]
        cond_2_val = unique_aud_cond[1]

    for other_stim_cond in marginal_stim_cond:

        for choice_cond in unique_choice_cond:

            if test_type == 'visLeftRight':

                cond_1_ds = cell_ds_time_sliced.where(
                (cell_ds_time_sliced['visDiff'] == cond_1_val) &
                (cell_ds_time_sliced['audDiff'] == other_stim_cond) &
                (cell_ds_time_sliced['responseMade'] == choice_cond), drop=True
                )

                cond_2_ds = cell_ds_time_sliced.where(
                (cell_ds_time_sliced['visDiff'] == cond_2_val) &
                (cell_ds_time_sliced['audDiff'] == other_stim_cond) &
                (cell_ds_time_sliced['responseMade'] == choice_cond), drop=True
                )

            elif test_type == 'audLeftRight':

                cond_1_ds = cell_ds_time_sliced.where(
                (cell_ds_time_sliced['visDiff'] == other_stim_cond) &
                (cell_ds_time_sliced['audDiff'] == cond_1_val) &
                (cell_ds_time_sliced['responseMade'] == choice_cond), drop=True
                )

                cond_2_ds = cell_ds_time_sliced.where(
                (cell_ds_time_sliced['visDiff'] == other_stim_cond) &
                (cell_ds_time_sliced['audDiff'] == cond_2_val) &
                (cell_ds_time_sliced['responseMade'] == choice_cond), drop=True
                )

            num_cond_1_ds_trials = len(cond_1_ds.Trial.values)
            num_cond_2_ds_trials = len(cond_2_ds.Trial.values)

            if (num_cond_1_ds_trials == 0) or (num_cond_2_ds_trials == 0):
                if verbose:
                    print('No trials for one of the stimulus conditions, skipping.')
                continue

            n_x_and_y = num_cond_1_ds_trials + num_cond_2_ds_trials
            n_x = num_cond_1_ds_trials

            # generate shuffle matrix
            ordered_matrix = np.tile(np.arange(n_x_and_y), (num_shuffle, 1)).T
            # ordered_matrix_shuffled = np.random.permutation(ordered_matrix) # shuffle rows
            ordered_matrix_shuffled = shuffle_each_column(ordered_matrix)
            shuffle_matrix = ordered_matrix_shuffled.copy()[0:n_x, :]
            # check shuffle and original matrix are different

            # make the first column the original order
            # TODO: somehow assert fails if the line below is placed
            # before the assert, which is weird...
            shuffle_matrix[:, 0] = np.arange(n_x)

            # check sum of each column is the same (ie. shuffling is within each column)
            column_sum = np.sum(ordered_matrix_shuffled, axis=0)
            assert np.all(column_sum == column_sum[0])

            # u_stat is equivalent to the 'n' in the matlab code
            u_stat_numerator = cal_mann_whit_numerator(
                x=cond_1_ds['firing_rate'].values,
                y=cond_2_ds['firing_rate'].values,
                shuffle_matrix=shuffle_matrix
            )

            total_possible_comparisons = num_cond_1_ds_trials * num_cond_2_ds_trials

            numerator_total += u_stat_numerator
            denominator_total += total_possible_comparisons


    if denominator_total > 0:
        choice_probability = numerator_total / denominator_total
        # This is basically equivalent to calculating the percentile
        # numerator_p_val = sstats.percentileofscore(numerator[1:], numerator[0]) / 100
        choice_probability_rel_to_shuffle = sstats.rankdata(choice_probability)[0]
        p_of_choice_probability = choice_probability_rel_to_shuffle / (num_shuffle + 1)
    else:
        if verbose:
            print('Warning: all stimulus conditions have only a single choice, '
                  'please double check this experiment')
        choice_probability = np.nan
        p_of_choice_probability = np.nan

    return choice_probability, p_of_choice_probability


def cal_combined_conditions_choice_stim_numpy(cell_fr,
                                        aud_cond_per_trial,
                                        vis_cond_per_trial, 
                                        response_per_trial,
                                        unique_vis_cond,
                                        unique_aud_cond,
                                        unique_choice_cond,
                                        num_shuffle=2000,
                                        verbose=True,
                                        test_type='visLeftRight'):

    """
    Calculates the combined conditions stimulus probability for a single neuron.

    Parameters
    ----------
    cell_ds_time_sliced
    unique_vis_cond
    unique_aud_cond
    num_shuffle
    verbose
    cond_1_val
    cond_2_val
    choice_cond_dict

    Returns
    -------

    """

    numerator_total = 0  # Note that this will become a vector of num_shuffle > 1
    denominator_total = 0

    # Select which stimulus condition to "marganlise out" by comparing the firing rate
    # by controlling for the other stimulus condition.
    if test_type == 'visLeftRight':
        marginal_stim_cond = unique_aud_cond
        cond_1_val = unique_vis_cond[0]
        cond_2_val = unique_vis_cond[1]
    elif test_type == 'audLeftRight':
        marginal_stim_cond = unique_vis_cond
        cond_1_val = unique_aud_cond[0]
        cond_2_val = unique_aud_cond[1]

    for other_stim_cond in marginal_stim_cond:

        for choice_cond in unique_choice_cond:

            if test_type == 'visLeftRight':

                cond_1_trial_idx = np.where(
                    (vis_cond_per_trial == cond_1_val) &
                    (aud_cond_per_trial == other_stim_cond) &
                    (response_per_trial == choice_cond)
                )[0]

                cond_2_trial_idx = np.where(
                    (vis_cond_per_trial == cond_2_val) &
                    (aud_cond_per_trial == other_stim_cond) &
                    (response_per_trial == choice_cond)
                )[0]

            elif test_type == 'audLeftRight':

                cond_1_trial_idx = np.where(
                    (vis_cond_per_trial == other_stim_cond) &
                    (aud_cond_per_trial == cond_1_val) &
                    (response_per_trial == choice_cond)
                )[0]

                cond_2_trial_idx = np.where(
                    (vis_cond_per_trial == other_stim_cond) &
                    (aud_cond_per_trial == cond_2_val) &
                    (response_per_trial == choice_cond)
                )[0]

            cond_1_fr = cell_fr[cond_1_trial_idx]
            cond_2_fr = cell_fr[cond_2_trial_idx]

            num_cond_1_ds_trials = len(cond_1_trial_idx)
            num_cond_2_ds_trials = len(cond_2_trial_idx)

            if (num_cond_1_ds_trials == 0) or (num_cond_2_ds_trials == 0):
                if verbose:
                    print('No trials for one of the stimulus conditions, skipping.')
                continue

            n_x_and_y = num_cond_1_ds_trials + num_cond_2_ds_trials
            n_x = num_cond_1_ds_trials

            # generate shuffle matrix
            ordered_matrix = np.tile(np.arange(n_x_and_y), (num_shuffle, 1)).T
            # ordered_matrix_shuffled = np.random.permutation(ordered_matrix) # shuffle rows
            ordered_matrix_shuffled = shuffle_each_column(ordered_matrix)
            shuffle_matrix = ordered_matrix_shuffled.copy()[0:n_x, :]
            # check shuffle and original matrix are different

            # make the first column the original order
            # TODO: somehow assert fails if the line below is placed
            # before the assert, which is weird...
            shuffle_matrix[:, 0] = np.arange(n_x)

            # check sum of each column is the same (ie. shuffling is within each column)
            column_sum = np.sum(ordered_matrix_shuffled, axis=0)
            assert np.all(column_sum == column_sum[0])

            # u_stat is equivalent to the 'n' in the matlab code
            u_stat_numerator = cal_mann_whit_numerator(
                x=cond_1_fr,
                y=cond_2_fr,
                shuffle_matrix=shuffle_matrix
            )

            total_possible_comparisons = num_cond_1_ds_trials * num_cond_2_ds_trials

            numerator_total += u_stat_numerator
            denominator_total += total_possible_comparisons


    if denominator_total > 0:
        choice_probability = numerator_total / denominator_total
        # This is basically equivalent to calculating the percentile
        # numerator_p_val = sstats.percentileofscore(numerator[1:], numerator[0]) / 100
        choice_probability_rel_to_shuffle = sstats.rankdata(choice_probability)[0]
        p_of_choice_probability = choice_probability_rel_to_shuffle / (num_shuffle + 1)
    else:
        if verbose:
            print('Warning: all stimulus conditions have only a single choice, '
                  'please double check this experiment')
        choice_probability = np.nan
        p_of_choice_probability = np.nan

    return choice_probability, p_of_choice_probability
    
    
    


def test_cal_mann_whit_numerator():
    # TODO: work on this
    x = [1.1, 3.1, 5.1, 7.1, 2.1]
    y = [4.2, 7.2, 10.2, 4.2, 5.2, 4, 3, 6.3]

    # shuffle_matrix =

    return 1


def permutation_test_ANOVA_single_cell(cell_ds, test_cond=['audLR', 'visLR', 'interaction'],
                                       peri_stim_time_range=[0.00, 0.23],
                                       num_shuffle=1000):
    """
    Performs ANOVA-like analysis to test for
    (1) visual term
    (2) auditory term
    (3) interaction term

    Parameters
    ----------
    cell_ds
    test_cond
    peri_stim_time_range

    Returns
    -------

    """

    cell_ds_time_sliced = cell_ds.where(
        (cell_ds['PeriEventTime'] >= peri_stim_time_range[0]) &
        (cell_ds['PeriEventTime'] <= peri_stim_time_range[1]),
        drop=True
    )

    if 'audLR' in test_cond:

        aud_left_mean_rate = cell_ds_time_sliced.where(
            cell_ds_time_sliced['audDiff'] == -60, drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        aud_right_mean_rate = cell_ds_time_sliced.where(
            cell_ds_time_sliced['audDiff'] == 60, drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        aud_left_minus_right = aud_left_mean_rate - aud_right_mean_rate

        num_aud_left_trial = len(np.where(cell_ds_time_sliced.isel(Time=0)['audDiff'] == -60)[0])
        num_aud_right_trial = len(np.where(cell_ds_time_sliced.isel(Time=0)['audDiff'] == 60)[0])

        total_trials = len(cell_ds.Trial.values)

        # Create matrix of values to shuffle (parallelising each shuffle)
        ordered_matrix = np.tile(np.arange(total_trials), (num_shuffle, 1)).T
        ordered_matrix_shuffled = shuffle_each_column(ordered_matrix)
        aud_left_shuffle_matrix = ordered_matrix_shuffled[0:num_aud_left_trial, :]
        aud_right_shuffle_matrix = ordered_matrix_shuffled[num_aud_left_trial:, :]

        cell_ds_time_sliced_array = cell_ds_time_sliced['firing_rate'].mean('Time').values

        random_aud_left_fr_matrix = cell_ds_time_sliced_array[aud_left_shuffle_matrix]
        random_aud_right_fr_matrix = cell_ds_time_sliced_array[aud_right_shuffle_matrix]

        random_aud_left_mean_rate = np.mean(random_aud_left_fr_matrix, axis=0)
        random_aud_right_mean_rate = np.mean(random_aud_right_fr_matrix, axis=0)

        random_aud_left_minus_right_list = random_aud_left_mean_rate - random_aud_right_mean_rate

        percentile_score = sstats.percentileofscore(random_aud_left_minus_right_list,
                                                    score=aud_left_minus_right)
        aud_lr_score = percentile_score

    if 'visLR' in test_cond:


        vis_left_mean_rate = cell_ds_time_sliced.where(
            cell_ds_time_sliced['visDiff'] < 0, drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        vis_right_mean_rate = cell_ds_time_sliced.where(
            cell_ds_time_sliced['visDiff'] > 0, drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        vis_left_minus_right = vis_left_mean_rate - vis_right_mean_rate

        num_vis_left_trial = len(np.where(cell_ds_time_sliced.isel(Time=0)['visDiff'] < 0)[0])
        num_vis_right_trial = len(np.where(cell_ds_time_sliced.isel(Time=0)['visDiff'] > 0)[0])

        total_trials = len(cell_ds.Trial.values)

        # Create matrix of values to shuffle (parallelising each shuffle)
        ordered_matrix = np.tile(np.arange(total_trials), (num_shuffle, 1)).T
        ordered_matrix_shuffled = shuffle_each_column(ordered_matrix)
        vis_left_shuffle_matrix = ordered_matrix_shuffled[0:num_vis_left_trial, :]
        vis_right_shuffle_matrix = ordered_matrix_shuffled[num_vis_left_trial:, :]

        cell_ds_time_sliced_array = cell_ds_time_sliced['firing_rate'].mean('Time').values

        random_vis_left_fr_matrix = cell_ds_time_sliced_array[vis_left_shuffle_matrix]
        random_vis_right_fr_matrix = cell_ds_time_sliced_array[vis_right_shuffle_matrix]

        random_vis_left_mean_rate = np.mean(random_vis_left_fr_matrix, axis=0)
        random_vis_right_mean_rate = np.mean(random_vis_right_fr_matrix, axis=0)

        random_vis_left_minus_right_list = random_vis_left_mean_rate - random_vis_right_mean_rate

        percentile_score = sstats.percentileofscore(random_vis_left_minus_right_list,
                                                    score=vis_left_minus_right)
        vis_lr_score = percentile_score

    if 'interaction' in test_cond:

        aud_left_vis_left_mean_rate = cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] < 0) &
            (cell_ds_time_sliced['visDiff'] < 0), drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        aud_right_vis_right_mean_rate = cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] > 0) &
            (cell_ds_time_sliced['visDiff'] > 0), drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        aud_left_vis_right_mean_rate = cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] < 0) &
            (cell_ds_time_sliced['visDiff'] > 0), drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        aud_right_vis_left_mean_rate = cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] > 0) &
            (cell_ds_time_sliced['visDiff'] < 0), drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        coh_minus_conflict = (aud_left_vis_left_mean_rate + aud_right_vis_right_mean_rate) - \
                             (aud_left_vis_right_mean_rate + aud_right_vis_left_mean_rate)


        num_aud_left_vis_left_trials = len(cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] < 0) &
            (cell_ds_time_sliced['visDiff'] < 0), drop=True
        ).Trial)

        num_aud_right_vis_right_trials = len(cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] > 0) &
            (cell_ds_time_sliced['visDiff'] > 0), drop=True
        ).Trial)

        num_aud_left_vis_right_trials = len(cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] < 0) &
            (cell_ds_time_sliced['visDiff'] > 0), drop=True
        ).Trial)

        num_aud_right_vis_left_trials = len(cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] > 0) &
            (cell_ds_time_sliced['visDiff'] < 0), drop=True
        ).Trial)

        total_trials = len(cell_ds.Trial.values)

        # Create matrix of values to shuffle (parallelising each shuffle)
        ordered_matrix = np.tile(np.arange(total_trials), (num_shuffle, 1)).T
        ordered_matrix_shuffled = shuffle_each_column(ordered_matrix)

        aud_left_vis_left_idx = np.arange(0, num_aud_left_vis_left_trials)
        aud_right_vis_right_idx = np.arange(num_aud_left_vis_left_trials,
                                            num_aud_left_vis_left_trials+num_aud_right_vis_right_trials)
        aud_left_vis_right_idx = np.arange(
            num_aud_left_vis_left_trials + num_aud_right_vis_right_trials,
            num_aud_left_vis_left_trials + num_aud_right_vis_right_trials + num_aud_left_vis_right_trials
        )

        aud_right_vis_left_idx = np.arange(
            num_aud_left_vis_left_trials + num_aud_right_vis_right_trials + num_aud_left_vis_right_trials,
            num_aud_left_vis_left_trials + num_aud_right_vis_right_trials + num_aud_left_vis_right_trials +
            num_aud_right_vis_left_trials
        )

        # pdb.set_trace()

        aud_left_vis_left_shuffle_matrix = ordered_matrix_shuffled[aud_left_vis_left_idx, :]
        aud_right_vis_right_shuffle_matrix = ordered_matrix_shuffled[aud_right_vis_right_idx, :]
        aud_left_vis_right_shuffle_matrix = ordered_matrix_shuffled[aud_left_vis_right_idx, :]
        aud_right_vis_left_shuffle_matrix = ordered_matrix_shuffled[aud_right_vis_left_idx, :]

        cell_ds_time_sliced_array = cell_ds_time_sliced['firing_rate'].mean('Time').values

        random_aud_left_vis_left_fr_matrix = cell_ds_time_sliced_array[aud_left_vis_left_shuffle_matrix]
        random_aud_right_vis_right_fr_matrix = cell_ds_time_sliced_array[aud_right_vis_right_shuffle_matrix]
        random_aud_left_vis_right_fr_matrix = cell_ds_time_sliced_array[aud_left_vis_right_shuffle_matrix]
        random_aud_right_vis_left_fr_matrix = cell_ds_time_sliced_array[aud_right_vis_left_shuffle_matrix]

        random_aud_left_vis_left_mean_rate = np.mean(random_aud_left_vis_left_fr_matrix, axis=0)
        random_aud_right_vis_right_mean_rate = np.mean(random_aud_right_vis_right_fr_matrix, axis=0)
        random_aud_left_vis_right_mean_rate = np.mean(random_aud_left_vis_right_fr_matrix, axis=0)
        random_aud_right_vis_left_mean_rate = np.mean(random_aud_right_vis_left_fr_matrix, axis=0)

        random_coh_minus_conflict_list = (random_aud_left_vis_left_mean_rate + random_aud_right_vis_right_mean_rate) - \
                                         (random_aud_left_vis_right_mean_rate + random_aud_right_vis_left_mean_rate)

        percentile_score = sstats.percentileofscore(random_coh_minus_conflict_list,
                                                    score=coh_minus_conflict)
        aud_vis_interaction_score = percentile_score


    return aud_lr_score, vis_lr_score, aud_vis_interaction_score


def permutation_test_df_to_anova_df(permutation_test_df, percentile_to_p_val_name_dict={
    'VisLR': 'visPval', 'AudLR': 'audPval', 'Interaction': 'interactionPval'}, test_type='two-sided'):
    """
    Convert permutation ANOVA dataframe to the same format as ANOVA dataframe.
    Parameters
    ----------
    permutation_test_df
    percentile_to_p_val_name_dict

    Returns
    -------

    """
    permutation_test_anova_df = permutation_test_df.copy()

    for permutation_col_name, anova_col_name in percentile_to_p_val_name_dict.items():
        permutation_test_anova_df[anova_col_name] = permutation_test_anova_df[permutation_col_name] / 100

        if test_type == 'one-sided':
            permutation_test_anova_df[anova_col_name].loc[
                permutation_test_anova_df[anova_col_name] >= 0.5
                ] = 1 - permutation_test_anova_df[anova_col_name]

    return permutation_test_anova_df

@ray.remote
def permutation_test_ANOVA_single_cell_parallel(cell_ds, test_cond=['audLR', 'visLR', 'interaction'],
                                       peri_stim_time_range=[0.00, 0.23],
                                       num_shuffle=1000, test_type='two-sided'):
    """
    Performs ANOVA-like analysis to test for
    (1) visual term
    (2) auditory term
    (3) interaction term

    Parameters
    ----------
    cell_ds : (xarray dataset)

    test_cond : (list of str)
        list of conditions to test for
        audLR : difference in mean activity between audio left and audio right trials
        visLR : difference in mean activiyt between visual left and visual right trials
    peri_stim_time_range : (list)
        first element is the starting point to take the time bin
        second element is the end point to take the time bin
    num_shuffle : (int)
        number of shuffles to perform for the permutation test
    test_type : (str)
        'two-sided' - permutation test on the absolute difference, p-values are easier to interpret here
        'one-sided' - permutation test on the signed difference
    Returns
    -------

    """

    cell_ds_time_sliced = cell_ds.where(
        (cell_ds['PeriEventTime'] >= peri_stim_time_range[0]) &
        (cell_ds['PeriEventTime'] <= peri_stim_time_range[1]),
        drop=True
    )

    if 'audLR' in test_cond:

        aud_left_mean_rate = cell_ds_time_sliced.where(
            cell_ds_time_sliced['audDiff'] == -60, drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        aud_right_mean_rate = cell_ds_time_sliced.where(
            cell_ds_time_sliced['audDiff'] == 60, drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        if test_type == 'two-sided':

            aud_left_minus_right = np.abs(aud_left_mean_rate - aud_right_mean_rate)

        elif test_type == 'one-sided':

            aud_left_minus_right = aud_left_mean_rate - aud_right_mean_rate

        num_aud_left_trial = len(np.where(cell_ds_time_sliced.isel(Time=0)['audDiff'] == -60)[0])
        num_aud_right_trial = len(np.where(cell_ds_time_sliced.isel(Time=0)['audDiff'] == 60)[0])

        total_trials = len(cell_ds.Trial.values)

        # Create matrix of values to shuffle (parallelising each shuffle)
        ordered_matrix = np.tile(np.arange(total_trials), (num_shuffle, 1)).T
        ordered_matrix_shuffled = shuffle_each_column(ordered_matrix)
        aud_left_shuffle_matrix = ordered_matrix_shuffled[0:num_aud_left_trial, :]
        aud_right_shuffle_matrix = ordered_matrix_shuffled[num_aud_left_trial:, :]

        cell_ds_time_sliced_array = cell_ds_time_sliced['firing_rate'].mean('Time').values

        random_aud_left_fr_matrix = cell_ds_time_sliced_array[aud_left_shuffle_matrix]
        random_aud_right_fr_matrix = cell_ds_time_sliced_array[aud_right_shuffle_matrix]

        random_aud_left_mean_rate = np.mean(random_aud_left_fr_matrix, axis=0)
        random_aud_right_mean_rate = np.mean(random_aud_right_fr_matrix, axis=0)

        if test_type == 'two-sided':
            random_aud_left_minus_right_list = np.abs(random_aud_left_mean_rate - random_aud_right_mean_rate)
        elif test_type == 'one-sided':
            random_aud_left_minus_right_list = random_aud_left_mean_rate - random_aud_right_mean_rate

        percentile_score = sstats.percentileofscore(random_aud_left_minus_right_list,
                                                    score=aud_left_minus_right)
        aud_lr_score = percentile_score

    if 'visLR' in test_cond:


        vis_left_mean_rate = cell_ds_time_sliced.where(
            cell_ds_time_sliced['visDiff'] < 0, drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        vis_right_mean_rate = cell_ds_time_sliced.where(
            cell_ds_time_sliced['visDiff'] > 0, drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        if test_type == 'two-sided':
            vis_left_minus_right = np.abs(vis_left_mean_rate - vis_right_mean_rate)
        elif test_type == 'one-sided':
            vis_left_minus_right = vis_left_mean_rate - vis_right_mean_rate

        num_vis_left_trial = len(np.where(cell_ds_time_sliced.isel(Time=0)['visDiff'] < 0)[0])
        num_vis_right_trial = len(np.where(cell_ds_time_sliced.isel(Time=0)['visDiff'] > 0)[0])

        total_trials = len(cell_ds.Trial.values)

        # Create matrix of values to shuffle (parallelising each shuffle)
        ordered_matrix = np.tile(np.arange(total_trials), (num_shuffle, 1)).T
        ordered_matrix_shuffled = shuffle_each_column(ordered_matrix)
        vis_left_shuffle_matrix = ordered_matrix_shuffled[0:num_vis_left_trial, :]
        vis_right_shuffle_matrix = ordered_matrix_shuffled[num_vis_left_trial:, :]

        cell_ds_time_sliced_array = cell_ds_time_sliced['firing_rate'].mean('Time').values

        random_vis_left_fr_matrix = cell_ds_time_sliced_array[vis_left_shuffle_matrix]
        random_vis_right_fr_matrix = cell_ds_time_sliced_array[vis_right_shuffle_matrix]

        random_vis_left_mean_rate = np.mean(random_vis_left_fr_matrix, axis=0)
        random_vis_right_mean_rate = np.mean(random_vis_right_fr_matrix, axis=0)

        if test_type == 'two-sided':
            random_vis_left_minus_right_list = np.abs(random_vis_left_mean_rate - random_vis_right_mean_rate)
        elif test_type == 'one-sided':
            random_vis_left_minus_right_list = random_vis_left_mean_rate - random_vis_right_mean_rate

        percentile_score = sstats.percentileofscore(random_vis_left_minus_right_list,
                                                    score=vis_left_minus_right)

        if test_type == 'two-sided':
            # for a two-sided test, we just want to know the chance of observing the absolute
            # difference, so if the observed value is at the 99th percentile, the chance of
            # observing smaller this big is 1 - 0.99 = 0.01
            vis_lr_score = 1 - percentile_score
        elif test_type == 'one-sided':
            # in a one-sided test, we care about the direction of change
            vis_lr_score = percentile_score

    if 'interaction' in test_cond:

        aud_left_vis_left_mean_rate = cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] < 0) &
            (cell_ds_time_sliced['visDiff'] < 0), drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        aud_right_vis_right_mean_rate = cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] > 0) &
            (cell_ds_time_sliced['visDiff'] > 0), drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        aud_left_vis_right_mean_rate = cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] < 0) &
            (cell_ds_time_sliced['visDiff'] > 0), drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        aud_right_vis_left_mean_rate = cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] > 0) &
            (cell_ds_time_sliced['visDiff'] < 0), drop=True
        )['firing_rate'].mean(['Time', 'Trial']).values

        if test_type == 'two-sided':
            coh_minus_conflict = np.abs((aud_left_vis_left_mean_rate + aud_right_vis_right_mean_rate) - \
                                 (aud_left_vis_right_mean_rate + aud_right_vis_left_mean_rate))
        elif test_type == 'one-sided':

            coh_minus_conflict = (aud_left_vis_left_mean_rate + aud_right_vis_right_mean_rate) - \
                                 (aud_left_vis_right_mean_rate + aud_right_vis_left_mean_rate)


        num_aud_left_vis_left_trials = len(cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] < 0) &
            (cell_ds_time_sliced['visDiff'] < 0), drop=True
        ).Trial)

        num_aud_right_vis_right_trials = len(cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] > 0) &
            (cell_ds_time_sliced['visDiff'] > 0), drop=True
        ).Trial)

        num_aud_left_vis_right_trials = len(cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] < 0) &
            (cell_ds_time_sliced['visDiff'] > 0), drop=True
        ).Trial)

        num_aud_right_vis_left_trials = len(cell_ds_time_sliced.where(
            (cell_ds_time_sliced['audDiff'] > 0) &
            (cell_ds_time_sliced['visDiff'] < 0), drop=True
        ).Trial)

        total_trials = len(cell_ds.Trial.values)

        # Create matrix of values to shuffle (parallelising each shuffle)
        ordered_matrix = np.tile(np.arange(total_trials), (num_shuffle, 1)).T
        ordered_matrix_shuffled = shuffle_each_column(ordered_matrix)

        aud_left_vis_left_idx = np.arange(0, num_aud_left_vis_left_trials)
        aud_right_vis_right_idx = np.arange(num_aud_left_vis_left_trials,
                                            num_aud_left_vis_left_trials+num_aud_right_vis_right_trials)
        aud_left_vis_right_idx = np.arange(
            num_aud_left_vis_left_trials + num_aud_right_vis_right_trials,
            num_aud_left_vis_left_trials + num_aud_right_vis_right_trials + num_aud_left_vis_right_trials
        )

        aud_right_vis_left_idx = np.arange(
            num_aud_left_vis_left_trials + num_aud_right_vis_right_trials + num_aud_left_vis_right_trials,
            num_aud_left_vis_left_trials + num_aud_right_vis_right_trials + num_aud_left_vis_right_trials +
            num_aud_right_vis_left_trials
        )


        aud_left_vis_left_shuffle_matrix = ordered_matrix_shuffled[aud_left_vis_left_idx, :]
        aud_right_vis_right_shuffle_matrix = ordered_matrix_shuffled[aud_right_vis_right_idx, :]
        aud_left_vis_right_shuffle_matrix = ordered_matrix_shuffled[aud_left_vis_right_idx, :]
        aud_right_vis_left_shuffle_matrix = ordered_matrix_shuffled[aud_right_vis_left_idx, :]

        cell_ds_time_sliced_array = cell_ds_time_sliced['firing_rate'].mean('Time').values

        random_aud_left_vis_left_fr_matrix = cell_ds_time_sliced_array[aud_left_vis_left_shuffle_matrix]
        random_aud_right_vis_right_fr_matrix = cell_ds_time_sliced_array[aud_right_vis_right_shuffle_matrix]
        random_aud_left_vis_right_fr_matrix = cell_ds_time_sliced_array[aud_left_vis_right_shuffle_matrix]
        random_aud_right_vis_left_fr_matrix = cell_ds_time_sliced_array[aud_right_vis_left_shuffle_matrix]

        random_aud_left_vis_left_mean_rate = np.mean(random_aud_left_vis_left_fr_matrix, axis=0)
        random_aud_right_vis_right_mean_rate = np.mean(random_aud_right_vis_right_fr_matrix, axis=0)
        random_aud_left_vis_right_mean_rate = np.mean(random_aud_left_vis_right_fr_matrix, axis=0)
        random_aud_right_vis_left_mean_rate = np.mean(random_aud_right_vis_left_fr_matrix, axis=0)

        random_coh_minus_conflict_list = (random_aud_left_vis_left_mean_rate + random_aud_right_vis_right_mean_rate) - \
                                         (random_aud_left_vis_right_mean_rate + random_aud_right_vis_left_mean_rate)

        if test_type == 'two-sided':
            random_coh_minus_conflict_list = np.abs(random_coh_minus_conflict_list)

        percentile_score = sstats.percentileofscore(random_coh_minus_conflict_list,
                                                    score=coh_minus_conflict)
        aud_vis_interaction_score = percentile_score

    output_score_df = pd.DataFrame.from_dict(
        {'Cell': [cell_ds.Cell.values],
         'AudLR': [aud_lr_score],
         'VisLR': [vis_lr_score],
         'Interaction': [aud_vis_interaction_score]})

    return output_score_df


def cal_mean_abs_diff_at_max_window(alignment_ds_fr, max_diff_time_idxs,
                                    cond_1_trial_idx, cond_2_trial_idx=None,
                                    window_width_back=12, window_width_forward=12,
                                    baseline_window_start=None, baseline_window_end=None):
    """

    Parameters
    ----------
    alignment_ds_fr
    max_diff_time_idxs
    left_trial_idx
    right_trial_idx
    window_width_back
    window_width_forward

    Returns
    -------

    """
    abs_diff_all_cell = np.zeros(np.shape(alignment_ds_fr)[0])
    abs_diff_all_sign = np.zeros(np.shape(alignment_ds_fr)[0])

    total_num_window = np.shape(alignment_ds_fr)[1]

    if cond_2_trial_idx is None:

        print('Implement something...')

    else:

        for cell_idx in np.arange(np.shape(alignment_ds_fr)[0]):

            window_start = max_diff_time_idxs[cell_idx] - window_width_back
            window_end = max_diff_time_idxs[cell_idx] + window_width_forward

            window_start = np.max([0, window_start])
            window_end = np.min([window_end, total_num_window])

            aud_left_cell_fr = alignment_ds_fr[cell_idx, window_start:window_end, cond_1_trial_idx]
            aud_right_cell_fr = alignment_ds_fr[cell_idx, window_start:window_end, cond_2_trial_idx]

            aud_left_right_abs_diff = np.abs(np.mean(aud_left_cell_fr) - np.mean(aud_right_cell_fr))
            abs_diff_all_cell[cell_idx] = aud_left_right_abs_diff

            aud_left_right_sign = np.sign(np.mean(aud_left_cell_fr) - np.mean(aud_right_cell_fr))
            abs_diff_all_sign[cell_idx] = aud_left_right_sign

    return abs_diff_all_cell, abs_diff_all_sign


def cal_mean_abs_diff_at_max_window_single_trial_type():

    abs_diff_all_cell = np.zeros(np.shape(alignment_ds_fr))


    return


def cal_two_cond_max_loc_diff(smoothed_alignment_ds, alignment_ds_fr, test_cond='audLeftRight',
                              num_shuffle=1000, window_width_sec=0.05, bin_width=0.002,
                              search_only_post_stim=False, test_stat='mean', include_baseline=False):
    """

    Parameters
    ----------
    smoothed_alignment_ds : (xarray dataset)
        alignment ds smoothed
    alignment_ds_fr : (numpy ndarray)
        array of shape (numCell, numTrials, numTimeBins)
    test_cond : (str)
        what condition to test
        'audLeftRight' : difference in firing rate between left and right
    num_shuffle : (int)
        number of shuffles to perform for the permutation test
    search_only_post_stim : (bool)
        whether to search the location of maximum difference only in the post-stimulus period
    test_stat : (str)
        'mean' : in the window of maximum mean difference, do permutation and calculate mean difference
        'max': in the window of maximum nean difference, do permutatio and calculate max difference
    include_baseline : (bool)
        whether to include baseline activity
    Returns
    -------

    """
    if not test_cond in ('audLeftRight', 'visLeftRight', 'visLeftRightAudCenter', 'audLeftRightMultimodal',
                         'visLeftRightMultimodal', 'audLeftRightAll'):
        raise AssertionError
    window_width_frames = window_width_sec / bin_width
    window_width_back = int(np.floor(window_width_frames / 2))
    window_width_forward = int(np.floor(window_width_frames / 2))
    if test_cond == 'audLeftRight':
        cond_1_ds = smoothed_alignment_ds.where(((smoothed_alignment_ds['audDiff'] == -60) & (smoothed_alignment_ds['visDiff'] == 0)),
          drop=True)['firing_rate']
        cond_2_ds = smoothed_alignment_ds.where(((smoothed_alignment_ds['audDiff'] == 60) & (smoothed_alignment_ds['visDiff'] == 0)),
          drop=True)['firing_rate']
        cond_1_idx = np.where((smoothed_alignment_ds['audDiff'].isel(Cell=0, Time=0) == -60) & (smoothed_alignment_ds['visDiff'].isel(Cell=0, Time=0) == 0))[0]
        cond_2_idx = np.where((smoothed_alignment_ds['audDiff'].isel(Cell=0, Time=0) == 60) & (smoothed_alignment_ds['visDiff'].isel(Cell=0, Time=0) == 0))[0]
    elif test_cond == 'visLeftRight':
        cond_1_ds = smoothed_alignment_ds.where((~np.isfinite(smoothed_alignment_ds['audDiff']) & (smoothed_alignment_ds['visDiff'] < 0)),
          drop=True)['firing_rate']
        cond_2_ds = smoothed_alignment_ds.where((~np.isfinite(smoothed_alignment_ds['audDiff']) & (smoothed_alignment_ds['visDiff'] > 0)),
          drop=True)['firing_rate']
        cond_1_idx = np.where((smoothed_alignment_ds['visDiff'].isel(Cell=0, Time=0) < 0) & ~np.isfinite(smoothed_alignment_ds['audDiff'].isel(Cell=0, Time=0)))[0]
        cond_2_idx = np.where((smoothed_alignment_ds['visDiff'].isel(Cell=0, Time=0) > 0) & ~np.isfinite(smoothed_alignment_ds['audDiff'].isel(Cell=0, Time=0)))[0]
    elif test_cond == 'visLeftRightAudCenter':
        cond_1_ds = smoothed_alignment_ds.where(((smoothed_alignment_ds['audDiff'] == 0) & (smoothed_alignment_ds['visDiff'] < 0)),
          drop=True)['firing_rate']
        cond_2_ds = smoothed_alignment_ds.where(((smoothed_alignment_ds['audDiff'] == 0) & (smoothed_alignment_ds['visDiff'] > 0)),
          drop=True)['firing_rate']
        cond_1_idx = np.where((smoothed_alignment_ds['visDiff'].isel(Cell=0, Time=0) < 0) & (smoothed_alignment_ds['audDiff'].isel(Cell=0, Time=0) == 0))[0]
        cond_2_idx = np.where((smoothed_alignment_ds['visDiff'].isel(Cell=0, Time=0) > 0) & (smoothed_alignment_ds['audDiff'].isel(Cell=0, Time=0) == 0))[0]
    elif test_cond == 'visLeftRightMultimodal':
        cond_1_ds = smoothed_alignment_ds.where((smoothed_alignment_ds['visDiff'] == -0.8),
          drop=True)['firing_rate']
        cond_2_ds = smoothed_alignment_ds.where((smoothed_alignment_ds['visDiff'] == 0.8),
          drop=True)['firing_rate']
        cond_1_idx = np.where(smoothed_alignment_ds['visDiff'].isel(Cell=0, Time=0) == -0.8)[0]
        cond_2_idx = np.where(smoothed_alignment_ds['visDiff'].isel(Cell=0, Time=0) == 0.8)[0]
    elif test_cond == 'audLeftRightMultimodal':
        cond_1_ds = smoothed_alignment_ds.where(((smoothed_alignment_ds['audDiff'] == -60) & smoothed_alignment_ds['visDiff'].isin([-0.8, 0, 0.8])),
          drop=True)['firing_rate']
        cond_2_ds = smoothed_alignment_ds.where(((smoothed_alignment_ds['audDiff'] == 60) & smoothed_alignment_ds['visDiff'].isin([-0.8, 0, 0.8])),
          drop=True)['firing_rate']
        cond_1_idx = np.where((smoothed_alignment_ds['audDiff'].isel(Cell=0, Time=0) == -60) & smoothed_alignment_ds['visDiff'].isel(Cell=0, Time=0).isin([-0.8, 0, 0.8]))[0]
        cond_2_idx = np.where((smoothed_alignment_ds['audDiff'].isel(Cell=0, Time=0) == 60) & smoothed_alignment_ds['visDiff'].isel(Cell=0, Time=0).isin([-0.8, 0, 0.8]))[0]
    elif test_cond == 'audLeftRightAll':
        cond_1_ds = smoothed_alignment_ds.where((smoothed_alignment_ds['audDiff'] == -60),
          drop=True)['firing_rate']
        cond_2_ds = smoothed_alignment_ds.where((smoothed_alignment_ds['audDiff'] == 60),
          drop=True)['firing_rate']
        cond_1_idx = np.where(smoothed_alignment_ds['audDiff'].isel(Cell=0, Time=0) == -60)[0]
        cond_2_idx = np.where(smoothed_alignment_ds['audDiff'].isel(Cell=0, Time=0) == 60)[0]

    two_cond_abs_diff = np.abs(cond_1_ds.mean('Trial') - cond_2_ds.mean('Trial'))
    peri_event_time = smoothed_alignment_ds.isel(Trial=0).PeriEventTime

    if search_only_post_stim:
        post_stim_peri_event_time_idx = np.where(peri_event_time >= 0)[0]
        post_stim_peri_event_time = peri_event_time[post_stim_peri_event_time_idx]
        two_cond_abs_diff_post_stim = two_cond_abs_diff.sel(Time=post_stim_peri_event_time_idx)
        max_diff_time_post_stim_frame = two_cond_abs_diff_post_stim.argmax(dim='Time')
        max_diff_time = post_stim_peri_event_time[max_diff_time_post_stim_frame]
        max_diff_time_idxs = [np.where(peri_event_time == x)[0][0] for x in max_diff_time.values]
    else:
        max_diff_time_frame = two_cond_abs_diff.argmax(dim='Time')
        max_diff_time = peri_event_time[max_diff_time_frame]
        max_diff_time_idxs = max_diff_time_frame.values
    if test_stat == 'mean':
        mean_abs_diff, mean_sign = cal_mean_abs_diff_at_max_window(alignment_ds_fr, max_diff_time_idxs, cond_1_idx,
          cond_2_idx, window_width_back=window_width_back,
          window_width_forward=window_width_forward)
    elif test_stat == 'max':
        max_abs_diff = two_cond_abs_diff.max(dim='Time').values
        two_cond_diff = cond_1_ds.mean('Trial') - cond_2_ds.mean('Trial')
        max_abs_diff_idx = two_cond_abs_diff.argmax(dim='Time').values
        max_sign = np.zeros(len(max_abs_diff_idx))
        for cell_n, cell_max_diff_idx in enumerate(max_abs_diff_idx):
            max_sign[cell_n] = np.sign(two_cond_diff.isel(Cell=cell_n, Time=cell_max_diff_idx))
    else:
        print('Warning: no valid test stat provided.')

    all_trial_idx = np.concatenate([cond_1_idx, cond_2_idx])
    num_cond_1_trial = len(cond_1_idx)
    if test_stat == 'max':
        two_cond_max_diff_all_cell_all_shuffled = np.zeros((
         num_shuffle, np.shape(alignment_ds_fr)[0]))
    elif test_stat == 'mean':
        two_cond_abs_diff_all_cell_all_shuffled = np.zeros((
         num_shuffle, np.shape(alignment_ds_fr)[0]))
        two_cond_abs_diff_all_cell_all_shuffled_sign = np.zeros((
         num_shuffle, np.shape(alignment_ds_fr)[0]))

    for shuffle in tqdm(np.arange(num_shuffle)):
        trial_idx_shuffled = np.random.permutation(all_trial_idx)
        cond_1_idx_shuffled = trial_idx_shuffled[0:num_cond_1_trial]
        cond_2_idx_shuffled = trial_idx_shuffled[num_cond_1_trial:]
        cond_1_smoothed_shuffled = smoothed_alignment_ds['firing_rate'].isel(Trial=cond_1_idx_shuffled)
        cond_2_smoothed_shuffled = smoothed_alignment_ds['firing_rate'].isel(Trial=cond_2_idx_shuffled)
        two_cond_abs_diff_shuffled = np.abs(cond_1_smoothed_shuffled.mean('Trial') - cond_2_smoothed_shuffled.mean('Trial'))
        max_diff_time_frame_shuffled = two_cond_abs_diff_shuffled.argmax(dim='Time')
        max_diff_time_idxs_shuffled = max_diff_time_frame_shuffled.values
        if test_stat == 'mean':
            two_cond_abs_diff_all_cell_shuffled, shuffled_sign = cal_mean_abs_diff_at_max_window(alignment_ds_fr=alignment_ds_fr,
              max_diff_time_idxs=max_diff_time_idxs_shuffled,
              cond_1_trial_idx=cond_1_idx_shuffled,
              cond_2_trial_idx=cond_2_idx_shuffled,
              window_width_back=window_width_back,
              window_width_forward=window_width_forward)
            two_cond_abs_diff_all_cell_all_shuffled[shuffle, :] = two_cond_abs_diff_all_cell_shuffled
            two_cond_abs_diff_all_cell_all_shuffled_sign[shuffle, :] = shuffled_sign
        elif test_stat == 'max':
            two_cond_max_diff_all_cell_all_shuffled[shuffle, :] = two_cond_abs_diff_shuffled.max(dim='Time').values

    all_cell_two_cond_abs_diff_percentile = np.zeros(np.shape(alignment_ds_fr)[0])
    for cell in np.arange(np.shape(alignment_ds_fr)[0]):
        if test_stat == 'mean':
            cell_two_cond_abs_diff_percentile = sstats.percentileofscore(two_cond_abs_diff_all_cell_all_shuffled[:, cell], mean_abs_diff[cell])
        elif test_stat == 'max':
            cell_two_cond_abs_diff_percentile = sstats.percentileofscore(two_cond_max_diff_all_cell_all_shuffled[:, cell], max_abs_diff[cell])
            all_cell_two_cond_abs_diff_percentile[cell] = cell_two_cond_abs_diff_percentile

    if test_stat == 'mean':
        abs_diff = mean_abs_diff
        sign = mean_sign
        shuffled_matrix = two_cond_abs_diff_all_cell_all_shuffled
        shuffled_matrix = shuffled_matrix * two_cond_abs_diff_all_cell_all_shuffled_sign
    elif test_stat == 'max':
        abs_diff = max_abs_diff
        sign = max_sign
        shuffled_matrix = two_cond_max_diff_all_cell_all_shuffled

    if include_baseline:
        cond_1_and_2_ds = xr.concat([cond_1_ds, cond_2_ds], dim='Trial')
        baseline_time_period = [0, 0.7]
        cond_1_and_2_baseline_mean = cond_1_and_2_ds.where(((cond_1_and_2_ds['PeriEventTime'] >= baseline_time_period[0]) & (cond_1_and_2_ds['PeriEventTime'] <= baseline_time_period[1])),
          drop=True).mean([
         'Trial', 'Time']).values
    else:
        cond_1_and_2_baseline_mean = np.nan

    return (abs_diff, sign, all_cell_two_cond_abs_diff_percentile, max_diff_time, shuffled_matrix, cond_1_and_2_baseline_mean)






def randomly_flip_alignment_ds_fr(smoothed_alignment_ds_fr, og_alignment_ds_fr):
    """
    Randomly choose 50% of the trials and flip the ordering of the times.
    Smoothed firing rate is the data used to find the time point with maximum difference.
    Original firing rate is used in the actual statistical test.
    Parameters
    ----------
    smoothed_alignment_ds_fr : (numpy ndarray)
        smoothed firing rate / some kind of activity
        Dimensions: (cell, time, trial)
    og_alignment_ds_fr : (numpy ndarray)
        original firing rate

    Returns
    -------

    """
    randomly_flipped_alignment_ds_fr = smoothed_alignment_ds_fr.copy()
    randomly_flipped_og_alignment_ds_fr = og_alignment_ds_fr.copy()

    num_cell = np.shape(smoothed_alignment_ds_fr)[0]
    num_trial = np.shape(smoothed_alignment_ds_fr)[2]

    num_trial_to_flip = int(num_trial / 2)

    for cell in np.arange(num_cell):
        trials_to_flip = np.random.permutation(np.arange(num_trial))[0:num_trial_to_flip]

        for flip_trial in trials_to_flip:
            randomly_flipped_alignment_ds_fr[cell, :, flip_trial] = np.flip(smoothed_alignment_ds_fr[cell, :, flip_trial])
            randomly_flipped_og_alignment_ds_fr[cell, :, flip_trial] = np.flip(og_alignment_ds_fr[cell, :, flip_trial])

    return randomly_flipped_alignment_ds_fr, randomly_flipped_og_alignment_ds_fr


def cal_one_cond_before_after_max_loc_diff(smoothed_alignment_ds, alignment_ds_fr,
                                           test_cond='audOnOff', window_width_sec=0.05,
                                           bin_width=0.002, pre_stim_time_range=[-0.3, 0],
                                           num_shuffle=1000):
    """
    Performs permutation test and audio on/off response (by comparing firing rate
    before and after stimulus).

    Parameters
    ----------
    smoothed_alignment_ds : (xarray dataset)
    alignment_ds_fr : (numpy nd array)
    test_cond
    window_width_sec
    bin_width
    pre_stim_time_range
    num_shuffle

    Returns
    -------

    """

    if test_cond == 'audOnOff':

        cond_1_ds = smoothed_alignment_ds.where(
            (smoothed_alignment_ds['audDiff'] == -60) &
            (smoothed_alignment_ds['visDiff'] == 0), drop=True)['firing_rate']

        cond_2_ds = smoothed_alignment_ds.where(
            (smoothed_alignment_ds['audDiff'] == 60) &
            (smoothed_alignment_ds['visDiff'] == 0), drop=True)['firing_rate']

        aud_on_smoothed_ds = xr.concat([cond_1_ds, cond_2_ds], dim='Trial')

        pre_stim_time_range = [-0.3, -0.01]
        pre_stim_da = aud_on_smoothed_ds.where(
            (aud_on_smoothed_ds['PeriEventTime'] >= pre_stim_time_range[0]) &
            (aud_on_smoothed_ds['PeriEventTime'] <= pre_stim_time_range[1]), drop=True
        ).mean(['Time', 'Trial'])

        peri_event_time = aud_on_smoothed_ds.PeriEventTime.isel(Trial=0).values

        post_stim_da = aud_on_smoothed_ds.where(aud_on_smoothed_ds['PeriEventTime'] >= 0, drop=True).mean('Trial')

        post_minus_pre_stim_da = np.abs(post_stim_da - pre_stim_da)
        max_diff_time_frame = post_stim_da.Time[post_minus_pre_stim_da.argmax(dim='Time')]
        max_diff_time_idxs = max_diff_time_frame.values
        max_diff_time = peri_event_time[max_diff_time_idxs]

    # Calculate pre-post abs diff
    num_cell = np.shape(alignment_ds_fr)[0]
    total_num_window = np.shape(alignment_ds_fr)[1]

    post_stim_cell_vec = np.zeros(num_cell)
    pre_stim_cell_vec = np.zeros(num_cell)

    window_width_back = 12
    window_width_forward = 12

    baseline_window_start = np.argmin(np.abs(peri_event_time - pre_stim_time_range[0]))
    baseline_window_end = np.argmin(np.abs(peri_event_time - pre_stim_time_range[1]))

    zero_window_start = np.argmin(np.abs(peri_event_time - 0))

    shuffle_abs_diff_matrix = np.zeros((num_cell, num_shuffle))

    for cell_idx in tqdm(np.arange(num_cell)):
        window_start = max_diff_time_idxs[cell_idx] - window_width_back
        window_end = max_diff_time_idxs[cell_idx] + window_width_forward

        window_start = np.max([zero_window_start, window_start])  # restrict to 0 ms post-stim
        window_end = np.min([window_end, total_num_window])

        post_stim_fr_per_trial = alignment_ds_fr[cell_idx, window_start:window_end, :]
        post_stim_fr = np.mean(post_stim_fr_per_trial)
        post_stim_cell_vec[cell_idx] = post_stim_fr

        pre_stim_fr_per_trial = alignment_ds_fr[cell_idx, baseline_window_start:baseline_window_end, :]
        pre_stim_fr = np.mean(pre_stim_fr_per_trial)
        pre_stim_cell_vec[cell_idx] = pre_stim_fr

    mean_abs_diff = np.abs(post_stim_cell_vec - pre_stim_cell_vec)
    mean_sign = np.sign(post_stim_cell_vec - pre_stim_cell_vec)

    smoothed_aud_on_fr = aud_on_smoothed_ds.transpose('Cell', 'Time', 'Trial').values
    pre_stim_idx = np.where((peri_event_time >= pre_stim_time_range[0]) &
                            (peri_event_time <= pre_stim_time_range[1])
                            )[0]
    post_stim_idx = np.where((peri_event_time >= 0))[0]
    total_frame = len(post_stim_idx)

    for n_shuffle in np.arange(num_shuffle):

        # Randomly flip the smoothed firing rate matrix to find time of max diff for each cell
        smoothed_random_flipped_fr, og_aud_on_fr_flipped = randomly_flip_alignment_ds_fr(
            smoothed_aud_on_fr, alignment_ds_fr)

        # Take the mean over trials : smoothed firing rates
        smoothed_pre_stim_cell_vec_random = np.mean(smoothed_random_flipped_fr[:, pre_stim_idx, :], axis=(1, 2))
        smoothed_post_stim_matrix = np.mean(smoothed_random_flipped_fr[:, post_stim_idx, :], axis=2)

        # Take the mean over trials: original firing rates
        pre_stim_cell_vec_random = np.mean(og_aud_on_fr_flipped[:, pre_stim_idx, :], axis=(1, 2))
        post_stim_matrix = np.mean(og_aud_on_fr_flipped[:, post_stim_idx, :], axis=2)

        # Note that max is determined from smoothed version (so it's less noisy / basically binned)
        post_minus_pre_fr = smoothed_post_stim_matrix - np.expand_dims(smoothed_pre_stim_cell_vec_random, axis=1)
        post_stim_max_diff_frame = np.argmax(post_minus_pre_fr, axis=1)

        post_stim_cell_vec_random = np.zeros(num_cell)

        for cell_idx in np.arange(num_cell):
            cell_max_frame = post_stim_max_diff_frame[cell_idx]
            window_start = np.max([0, cell_max_frame - window_width_back])
            window_end = np.min([cell_max_frame + window_width_forward, total_frame])
            post_stim_cell_vec_random[cell_idx] = np.mean(
                post_stim_matrix[cell_idx, window_start:window_end], axis=0)

        # Mean absolute difference across each cell
        mean_abs_diff_random = np.abs(post_stim_cell_vec_random - pre_stim_cell_vec_random)
        shuffle_abs_diff_matrix[:, n_shuffle] = mean_abs_diff_random

    abs_diff_percentile = np.zeros(num_cell)

    for cell_idx in np.arange(num_cell):
        abs_diff_percentile[cell_idx] = sstats.percentileofscore(
            shuffle_abs_diff_matrix[cell_idx, :],
            mean_abs_diff[cell_idx]
        )

    return mean_abs_diff, mean_sign, abs_diff_percentile, max_diff_time


def test_abs_fr_diff(alignment_ds, conds_to_test=[
 'audLR', 'visLR', 'audOnOff'], search_only_post_stim=False, test_stat='mean', include_auc_score=False, auc_window_width=0.03,
 include_total_fr=True, num_shuffle=1000, include_baseline=False, sigma=30, window_width=50):
    """
    Test the maximum firing rate difference
    Relies on looping through cal_two_cond_max_loc_diff across stimulus condtions
    Parameters
    ----------
    alignment_ds
    conds_to_test
    search_only_post_stim
    test_stat : (str)
    include_baseline : (bool)
        whether to include baseline activity
    Returns
    -------

    """
    smooth_multiplier = 5
    alignment_ds_stacked = alignment_ds.stack(trialTime=['Trial', 'Time'])
    alignment_ds_stacked['firing_rate'] = (
     [
      'Cell', 'trialTime'],
     anaspikes.smooth_spikes((alignment_ds_stacked['firing_rate']),
       method='half_gaussian',
       sigma=sigma,
       window_width=window_width,
       custom_window=None))
    smoothed_alignment_ds = alignment_ds_stacked.unstack()
    alignment_ds_fr = alignment_ds['firing_rate'].values
    test_results_dict = dict()
    shuffle_test_results_dict = dict()

    if 'audLeftRightMultimodal' in conds_to_test or 'audLeftRightAll' in conds_to_test:
        if 'audLeftRightMultimodal' in conds_to_test:
            aud_lr_mean_abs_diff, aud_lr_mean_sign, all_cell_aud_lr_abs_diff_percentile, aud_lr_max_diff_time, shuffled_matrix, cond_1_and_2_baseline_mean = cal_two_cond_max_loc_diff(smoothed_alignment_ds, alignment_ds_fr,
              test_cond='audLeftRightMultimodal', num_shuffle=num_shuffle, search_only_post_stim=search_only_post_stim,
              test_stat=test_stat,
              include_baseline=include_baseline)
        elif 'audLeftRightAll' in conds_to_test:
            aud_lr_mean_abs_diff, aud_lr_mean_sign, all_cell_aud_lr_abs_diff_percentile, aud_lr_max_diff_time, shuffled_matrix, cond_1_and_2_baseline_mean = cal_two_cond_max_loc_diff(smoothed_alignment_ds, alignment_ds_fr,
              test_cond='audLeftRightAll', num_shuffle=num_shuffle, search_only_post_stim=search_only_post_stim,
              test_stat=test_stat,
              include_baseline=include_baseline)

        test_results_dict['audLRabsDiff'] = aud_lr_mean_abs_diff
        test_results_dict['audLRsign'] = aud_lr_mean_sign
        test_results_dict['audLRabsPercentile'] = all_cell_aud_lr_abs_diff_percentile
        test_results_dict['audLRmaxDiffTime'] = aud_lr_max_diff_time
        if include_baseline:
            test_results_dict['audLRbaseline'] = cond_1_and_2_baseline_mean
        else:
            shuffle_test_results_dict['audLRabsDiff'] = list()
            for n_shuffle in np.arange(np.shape(shuffled_matrix)[0]):
                shuffle_test_results_dict['audLRabsDiff'].extend(shuffled_matrix[n_shuffle, :])

        if include_auc_score:
            print('Calculating AUC score as well')
            auc_score_list = list()
            for n_cell, cell in tqdm(enumerate(smoothed_alignment_ds.Cell.values)):
                cell_aud_lr_max_diff_time = aud_lr_max_diff_time.values[n_cell]
                cell_ds = smoothed_alignment_ds.sel(Cell=cell)
                test_time_range = [cell_aud_lr_max_diff_time - auc_window_width / 2,
                 cell_aud_lr_max_diff_time + auc_window_width / 2]
                auc_score = cal_auc_at_window(cell_ds, test_cond='audLeftRightMultimodal', window_range=test_time_range)
                auc_score_list.append(auc_score)

            test_results_dict['audLRaucScore'] = auc_score_list

    if 'audLR' in conds_to_test or 'audLeftRight' in conds_to_test:
        aud_lr_mean_abs_diff, aud_lr_mean_sign, all_cell_aud_lr_abs_diff_percentile, aud_lr_max_diff_time, shuffled_matrix, cond_1_and_2_baseline_mean = cal_two_cond_max_loc_diff(smoothed_alignment_ds, alignment_ds_fr,
          test_cond='audLeftRight', num_shuffle=num_shuffle, search_only_post_stim=search_only_post_stim,
          test_stat=test_stat)
        test_results_dict['audLRabsDiff'] = aud_lr_mean_abs_diff
        test_results_dict['audLRsign'] = aud_lr_mean_sign
        test_results_dict['audLRabsPercentile'] = all_cell_aud_lr_abs_diff_percentile
        test_results_dict['audLRmaxDiffTime'] = aud_lr_max_diff_time
        if include_auc_score:
            print('Calculating AUC score as well')
            auc_score_list = list()
            for n_cell, cell in tqdm(enumerate(smoothed_alignment_ds.Cell.values)):
                cell_aud_lr_max_diff_time = aud_lr_max_diff_time.values[n_cell]
                cell_ds = smoothed_alignment_ds.sel(Cell=cell)
                test_time_range = [cell_aud_lr_max_diff_time - auc_window_width / 2,
                 cell_aud_lr_max_diff_time + auc_window_width / 2]
                auc_score = cal_auc_at_window(cell_ds, test_cond='audLeftRight', window_range=test_time_range)
                auc_score_list.append(auc_score)

            test_results_dict['audLRaucScore'] = auc_score_list


    if 'visLeftRightMultimodal' in conds_to_test:
            vis_lr_mean_abs_diff, vis_lr_mean_sign, all_cell_vis_lr_abs_diff_percentile, vis_lr_max_diff_time, shuffled_matrix, cond_1_and_2_baseline_mean = cal_two_cond_max_loc_diff(smoothed_alignment_ds, alignment_ds_fr,
              test_cond='visLeftRightMultimodal', num_shuffle=num_shuffle,
              search_only_post_stim=search_only_post_stim,
              test_stat=test_stat,
              include_baseline=include_baseline)
            test_results_dict['visLRabsDiff'] = vis_lr_mean_abs_diff
            test_results_dict['visLRsign'] = vis_lr_mean_sign
            test_results_dict['visLRabsPercentile'] = all_cell_vis_lr_abs_diff_percentile
            test_results_dict['vis_lr_max_diff_time'] = vis_lr_max_diff_time
            if include_baseline:
                test_results_dict['visLRbaseline'] = cond_1_and_2_baseline_mean
            shuffle_test_results_dict['visLRabsDiff'] = list()
            for n_shuffle in np.arange(np.shape(shuffled_matrix)[0]):
                shuffle_test_results_dict['visLRabsDiff'].extend(shuffled_matrix[n_shuffle, :])

            if include_auc_score:
                auc_score_list = list()
                print('Calculating AUC score as well')
                for n_cell, cell in tqdm(enumerate(smoothed_alignment_ds.Cell.values)):
                    cell_ds = smoothed_alignment_ds.sel(Cell=cell)
                    cell_vis_lr_max_diff_time = vis_lr_max_diff_time.values[n_cell]
                    test_time_range = [cell_vis_lr_max_diff_time - auc_window_width / 2,
                     cell_vis_lr_max_diff_time + auc_window_width / 2]
                    auc_score = cal_auc_at_window(cell_ds, test_cond='visLeftRightMultimodal', window_range=test_time_range)
                    auc_score_list.append(auc_score)

                test_results_dict['visLRaucScore'] = auc_score_list
    elif 'visLR' in conds_to_test or 'visLeftRight' in conds_to_test:
        vis_lr_mean_abs_diff, vis_lr_mean_sign, all_cell_vis_lr_abs_diff_percentile, vis_lr_max_diff_time, shuffled_matrix, c
        ond_1_and_2_baseline_mean = cal_two_cond_max_loc_diff(smoothed_alignment_ds, alignment_ds_fr,
          test_cond='visLeftRight', num_shuffle=num_shuffle, search_only_post_stim=search_only_post_stim,
          test_stat=test_stat)
        test_results_dict['visLRabsDiff'] = vis_lr_mean_abs_diff
        test_results_dict['visLRsign'] = vis_lr_mean_sign
        test_results_dict['visLRabsPercentile'] = all_cell_vis_lr_abs_diff_percentile
        test_results_dict['vis_lr_max_diff_time'] = vis_lr_max_diff_time
        if include_auc_score:
            auc_score_list = list()
            print('Calculating AUC score as well')
            for n_cell, cell in tqdm(enumerate(smoothed_alignment_ds.Cell.values)):
                cell_ds = smoothed_alignment_ds.sel(Cell=cell)
                cell_vis_lr_max_diff_time = vis_lr_max_diff_time.values[n_cell]
                test_time_range = [cell_vis_lr_max_diff_time - auc_window_width / 2,
                 cell_vis_lr_max_diff_time + auc_window_width / 2]
                auc_score = cal_auc_at_window(cell_ds, test_cond='visLeftRight', window_range=test_time_range)
                auc_score_list.append(auc_score)

            test_results_dict['visLRaucScore'] = auc_score_list
    elif 'visLeftRightAudCenter' in conds_to_test:
        vis_lr_mean_abs_diff, vis_lr_mean_sign, all_cell_vis_lr_abs_diff_percentile, vis_lr_max_diff_time = cal_two_cond_max_loc_diff(smoothed_alignment_ds, alignment_ds_fr,
          test_cond='visLeftRightAudCenter', num_shuffle=1000, search_only_post_stim=search_only_post_stim,
          test_stat=test_stat)
        test_results_dict['visLRabsDiff'] = vis_lr_mean_abs_diff
        test_results_dict['visLRsign'] = vis_lr_mean_sign
        test_results_dict['visLRabsPercentile'] = all_cell_vis_lr_abs_diff_percentile
        test_results_dict['vis_lr_max_diff_time'] = vis_lr_max_diff_time
    if 'audOnOff' in conds_to_test:
        aud_onoff_mean_abs_diff, aud_onoff_mean_sign, all_cell_aud_onoff_abs_diff_percentile, \
        aud_onoff_max_diff_time = cal_one_cond_before_after_max_loc_diff(smoothed_alignment_ds, alignment_ds_fr,
          test_cond='audOnOff', num_shuffle=num_shuffle)
        test_results_dict['audOnOffAbsDiff'] = aud_onoff_mean_abs_diff
        test_results_dict['audOnOffSign'] = aud_onoff_mean_sign
        test_results_dict['audOnOffAbsDiffPercentile'] = all_cell_aud_onoff_abs_diff_percentile
        test_results_dict['audOnoffMaxDiffTime'] = aud_onoff_max_diff_time


    test_results_df = pd.DataFrame.from_dict(test_results_dict)
    test_results_df['cellIdx'] = np.arange(len(test_results_df))
    test_results_df['shuffle'] = np.nan
    shuffle_results_df = pd.DataFrame.from_dict(shuffle_test_results_dict)
    shuffle_results_df['shuffle'] = np.repeat(np.arange(num_shuffle), len(test_results_df))
    shuffle_results_df['cellIdx'] = np.tile(np.arange(len(test_results_df)), num_shuffle)
    test_results_df = pd.concat([test_results_df, shuffle_results_df])

    return test_results_df

def test_abs_fr_diff_old(alignment_ds, conds_to_test=['audLR', 'visLR', 'audOnOff'],
                     search_only_post_stim=False, test_stat='mean',
                     include_auc_score=False, auc_window_width=0.03,
                     include_total_fr=True):
    """

    Parameters
    ----------
    alignment_ds
    conds_to_test
    search_only_post_stim
    test_stat : (str)
    Returns
    -------

    """
    # Smooth spikes (just for peak finding, not for stats)
    smooth_multiplier = 5
    alignment_ds_stacked = alignment_ds.stack(trialTime=['Trial', 'Time'])
    sigma = 3 * smooth_multiplier
    window_width = 20 * smooth_multiplier

    # also do some smoothing
    alignment_ds_stacked['firing_rate'] = (['Cell', 'trialTime'],
                                           anaspikes.smooth_spikes(
                                               alignment_ds_stacked['firing_rate'],
                                               method='half_gaussian',
                                               sigma=sigma, window_width=window_width,
                                               custom_window=None))
    smoothed_alignment_ds = alignment_ds_stacked.unstack()

    alignment_ds_fr = alignment_ds['firing_rate'].values

    test_results_dict = dict()

    if ('audLR' in conds_to_test) or ('audLeftRight' in conds_to_test):
        # Aud LR
        aud_lr_mean_abs_diff, aud_lr_mean_sign, all_cell_aud_lr_abs_diff_percentile, aud_lr_max_diff_time, shuffled_matrix = \
            cal_two_cond_max_loc_diff(smoothed_alignment_ds,
                                      alignment_ds_fr, test_cond='audLeftRight', num_shuffle=1000,
                                      search_only_post_stim=search_only_post_stim,
                                      test_stat=test_stat)

        test_results_dict['audLRabsDiff'] = aud_lr_mean_abs_diff
        test_results_dict['audLRsign'] = aud_lr_mean_sign
        test_results_dict['audLRabsPercentile'] = all_cell_aud_lr_abs_diff_percentile
        test_results_dict['audLRmaxDiffTime'] = aud_lr_max_diff_time

        if include_auc_score:
            print('Calculating AUC score as well')
            auc_score_list = list()
            for n_cell, cell in tqdm(enumerate(smoothed_alignment_ds.Cell.values)):

                cell_aud_lr_max_diff_time = aud_lr_max_diff_time.values[n_cell]
                cell_ds = smoothed_alignment_ds.sel(Cell=cell)
                test_time_range = [cell_aud_lr_max_diff_time - auc_window_width/2,
                                   cell_aud_lr_max_diff_time + auc_window_width/2]
                auc_score = cal_auc_at_window(cell_ds, test_cond='audLeftRight',
                                              window_range=test_time_range)
                auc_score_list.append(auc_score)

            test_results_dict['audLRaucScore'] = auc_score_list


    if ('visLR' in conds_to_test) or ('visLeftRight' in conds_to_test):
        # Do the same for Vis LR
        vis_lr_mean_abs_diff, vis_lr_mean_sign, all_cell_vis_lr_abs_diff_percentile, vis_lr_max_diff_time, shuffled_matrix = \
            cal_two_cond_max_loc_diff(smoothed_alignment_ds,
                                      alignment_ds_fr, test_cond='visLeftRight', num_shuffle=1000,
                                      search_only_post_stim=search_only_post_stim,
                                      test_stat=test_stat)

        test_results_dict['visLRabsDiff'] = vis_lr_mean_abs_diff
        test_results_dict['visLRsign'] = vis_lr_mean_sign
        test_results_dict['visLRabsPercentile'] = all_cell_vis_lr_abs_diff_percentile
        test_results_dict['vis_lr_max_diff_time'] = vis_lr_max_diff_time

        if include_auc_score:
            auc_score_list = list()
            print('Calculating AUC score as well')
            for n_cell, cell in tqdm(enumerate(smoothed_alignment_ds.Cell.values)):

                cell_ds = smoothed_alignment_ds.sel(Cell=cell)
                cell_vis_lr_max_diff_time = vis_lr_max_diff_time.values[n_cell]
                test_time_range = [cell_vis_lr_max_diff_time - auc_window_width/2,
                                   cell_vis_lr_max_diff_time + auc_window_width/2]
                auc_score = cal_auc_at_window(cell_ds, test_cond='visLeftRight',
                                              window_range=test_time_range)
                auc_score_list.append(auc_score)

            test_results_dict['visLRaucScore'] = auc_score_list

    elif ('visLeftRightAudCenter' in conds_to_test):
        vis_lr_mean_abs_diff, vis_lr_mean_sign, all_cell_vis_lr_abs_diff_percentile, vis_lr_max_diff_time = \
            cal_two_cond_max_loc_diff(smoothed_alignment_ds,
                                      alignment_ds_fr, test_cond='visLeftRightAudCenter', num_shuffle=1000,
                                      search_only_post_stim=search_only_post_stim,
                                      test_stat=test_stat)

        test_results_dict['visLRabsDiff'] = vis_lr_mean_abs_diff
        test_results_dict['visLRsign'] = vis_lr_mean_sign
        test_results_dict['visLRabsPercentile'] = all_cell_vis_lr_abs_diff_percentile
        test_results_dict['vis_lr_max_diff_time'] = vis_lr_max_diff_time

    # Aud On/OFF
    if 'audOnOff' in conds_to_test:
        aud_onoff_mean_abs_diff, aud_onoff_mean_sign, all_cell_aud_onoff_abs_diff_percentile, \
        aud_onoff_max_diff_time = cal_one_cond_before_after_max_loc_diff(smoothed_alignment_ds,
                                      alignment_ds_fr, test_cond='audOnOff',
                                      num_shuffle=1000)

        test_results_dict['audOnOffAbsDiff'] = aud_onoff_mean_abs_diff
        test_results_dict['audOnOffSign'] = aud_onoff_mean_sign
        test_results_dict['audOnOffAbsDiffPercentile'] = all_cell_aud_onoff_abs_diff_percentile
        test_results_dict['audOnoffMaxDiffTime'] = aud_onoff_max_diff_time


    test_results_df = pd.DataFrame.from_dict(test_results_dict)

    return test_results_df


def windowed_nonparametric_test(alignment_ds, conds_to_test=['audLR', 'visLR', 'audOnOff'],
                                search_only_post_stim=False, test_window_step_sec=0.03,
                                test_window_width_sec=0.1, test_type='mannU',
                                vis_left_levels=[-0.8, -0.4, -0.2, -0.1],
                                vis_right_levels=[0.1, 0.2, 0.4, 0.8],
                                smooth_spikes=True):
    """

    Parameters
    ----------
    alignment_ds : (xarray dataset)
    conds_to_test : (list of str)
    search_only_post_stim : (bool)
    test_window_step_sec : (float)
    test_window_width_sec : (float)

    Returns
    -------

    """

    if smooth_spikes:
        # Smooth spikes (just for peak finding, not for stats)
        smooth_multiplier = 5
        alignment_ds_stacked = alignment_ds.stack(trialTime=['Trial', 'Time'])
        sigma = 3 * smooth_multiplier
        window_width = 20 * smooth_multiplier

        # also do some smoothing
        alignment_ds_stacked['firing_rate'] = (['Cell', 'trialTime'],
                                               anaspikes.smooth_spikes(
                                                   alignment_ds_stacked['firing_rate'],
                                                   method='half_gaussian',
                                                   sigma=sigma, window_width=window_width,
                                                   custom_window=None))

        smoothed_alignment_ds = alignment_ds_stacked.unstack()

        peri_event_time = smoothed_alignment_ds.PeriEventTime.isel(Trial=0).values

    else:
        smoothed_alignment_ds = alignment_ds
        peri_event_time = alignment_ds.PeriEventTime.values

    if ('audLR' in conds_to_test) or ('audLeftRight' in conds_to_test):

        cond_1_ds = smoothed_alignment_ds.where(
            (smoothed_alignment_ds['audDiff'] == -60) &
            (smoothed_alignment_ds['visDiff'] == 0), drop=True)['firing_rate']

        cond_2_ds = smoothed_alignment_ds.where(
            (smoothed_alignment_ds['audDiff'] == 60) &
            (smoothed_alignment_ds['visDiff'] == 0), drop=True)['firing_rate']

        cell_list, window_peri_event_time_list, aud_lr_stat = \
            cal_two_cond_sliding_window_diff_all_cells(cond_1_ds, cond_2_ds, peri_event_time,
                                               test_window_step_sec, test_window_width_sec,
                                                test_type=test_type)

    if ('visLR' in conds_to_test) or ('visLeftRight' in conds_to_test):

        cond_1_ds = smoothed_alignment_ds.where(
            ~np.isfinite(smoothed_alignment_ds['audDiff']) &
            (smoothed_alignment_ds['visDiff'].isin(vis_left_levels)), drop=True
        )['firing_rate']

        cond_2_ds = smoothed_alignment_ds.where(
            ~np.isfinite(smoothed_alignment_ds['audDiff']) &
            (smoothed_alignment_ds['visDiff'].isin(vis_right_levels)), drop=True
        )['firing_rate']

        cell_list, window_peri_event_time_list, vis_lr_stat = \
            cal_two_cond_sliding_window_diff_all_cells(cond_1_ds, cond_2_ds, peri_event_time,
                                               test_window_step_sec, test_window_width_sec,
                                                test_type=test_type)

    if ('audOnOff' in conds_to_test):

        cond_1a_ds = smoothed_alignment_ds.where(
            (smoothed_alignment_ds['audDiff'] == -60) &
            (smoothed_alignment_ds['visDiff'] == 0), drop=True)['firing_rate']

        cond_1b_ds = smoothed_alignment_ds.where(
            (smoothed_alignment_ds['audDiff'] == 60) &
            (smoothed_alignment_ds['visDiff'] == 0), drop=True)['firing_rate']

        cond_1_ds = xr.concat([cond_1a_ds, cond_1b_ds], dim='Trial')

        pre_stim_time_range = [-0.3, -0.01]

        cond_2_ds = cond_1_ds.where(
            (cond_1_ds['PeriEventTime'] >= pre_stim_time_range[0]) &
            (cond_1_ds['PeriEventTime'] <= pre_stim_time_range[1]), drop=True
        ).mean(['Time'])

        cell_list, window_peri_event_time_list, aud_on_stat = \
            cal_two_cond_sliding_window_diff_all_cells(cond_1_ds, cond_2_ds, peri_event_time,
                                               test_window_step_sec, test_window_width_sec,
                                               test_type=test_type)

    test_results_df = pd.DataFrame.from_dict({'Cell': cell_list,
                                              'windowEnd': window_peri_event_time_list,
                                              'audLRtestStat': aud_lr_stat,
                                              'visLRtestStat': vis_lr_stat,
                                              'audOnTestStat': aud_on_stat})

    return test_results_df


def cal_two_cond_sliding_window_diff_all_cells(cond_1_ds, cond_2_ds, peri_event_time,
                                               test_window_step_sec, test_window_width_sec,
                                               test_type='mannU'):


    cell_list = list()
    window_peri_event_time_list = list()
    test_stat_per_window_list = list()

    for cell in tqdm(cond_1_ds.Cell.values):
        cell_cond_1 = cond_1_ds.sel(Cell=cell).values
        cell_cond_2 = cond_2_ds.sel(Cell=cell).values

        test_stat_per_window, peri_event_time_window_end = cal_two_cond_sliding_window_diff(
            cell_fr_cond_1=cell_cond_1,
            cell_fr_cond_2=cell_cond_2, peri_event_time=peri_event_time,
            test_window_step_sec=test_window_step_sec,
            test_window_width_sec=test_window_width_sec,
            test_type=test_type)

        window_peri_event_time_list.extend(peri_event_time_window_end)
        cell_list.extend(np.repeat(cell, len(peri_event_time_window_end)))
        test_stat_per_window_list.extend(test_stat_per_window)

    return cell_list, window_peri_event_time_list, test_stat_per_window_list


def cal_two_cond_sliding_window_diff(cell_fr_cond_1, cell_fr_cond_2, peri_event_time,
                                          test_window_step_sec=0.03, test_window_width_sec=0.1,
                                     test_type='mannU'):
    """

    Parameters
    ----------
    cell_fr_cond_1 : (numpy ndarray)
        matrix with dimensions trial x time bins
    cell_fr_cond_2 : (numpy ndarray)
         matrix with dimensions trial x time bins
    peri_event_time
    test_window_step_sec : (float)
    test_window_width_sec : (float)
    test_type : (str)

    Returns
    -------

    """

    total_num_bins = len(peri_event_time)
    time_range = peri_event_time[-1] - peri_event_time[0]
    sec_per_bin = time_range / total_num_bins
    test_window_step_bins = int(test_window_step_sec / sec_per_bin)
    test_window_size_bins = int(test_window_width_sec / sec_per_bin)

    window_start_idxs = np.arange(0, total_num_bins - test_window_size_bins,
                                  test_window_step_bins)

    window_end_idxs = window_start_idxs + test_window_size_bins
    peri_event_time_window_end = peri_event_time[window_end_idxs]

    test_stat_per_window = np.zeros(len(window_start_idxs))

    if (test_type == 'LRdecoding') or (test_type == 'LRdecodingRel'):
        num_cond_1_trials = np.shape(cell_fr_cond_1)[0]
        num_cond_2_trials = np.shape(cell_fr_cond_2)[0]
        clf = sklinear.LogisticRegression(solver='lbfgs')
        y = np.concatenate([np.repeat(0, num_cond_1_trials),
                            np.repeat(1, num_cond_2_trials)])

        if cell_fr_cond_2.ndim == 2:
            X_all_time_bins = np.concatenate([cell_fr_cond_1, cell_fr_cond_2])
        else:
            X_all_time_bins = cell_fr_cond_1

    for n_window, (win_start, win_end) in enumerate(zip(window_start_idxs, window_end_idxs)):

        cond_1_windowed_fr = np.mean(cell_fr_cond_1[:, win_start:win_end], axis=1)

        if cell_fr_cond_2.ndim == 2:
            cond_2_windowed_fr = np.mean(cell_fr_cond_2[:, win_start:win_end], axis=1)
        else:
            cond_2_windowed_fr = cell_fr_cond_2

        if np.array_equal(cond_1_windowed_fr, cond_2_windowed_fr):
            # print('Arrays are equal, skipping tests')
            test_stat_per_window[n_window] = np.nan
        elif test_type == 'mannU':
            if (len(np.unique(cond_1_windowed_fr)) == 1) & (len(np.unique(cond_2_windowed_fr)) == 1):
                test_stat_per_window[n_window] = np.nan
            else:
                test_stat, p_val = sstats.mannwhitneyu(cond_1_windowed_fr,
                                                       cond_2_windowed_fr)
                test_stat_per_window[n_window] = p_val
        elif test_type == 'ks2samp':
            test_stat, p_val = sstats.ks_2samp(cond_1_windowed_fr, cond_2_windowed_fr)
            test_stat_per_window[n_window] = p_val
        elif test_type == 'diff':
            test_stat_per_window[n_window] = np.mean(cond_1_windowed_fr) - np.mean(cond_2_windowed_fr)
        elif test_type == 'LRdecoding':
            if cell_fr_cond_2.ndim == 2:
                X_binned = X_all_time_bins[:, win_start:win_end]
                X = np.mean(X_binned, axis=1).reshape(-1, 1)
            else:
                X_binned = X_all_time_bins[:, win_start:win_end]
                X_cond_1 = np.mean(X_binned, axis=1)
                X = np.concatenate([X_cond_1, cell_fr_cond_2]).reshape(-1, 1)
            test_stat_per_window[n_window] = np.mean(sklselect.cross_val_score(clf, X, y, cv=5))
        elif test_type == 'LRdecodingRel':
            if cell_fr_cond_2.ndim == 2:
                X_binned = X_all_time_bins[:, win_start:win_end]
                X = np.mean(X_binned, axis=1).reshape(-1, 1)
            else:
                X_binned = X_all_time_bins[:, win_start:win_end]
                X_cond_1 = np.mean(X_binned, axis=1)
                X = np.concatenate([X_cond_1, cell_fr_cond_2]).reshape(-1, 1)

            mean_acc = np.mean(sklselect.cross_val_score(clf, X, y, cv=5))
            y_unique, y_counts = np.unique(y, return_counts=True)
            y_bias = np.max(y_counts) / len(y)
            mean_acc_rel_bias = (mean_acc - y_bias) / (1 - y_bias)
            test_stat_per_window[n_window] = mean_acc_rel_bias

        elif test_type == 'permutationTest':
            cond_1_cond_2_abs_diff = np.abs(np.mean(cond_1_windowed_fr) - np.mean(cond_2_windowed_fr))
            both_cond_windowed_fr = np.concatenate([cond_1_windowed_fr, cond_2_windowed_fr])
            num_cond_1 = len(cond_1_windowed_fr)
            num_cond_2 = len(cond_2_windowed_fr)
            num_shuffle = 1000
            shuffled_abs_diff = np.zeros(num_shuffle)
            for shuffle in np.arange(num_shuffle):
                random_idx = np.random.permutation(np.arange(num_cond_1+num_cond_2))
                cond_1_idx = random_idx[:num_cond_1]
                cond_2_idx = random_idx[num_cond_1:]
                cond_1_shuffled = both_cond_windowed_fr[cond_1_idx]
                cond_2_shuffled = both_cond_windowed_fr[cond_2_idx]
                shuffled_abs_diff[shuffle] = np.abs(np.mean(cond_1_shuffled) - np.mean(cond_2_shuffled))

            test_stat_per_window[n_window] = sstats.percentileofscore(shuffled_abs_diff,
                                                                      cond_1_cond_2_abs_diff)


    return test_stat_per_window, peri_event_time_window_end


def permutation_F_stat_test(all_cell_all_aud_vis_cond_df, anova_type=3, num_permutation=1000):
    """

    Parameters
    ----------
    all_cell_all_aud_vis_cond_df
    num_permutation (int)
    anova_type (int)
        which type of ANOVA test to perform (type 3 is most sensible in most cases)
    Returns
    -------

    """

    cell_list = list()
    permutation_list = list()
    vis_effect_list = list()
    aud_effect_list = list()
    aud_vis_interaction_effect_list = list()

    formula = 'firing_rate ~ C(visCond) + C(audCond) + C(visCond):C(audCond)'

    for n_permute in tqdm(np.arange(num_permutation)):

        # all_cell_all_aud_vis_cond_permutated_list = list()

        for cell in np.unique(all_cell_all_aud_vis_cond_df['Cell']):
            cell_df = all_cell_all_aud_vis_cond_df.loc[
                all_cell_all_aud_vis_cond_df['Cell'] == cell
                ]

            cell_df['firing_rate'] = np.random.permutation(cell_df['firing_rate'])

            cell_model = ols(formula, cell_df).fit()
            aov_table = anova_lm(cell_model, typ=anova_type)

            cell_list.append(cell)
            permutation_list.append(n_permute)

            aud_effect_list.append(aov_table['F']['C(audCond)'])
            vis_effect_list.append(aov_table['F']['C(visCond)'])
            aud_vis_interaction_effect_list.append(aov_table['F']['C(visCond):C(audCond)'])

    F_test_result_df = pd.DataFrame.from_dict({
        'Cell': cell_list,
        'permutation': permutation_list,
        'visF': vis_effect_list,
        'audF': aud_effect_list,
        'interactionF': aud_vis_interaction_effect_list
    })

    return F_test_result_df


def cal_auc_at_window(cell_ds, test_cond='audLeftRight', window_range=[0, 0.1]):
    """

    Parameters
    ----------
    cell_ds
    test_cond
    window_range

    Returns
    -------

    """

    cell_ds_windowed_mean = cell_ds.where(
        (cell_ds['PeriEventTime'] >= window_range[0]) &
        (cell_ds['PeriEventTime'] <= window_range[1]), drop=True
    ).mean('Time')

    if test_cond == 'audLeftRight':

        cond_1_ds = cell_ds_windowed_mean.where(
            (cell_ds_windowed_mean['audDiff'] == -60) &
            (cell_ds_windowed_mean['visDiff'] == 0), drop=True)['firing_rate']

        cond_2_ds = cell_ds_windowed_mean.where(
            (cell_ds_windowed_mean['audDiff'] == 60) &
            (cell_ds_windowed_mean['visDiff'] == 0), drop=True)['firing_rate']

        cond_1_idx = np.where(
            (cell_ds_windowed_mean['audDiff'] == -60) &
            (cell_ds_windowed_mean['visDiff'] == 0))[0]

        cond_2_idx = np.where(
            (cell_ds_windowed_mean['audDiff'] == 60) &
            (cell_ds_windowed_mean['visDiff'] == 0))[0]

    elif test_cond == 'visLeftRight':

        cond_1_ds = cell_ds_windowed_mean.where(
            ~np.isfinite(cell_ds_windowed_mean['audDiff']) &
            (cell_ds_windowed_mean['visDiff'] < 0), drop=True
        )['firing_rate']

        cond_2_ds = cell_ds_windowed_mean.where(
            ~np.isfinite(cell_ds_windowed_mean['audDiff']) &
            (cell_ds_windowed_mean['visDiff'] > 0), drop=True
        )['firing_rate']

        cond_1_idx = np.where(
            (cell_ds_windowed_mean['visDiff'] < 0) &
            ~np.isfinite(cell_ds_windowed_mean['audDiff']))[0]

        cond_2_idx = np.where(
            (cell_ds_windowed_mean['visDiff'] > 0) &
            ~np.isfinite(cell_ds_windowed_mean['audDiff']))[0]

    else:
        print('Error: no valid test cond specified, returning None')
        return None

    X = np.concatenate([cond_1_ds.values, cond_2_ds.values]).reshape(-1, 1)
    y = np.concatenate([np.repeat(0, len(cond_1_idx)), np.repeat(1, len(cond_2_idx))])

    model = sklinear.LogisticRegression(solver='lbfgs')
    model.fit(X, y)

    # compute class probability
    probs = model.predict_proba(X)

    # keep only 'positive class', ie. y = 1
    probs_1 = probs[:, 1]

    auc = sklmetrics.roc_auc_score(y, probs_1)
    auc_score = 2 * (auc - 0.5)

    return auc_score


def passive_time_of_separation_test(test_df, alignment_folder,
                                    test_modality='visLR', test_type='LRdecoding',
                                    test_stat_threshold=0.5, cell_idx_method='name',
                                    test_window_step_sec=0.03, test_window_width_sec=0.1,
                                    smooth_spikes=True):

    first_threshold_crossing_time_post_stim_list = list()

    for df_index, cell_df in tqdm(test_df.iterrows()):

        subject = cell_df['subjectRef']
        exp = cell_df['expRef']
        brain_region = cell_df['brainRegion']
        if cell_idx_method == 'name':
            cell_idx = cell_df.name
        else:
            cell_idx = cell_df['Cell']

        aligned_event = 'stimOnTime'

        alignment_ds = pephys.load_subject_exp_alignment_ds(
            alignment_folder=alignment_folder,
            subject_num=subject,
            exp_num=exp,
            target_brain_region=brain_region,
            aligned_event=aligned_event,
            alignment_file_ext='.nc')

        cell_ds = alignment_ds.isel(Cell=cell_idx)

        if smooth_spikes:
            smooth_multiplier = 5
            cell_ds_stacked = cell_ds.stack(trialTime=['Trial', 'Time'])
            sigma = 3 * smooth_multiplier
            window_width = 20 * smooth_multiplier
            # also do some smoothing
            cell_ds_stacked['firing_rate'] = (['trialTime'],
                                              anaspikes.smooth_spikes(
                                                  cell_ds_stacked['firing_rate'],
                                                  method='half_gaussian',
                                                  sigma=sigma, window_width=window_width,
                                                  custom_window=None))

            smoothed_cell_ds = cell_ds_stacked.unstack()
            peri_event_time = smoothed_cell_ds.PeriEventTime.isel(Trial=0).values
        else:
            smoothed_cell_ds = cell_ds
            peri_event_time = smoothed_cell_ds.PeriEventTime

        if test_modality == 'visLR':

            cond_1_ds = smoothed_cell_ds.where(
                (~np.isfinite(smoothed_cell_ds['audDiff'])) &
                (smoothed_cell_ds['visDiff'] == -0.8), drop=True
            )['firing_rate']

            cond_2_ds = smoothed_cell_ds.where(
                (~np.isfinite(smoothed_cell_ds['audDiff'])) &
                (smoothed_cell_ds['visDiff'] == 0.8), drop=True
            )['firing_rate']

        elif test_modality == 'audLR':

            cond_1_ds = smoothed_cell_ds.where(
                (smoothed_cell_ds['audDiff'] == -60) &
                (smoothed_cell_ds['visDiff'] == 0), drop=True)['firing_rate']

            cond_2_ds = smoothed_cell_ds.where(
                (smoothed_cell_ds['audDiff'] == 60) &
                (smoothed_cell_ds['visDiff'] == 0), drop=True)['firing_rate']

        elif test_modality == 'audOnOff':

            cond_1a_ds = smoothed_cell_ds.where(
                (smoothed_cell_ds['audDiff'] == -60) &
                (smoothed_cell_ds['visDiff'] == 0), drop=True)['firing_rate']

            cond_1b_ds = smoothed_cell_ds.where(
                (smoothed_cell_ds['audDiff'] == 60) &
                (smoothed_cell_ds['visDiff'] == 0), drop=True)['firing_rate']

            cond_1_ds = xr.concat([cond_1a_ds, cond_1b_ds], dim='Trial')

            pre_stim_time_range = [-0.3, -0.01]

            cond_2_ds = cond_1_ds.where(
                (cond_1_ds['PeriEventTime'] >= pre_stim_time_range[0]) &
                (cond_1_ds['PeriEventTime'] <= pre_stim_time_range[1]), drop=True
            ).mean(['Time'])

        cell_fr_cond_1 = cond_1_ds.values
        cell_fr_cond_2 = cond_2_ds.values

        if not smooth_spikes:
            cell_fr_cond_1 = cell_fr_cond_1.T
            cell_fr_cond_2 = cell_fr_cond_2.T

        test_stat_per_window, peri_event_time_window_end = cal_two_cond_sliding_window_diff(
            cell_fr_cond_1, cell_fr_cond_2, peri_event_time,
            test_window_step_sec=test_window_step_sec, test_window_width_sec=test_window_width_sec,
            test_type=test_type)

        if test_type in ['LRdecoding', 'permutationTest']:
            threshold_crossing_frame = np.where(test_stat_per_window >= test_stat_threshold)[0]
        else:
            threshold_crossing_frame = np.where(test_stat_per_window <= test_stat_threshold)[0]


        if len(threshold_crossing_frame) > 0:
            threshold_crossing_time = peri_event_time_window_end[threshold_crossing_frame]

            # Get only threshold crossing time after stimulus
            threshold_crossing_time_post_stim = threshold_crossing_time[threshold_crossing_time >= 0]

            if len(threshold_crossing_time_post_stim) > 0:
                first_threshold_crossing_time_post_stim = threshold_crossing_time_post_stim[0]
            else:
                first_threshold_crossing_time_post_stim = np.nan
        else:
            first_threshold_crossing_time_post_stim = np.nan

        first_threshold_crossing_time_post_stim_list.append(first_threshold_crossing_time_post_stim)

    return first_threshold_crossing_time_post_stim_list


def cal_dprime(stim_aligned_ds, comparision_type='audLeftRight',
               activity_name='zscored_spikes_rel_pre_stim_smoothed',
               dprime_metric='ave_std', timing='max_window', max_window_width_sec=0.1,
               loading_bar=False, min_trial_per_cond=5, fixed_window=None):
    """
    Code to calculate the discrimination index, or dprime (d') between two conditions for each neuron.
    fixed_window : (list or None)
        if None, then window is defined per neuron by taking time of maximum difference in the smoothed psth

    """
    cell_d_prime = []
    peri_event_time = stim_aligned_ds.PeriEventTime.values
    disable = not loading_bar

    for cell in tqdm(np.arange(len(stim_aligned_ds.Cell)), disable=disable):
        cell_ds = stim_aligned_ds.isel(Cell=cell)
        if comparision_type == 'audLeftRight':
            cond_1_ds = cell_ds.where((cell_ds['audDiff'] == -60), drop=True)
            cond_2_ds = cell_ds.where((cell_ds['audDiff'] == 60), drop=True)
        elif comparision_type == 'visLeftRight':
            cond_1_ds = cell_ds.where((cell_ds['visDiff'] <= -0.4), drop=True)
            cond_2_ds = cell_ds.where((cell_ds['visDiff'] >= 0.4), drop=True)
        elif comparision_type == 'audOnOff':
            cond_1_ds = cell_ds.where((cell_ds['audDiff'] == np.inf), drop=True)
            cond_2_ds = cell_ds.where((cell_ds['audDiff'] != np.inf), drop=True)
        elif comparision_type == 'moveLeftRight':
            cond_1_ds = cell_ds.where((cell_ds['responseMade'] == 1), drop=True)
            cond_2_ds = cell_ds.where((cell_ds['responseMade'] == 2), drop=True)

        if (len(cond_1_ds.Trial) <= min_trial_per_cond) | (len(cond_2_ds.Trial) <= min_trial_per_cond):
            cell_d_prime.append(np.nan)
        else:
            if fixed_window is None:
                cond_1_mean_fr = cond_1_ds[activity_name].mean('Trial')
                cond_2_mean_fr = cond_2_ds[activity_name].mean('Trial')
                post_stim_time_idx = np.where(peri_event_time >= 0)[0]
                cond_1_mean_fr_post_stim = cond_1_mean_fr[post_stim_time_idx].values
                cond_2_mean_fr_post_stim = cond_2_mean_fr[post_stim_time_idx].values
                time_idx_max_diff = np.argmax(np.abs(cond_1_mean_fr_post_stim - cond_2_mean_fr_post_stim))
                time_of_max_diff = peri_event_time[(time_idx_max_diff + post_stim_time_idx[0])]
                max_window_start = time_of_max_diff - max_window_width_sec / 2
                max_window_end = time_of_max_diff + max_window_width_sec / 2
            elif timing == 'max_window':
                max_window_start = fixed_window[0]
                max_window_end = fixed_window[1]
            else:
                print('Warning: no valid window method specified')

            max_window_idx = np.where((peri_event_time >= max_window_start) & (max_window_start <= max_window_end))[0]
            cond_1_trial_fr_around_window = cond_1_ds.isel(Time=(slice(max_window_idx[0], max_window_idx[(-1)])))[activity_name].mean('Time')
            cond_2_trial_fr_around_window = cond_2_ds.isel(Time=(slice(max_window_idx[0], max_window_idx[(-1)])))[activity_name].mean('Time')
            u_1 = np.mean(cond_1_trial_fr_around_window)
            u_2 = np.mean(cond_2_trial_fr_around_window)
            ss_1 = np.std(cond_1_trial_fr_around_window)
            ss_2 = np.std(cond_2_trial_fr_around_window)
            d_prime = (u_1 - u_2) / ((ss_1 + ss_2) / 2)
            cell_d_prime.append(d_prime.values)

    dprime_df = pd.DataFrame.from_dict(
        {'Cell_idx':np.arange(len(stim_aligned_ds.Cell)),
         'd_prime':cell_d_prime})

    return dprime_df


def cal_pairwise_decoding_sig(all_exp_classification_results_df, acc_metric='rel_score',
                              exp_var_name='Exp', brain_region_var_name='brainRegion',
                              cv_ave_method='mean',
                              brain_order_to_compare=['MOs', 'ACA', 'ORB', 'ILA', 'PL', 'OLF'],
                              correction_method='BH'):

    brain_grouped_dprime = []
    exp_and_brain_region_grouped_df = all_exp_classification_results_df.groupby(
        [exp_var_name, brain_region_var_name]).agg(
        cv_ave_method).reset_index()

    all_exp_grouped_accuracy = exp_and_brain_region_grouped_df.groupby('brainRegion').agg(cv_ave_method)

    brain_region_order = all_exp_grouped_accuracy.sort_values(acc_metric, ascending=False).reset_index()[
        'brainRegion'].values

    # subset brain region used in paper
    exp_and_brain_region_grouped_df = exp_and_brain_region_grouped_df.loc[
        exp_and_brain_region_grouped_df[brain_region_var_name].isin(['MOs', 'PL', 'ORB', 'ACA', 'ILA', 'OLF'])
    ]

    brain_new_name = list(string.ascii_lowercase)[0:len(brain_order_to_compare)]
    # exp_and_brain_region_grouped_df[brain_region_var_name] = exp_and_brain_region_grouped_df[brain_region_var_name].astype(str)
    map_to_new_name = dict(zip(brain_order_to_compare, brain_new_name))

    # Map to new name to reinforce order of comparison I want
    exp_and_brain_region_grouped_df[brain_region_var_name] = [map_to_new_name[x] for x in
                                                              exp_and_brain_region_grouped_df[
                                                                  brain_region_var_name].values]

    # metric_name = 'abs_d_prime'
    # brain_region_var_name = 'brain_region'

    new_brain_order = [map_to_new_name[x] for x in brain_order_to_compare]
    for brain_region in new_brain_order:
        brain_region_df = exp_and_brain_region_grouped_df.loc[
            exp_and_brain_region_grouped_df[brain_region_var_name] == brain_region
            ]

        brain_region_dprime = brain_region_df[acc_metric]

        brain_grouped_dprime.append(brain_region_dprime.values)

    stats, p_val = sstats.f_oneway(*brain_grouped_dprime)

    print('One way anova p-value: %.4f' % p_val)
    print(p_val)

    print('F = %.4f' % stats)
    print('One way anova p-value: %.4f' % p_val)
    print(p_val)

    brian_region_names = np.unique(exp_and_brain_region_grouped_df[brain_region_var_name]).tolist()

    group_1_list = []
    group_2_list = []
    p_val_list = []
    mean_diff_list = []

    for brain_region_1, brain_region_2 in itertools.combinations(brian_region_names, 2):
        brain_region_1_acc = exp_and_brain_region_grouped_df.loc[
            exp_and_brain_region_grouped_df['brainRegion'] == brain_region_1
            ][acc_metric]
        brain_region_2_acc = exp_and_brain_region_grouped_df.loc[
            exp_and_brain_region_grouped_df['brainRegion'] == brain_region_2
            ][acc_metric]

        # do unpaired ttest between those two regions
        stat, p_val = sstats.ttest_ind(brain_region_1_acc, brain_region_2_acc)

        group_1_list.append(brain_region_1)
        group_2_list.append(brain_region_2)
        p_val_list.append(p_val)
        mean_diff_list.append(np.mean(brain_region_2_acc) - np.mean(brain_region_1_acc))

    stat_results_df = pd.DataFrame.from_dict({
        'group1': group_1_list,
        'group2': group_2_list,
        'pvalues': p_val_list,
        'meandiffs': mean_diff_list
    })

    if correction_method == 'BH':
        # get the adjusted p-val
        from statsmodels.stats.multitest import fdrcorrection
        _, adjusted_pval = fdrcorrection(
            stat_results_df['pvalues'].values
        )

        stat_results_df['adjusted_pval'] = adjusted_pval
        stat_results_df['reject'] = (stat_results_df['adjusted_pval'] < 0.05)
        m_comp = stat_results_df
    else:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        # perform multiple pairwise comparison (Tukey HSD)
        m_comp = pairwise_tukeyhsd(endog=exp_and_brain_region_grouped_df[acc_metric],
                                groups=exp_and_brain_region_grouped_df[brain_region_var_name], alpha=0.05)

    return m_comp


def cal_prop_above_threshold_dprime(all_dprime_df_naive, all_dprime_df_trained,
                                   all_dprime_df_naive_shuffled, comparision_type='audLeftRight', threshold=0.25):

    naive_aud_lr = all_dprime_df_naive.loc[
        all_dprime_df_naive['comparison_type'] == comparision_type
        ]

    trained_aud_lr = all_dprime_df_trained.loc[
        all_dprime_df_trained['comparison_type'] == comparision_type
        ]

    naive_shuffled_aud_lr = all_dprime_df_naive_shuffled.loc[
        all_dprime_df_naive_shuffled['comparison_type'] == comparision_type
        ]

    sig_naive_aud_lr = naive_aud_lr.loc[
        np.abs(naive_aud_lr['d_prime']) > threshold
        ]

    sig_trained_aud_lr = trained_aud_lr.loc[
        np.abs(trained_aud_lr['d_prime'] > threshold)
    ]

    sig_naive_shuffled_aud_lr = naive_shuffled_aud_lr.loc[
        np.abs(naive_shuffled_aud_lr['d_prime'] > threshold)
    ]

    naive_shuffled_aud_lr_neurons = len(sig_naive_shuffled_aud_lr) / len(naive_shuffled_aud_lr)
    naive_aud_lr_neurons = len(sig_naive_aud_lr) / len(naive_aud_lr)
    trained_aud_lr_neurons = len(sig_trained_aud_lr) / len(trained_aud_lr)

    return naive_shuffled_aud_lr_neurons, naive_aud_lr_neurons, trained_aud_lr_neurons


def cal_prop_above_threshold_dprime_per_exp(all_dprime_df_naive, all_dprime_df_trained,
                                   all_dprime_df_naive_shuffled, comparision_type='audLeftRight', threshold=0.25):

    naive_aud_lr_neurons = []
    for exp in np.unique(all_dprime_df_naive['Exp']):
        naive_aud_lr = all_dprime_df_naive.loc[
            (all_dprime_df_naive['comparison_type'] == comparision_type) &
            (all_dprime_df_naive['Exp'] == exp)
            ]
        sig_naive_aud_lr = naive_aud_lr.loc[
            np.abs(naive_aud_lr['d_prime']) > threshold
            ]
        naive_aud_lr_neurons.append(len(sig_naive_aud_lr) / len(naive_aud_lr))

    trained_aud_lr_neurons = []
    for exp in np.unique(all_dprime_df_trained['Exp']):

        trained_aud_lr = all_dprime_df_trained.loc[
            (all_dprime_df_trained['comparison_type'] == comparision_type) &
            (all_dprime_df_trained['Exp'] == exp)
            ]
        sig_trained_aud_lr = trained_aud_lr.loc[
            np.abs(trained_aud_lr['d_prime'] > threshold)
        ]

        trained_aud_lr_neurons.append(len(sig_trained_aud_lr) / len(trained_aud_lr))

    naive_shuffled_aud_lr_neurons = []
    for exp in np.unique(all_dprime_df_naive_shuffled['Exp']):
        naive_shuffled_aud_lr = all_dprime_df_naive_shuffled.loc[
            (all_dprime_df_naive_shuffled['comparison_type'] == comparision_type) &
            (all_dprime_df_naive_shuffled['Exp'] == exp)
            ]

        sig_naive_shuffled_aud_lr = naive_shuffled_aud_lr.loc[
            np.abs(naive_shuffled_aud_lr['d_prime'] > threshold)
        ]

        naive_shuffled_aud_lr_neurons.append(len(sig_naive_shuffled_aud_lr) / len(naive_shuffled_aud_lr))



    return np.array(naive_shuffled_aud_lr_neurons), np.array(naive_aud_lr_neurons), np.array(trained_aud_lr_neurons)