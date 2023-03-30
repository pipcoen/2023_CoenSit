import os
import numpy as np
import pickle as pkl
import pandas as pd
import src.models.kernel_regression as kernel_regression
import src.visualization.vizregression as vizregression
import src.data.process_ephys_data as pephys
import src.data.analyse_spikes as anaspikes

import sklearn.model_selection as sklselection
import sklearn.metrics as sklmetrics
import xarray as xr

import matplotlib.pyplot as plt
import sciplotlib.polish as splpolish
import sciplotlib.style as splstyle
from tqdm import tqdm
import glob

import matplotlib as mpl
import pdb

import collections
import scipy.stats as sstats


def main():


    # Linux systems
    # behave_df_path = '/media/timsit/Partition 1/data/interim/passive-m2-new-parent/subset/ephys_behaviour_df.pkl'
    # alignment_folder = '/media/timsit/Partition 1/data/interim/passive-m2-new-parent-alignment-2ms/'
    # model_fit_result_folder = '/media/timsit/Partition 1/reports/figures/passive-kernel-regression/regression-psth-cv/'
    # model_fit_result_folder = '/media/timsit/Partition 1/reports/figures/passive-kernel-regression/regression-psth-cv-include-unimodal-models-2-fold-cv-include-mse-w-kernels-2021-03-16/'

    # Mac
    behave_df_path = '/Volumes/Partition 1/data/interim/passive-m2-new-parent/subset/ephys_behaviour_df.pkl'
    alignment_folder = '/Volumes/Partition 1/data/interim/passive-m2-new-parent-alignment-2ms/'
    model_fit_result_folder = '/Volumes/Partition 1/reports/figures/passive-kernel-regression/regression-psth-zscore-pre-stim-divided-cv-include-unimodal-models-2-fold-cv-include-mse-w-kernels/'


    feature_sets = {
        'audOnOnly': ['stimOn'],
        'visOnly': ['stimOn', 'visSign'],
        'audOnly': ['stimOn', 'audSign'],
        'addition': ['stimOn', 'audSign', 'visSign'],
        'interaction': ['stimOn', 'audSign', 'visSign', 'audVis']}

    if not os.path.exists(model_fit_result_folder):
        os.mkdir(model_fit_result_folder)

    passive_behave_df = pd.read_pickle(behave_df_path)
    passive_behave_df = passive_behave_df.loc[
        passive_behave_df['subjectRef'] != 1
    ]
    target_exp = np.unique(passive_behave_df['expRef'])

    smooth_sigma = 30
    smooth_window = 50
    split_random_state = 1
    evaluation_method = 'cv-eval'
    cv_n_splits = 2
    activity_normalization_method = 'zscore_rel_pre_stim'  # 'divide_by_overall_mean', or 'divide_by_pre_stim_mean'
                                                               # 'zscore_rel_pre_stim'

    batch_alignment_ds_to_regression_results(alignment_folder=alignment_folder,
                                             model_fit_result_folder=model_fit_result_folder,
                                             smooth_sigma=smooth_sigma, smooth_window=smooth_window,
                                             split_random_state=split_random_state,
                                             evaluation_method=evaluation_method, cv_n_splits=cv_n_splits,
                                             target_exp=target_exp,
                                             passive_behave_df=passive_behave_df, feature_sets=feature_sets,
                                             activity_normalization_method=activity_normalization_method)


def batch_alignment_ds_to_regression_results(alignment_folder, model_fit_result_folder,
                                             target_brain_region='MOs', smooth_window=50,
                                             smooth_sigma=30, split_random_state=None,
                                             evaluation_method='fit-all', cv_n_splits=5, target_exp=None,
                                             passive_behave_df=None, feature_sets={
                                        'addition':  ['stimOn', 'audSign', 'visSign'],
                                       'interaction': ['stimOn', 'audSign', 'visSign', 'audVis']},
                                             activity_name='smoothed_fr', test_size=0.5,
                                             time_window=[-0.05, 0.4],
                                             activity_normalization_method=None):
    """

    Parameters
    ----------
    alignment_folder
    model_fit_result_folder
    target_brain_region
    smooth_window
    smooth_sigma
    split_random_state
    evaluation_method
    target_exp
    activity_normalization_method : str or None
        how to control for difference in overall or mean activity of neurons
        if None, then use spike activity
        'divide_by_mean' : divide the activity of each neuron by it's mean activity before stimulus onset
    Returns
    -------

    """

    alignment_ds_fname_list = glob.glob(os.path.join(alignment_folder, '*%s*.nc') % target_brain_region)
    print('Number of files found: %.f' % len(alignment_ds_fname_list))

    for alignment_ds_fname in alignment_ds_fname_list:

        alignment_ds = xr.open_dataset(alignment_ds_fname).load()
        exp = alignment_ds.Exp.values[0]

        """
        if passive_behave_df is not None:
            subject = np.unique(passive_behave_df.loc[
                passive_behave_df['expRef'] == exp
            ]['subjectRef'])[0]
        """

        if exp not in target_exp:
            continue

        alignment_ds = alignment_ds.isel(Exp=0)
        alignment_ds = alignment_ds.stack(trialTime=['Trial', 'Time'])
        alignment_ds['smoothed_fr'] = (['Cell', 'trialTime'],
                                       anaspikes.smooth_spikes(
                                           alignment_ds['firing_rate'], method='half_gaussian',
                                           sigma=smooth_sigma, window_width=smooth_window))
        alignment_ds = alignment_ds.unstack()

        # Do clean up of some dimensions
        vars_to_clean = ['audDiff', 'visDiff']

        for var_name in vars_to_clean:
            if 'Cell' in alignment_ds[var_name].dims:
                alignment_ds[var_name] = alignment_ds[var_name].isel(Cell=0)
            if 'Time' in alignment_ds[var_name].dims:
                alignment_ds[var_name] = alignment_ds[var_name].isel(Time=0)

        if activity_normalization_method == 'divide_by_pre_stim_mean':
            print('Dividing exp %.f %s by pre-stimulus mean' % (exp, activity_name))
            pre_stim_mean_per_cell_val = alignment_ds.where(
                alignment_ds['PeriEventTime'] < 0, drop=True
            )[activity_name].mean(['Trial', 'Time'])

            pre_stim_mean_per_cell_val[pre_stim_mean_per_cell_val == 0] = 1  # handle cases of division by zero
            alignment_ds[activity_name] = alignment_ds[activity_name] / pre_stim_mean_per_cell_val

        elif activity_normalization_method == 'divide_by_overall_mean':
            overall_mean_per_cell_val = alignment_ds[activity_name].mean(['Trial', 'Time'])
            alignment_ds[activity_name] = alignment_ds[activity_name] / overall_mean_per_cell_val

        elif activity_normalization_method == 'zscore_rel_pre_stim':
            print('zscoring exp %.f %s by pre-stimulus activity' % (exp, activity_name))
            pre_stim_mean_per_cell_val = alignment_ds.where(
                alignment_ds['PeriEventTime'] < 0, drop=True
            )[activity_name].mean(['Trial', 'Time'])

            pre_stim_std_per_cell_val = alignment_ds.where(
                alignment_ds['PeriEventTime'] < 0, drop=True
            )[activity_name].std(['Trial', 'Time'])
            pre_stim_std_per_cell_val[pre_stim_std_per_cell_val == 0] = 1 # handle cases of division by zero

            alignment_ds[activity_name] = (alignment_ds[activity_name] - pre_stim_mean_per_cell_val) / pre_stim_std_per_cell_val



        regression_results = alignment_ds_to_regression_results(alignment_ds, mean_same_stim=True,
                                       evaluation_method=evaluation_method, cv_n_splits=cv_n_splits,
                                       feature_sets=feature_sets,
                                       subset_stim_cond=[{'audDiff': 60, 'visDiff': 0.8},
                                                         {'audDiff': -60, 'visDiff': -0.8},
                                                         {'audDiff': 60, 'visDiff': -0.8},
                                                         {'audDiff': -60, 'visDiff': 0.8}],
                                       stim_cond_names=['arvr', 'alvl', 'arvl', 'alvr'],
                                       split_random_state=split_random_state, time_window=time_window,
                                      activity_name=activity_name, test_size=test_size)

        save_name = 'exp%.f_regression_results.pkl' % exp
        save_path = os.path.join(model_fit_result_folder, save_name)
        with open(save_path, 'wb') as handle:
            pkl.dump(regression_results, handle)


def alignment_ds_to_regression_results(alignment_ds, mean_same_stim=True, evaluation_method='fit-all',
                                       cv_n_splits=5, test_size=0.5, feature_sets={'addition':['stimOn', 'audSign', 'visSign'],
 'interaction':['stimOn', 'audSign', 'visSign', 'audVis']}, subset_stim_cond=[{'audDiff':60,
  'visDiff':0.8}, {'audDiff':-60, 'visDiff':-0.8}, {'audDiff':60, 'visDiff':-0.8}, {'audDiff':-60, 'visDiff':0.8}],
    stim_cond_names=['arvr', 'alvl', 'arvl', 'alvr'], split_random_state=None, time_window=[-0.05, 0.4],
    quantify_temporal_kernel=True, eval_metrics=['varExplained', 'mse'], kernel_quantification_window=None,
    activity_name='smoothed_fr', multiple_cells=True, include_single_trials=True, model='least-squares',
    error_window=[0.0, 0.4], kernel_quantification_set='cv', make_blank_trials=False):
    """

    Parameters
    ----------
    alignment_ds : (xarray dataset)
    mean_same_stim : (bool)
        whether to take the mean PSTH and fit to that (rather than fitting individual trials)
    evaluation_method : (str)
    test_size : (float)
        value from 0 - 1
        test set size of doing train-test-split
    feature_sets
    subset_stim_cond
    split_random_state : (int)
        random seed to do train-test split
    eval_metrics : (list)
        list of evaluation metrics to use
    activity_name : (str)
        what activity to fit the model to.
    model : (str or sklearn model object)
        what model to use to fit the regression
        'least-squares' : linear regression using least-squares
    Returns
    -------

    """
    regression_results = dict()
    regression_results['X_set_results'] = dict()
    if time_window is not None:
        alignment_ds = alignment_ds.where(
            (alignment_ds['PeriEventTime'] >= time_window[0]) &
            (alignment_ds['PeriEventTime'] <= time_window[1]), drop=True
        )
        alignment_ds['audDiff'] = alignment_ds['audDiff'].isel(Time=0)
        alignment_ds['visDiff'] = alignment_ds['visDiff'].isel(Time=0)

    if make_blank_trials:
        time_window_width = time_window[1] - time_window[0]
        baseline_activity_start = time_window[0] - time_window_width
        baseline_activity_end = time_window[0]
        baseline_alignment_ds = alignment_ds.where(((alignment_ds['PeriEventTime'] >= baseline_activity_start) & (alignment_ds['PeriEventTime'] <= baseline_activity_end)),
          drop=True).copy()
        num_subset_trials = 50
        trial_idx_to_get = np.random.choice(baseline_alignment_ds.Trial.values, num_subset_trials)
        baseline_alignment_ds = baseline_alignment_ds.sel(Trial=trial_idx_to_get)


    if make_blank_trials:
        baseline_alignment_ds['PeriEventTime'] = (['Trial', 'Time'],
         np.tile(alignment_ds['PeriEventTime'].isel(Trial=0).values, [
          num_subset_trials, 1]))
        baseline_alignment_ds['Time'] = (
         'Time', alignment_ds['Time'].values)
        alignment_ds['PeriEventTime'] = ('Time', alignment_ds['PeriEventTime'].isel(Trial=0).values)
        baseline_alignment_ds['audDiff'] = (
         'Trial', np.repeat(np.inf, num_subset_trials))
        baseline_alignment_ds['visDiff'] = ('Trial', np.repeat(0, num_subset_trials))
        baseline_alignment_ds['Trial'] = ('Trial',
         np.arange(np.max(alignment_ds.Trial.values) + 1, np.max(alignment_ds.Trial.values) + 1 + num_subset_trials))
        alignment_ds = xr.concat([alignment_ds, baseline_alignment_ds], dim='Trial')
    if evaluation_method == 'fit-all':
        subset_alignment_ds = subset_stim_cond_alignment_ds(subset_stim_cond=subset_stim_cond, alignment_ds=alignment_ds,
          mean_same_stim=mean_same_stim,
          get_subset_cond_idx=False)
        subset_alignment_ds = subset_alignment_ds.load()
        if multiple_cells:
            num_cell = len(subset_alignment_ds.Cell)
        else:
            num_cell = 1
        num_trial = len(subset_alignment_ds.Trial)
        num_time_bin = len(subset_alignment_ds.Time)
        for feature_set_name, feature_set in feature_sets.items():
            X = make_trial_feature_matrix(alignment_ds=subset_alignment_ds, feature_set=feature_set)
            Y_predict_all_cell = np.zeros((num_cell, num_trial, num_time_bin))
            Y_all_cell = np.zeros((num_cell, num_trial, num_time_bin))
            W_train_all_cell = np.zeros((num_cell, len(feature_set), num_time_bin))
            if multiple_cells:
                for n_cell, cell in enumerate(subset_alignment_ds.Cell):
                    Y = subset_alignment_ds.isel(Cell=cell)[activity_name].values
                    W, Y_hat = do_regression(X, Y)
                    Y_predict_all_cell[n_cell, :, :] = Y_hat
                    Y_all_cell[n_cell, :, :] = Y
                    W_train_all_cell[n_cell, :, :] = W

            else:
                n_cell = 0
                Y = subset_alignment_ds[activity_name].values
                W, Y_hat = do_regression(X, Y)
                Y_predict_all_cell[n_cell, :, :] = Y_hat
                Y_all_cell[n_cell, :, :] = Y
                W_train_all_cell[n_cell, :, :] = W
            regression_results['X_set_results'][feature_set_name] = dict()
            regression_results['X_set_results'][feature_set_name]['X'] = X
            regression_results['X_set_results'][feature_set_name]['Y_predict'] = Y_predict_all_cell
            regression_results['X_set_results'][feature_set_name]['Y'] = Y_all_cell
            regression_results['X_set_results'][feature_set_name]['W_train'] = W_train_all_cell

        regression_results['Y'] = Y
    elif evaluation_method == 'cv-eval':
        subset_alignment_ds, stim_cond_idx = subset_stim_cond_alignment_ds(subset_stim_cond=subset_stim_cond,
          alignment_ds=alignment_ds,
          mean_same_stim=False,
          get_subset_cond_idx=True,
          sort_trials=False,
          max_trial=None)
        if len(subset_alignment_ds.Trial) == 0:
            print('No trials for subset stimulus condition, entering debug mode')
            pdb.set_trace()
        else:
            all_stim_cond_idx = np.concatenate(stim_cond_idx)
            subset_alignment_ds = subset_alignment_ds.load()
            if multiple_cells:
                num_cell = len(subset_alignment_ds.Cell)
            else:
                num_cell = 1
        num_trial = len(subset_alignment_ds.Trial)
        num_time_bin = len(subset_alignment_ds.Time)
        splitter = sklselection.StratifiedKFold(n_splits=cv_n_splits, shuffle=True,
          random_state=split_random_state)
        stim_group_per_trial = np.zeros(num_trial)
        for n_sc, sc_idx in enumerate(stim_cond_idx):
            stim_group_per_trial[sc_idx] = n_sc

        for feature_set_name, feature_set in feature_sets.items():
            X = make_trial_feature_matrix(alignment_ds=subset_alignment_ds, feature_set=feature_set)
            num_stim_cond = len(stim_cond_idx)
            num_features = len(feature_set)
            Y_predict_all_cell = np.zeros((cv_n_splits, num_cell, num_stim_cond, num_time_bin))
            Y_test_actual_all_cell = np.zeros((cv_n_splits, num_cell, num_stim_cond, num_time_bin))
            cell_fit_df_list = list()
            peri_event_time_bins = np.linspace(time_window[0], time_window[1], num_time_bin)
            all_feature_peri_event_time_bin = np.tile(peri_event_time_bins, num_features)
            W_train_all_cell = np.zeros((num_cell, cv_n_splits, num_features, num_time_bin))
            for n_cell, cell in enumerate(subset_alignment_ds.Cell):
                y = subset_alignment_ds.isel(Cell=cell)[activity_name].values
                n_split = cv_n_splits
                var_explained_per_split = np.zeros(n_split)
                test_set_explainable_variance_per_split = np.zeros(n_split)
                if 'mse' in eval_metrics:
                    test_set_mse_per_split = np.zeros(n_split)
                if 'ss_residuals' in eval_metrics:
                    test_set_ss_residuals_per_split = np.zeros(n_split)
                if quantify_temporal_kernel:
                    temporal_kernel_bias_mean = np.zeros(n_split)
                    temporal_kernel_aud_mean = np.zeros(n_split)
                    temporal_kernel_vis_mean = np.zeros(n_split)
                    temporal_kernel_av_mean = np.zeros(n_split)
                    temporal_kernel_bias_signed_max = np.zeros(n_split)
                    temporal_kernel_aud_signed_max = np.zeros(n_split)
                    temporal_kernel_vis_signed_max = np.zeros(n_split)
                    temporal_kernel_av_signed_max = np.zeros(n_split)
                single_cell_y_predict_cv = list()
                single_cell_y_test_actual_cv = list()
                for n_cv, (train_idx, test_index) in enumerate(splitter.split(X, y=stim_group_per_trial)):
                    X_train, X_test = X[train_idx], X[test_index]
                    y_train, y_test = y[train_idx], y[test_index]
                    stim_cond_idx_train, stim_cond_idx_test = all_stim_cond_idx[train_idx], all_stim_cond_idx[test_index]
                    if mean_same_stim:
                        X_train = np.zeros((num_stim_cond, num_features))
                        y_train = np.zeros((num_stim_cond, num_time_bin))
                        X_test = np.zeros((num_stim_cond, num_features))
                        y_test = np.zeros((num_stim_cond, num_time_bin))
                        for n_sc, sc_idx in enumerate(stim_cond_idx):
                            sc_test_idx = stim_cond_idx_test[np.isin(stim_cond_idx_test, sc_idx)]
                            X_test[n_sc, :] = np.mean((X[sc_test_idx, :]), axis=0)
                            y_test[n_sc, :] = np.mean((y[sc_test_idx, :]), axis=0)
                            sc_train_idx = stim_cond_idx_train[np.isin(stim_cond_idx_train, sc_idx)]
                            X_train[n_sc, :] = np.mean((X[sc_train_idx, :]), axis=0)
                            y_train[n_sc, :] = np.mean((y[sc_train_idx, :]), axis=0)

                    else:
                        if np.sum(np.isnan(y_test)) > 0:
                            print('outputs contain nan, entering debug mode')
                            print('Feature set name: %s' % feature_set_name)
                            pdb.set_trace()
                        elif model == 'least-squares':
                            W_train, y_train_hat = do_regression(X_train, y_train)
                            W_train_all_cell[n_cell, n_cv, :, :] = W_train
                            y_test_hat = np.matmul(X_test, W_train)
                        else:
                            train_set_model = model.fit(X_train, y_train)
                            W_train = train_set_model.coef_.T
                            W_train_all_cell[n_cell, n_cv, :, :] = W_train
                            y_test_hat = train_set_model.predict(X_test)
                        Y_predict_all_cell[n_cv, n_cell, :, :] = y_test_hat
                        Y_test_actual_all_cell[n_cv, n_cell, :, :] = y_test
                        var_explained = sklmetrics.explained_variance_score(y_true=(y_test.flatten()), y_pred=(y_test_hat.flatten()))
                        if model == 'least-squares':
                            _, non_validated_model_prediction = do_regression(X_test, y_test)
                        else:
                            test_set_model = model.fit(X_test, y_test)
                        non_validated_model_prediction = test_set_model.predict(X_test)
                    test_set_explainable_variance = sklmetrics.explained_variance_score(y_true=(y_test.flatten()), y_pred=(non_validated_model_prediction.flatten()))
                    var_explained_per_split[n_cv] = var_explained
                    test_set_explainable_variance_per_split[n_cv] = test_set_explainable_variance
                    if 'mse' in eval_metrics:
                        if error_window is not None:
                            peri_event_time = subset_alignment_ds.PeriEventTime.isel(Trial=0).values
                            subset_time_idx = np.where((peri_event_time >= error_window[0]) & (peri_event_time <= error_window[1]))[0]
                            y_test_subset = y_test[:, subset_time_idx]
                            y_test_hat_subset = y_test_hat[:, subset_time_idx]
                            test_set_mse_per_split[n_cv] = sklmetrics.mean_squared_error(y_true=(y_test_subset.flatten()), y_pred=(y_test_hat_subset.flatten()))
                        else:
                            test_set_mse_per_split[n_cv] = sklmetrics.mean_squared_error(y_true=(y_test.flatten()), y_pred=(y_test_hat.flatten()))
                    if 'ss_residuals' in eval_metrics:
                        if error_window is not None:
                            test_set_ss_residuals_per_split[n_cv] = np.sum((y_test_subset.flatten() - y_test_hat_subset.flatten()) ** 2)
                        else:
                            test_set_ss_residuals_per_split[n_cv] = np.sum((y_test.flatten() - y_test_hat.flatten()) ** 2)
                    if kernel_quantification_set == 'all':
                        W_all, y_all = do_regression(X, y)
                        W_to_quantify = W_all
                    elif kernel_quantification_set == 'cv':
                        W_to_quantify = W_train
                    if quantify_temporal_kernel:
                        if kernel_quantification_window is not None:
                            subset_time_idx = np.where((peri_event_time >= kernel_quantification_window[0]) & (peri_event_time <= kernel_quantification_window[1]))[0]
                        else:
                            subset_time_idx = np.arange(np.shape(W_to_quantify)[1])
                        if len(feature_set) == 1:
                            temporal_kernel_bias_mean[n_cv] = np.mean(W_to_quantify[(0, subset_time_idx)])
                        else:
                            if len(feature_set) == 2:
                                temporal_kernel_bias_mean[n_cv] = np.mean(W_to_quantify[(0, subset_time_idx)])
                                if 'visSign' in feature_set:
                                    temporal_kernel_vis_mean[n_cv] = np.mean(W_to_quantify[(1, subset_time_idx)])
                                else:
                                    if 'audSign' in feature_set:
                                        temporal_kernel_aud_mean[n_cv] = np.mean(W_to_quantify[(1, subset_time_idx)])
                            else:
                                temporal_kernel_bias_mean[n_cv] = np.mean(W_to_quantify[(0, subset_time_idx)])
                                temporal_kernel_aud_mean[n_cv] = np.mean(W_to_quantify[(1, subset_time_idx)])
                                temporal_kernel_vis_mean[n_cv] = np.mean(W_to_quantify[(2, subset_time_idx)])
                        if 'audVis' in feature_set:
                            temporal_kernel_av_mean[n_cv] = np.mean(W_to_quantify[(3, subset_time_idx)])
                        else:
                            bias_max_loc = np.argmax(np.abs(W_to_quantify[(0, subset_time_idx)]))
                            temporal_kernel_bias_signed_max[n_cv] = W_to_quantify[(0, bias_max_loc)]
                            if len(feature_set) == 2:
                                if 'visSign' in feature_set:
                                    vis_max_loc = np.argmax(np.abs(W_to_quantify[1, :]))
                                    temporal_kernel_vis_signed_max[n_cv] = W_to_quantify[(1, vis_max_loc)]
                                else:
                                    if 'audSign' in feature_set:
                                        aud_max_loc = np.argmax(np.abs(W_to_quantify[1, :]))
                                        temporal_kernel_aud_signed_max[n_cv] = W_to_quantify[(1, aud_max_loc)]
                            else:
                                if len(feature_set) >= 3:
                                    aud_max_loc = np.argmax(np.abs(W_to_quantify[1, :]))
                                    temporal_kernel_aud_signed_max[n_cv] = W_to_quantify[(1, aud_max_loc)]
                                    vis_max_loc = np.argmax(np.abs(W_to_quantify[2, :]))
                                    temporal_kernel_vis_signed_max[n_cv] = W_to_quantify[(2, vis_max_loc)]
                        if 'audVis' in feature_set:
                            av_max_loc = np.argmax(np.abs(W_to_quantify[3, :]))
                            temporal_kernel_av_signed_max[n_cv] = W_to_quantify[(3, av_max_loc)]

                cell_fit_df = pd.DataFrame.from_dict({'Cell':np.repeat(cell.values, n_split),
                 'cv':np.arange(n_split),
                 'varExplained':var_explained_per_split,
                 'explainableVar':test_set_explainable_variance_per_split})
                if 'mse' in eval_metrics:
                    cell_fit_df['mse'] = test_set_mse_per_split
                if 'ss_residuals' in eval_metrics:
                    cell_fit_df['ss_residuals'] = test_set_ss_residuals_per_split
                if quantify_temporal_kernel:
                    cell_fit_df['biasKernelMean'] = temporal_kernel_bias_mean
                    cell_fit_df['audKernelMean'] = temporal_kernel_aud_mean
                    cell_fit_df['visKernelMean'] = temporal_kernel_vis_mean
                    cell_fit_df['avKernelMean'] = temporal_kernel_av_mean
                    cell_fit_df['biasKernelSignedMax'] = temporal_kernel_bias_signed_max
                    cell_fit_df['audKernelSignedMax'] = temporal_kernel_aud_signed_max
                    cell_fit_df['visKernelSignedMax'] = temporal_kernel_vis_signed_max
                    cell_fit_df['avKernelSignedMax'] = temporal_kernel_av_signed_max
                cell_fit_df_list.append(cell_fit_df)

            regression_results['X_set_results'][feature_set_name] = dict()
            all_cell_fit_df = pd.concat(cell_fit_df_list)
            regression_results['X_set_results'][feature_set_name]['model_performance_df'] = all_cell_fit_df
            regression_results['X_set_results'][feature_set_name]['Y_test_actual_all_cell'] = np.mean(Y_test_actual_all_cell, axis=0)
            regression_results['X_set_results'][feature_set_name]['Y_test_predict_all_cell'] = np.mean(Y_predict_all_cell, axis=0)
            if include_single_trials:
                regression_results['Y_original'] = subset_alignment_ds
            try:
                W_train_all_cell_array = xr.DataArray(W_train_all_cell, dims=['Cell', 'Cv', 'Feature', 'Time'], coords={'Cell':np.arange(num_cell),
                 'Cv':np.arange(cv_n_splits),
                 'Feature':feature_set,
                 'Time':peri_event_time_bins})
            except:
                pdb.set_trace()

            regression_results['X_set_results'][feature_set_name]['kernels'] = W_train_all_cell_array

    elif evaluation_method == 'train-test-split':
        subset_alignment_ds, stim_cond_idx = subset_stim_cond_alignment_ds(subset_stim_cond=subset_stim_cond, alignment_ds=alignment_ds,
          mean_same_stim=False,
          get_subset_cond_idx=True)
        all_stim_cond_idx = np.concatenate(stim_cond_idx)
        num_cell = len(subset_alignment_ds.Cell)
        num_trial = len(subset_alignment_ds.Trial)
        num_time_bin = len(subset_alignment_ds.Time)
        cell_fit_df_list = list()
        for feature_set_name, feature_set in feature_sets.items():
            X = make_trial_feature_matrix(alignment_ds=subset_alignment_ds, feature_set=feature_set)
            num_stim_cond = len(stim_cond_idx)
            num_features = len(feature_set)
            Y_train_actual_all_cell = np.zeros((num_cell, num_stim_cond, num_time_bin))
            Y_test_actual_all_cell = np.zeros((num_cell, num_stim_cond, num_time_bin))
            Y_train_predict_all_cell = np.zeros((num_cell, num_stim_cond, num_time_bin))
            Y_test_predict_all_cell = np.zeros((num_cell, num_stim_cond, num_time_bin))
            W_train_all_cell = np.zeros((num_cell, num_features, num_time_bin))
            W_test_all_cell = np.zeros((num_cell, num_features, num_time_bin))
            cell_fit_df_list = list()
            stim_cond_name_to_idx = dict()
            for n_cell, cell in enumerate(subset_alignment_ds.Cell):
                y = subset_alignment_ds.isel(Cell=cell)['smoothed_fr'].values
                train_idx, test_index = sklselection.train_test_split((np.arange(num_trial)), test_size=test_size,
                  shuffle=True,
                  random_state=split_random_state)
                X_train, X_test = X[train_idx], X[test_index]
                y_train, y_test = y[train_idx], y[test_index]
                stim_cond_idx_train, stim_cond_idx_test = all_stim_cond_idx[train_idx], all_stim_cond_idx[test_index]
                if mean_same_stim:
                    X_train = np.zeros((num_stim_cond, num_features))
                    y_train = np.zeros((num_stim_cond, num_time_bin))
                    X_test = np.zeros((num_stim_cond, num_features))
                    y_test = np.zeros((num_stim_cond, num_time_bin))
                    for n_sc, sc_idx in enumerate(stim_cond_idx):
                        sc_test_idx = stim_cond_idx_test[np.isin(stim_cond_idx_test, sc_idx)]
                        X_test[n_sc, :] = np.mean((X[sc_test_idx, :]), axis=0)
                        y_test[n_sc, :] = np.mean((y[sc_test_idx, :]), axis=0)
                        sc_train_idx = stim_cond_idx_train[np.isin(stim_cond_idx_train, sc_idx)]
                        X_train[n_sc, :] = np.mean((X[sc_train_idx, :]), axis=0)
                        y_train[n_sc, :] = np.mean((y[sc_train_idx, :]), axis=0)
                        stim_cond_name_to_idx[stim_cond_names[n_sc]] = dict()
                        stim_cond_name_to_idx[stim_cond_names[n_sc]]['train'] = sc_train_idx
                        stim_cond_name_to_idx[stim_cond_names[n_sc]]['test'] = sc_test_idx

                W_train, y_train_hat = do_regression(X_train, y_train)
                y_test_hat = np.matmul(X_test, W_train)
                W_train_all_cell[n_cell, :, :] = W_train
                Y_train_predict_all_cell[n_cell, :, :] = y_train_hat
                Y_test_predict_all_cell[n_cell, :, :] = y_test_hat
                Y_train_actual_all_cell[n_cell, :, :] = y_train
                Y_test_actual_all_cell[n_cell, :, :] = y_test
                var_explained = sklmetrics.explained_variance_score(y_true=(y_test.flatten()), y_pred=(y_test_hat.flatten()))
                W_test, non_validated_model_prediction = do_regression(X_test, y_test)
                W_test_all_cell[n_cell, :, :] = W_test
                test_set_explainable_variance = sklmetrics.explained_variance_score(y_true=(y_test.flatten()), y_pred=(non_validated_model_prediction.flatten()))
                cell_fit_df = pd.DataFrame.from_dict({'Cell':[
                  cell.values],
                 'varExplained':[
                  var_explained],
                 'explainableVar':[
                  test_set_explainable_variance]})
                cell_fit_df_list.append(cell_fit_df)

            all_cell_fit_df = pd.concat(cell_fit_df_list)
            regression_results['X_set_results'][feature_set_name] = dict()
            regression_results['X_set_results'][feature_set_name]['model_performance_df'] = all_cell_fit_df
            regression_results['X_set_results'][feature_set_name]['Y_train_actual_all_cell'] = Y_train_actual_all_cell
            regression_results['X_set_results'][feature_set_name]['Y_test_actual_all_cell'] = Y_test_actual_all_cell
            regression_results['X_set_results'][feature_set_name]['Y_train_predict_all_cell'] = Y_train_predict_all_cell
            regression_results['X_set_results'][feature_set_name]['Y_test_predict_all_cell'] = Y_train_predict_all_cell
            regression_results['X_set_results'][feature_set_name]['W_train'] = W_train_all_cell
            regression_results['X_set_results'][feature_set_name]['W_test'] = W_test_all_cell
            regression_results['X_set_results'][feature_set_name]['stim_cond_idx'] = stim_cond_name_to_idx
            if include_single_trials:
                regression_results['Y_original'] = subset_alignment_ds

    regression_results['params'] = dict()
    regression_results['params']['evaluation_method'] = evaluation_method
    regression_results['params']['cv_n_splits'] = cv_n_splits
    regression_results['params']['peri_stimulus_time'] = alignment_ds.PeriEventTime.isel(Trial=0).values
    regression_results['params']['stim_cond_names'] = stim_cond_names
    return regression_results


def subset_stim_cond_alignment_ds(alignment_ds, subset_stim_cond=[{'audDiff': 60, 'visDiff': 0.8},
                                                                    {'audDiff': -60, 'visDiff': -0.8},
                                                                    {'audDiff': 60, 'visDiff': -0.8},
                                                                    {'audDiff': -60, 'visDiff': 0.8}],
                                  sort_trials=False, mean_same_stim=False, get_subset_cond_idx=False):
    """

    Parameters
    ----------
    alignment_ds
    subset_stim_cond
    sort_trials
    mean_same_stim
    get_subset_cond_idx

    Returns
    -------

    """

    subset_aligment_ds_list = list()

    if get_subset_cond_idx:
        subset_cond_idx = list()

    for stim_cond_dict in subset_stim_cond:
        stim_cond_ds = alignment_ds.where(
            (alignment_ds['visDiff'] == stim_cond_dict['visDiff']) &
            (alignment_ds['audDiff'] == stim_cond_dict['audDiff']), drop=True
        )

        if mean_same_stim:
            stim_cond_ds = stim_cond_ds.mean('Trial')

        subset_aligment_ds_list.append(stim_cond_ds)

    subset_aligment_ds = xr.concat(subset_aligment_ds_list, dim='Trial')

    if get_subset_cond_idx:

        for stim_cond_dict in subset_stim_cond:
            cond_idx = np.where(
                (subset_aligment_ds['visDiff'] == stim_cond_dict['visDiff']) &
                (subset_aligment_ds['audDiff'] == stim_cond_dict['audDiff'])
            )[0]
            subset_cond_idx.append(cond_idx)


    if sort_trials:
        subset_aligment_ds = subset_aligment_ds.sortby('Trial')

    if get_subset_cond_idx:
        return subset_aligment_ds, subset_cond_idx
    else:
        return subset_aligment_ds


def make_trial_feature_matrix(alignment_ds, feature_set=['stimOn', 'audSign', 'visSign']):
    """

    Parameters
    ----------
    alignment_ds : (xarray dataset)
    feature_set
    include_stim_cond_idx

    Returns
    -------

    """

    num_trial = len(alignment_ds.Trial)
    X = np.zeros((num_trial, len(feature_set)))

    for n_feature, feature_name in enumerate(feature_set):

        if feature_name == 'stimOn':
            feature_values = np.repeat(1, num_trial)
        elif feature_name == 'audSign':
            feature_values = np.sign(alignment_ds['audDiff'].values)
        elif feature_name == 'visSign':
            feature_values = np.sign(alignment_ds['visDiff'].values)
        elif feature_name == 'audVis':
            feature_values = np.sign(alignment_ds['audDiff'].values) * np.sign(alignment_ds['visDiff'].values)

        X[:, n_feature] = feature_values

    return X



def do_regression(X, Y):

    W, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    Y_hat = np.matmul(X, W)

    return W, Y_hat


def load_combined_models_df(model_results_folder, behave_df_path, filetype='pickle'):
    """

    Parameters
    ----------
    model_results_folder
    behave_df_path

    Returns
    -------

    """

    if filetype == 'pickle':
        model_result_fname_list = glob.glob(os.path.join(model_results_folder, '*.pkl'))
    else:
        model_result_fname_list = glob.glob(os.path.join(model_results_folder, '*/'))

    passive_behave_df = pd.read_pickle(behave_df_path)
    total_neuron_count = 0
    df_list = list()

    for model_result_fname in model_result_fname_list:

        if filetype == 'pickle':
            model_result = pd.read_pickle(model_result_fname)
            X_set_names = list(model_result['X_set_results'].keys())
        else:
            X_set_names = glob.glob(os.path.join(model_result_fname, '*.csv'))

        for x_set in X_set_names:

            if filetype == 'pickle':
                model_df = model_result['X_set_results'][x_set]['model_performance_df']
            else:
                model_df = pd.read_csv(x_set)

            # reindex neuron
            n_neuron = len(pd.unique(model_df['Cell']))

            model_df['neuron'] = model_df['Cell']
            model_df['neuron'] = model_df['neuron'].replace(
                pd.unique(model_df['Cell']),
                np.arange(total_neuron_count, total_neuron_count + n_neuron)
            )

            if filetype == 'pickle':
                model_df['model'] = x_set
            else:
                model_df['model'] = os.path.basename(x_set).split('_')[0]

            # Add exp info
            if filetype == 'pickle':
                exp_num = int(os.path.basename(model_result_fname)[3:5])  # currently assumes two digits after 'exp'
            else:
                exp_num = int(os.path.basename(os.path.normpath(model_result_fname))[3:5])
            model_df['Exp'] = exp_num

            # Subject as well
            subject = np.unique(passive_behave_df.loc[
                                    passive_behave_df['expRef'] == exp_num
                                    ]['subjectRef'])[0]
            model_df['Subject'] = subject
            df_list.append(model_df)

        total_neuron_count += n_neuron

    all_models_df = pd.concat(df_list)

    return all_models_df

def load_active_combined_models_df(model_results_folder, behave_df_path, take_kernel_abs=True,
                                   kernel_mean_window=[0, 0.4],
                                   filetype='pickle'):
    """

    """

    # Get kernel results
    kernel_info = collections.defaultdict(list)
    model_result_folder_list = glob.glob(os.path.join(model_results_folder, '*/'))

    for model_results_folder in model_result_folder_list:

        movement_kernel_path = glob.glob(os.path.join(model_results_folder, '*mov_kernels.nc'))[0]
        movement_kernel_ds = xr.open_dataset(movement_kernel_path)
        movement_kernel_ds = movement_kernel_ds.to_array()

        stim_kernel_path = glob.glob(os.path.join(model_results_folder, '*stim_kernels.nc'))[0]
        stim_kernel_ds = xr.open_dataset(stim_kernel_path).to_array()

        num_cell = len(movement_kernel_ds.Cell)

        if take_kernel_abs:
            move_left_kernel_mean_vals = np.abs(movement_kernel_ds.sel(Feature='moveLeft',
                                                                       Time=slice(kernel_mean_window[0],
                                                                                  kernel_mean_window[1]),
                                                                       )).mean('Time').values

            move_right_kernel_mean_vals = np.abs(movement_kernel_ds.sel(Feature='moveRight',
                                                                        Time=slice(kernel_mean_window[0],
                                                                                   kernel_mean_window[1]),
                                                                        )).mean('Time').values

            aud_sign_kernel_mean_vals = np.abs(stim_kernel_ds.sel(Feature='audSign',
                                                                  Time=slice(kernel_mean_window[0],
                                                                             kernel_mean_window[1]),
                                                                  )).mean('Time').values
            vis_sign_kernel_mean_vals = np.abs(stim_kernel_ds.sel(Feature='visSign',
                                                                  Time=slice(kernel_mean_window[0],
                                                                             kernel_mean_window[1]),
                                                                  )).mean('Time').values


        else:
            move_left_kernel_mean_vals = movement_kernel_ds.sel(Feature='moveLeft',
                                                                Time=slice(kernel_mean_window[0],
                                                                           kernel_mean_window[1]),
                                                                ).mean('Time').values
            move_right_kernel_mean_vals = movement_kernel_ds.sel(Feature='moveRight',
                                                                 Time=slice(kernel_mean_window[0],
                                                                            kernel_mean_window[1]),
                                                                 ).mean('Time').values

        exp = int(os.path.basename(os.path.normpath(model_results_folder))[3:])
        kernel_info['Exp'].extend(np.repeat(exp, num_cell))
        kernel_info['Cell'].extend(movement_kernel_ds.Cell.values)
        kernel_info['movLeftKernelMean'].extend(move_left_kernel_mean_vals.flatten())
        kernel_info['movRightKernelMean'].extend(move_right_kernel_mean_vals.flatten())
        kernel_info['audSignKernelMean'].extend(aud_sign_kernel_mean_vals.flatten())
        kernel_info['visSignKernelMean'].extend(vis_sign_kernel_mean_vals.flatten())

    kernel_info_df = pd.DataFrame.from_dict(kernel_info)
    kernel_info_df = kernel_info_df.sort_values(by=['Exp', 'Cell'])

    # Get hemisphere info
    """
    passive_neuron_df_w_hemisphere = pd.read_pickle(
        param_dict['neuron_df_w_hem_path']
    )

    passive_MOs_df = passive_neuron_df_w_hemisphere.loc[
        (passive_neuron_df_w_hemisphere['cellLoc'] == 'MOs') &
        (passive_neuron_df_w_hemisphere['subjectRef'] != 1)
        ]
    """

    return kernel_info_df


def convert_model_pickle_to_better_format(model_results_folder):
    """
    One time code to convert some old pickled model results into something more stable, to allow for
    reading using different versions of eg. xarray
    """

    model_result_fname_list = glob.glob(os.path.join(model_results_folder, '*.pkl'))

    for model_result_fname in model_result_fname_list:
        model_result = pd.read_pickle(model_result_fname)

        # Get the exp number
        exp_str = os.path.basename(model_result_fname).split('_')[0]

        if '.' in exp_str:
            exp_str = exp_str.split('.')[0]

        model_results_exp_folder = os.path.join(model_results_folder, exp_str)

        if not os.path.isdir(model_results_exp_folder):
            os.makedirs(model_results_exp_folder)

        # 'params' is in passive modelling results, but not in some active model results
        if 'params' in model_result.keys():
            # save params as one npz file
            param_file_name = '%s_params.npz' % exp_str

            np.savez(os.path.join(model_results_exp_folder, param_file_name), **model_result['params'])


        if 'X_set_results' in model_result.keys():
            # save the individual model files
            for model_name in model_result['X_set_results'].keys():

                model_performance_df = model_result['X_set_results'][model_name]['model_performance_df']
                model_performance_df_save_path = os.path.join(model_results_exp_folder, '%s_model_performance.csv' % model_name)
                model_performance_df.to_csv(model_performance_df_save_path)

                Y_test_actual_all_cell = model_result['X_set_results'][model_name]['Y_test_actual_all_cell']
                np.save(os.path.join(model_results_exp_folder, '%s_Y_test_actual_all_cell' % model_name), Y_test_actual_all_cell)

                Y_test_predict_all_cell = model_result['X_set_results'][model_name]['Y_test_predict_all_cell']
                np.save(os.path.join(model_results_exp_folder, '%s_Y_test_predict_all_cell' % model_name),
                        Y_test_predict_all_cell)

                kernels = model_result['X_set_results'][model_name]['kernels']
                kernel_save_path = os.path.join(model_results_exp_folder, '%s_kernels.nc' % model_name)
                kernels.to_netcdf(kernel_save_path)


        # This is for active modelling results
        if 'feat_idx_dict' in model_result.keys():
            feat_idx_dict_file_name = '%s_feat_idx_dict.npz' % exp_str
            np.savez(os.path.join(model_results_exp_folder, feat_idx_dict_file_name), **model_result['feat_idx_dict'])

        # Active modeling results
        if 'peri_event_time' in model_result.keys():
            peri_event_time_file_name = '%s_peri_event_time.npy' % exp_str
            np.save(os.path.join(model_results_exp_folder, peri_event_time_file_name), model_result['peri_event_time'])

        # Active modeling results
        if 'stim_kernels' in model_result.keys():
            stim_kernels_save_name = '%s_stim_kernels.nc' % exp_str
            stim_kernels = model_result['stim_kernels'].load()
            stim_kernels.to_netcdf(os.path.join(model_results_exp_folder, stim_kernels_save_name))

        # Active modeling results
        if 'mov_kernels' in model_result.keys():
            mov_kernels_save_name = '%s_mov_kernels.nc' % exp_str
            mov_kernels = model_result['mov_kernels'].load()
            mov_kernels.to_netcdf(os.path.join(model_results_exp_folder, mov_kernels_save_name))


    return 0




def plot_model_fit_comparison(all_models_df, model_1_name='addition', model_2_name='interaction',
    fit_metric='varExplained', unity_min=-2, unity_max=1, print_stat=False, xlabel_name=None,
    ylabel_name=None, fig=None, ax=None, transform_vals=None, highlight_neurons=None, pad_ratio=0.1, verbose=True):
    """

    Parameters
    ----------
    all_models_df
    model_1_name
    model_2_name
    xlabel_name
    ylabel_name
    fig
    ax

    Returns
    -------

    """
    model_1_df = all_models_df.loc[(all_models_df['model'] == model_1_name)]
    model_2_df = all_models_df.loc[(all_models_df['model'] == model_2_name)]
    model_1_cv_mean_metric = model_1_df.groupby('neuron').agg('mean')[fit_metric]
    model_2_cv_mean_metric = model_2_df.groupby('neuron').agg('mean')[fit_metric]
    if fig is None:
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(4, 4)
    two_model_cv_mean_metric = np.concatenate([model_1_cv_mean_metric,
     model_2_cv_mean_metric])
    if unity_min is None:
        unity_min = np.min(two_model_cv_mean_metric)
    if unity_max is None:
        unity_max = np.max(two_model_cv_mean_metric)
    unity_vals = np.linspace(unity_min, unity_max, 1000)
    ax.scatter(model_1_cv_mean_metric, model_2_cv_mean_metric,
      s=3, color='black', edgecolor='none')
    ax.plot(unity_vals, unity_vals, linestyle='--', color='gray', zorder=0)
    if highlight_neurons is not None:
        for neuron_info in highlight_neurons:
            neuron_model_1_df = model_1_df.loc[((model_1_df['Exp'] == neuron_info['Exp']) & (model_1_df['Cell'] == neuron_info['neuron']))]
            neuron_model_2_df = model_2_df.loc[((model_2_df['Exp'] == neuron_info['Exp']) & (model_2_df['Cell'] == neuron_info['neuron']))]
            neuron_model_1_fit_metric = np.mean(neuron_model_1_df[fit_metric])
            neuron_model_2_fit_metric = np.mean(neuron_model_2_df[fit_metric])
            if verbose:
                print('Example neuron model 1 metric: %.3f' % neuron_model_1_fit_metric)
                print('Example neuron model 2 metric: %.3f' % neuron_model_2_fit_metric)
            ax.scatter(neuron_model_1_fit_metric, neuron_model_2_fit_metric,
              s=10, color='red', edgecolor='none', zorder=9)

    ax.set_xlim([unity_min, unity_max + unity_max * pad_ratio])
    ax.set_ylim([unity_min, unity_max + unity_max * pad_ratio])
    if print_stat:
        test_stat, p_val = sstats.ttest_rel(model_1_cv_mean_metric, model_2_cv_mean_metric)
        ax.text(0.8, 0.1, s=('$t=%.4f$' % test_stat), transform=(ax.transAxes))
        if p_val < 0.0001:
            p_val_str = '$p < 10^{%.f}$' % np.ceil(np.log10(p_val))
        else:
            p_val_str = '$p = %.4f$' % p_val
        ax.text(0.8, 0.15, s=p_val_str, transform=(ax.transAxes))
    if verbose:
        print('Model 1 mean metric: %.3f' % np.mean(model_1_cv_mean_metric))
        print('Model 2 mean metric: %.3f' % np.mean(model_2_cv_mean_metric))
    if xlabel_name is None:
        xlabel_name = model_1_name
    if ylabel_name is None:
        ylabel_name = model_2_name
    ax.set_xlabel(xlabel_name, size=12)
    ax.set_ylabel(ylabel_name, size=12)
    return fig, ax


def plot_single_cell_fit(regression_results, plot_type='heatmap', train_test_split=True, cell_idx=0, x_sets_to_plot=None,
                         og_data_color='black', model_prediction_colors=['gray', 'green'], include_legend=False,
                         legend_labels=['Data', 'Model'], include_trial_variation_shade=False,
                         variation_shade='std', fig=None, axs=None):
    """

    Parameters
    ----------
    regression_results
    plot_type
    train_test_split
    cell_idx
    x_sets_to_plot
    og_data_color
    model_prediction_colors
    include_legend
    legend_labels
    include_trial_variation_shade
    variation_shade : (str)
        what measure to plot the shading
        'std' : +/- 1 standard deviation
        'sem' : +/- 1 standard error
    fig
    axs

    Returns
    -------

    """
    if plot_type == 'heatmap':
        if fig is None:
            if axs is None:
                fig, axs = plt.subplots(1, 4)
                fig.set_size_inches(15, 4)
        axs[0].imshow(Y, aspect='auto')
        axs[0].set_ylabel('Trials', size=12)
        axs[1].imshow(X, aspect='auto')
        axs[1].set_xticks(np.arange(len(feature_set)))
        axs[1].set_xticklabels(feature_set)
        axs[2].imshow(W, aspect='auto')
        axs[2].set_yticks(np.arange(len(feature_set)))
        axs[2].set_yticklabels(feature_set)
        axs[2].set_ylim([-0.5, len(feature_set) - 0.5])
        axs[3].imshow(Y_hat, aspect='auto')
        fig.tight_layout()
    elif plot_type == 'four-cond-psth':
        if fig is None and axs is None:
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
            fig.set_size_inches(5, 5)
        peri_event_time = regression_results['params']['peri_stimulus_time']
        X_set_results = regression_results['X_set_results']
        stim_cond_names = regression_results['params']['stim_cond_names']
        if x_sets_to_plot is None:
            x_sets_to_plot = list(X_set_results.keys())
        stim_cond_to_plot_loc = {'alvr':[0, 0],  'alvl':[1, 0],  'arvr':[0, 1], 'arvl':[1, 1]}

        for n_x_set, x_set in enumerate(x_sets_to_plot):
            x_set_data = X_set_results[x_set]
            if train_test_split:
                test_set_actual_data = x_set_data['Y_test_actual_all_cell']
                test_set_prediction = x_set_data['Y_test_predict_all_cell']
            else:
                test_set_actual_data = x_set_data['Y']
                test_set_prediction = x_set_data['Y_predict']
            for n_stim_cond, stim_cond in enumerate(stim_cond_names):
                x_loc, y_loc = stim_cond_to_plot_loc[stim_cond]
                if n_x_set == 0:
                    axs[(x_loc, y_loc)].plot(peri_event_time, (test_set_actual_data[cell_idx, n_stim_cond, :]), color=og_data_color)
                axs[(x_loc, y_loc)].plot(peri_event_time, (test_set_prediction[cell_idx, n_stim_cond, :]), color=(model_prediction_colors[n_x_set]))
                if include_trial_variation_shade:
                    Y_original_ds = regression_results['Y_original']
                    test_set_actual_data_vals = test_set_actual_data[cell_idx, n_stim_cond, :]
                    if stim_cond == 'alvr':
                        Y_vals = Y_original_ds.where(((Y_original_ds['audDiff'] < 0) & (Y_original_ds['visDiff'] > 0)),
                          drop=True).isel(Cell=cell_idx)['smoothed_fr'].values
                    elif stim_cond == 'arvl':
                        Y_vals = Y_original_ds.where(((Y_original_ds['audDiff'] > 0) & (Y_original_ds['visDiff'] < 0)),
                              drop=True).isel(Cell=cell_idx)['smoothed_fr'].values
                    elif stim_cond == 'arvr':
                        Y_vals = Y_original_ds.where(((Y_original_ds['audDiff'] > 0) & (Y_original_ds['visDiff'] > 0)),
                                  drop=True).isel(Cell=cell_idx)['smoothed_fr'].values
                    elif stim_cond == 'alvl':
                        Y_vals = Y_original_ds.where(((Y_original_ds['audDiff'] < 0) & (Y_original_ds['visDiff'] < 0)),
                                      drop=True).isel(Cell=cell_idx)['smoothed_fr'].values
                    num_trials = np.shape(Y_vals)[0]
                    if variation_shade == 'std':
                        test_set_actual_data_variation = np.std(Y_vals, axis=0)
                    elif variation_shade == 'sem':
                        test_set_actual_data_variation = np.std(Y_vals, axis=0) / np.sqrt(num_trials)
                        test_set_actual_data_upper = test_set_actual_data_vals + test_set_actual_data_variation
                        test_set_actual_data_lower = test_set_actual_data_vals - test_set_actual_data_variation
                        axs[(x_loc, y_loc)].fill_between(peri_event_time, test_set_actual_data_lower, test_set_actual_data_upper,
                          color='gray',
                          alpha=0.3,
                          zorder=0,
                          lw=0.0)

            axs[(0, 0)].set_title('$A_LV_R$', size=12)
            axs[(1, 0)].set_title('$A_LV_L$', size=12)
            axs[(0, 1)].set_title('$A_RV_R$', size=12)
            axs[(1, 1)].set_title('$A_RV_L$', size=12)
            if include_legend:
                all_colors = [
                 og_data_color]
                all_colors.extend(model_prediction_colors[0:len(x_sets_to_plot)])
                custom_lines = [mpl.lines.Line2D([0], [0], color=x, lw=2) for x in all_colors]
                fig.legend(custom_lines, legend_labels, bbox_to_anchor=(1.04, 0.5))
            fig.text(0.5, 0, 'Peri-stimulus time (s)', size=12, ha='center')
            fig.text(0, 0.5, 'Spike/s', size=12, va='center', rotation=90)
    elif plot_type == 'nine-cond-psth':
        if fig is None:
            if axs is None:
                fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
                fig.set_size_inches(8, 8)
        peri_event_time = regression_results['params']['peri_stimulus_time']
        X_set_results = regression_results['X_set_results']
        stim_cond_names = regression_results['params']['stim_cond_names']
        if x_sets_to_plot is None:
            x_sets_to_plot = list(X_set_results.keys())
        stim_cond_to_plot_loc = {'alvr':[0, 0],  'alvl':[2, 0],  'arvr':[
          0, 2],
         'arvl':[2, 2],  'a0vr':[
          0, 1],
         'a0vl':[2, 1],  'arv0':[
          1, 2],
         'alv0':[1, 0]}
        for n_x_set, x_set in enumerate(x_sets_to_plot):
            x_set_data = X_set_results[x_set]
            if train_test_split:
                test_set_actual_data = x_set_data['Y_test_actual_all_cell']
                test_set_prediction = x_set_data['Y_test_predict_all_cell']
            else:
                test_set_actual_data = x_set_data['Y']
                test_set_prediction = x_set_data['Y_predict']
            for n_stim_cond, stim_cond in enumerate(stim_cond_names):
                x_loc, y_loc = stim_cond_to_plot_loc[stim_cond]
                if n_x_set == 0:
                    axs[(x_loc, y_loc)].plot(peri_event_time, (test_set_actual_data[cell_idx, n_stim_cond, :]), color=og_data_color)
                axs[(x_loc, y_loc)].plot(peri_event_time, (test_set_prediction[cell_idx, n_stim_cond, :]), color=(model_prediction_colors[n_x_set]))
                if include_trial_variation_shade:
                    Y_original_ds = regression_results['Y_original']
                    test_set_actual_data_vals = test_set_actual_data[cell_idx, n_stim_cond, :]
                    if stim_cond == 'alvr':
                        Y_vals = Y_original_ds.where(((Y_original_ds['audDiff'] < 0) & (Y_original_ds['visDiff'] > 0)),
                          drop=True).isel(Cell=cell_idx)['smoothed_fr'].values
                    else:
                        if stim_cond == 'arvl':
                            Y_vals = Y_original_ds.where(((Y_original_ds['audDiff'] > 0) & (Y_original_ds['visDiff'] < 0)),
                              drop=True).isel(Cell=cell_idx)['smoothed_fr'].values
                        else:
                            if stim_cond == 'arvr':
                                Y_vals = Y_original_ds.where(((Y_original_ds['audDiff'] > 0) & (Y_original_ds['visDiff'] > 0)),
                                  drop=True).isel(Cell=cell_idx)['smoothed_fr'].values
                            else:
                                if stim_cond == 'alvl':
                                    Y_vals = Y_original_ds.where(((Y_original_ds['audDiff'] < 0) & (Y_original_ds['visDiff'] < 0)),
                                      drop=True).isel(Cell=cell_idx)['smoothed_fr'].values
                                else:
                                    if stim_cond == 'a0vl':
                                        Y_vals = Y_original_ds.where(((Y_original_ds['audDiff'] == np.inf) & (Y_original_ds['visDiff'] < 0)),
                                          drop=True).isel(Cell=cell_idx)['smoothed_fr'].values
                                    else:
                                        if stim_cond == 'a0vr':
                                            Y_vals = Y_original_ds.where(((Y_original_ds['audDiff'] == np.inf) & (Y_original_ds['visDiff'] > 0)),
                                              drop=True).isel(Cell=cell_idx)['smoothed_fr'].values
                                        else:
                                            if stim_cond == 'alv0':
                                                Y_vals = Y_original_ds.where(((Y_original_ds['audDiff'] < 0) & (Y_original_ds['visDiff'] == 0)),
                                                  drop=True).isel(Cell=cell_idx)['smoothed_fr'].values
                                            else:
                                                if stim_cond == 'arv0':
                                                    Y_vals = Y_original_ds.where(((Y_original_ds['audDiff'] > 0) & (Y_original_ds['visDiff'] == 0)),
                                                      drop=True).isel(Cell=cell_idx)['smoothed_fr'].values
                    num_trials = np.shape(Y_vals)[0]
                    if variation_shade == 'std':
                        test_set_actual_data_variation = np.std(Y_vals, axis=0)
                    else:
                        if variation_shade == 'sem':
                            test_set_actual_data_variation = np.std(Y_vals, axis=0) / np.sqrt(num_trials)
                        test_set_actual_data_upper = test_set_actual_data_vals + test_set_actual_data_variation
                        test_set_actual_data_lower = test_set_actual_data_vals - test_set_actual_data_variation
                        axs[(x_loc, y_loc)].fill_between(peri_event_time, test_set_actual_data_lower, test_set_actual_data_upper,
                          color='gray',
                          alpha=0.3,
                          zorder=0,
                          lw=0.0)

        axs[(0, 0)].set_title('$A_LV_R$', size=12)
        axs[(2, 0)].set_title('$A_LV_L$', size=12)
        axs[(0, 2)].set_title('$A_RV_R$', size=12)
        axs[(2, 2)].set_title('$A_RV_L$', size=12)
        axs[(0, 1)].set_title('$A_0V_R$', size=12)
        axs[(2, 1)].set_title('$A_0V_L$', size=12)
        axs[(1, 2)].set_title('$A_RV_0$', size=12)
        axs[(1, 0)].set_title('$A_LV_0$', size=12)
        if include_legend:
            all_colors = [
             og_data_color]
            all_colors.extend(model_prediction_colors[0:len(x_sets_to_plot)])
            custom_lines = [mpl.lines.Line2D([0], [0], color=x, lw=2) for x in all_colors]
            fig.legend(custom_lines, legend_labels, bbox_to_anchor=(1.04, 0.5))
        fig.text(0.5, 0, 'Peri-stimulus time (s)', size=12, ha='center')
        fig.text(0, 0.5, 'Spike/s', size=12, va='center', rotation=90)
    return fig, axs


def plot_fitted_kernels(regression_results, cell_idx, x_sets_to_plot=None, kernel_colors=['red', 'red', 'red'],
                        kernel_names=['StimOn', 'Aud L/R', 'Vis L/R'], train_test_split=True,
                        fig=None, axs=None):


    if x_sets_to_plot is None:
        x_sets_to_plot = list(regression_results['X_set_results'].keys())

    peri_event_time = regression_results['params']['peri_stimulus_time']

    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(1, 3, sharey=True)
        fig.set_size_inches(7, 3)

    for x_set in x_sets_to_plot:
        W_train = regression_results['X_set_results'][x_set]['W_train']
        num_kernels = np.shape(W_train)[1]

        for n_kernel in np.arange(num_kernels):
            axs[n_kernel].plot(peri_event_time,
                               W_train[cell_idx, n_kernel, :], color=kernel_colors[n_kernel])

            axs[n_kernel].set_title(kernel_names[n_kernel], size=12)

    axs[0].set_ylabel('Spikes/s', size=12)
    fig.text(0.5, 0, 'Peri-stimulus time (s)', size=12, ha='center')
    fig.tight_layout()

    return fig, axs


if __name__ == '__main__':
    main()