"""
This script produces Figure 4B of the paper.
This is the regression of a neuron during the active condition : Example fit

Internal notes:
This is from : 14.5d-active-only-3-plus-2-versus-4-plus-2-psth-fit-checkpoint
"""

import os
import numpy as np
import pickle as pkl
import pandas as pd
import src.models.kernel_regression as kernel_regression
import src.data.process_ephys_data as pephys
import src.data.analyse_spikes as anaspikes
import src.data.alignment_mean_subtraction as ams
import src.models.psth_regression as psth_regression
import sklearn.linear_model as sklinear
import scipy.spatial as sspatial

import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sciplotlib.polish as splpolish
import sciplotlib.style as splstyle

import glob
import matplotlib as mpl

from tqdm import tqdm

from collections import defaultdict
import itertools
import scipy.stats as sstats

import src.visualization.vizpikes as vizpikes

import pdb


main_data_folder = '/Volumes/Partition 1/data/interim'
fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
fig_name = 'fig-4b.pdf'

def main():
    exp = 15
    cell_idx = 23
    random_seed = 30  # on my M1 Mac, just in case this change with operating systems (usually trying first 40 is sufficient)

    # load aligned data
    active_MOs_alignment_fname = '/Volumes/Partition 1/data/interim/active-m2-choice-init-alignment/alignedToStim2ms/subject_2_exp_15_MOs_aligned_to_stimOnTime.nc'

    train_test_split = 'per-stim-cond'
    stim_cond_list = ['arvr', 'arvl', 'alvr', 'alvl']

    rt_var_name = 'choiceInitTimeRelStim'
    max_rt = 0.3

    smooth_sigma = 30
    smooth_window_width = 50
    subset_time_window = [-0.2, 1]

    # active_only_model_feature_set = ['baseline', 'audSign', 'visSign',
    #                                  'moveLeft', 'moveRight']

    active_only_model_feature_set = ['stimOn', 'audSign', 'visSign',
                                     'moveLeft', 'moveRight']

    custom_model = sklinear.Ridge(alpha=0.01, fit_intercept=False)

    vis_contrast_levels = np.array([0.8])
    coherent_vis_contrast_levels = np.array([0.1, 0.2, 0.4, 0.8])
    conflict_vis_contrast_levels = np.array([0.1, 0.2, 0.4, 0.8])

    test_size = 0.5  # size of test set (proportion of trials)
    # random_seed = 4 (original when I looked at this on 2023-03-27)


    active_align_to_stim_ds = xr.open_dataset(active_MOs_alignment_fname)
    active_align_to_stim_ds = active_align_to_stim_ds.isel(Exp=0)

    # smooth spikes : active
    cell_active_stim_aligned_ds = active_align_to_stim_ds.stack(trialTime=['Trial', 'Time'])
    cell_active_stim_aligned_ds['smoothed_fr'] = (['Cell', 'trialTime'], anaspikes.smooth_spikes(
        cell_active_stim_aligned_ds['firing_rate'],
        method='half_gaussian',
        sigma=smooth_sigma, window_width=smooth_window_width,
        custom_window=None))
    active_stim_aligned_ds = cell_active_stim_aligned_ds.unstack()

    event_start_ends = {'baseline': [-0.2, 0.7],
                        'stimOn': [-0.2, 0.7],
                        'audSign': [-0.2, 0.7],
                        'visSign': [-0.2, 0.7],
                        'audSignVisSign': [-0.2, 0.7],
                        'moveLeft': [-0.2, 0.7],
                        'moveRight': [-0.2, 0.7]}

    # Fit regression model
    fit_active_model_error, three_plus_two_model_fits, model_data = ams.fit_active_only_model(
        active_stim_aligned_ds, stim_cond_list,
        coherent_vis_contrast_levels,
        conflict_vis_contrast_levels, vis_contrast_levels,
        subset_time_window, test_size=test_size,
        rt_variable_name='choiceInitTimeRelStim',
        activity_name='smoothed_fr',
        random_seed=random_seed, return_fits=True,
        error_metric_method='ave-per-stimulus',
        feature_set=active_only_model_feature_set,
        return_model_data=True,
        event_start_ends=event_start_ends,
        custom_model=custom_model)

    fit_active_model_error, four_plus_two_model_fits, model_data = ams.fit_active_only_model(
        active_stim_aligned_ds, stim_cond_list,
        coherent_vis_contrast_levels,
        conflict_vis_contrast_levels, vis_contrast_levels,
        subset_time_window, test_size=test_size,
        rt_variable_name='choiceInitTimeRelStim',
        activity_name='smoothed_fr',
        random_seed=random_seed, return_fits=True,
        error_metric_method='ave-per-stimulus',
        feature_set=['stimOn', 'audSign', 'visSign', 'audSignVisSign'
                                                     'moveLeft', 'moveRight'],
        return_model_data=True,
        event_start_ends=event_start_ends,
        custom_model=custom_model)

    choice_var_name = 'choiceThreshDir'

    three_plus_two_cell_ds = three_plus_two_model_fits.isel(Cell=cell_idx)
    four_plus_two_cell_ds = four_plus_two_model_fits.isel(Cell=cell_idx)

    three_plus_two_cell_ds['Y_test_predict_4plus2'] = four_plus_two_cell_ds['Y_test_predict']

    alvr_cl = three_plus_two_cell_ds.where(
        (three_plus_two_cell_ds['visDiff'] > 0) &
        (three_plus_two_cell_ds['audDiff'] == -60) &
        (three_plus_two_cell_ds[choice_var_name] == 1), drop=True
    )

    alvr_cr = three_plus_two_cell_ds.where(
        (three_plus_two_cell_ds['visDiff'] > 0) &
        (three_plus_two_cell_ds['audDiff'] == -60) &
        (three_plus_two_cell_ds[choice_var_name] == 2), drop=True
    )

    arvl_cl = three_plus_two_cell_ds.where(
        (three_plus_two_cell_ds['visDiff'] < 0) &
        (three_plus_two_cell_ds['audDiff'] == 60) &
        (three_plus_two_cell_ds[choice_var_name] == 1), drop=True
    )

    arvl_cr = three_plus_two_cell_ds.where(
        (three_plus_two_cell_ds['visDiff'] < 0) &
        (three_plus_two_cell_ds['audDiff'] == 60) &
        (three_plus_two_cell_ds[choice_var_name] == 2), drop=True
    )

    alvl_cl = three_plus_two_cell_ds.where(
        (three_plus_two_cell_ds['visDiff'] < 0) &
        (three_plus_two_cell_ds['audDiff'] == -60) &
        (three_plus_two_cell_ds[choice_var_name] == 1), drop=True
    )

    arvr_cr = three_plus_two_cell_ds.where(
        (three_plus_two_cell_ds['visDiff'] > 0) &
        (three_plus_two_cell_ds['audDiff'] == 60) &
        (three_plus_two_cell_ds[choice_var_name] == 2), drop=True
    )

    peri_event_time = alvr_cl.PeriEventTime.isel(Trial=0).values

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)
        fig.set_size_inches(10, 5)

        # ALVR CL
        axs[1, 1].plot(peri_event_time, alvr_cl['Y_test'].mean('Trial'), color='black')
        axs[1, 1].plot(peri_event_time, alvr_cl['Y_test_predict'].mean('Trial'), color='gray')
        axs[1, 1].plot(peri_event_time, alvr_cl['Y_test_predict_4plus2'].mean('Trial'), color='red')
        axs[1, 1].set_title(r'$A_LV_R$ Choose left', size=12)

        # ALVR CR
        axs[1, 3].plot(peri_event_time, alvr_cr['Y_test'].mean('Trial'), color='black')
        axs[1, 3].plot(peri_event_time, alvr_cr['Y_test_predict'].mean('Trial'), color='gray')
        axs[1, 3].plot(peri_event_time, alvr_cr['Y_test_predict_4plus2'].mean('Trial'), color='red')
        axs[1, 3].set_title(r'$A_LV_R$ Choose right', size=12)

        # ARVL CL
        axs[0, 0].set_title(r'$A_RV_L$ Choose left', size=12)
        axs[0, 0].plot(peri_event_time, arvl_cl['Y_test'].mean('Trial'), color='black')
        axs[0, 0].plot(peri_event_time, arvl_cl['Y_test_predict'].mean('Trial'), color='gray')
        axs[0, 0].plot(peri_event_time, arvl_cl['Y_test_predict_4plus2'].mean('Trial'), color='red')

        # ARVL CR
        axs[0, 2].set_title(r'$A_RV_L$ Choose right', size=12)
        axs[0, 2].plot(peri_event_time, arvl_cr['Y_test'].mean('Trial'), color='black')
        axs[0, 2].plot(peri_event_time, arvl_cr['Y_test_predict'].mean('Trial'), color='gray')
        axs[0, 2].plot(peri_event_time, arvl_cr['Y_test_predict_4plus2'].mean('Trial'), color='red')

        # ALVL CL
        axs[1, 0].set_title(r'$A_LV_L$ Choose left', size=12)
        axs[1, 0].plot(peri_event_time, alvl_cl['Y_test'].mean('Trial'), color='black')
        axs[1, 0].plot(peri_event_time, alvl_cl['Y_test_predict'].mean('Trial'), color='gray')
        axs[1, 0].plot(peri_event_time, alvl_cl['Y_test_predict_4plus2'].mean('Trial'), color='red')

        # ARVR CR
        axs[0, 3].set_title(r'$A_RV_R$ Choose right', size=12)
        axs[0, 3].plot(peri_event_time, arvr_cr['Y_test'].mean('Trial'), color='black')
        axs[0, 3].plot(peri_event_time, arvr_cr['Y_test_predict'].mean('Trial'), color='gray')
        axs[0, 3].plot(peri_event_time, arvr_cr['Y_test_predict_4plus2'].mean('Trial'), color='red')

        # Remove axis with no condition
        axs[0, 1].axis('off')
        axs[1, 2].axis('off')

        fig.text(0.55, 0, 'Peri-stimulus time (s)', ha='center', size=12)
        fig.text(0.0, 0.5, 'Firing rate (spikes/s)', va='center', rotation=90, size=12)

        custom_legend = [Line2D([0], [0], color='black', lw=2),
                         Line2D([0], [0], color='gray', lw=2),
                         Line2D([0], [0], color='red', lw=2)]

        fig.legend(custom_legend, ['Test set data', '3 + 2 model', '4 + 2 model'],
                   bbox_to_anchor=(1.3, 0.6))

        fig.tight_layout()

        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
