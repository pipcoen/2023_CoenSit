"""
This script generates figure 4A in the paper
This is the example additive model fit to a neuron during the active condition, showing the kernels fitted.
Internal notes:
This is from src/data/figure_for_paper_regression_model.py
"""
import os
import numpy as np
import pickle as pkl
import pandas as pd
import src.models.kernel_regression as kernel_regression
import src.models.psth_regression as psth_regression
import src.visualization.vizregression as vizregression
import src.visualization.vizmodel as vizmodel
import src.data.process_ephys_data as pephys
import src.data.analyse_spikes as anaspikes

import sklearn.linear_model as sklinear
import src.data.alignment_mean_subtraction as ams

from tqdm import tqdm
import xarray as xr

import scipy.stats as sstats

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sciplotlib.polish as splpolish
import sciplotlib.style as splstyle
import pdb
import glob

from collections import defaultdict


main_data_folder = '/Volumes/Partition 1/data/interim'
fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
fig_name = 'fig-4a.pdf'
subject = 2
exp = 15
cell_idx = 23
split_random_state = [18]  # np.arange(0, 40).tolist()
random_seed = split_random_state[0]
include_trial_variation_shade = False
variation_shade = 'std'
x_sets_to_plot = ['addition', 'interaction']
model_prediction_colors = ['gray', 'orange', 'purple', 'green', 'red']
plot_type = 'four-cond-psth'
smooth_sigma = 30
smooth_window_width = 50

def main():
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

    # active_only_model_feature_set = ['stimOn', 'audSign', 'visSign',
    #                                 'moveLeft', 'moveRight']

    active_only_model_feature_set = ['stimOn', 'audSign', 'visSign', 'moveOnset', 'moveDiff']
    event_start_ends = {'baseline': [-0.2, 0.7],
                        'stimOn': [-0.2, 0.7],
                        'audSign': [-0.2, 0.7],
                        'visSign': [-0.2, 0.7],
                        'audSignVisSign': [-0.2, 0.7],
                        'moveLeft': [-0.2, 0.7],
                        'moveRight': [-0.2, 0.7],
                        'moveOnset': [-0.2, 0.7],
                        'moveDiff': [-0.2, 0.7]}

    custom_model = sklinear.Ridge(alpha=0.01, fit_intercept=False)

    vis_contrast_levels = np.array([0.8])
    coherent_vis_contrast_levels = np.array([0.1, 0.2, 0.4, 0.8])
    conflict_vis_contrast_levels = np.array([0.1, 0.2, 0.4, 0.8])

    test_size = 0.5  # size of test set (proportion of trials)
    # random_seed = 4 (original when I looked at this on 2023-03-27)
    random_seed = 1  # 18, 20, 30, 14, 23 is okay, 27 is quite good, 35 is quite good

    # for moveOnset + moveDiff
    # 1 is good

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

    model_weights = model_data['model'].coef_

    line_color = 'gray'


    peri_event_time = model_data['peri_event_time']

    # 2021-03-25 Adding a subset of the stimulus time so it matches that of the passive stimulus kernels
    subset_peri_event_time_window = [-0.1, 0.4]

    if subset_peri_event_time_window is not None:
        subset_time_index = np.where(
            (peri_event_time >= subset_peri_event_time_window[0]) &
            (peri_event_time <= subset_peri_event_time_window[1])
        )[0]

    with plt.style.context(splstyle.get_style('nature-reviews')):

        fig, axs = plt.subplots(1, len(model_data['feat_idx_dict']), sharex=False, sharey=True)
        fig.set_size_inches(12, 3)
        for n_feat, (feature, feat_idx) in enumerate(model_data['feat_idx_dict'].items()):

            if feature in ['moveLeft', 'moveRight', 'audSignVisSignmoveLeft']:
                # num_idx_to_start = np.argmin(np.abs(peri_event_time)).values
                num_idx_to_start = np.argmin(np.abs(peri_event_time.values))
                axs[n_feat].plot(peri_event_time[num_idx_to_start:] - 0.2,
                                 model_weights[cell_idx, feat_idx][num_idx_to_start:],
                                 color=line_color)
                axs[n_feat].set_xlabel('Peri-movement time (s)', size=11)
            else:

                axs[n_feat].plot(peri_event_time, model_weights[cell_idx, feat_idx],
                                 color=line_color)
                axs[n_feat].set_xlabel('Peri-stimulus time (s)', size=11)

        axs[0].set_title('Stimulus onset', size=12)
        axs[1].set_title('Auditory left/right', size=12)
        axs[2].set_title('Visual left/right', size=12)
        # axs[3].set_title('Move left', size=12)
        # axs[4].set_title('Move right', size=12)

        axs[3].set_title('Move onset', size=12)
        axs[4].set_title('Move direction', size=12)

        axs[0].set_ylabel('Firing rate (spike/s)', size=12)

        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()