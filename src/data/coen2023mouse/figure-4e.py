"""
This script generates figure 4e in the paper
This is the example additive model fit to a neuron during the passive condition.

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
fig_name = 'fig-4e.pdf'


def main():

    subject = 2
    exp = 15
    cell_idx = 23
    # split_random_state = [20]
    split_random_state = 20
    custom_ylim = None
    include_trial_variation_shade = True
    variation_shade = 'std'


    # fig_folder = '/media/timsit/Partition 1/reports/figures/figure-5-for-pip/example_passive_psth_fit',
    stim_alignment_folder = os.path.join(main_data_folder, 'passive-m2-new-parent-alignment-2ms')
    x_sets_to_plot = ['addition', 'interaction']

    # split_random_state = param_dict['split_random_state']
    # Good random seeds: 20
    # stim_alignment_folder = param_dict['stim_alignemnt_folder']
    # fig_folder = param_dict['fig_folder']
    # subject = param_dict['subject']
    # exp = param_dict['exp']
    # cell_idx = param_dict['cell_idx']
    # custom_ylim = param_dict['custom_ylim']
    legend_labels = ['Data', 'Addition', 'Interaction']
    plot_type = 'four-cond-psth'
    model_prediction_colors = ['gray', 'orange', 'purple', 'green', 'red']
    # include_trial_variation_shade = param_dict['include_trial_variation_shade']



    target_brain_region = 'MOs'
    smooth_spikes = True
    # time_bin_width = 2 / 1000  # 2 ms time bins
    # smooth_sigma_in_sec = 25 / 1000
    # smooth_sigma_in_frames = smooth_sigma_in_sec / time_bin_width
    smooth_window = 50
    smooth_sigma = 30

    # smooth_window = 10
    # smooth_sigma = 25

    feature_sets = {
       'audOnOnly': ['stimOn'],
       'visOnly': ['stimOn', 'visSign'],
       'audOnly': ['stimOn', 'audSign'],
       'addition':  ['stimOn', 'audSign', 'visSign'],
       'interaction': ['stimOn', 'audSign', 'visSign', 'audVis']}


    # feature_sets={'addition':  ['stimOn', 'audSign', 'visSign'],
    #                                        'interaction': ['stimOn', 'audSign', 'visSign', 'audVis']}

    alignment_ds = pephys.load_subject_exp_alignment_ds(alignment_folder=stim_alignment_folder,
                              subject_num=subject, exp_num=exp,
                              target_brain_region=target_brain_region,
                              aligned_event='stimOnTime',
                              alignment_file_ext='.nc')

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


    evaluation_method = 'train-test-split'
    # evaluation_method = 'fit-all'
    train_test_split = True
    test_size = 0.5

    regression_results = psth_regression.alignment_ds_to_regression_results(
                                           alignment_ds.load(), mean_same_stim=True,
                                           evaluation_method=evaluation_method, split_random_state=split_random_state,
                                           feature_sets=feature_sets, test_size=test_size,
                                           subset_stim_cond=[{'audDiff': 60, 'visDiff': 0.8},
                                                             {'audDiff': -60, 'visDiff': -0.8},
                                                             {'audDiff': 60, 'visDiff': -0.8},
                                                             {'audDiff': -60, 'visDiff': 0.8}],
                                           include_single_trials=True)

    # if fig_folder is None:
    #     fig_folder = '/media/timsit/Partition 1/reports/figures/figure-5-for-pip/'
    include_legend = True

    """
    if train_test_split:
        fig_name = '5_s%.fe%.fMOs%.f_four_cond_plot_test_set_addition_interaction_random_seed_%.f.pdf' % (
        subject, exp, cell_idx, split_random_state)
    else:
        fig_name = '5_s%.fe%.fMOs%.f_four_cond_plot_fit_all.pdf' % (subject, exp, cell_idx)
    """

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, axs = psth_regression.plot_single_cell_fit(regression_results, plot_type=plot_type,
                                                        train_test_split=train_test_split,
                                                        model_prediction_colors=model_prediction_colors,
                                                        x_sets_to_plot=x_sets_to_plot,
                                                        include_legend=include_legend,
                                                        cell_idx=cell_idx, fig=None, axs=None,
                                                        legend_labels=legend_labels,
                                                        include_trial_variation_shade=include_trial_variation_shade,
                                                        variation_shade=variation_shade)

        if custom_ylim is not None:
            axs[0, 0].set_ylim(custom_ylim)

        if fig_folder is not None:
            fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()