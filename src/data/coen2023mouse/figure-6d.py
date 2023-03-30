"""
This script plots figure 6d of the paper.
This is the mouse reaction time + trained model reaction time + naive model reaction time
"""

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

import jax
# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')

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

import src.visualization.report_plot_model_vs_mouse_behaviour as plot_mouse_vs_model
import src.data.analyse_behaviour as anabehave
import src.data.struct_to_dataframe as stdf



fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
fig_name = 'fig-6d.pdf'

def main():

    # Load naive model data
    model_number = 47
    target_random_seed = 0
    model_combined_df_path = '/Volumes/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f/model_output_and_behaviour/model_behaviour_df_test_seed_%.f.pkl' % (
    model_number, target_random_seed)

    model_combined_df = pd.read_pickle(model_combined_df_path)

    model_combined_df = anabehave.get_df_stim_cond(
        subset_active_behaviour_df=model_combined_df,
        include_mouse_number=False,
        include_exp_number=False,
        mouse_rt_var_name=None,
        mouse_choice_var_name=None,
        fields_to_include=['audCond', 'visCond', 'reactionTime'],
        vis_cond_name='visCond', aud_cond_name='audCond',
        verbose=False)

    # Load mouse data
    save_folder = '/Volumes/Partition 1/data/interim/model-vs-mouse-rt/'
    mouse_four_cond_rt = stdf.loadmat(
        '/Volumes/Partition 1/data/interim/mouse-rt/four_tim_cond_rt_diff_contrast_sets.mat')
    model_all_cond_rt = np.median(model_combined_df['reactionTime'])
    model_coherent_rt = np.median(model_combined_df.loc[
                                      model_combined_df['condType'] == 'coherent'
                                      ]['reactionTime'])

    model_vis_rt = np.median(model_combined_df.loc[
                                 model_combined_df['condType'] == 'vis_only'
                                 ]['reactionTime'])

    print('Overall median: %.3f' % model_all_cond_rt)
    print('Vis only median: %.3f' % model_vis_rt)
    print('Coherent median: %.3f' % model_coherent_rt)

    # load trained model data
    model_number = 20
    model_result_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/drift-model-%.f/' % model_number
    # alignment_ds_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/kernel_sig_66_neurons_samples.nc'
    # subset_neurons_df_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/kernel_66_neurons_df.pkl'

    # alignment_ds_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/kernel_sig_133_neurons_samples.nc'
    # subset_neurons_df_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/kernel_133_neurons_df.pkl'

    extract_model_behaviour_from_output = False
    get_mouse_combined_df = False  # This is for ephys behaviour (with matching neurons)
    get_all_mouse_combined_df = False  # This is for all behaviour (either 17 mice, or all behaviour from 5 mice)
    save_format = 'csv'
    get_mouse_conflict_combined_df = False
    mouse_active_behave_df_path = '/media/timsit/Partition 1/data/interim/active-m2-choice-init/subset/ephys_behaviour_df.pkl'
    target_random_seed = None
    # Name of the files to save model output and behaviour dataframe
    model_output_fname = 'model_output_per_stim_cond_test_seed_0_1_2.pkl'
    model_behaviour_output_fname = 'model_behaviour_df_test_seed_0_1_2.pkl'

    og_model_combined_df_path = '/Volumes/Ultra Touch/multispaceworld-rnn-models/drift-model-20/model_output_and_behaviour/model_behaviour_df_test_seed_0_1.pkl'
    # mouse_combined_df_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/drift-model-22/model_output_and_behaviour/matching_mouse_behaviour_df.pkl'

    # og_model_combined_df_path = '/Volumes/Ultra Touch/multispaceworld-rnn-models/drift-model-22/model_output_and_behaviour/model_behaviour_df_test_seed_0.pkl'
    # mouse_combined_df_path = '/Volumes/Ultra Touch/multispaceworld-rnn-models/drift-model-23/model_output_and_behaviour/matching_mouse_behaviour_df.pkl'
    og_model_combined_df = pd.read_pickle(og_model_combined_df_path)
    # mouse_combined_df = pd.read_pickle(mouse_combined_df_path)

    # some processing of model_combined_df

    og_model_combined_df = anabehave.get_df_stim_cond(
        subset_active_behaviour_df=og_model_combined_df,
        include_mouse_number=False,
        include_exp_number=False,
        mouse_rt_var_name=None,
        mouse_choice_var_name=None,
        fields_to_include=['audCond', 'visCond', 'reactionTime'],
        vis_cond_name='visCond', aud_cond_name='audCond',
        verbose=False)

    og_model_all_cond_rt = np.median(og_model_combined_df['reactionTime'])
    og_model_coherent_rt = np.median(og_model_combined_df.loc[
                                         og_model_combined_df['condType'] == 'coherent'
                                         ]['reactionTime'])

    og_model_vis_rt = np.median(og_model_combined_df.loc[
                                    og_model_combined_df['condType'] == 'vis_only'
                                    ]['reactionTime'])


    # Process mouse data
    all_active_behaviour = pd.read_pickle(
        '/Volumes/Partition 1/data/interim/multispaceworld-all-active-behaviour/ephys_behaviour_df.pkl')
    min_rt = 0.0
    max_rt = 10.0
    # rt_var_name = 'timeToFirstMove'
    # rt_var_name = 'threshMoveTime'
    rt_var_name = 'reactionTime'
    # choice_var_name = 'threshMoveDirection'
    choice_var_name = 'responseCalc'

    max_time_to_feedback = 1.5

    remove_laser_power_on_trials = True

    subset_active_behaviour_df = all_active_behaviour.loc[
        ~all_active_behaviour['subjectId'].isin(['PC011', 'PC012', 'PC013', 'PC015'])
    ]

    subset_active_behaviour_df = subset_active_behaviour_df.loc[
        subset_active_behaviour_df['validTrial'] == 1
        ]

    subset_active_behaviour_df = subset_active_behaviour_df.loc[
        subset_active_behaviour_df[rt_var_name] <= max_rt
        ]

    if max_time_to_feedback is not None:
        subset_active_behaviour_df = subset_active_behaviour_df.loc[
            subset_active_behaviour_df['timeToFeedback'] <= max_time_to_feedback
            ]

    if remove_laser_power_on_trials:

        if 'laserPower' in subset_active_behaviour_df.columns:
            print('Removing laser power on trials')
            subset_active_behaviour_df = subset_active_behaviour_df.loc[
                (subset_active_behaviour_df['laserType'] == 0) |
                np.isnan(subset_active_behaviour_df['laserType'])
                ]

    include_mouse_number = True

    if include_mouse_number:

        rt_df = subset_active_behaviour_df[['audDiff', 'visDiff', rt_var_name, 'subjectRef', 'expRef',
                                            choice_var_name]].dropna()

    else:

        rt_df = subset_active_behaviour_df[['audDiff', 'visDiff', rt_var_name,
                                            choice_var_name]].dropna()

    aud_only_df = rt_df.loc[
        (rt_df['audDiff'].isin([-60, 60])) &
        (rt_df['visDiff'] == 0)
        ]

    # NOTE: vis only here can be either center or off
    vis_only_df = rt_df.loc[
        (~rt_df['audDiff'].isin([-60, 60])) &
        (rt_df['visDiff'] != 0)
        ]

    ar_vr_df = rt_df.loc[
        (rt_df['audDiff'] == 60) &
        (rt_df['visDiff'] > 0)
        ]

    al_vl_df = rt_df.loc[
        (rt_df['audDiff'] == -60) &
        (rt_df['visDiff'] < 0)
        ]

    ar_vl_df = rt_df.loc[
        (rt_df['audDiff'] == 60) &
        (rt_df['visDiff'] < 0)
        ]

    al_vr_df = rt_df.loc[
        (rt_df['audDiff'] == -60) &
        (rt_df['visDiff'] > 0)
        ]

    coherent_df = pd.concat([al_vl_df, ar_vr_df])
    conflict_df = pd.concat([al_vr_df, ar_vl_df])
    # conflict_df = pd.concat([al_vr_df, al_vr_df])

    aud_only_df['condType'] = 'aud_only'
    vis_only_df['condType'] = 'vis_only'
    coherent_df['condType'] = 'coherent'
    conflict_df['condType'] = 'conflict'

    print('Aud only median: %.3f' % np.median(aud_only_df[rt_var_name]))
    print('Vis only median: %.3f' % np.median(vis_only_df[rt_var_name]))
    print('Coherent median: %.3f' % np.median(coherent_df[rt_var_name]))
    print('Conflict median: %.3f' % np.median(conflict_df[rt_var_name]))

    mouse_combined_df = pd.concat([aud_only_df, vis_only_df, coherent_df, conflict_df])

    ## Remove audio off trials from mice
    mouse_combined_df = mouse_combined_df.loc[
        np.isfinite(mouse_combined_df['audDiff'])
    ]

    rt_var_name = 'reactionTime'
    # choice_var_name = 'threshMoveDirection'
    choice_var_name = 'responseCalc'
    contrast_40_rts = mouse_four_cond_rt['reac40']
    line_connection_spacing = 0.1
    # Method 1: Each mice offset calculated by subtracting overall median
    # fig_name = 'subject_level_model_vs_mouse_rt_3lines_remove_aud_off_in_mouse_customThresholdFitted17mice_reactionTime'

    # Method 2: Each mice offset calculated by getting median per stim cond, then mean across stim cond
    # fig_name = '40p_contrast_subject_level_model_vs_mouse_rt_3lines_remove_aud_off_in_mouse_customThresholdFitted17mice_reactionTime_median_then_mean'

    fig_ext = '.pdf'
    offset_cal_method = 'group_mice'
    mouse_rt_cal_method = 'median-then-mean'

    aud_offset = (np.array(contrast_40_rts['aud']) - np.array(contrast_40_rts['offset'])) * 1000
    vis_offset = (np.array(contrast_40_rts['vis']) - np.array(contrast_40_rts['offset'])) * 1000
    coherent_offset = (np.array(contrast_40_rts['coh']) - np.array(contrast_40_rts['offset'])) * 1000
    conflict_offset = (np.array(contrast_40_rts['con']) - np.array(contrast_40_rts['offset'])) * 1000

    with plt.style.context(splstyle.get_style('nature-reviews')):

        fig, ax = vizbehave.plot_model_vs_mouse_four_cond_plot(model_combined_df,
                                                               mouse_combined_df, plot_mouse_lines=False,
                                                               mouse_rt_var_name=rt_var_name,
                                                               mouse_rt_cal_method=mouse_rt_cal_method,
                                                               offset_cal_method=offset_cal_method,
                                                               fig=None, ax=None)

        num_mice = len(contrast_40_rts['subject'])

        for n_mouse in np.arange(num_mice):
            ax.plot([0, 1 - line_connection_spacing],
                    [aud_offset[n_mouse], vis_offset[n_mouse]], color='gray', zorder=0)
            ax.plot([1, 2 - line_connection_spacing],
                    [vis_offset[n_mouse], coherent_offset[n_mouse]], color='gray', zorder=0)
            ax.plot([2, 3 - line_connection_spacing],
                    [coherent_offset[n_mouse], conflict_offset[n_mouse]], color='gray', zorder=0)

        fig, ax = vizbehave.plot_model_vs_mouse_four_cond_plot(og_model_combined_df,
                                                               mouse_combined_df, plot_mouse_lines=False,
                                                               mouse_rt_var_name=rt_var_name,
                                                               mouse_rt_cal_method=mouse_rt_cal_method,
                                                               offset_cal_method=offset_cal_method,
                                                               fig=fig, ax=ax, model_linestyle='-')

        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()

