"""
This scripts generates figure 3c : Example Choice neuron

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


import src.models.predict_model as pmodel
import time

import src.data.analyse_spikes as anaspikes
import src.data.process_ephys_data as pephys

import src.visualization.vizmodel as vizmodel
import src.visualization.vizpikes as vizpikes
import src.visualization.vizstat as vizstat

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

fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
fig_ext = '.pdf'

def main():

    # RASTER
    fig_name = 'fig-3d-top.pdf'
    # fig_name = '4_example_active_visualLR_neuron_raster_s3e31c6-all-visual-contrast'
    # Previous good ones: s3e31c6 (2020-11-22)
    # s2e18c14 (not good, low fr)
    # s2e15c23  (also active when visual off...)

    # Visual CCCP p <= 0.01
    # s6e59MOs4
    # s3e30MOs38 (firing rate too low)

    # active_alignment_folder = '/media/timsit/Partition 1/data/interim/active-new-parent-alignment/alignToStimOnTime2ms/'
    # active_alignment_folder = '/media/timsit/Partition 1/data/interim/active-m2-choice-init-alignment/alignedToStim2ms/'
    active_alignment_folder = '/Volumes/Partition 1/data/interim/active-m2-choice-init-alignment/alignedToStim2ms/'

    subject = 3
    experiment = 21
    brain_region = 'MOs'
    exp_cell_idx = 95

    time_start = -0.1
    time_end = 0.5
    vis_cond_list = [[-0.8, -0.4, -0.2, -0.1], [0], [0.1, 0.2, 0.4, 0.8]]
    rt_variable_name = 'choiceInitTimeRelStim'
    square_axis = False
    include_stim_on_line = True
    scatter_dot_size = 2
    movement_scatter_dot_size = 2

    alignment_ds = pephys.load_subject_exp_alignment_ds(
        alignment_folder=active_alignment_folder,
        subject_num=subject, exp_num=experiment,
        target_brain_region=brain_region,
        aligned_event='stimOnTime',
        alignment_file_ext='.nc')

    cell_ds = alignment_ds.isel(Cell=exp_cell_idx)

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, axs = vizpikes.plot_grid_raster(cell_ds, vis_cond_list=vis_cond_list,
                                             time_start=time_start, time_end=time_end,
                                             scatter_dot_size=scatter_dot_size,
                                             movement_scatter_dot_size=movement_scatter_dot_size,
                                             include_stim_on_line=include_stim_on_line,
                                             rt_variable_name=rt_variable_name,
                                             square_axis=square_axis)
        fig.set_size_inches(4, 4)
        fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), bbox_inches='tight')


    # LEFT PSTH, MIDDLE PSTH, RIGHT PSTH
    fig_name = 'fig-3d-bottom.pdf'
    active_alignment_folder = '/Volumes/Partition 1/data/interim/active-m2-choice-init-alignment/alignedToStim2ms/'

    subject = 3
    experiment = 21
    brain_region = 'MOs'
    exp_cell_idx = 95

    time_start = -0.1
    time_end = 0.5
    vis_cond_list = [[-0.8, -0.4, -0.2, -0.1], [0], [0.1, 0.2, 0.4, 0.8]]
    rt_variable_name = 'choiceInitTimeRelStim'
    square_axis = False
    include_stim_on_line = True
    scatter_dot_size = 2
    movement_scatter_dot_size = 2

    alignment_ds = pephys.load_subject_exp_alignment_ds(
        alignment_folder=active_alignment_folder,
        subject_num=subject, exp_num=experiment,
        target_brain_region=brain_region,
        aligned_event='stimOnTime',
        alignment_file_ext='.nc')

    cell_ds = alignment_ds.isel(Cell=exp_cell_idx)

    # Smoothing
    smooth_sigma = 30
    smooth_window_width = 300
    time_start = -0.2
    time_end = 0.7

    # smooth spikes
    smoothed_cell_ds = cell_ds.stack(trialTime=['Trial', 'Time'])
    smoothed_cell_ds['smoothed_fr'] = ('trialTime', anaspikes.smooth_spikes(
        smoothed_cell_ds['firing_rate'],
        method='half_gaussian',
        sigma=smooth_sigma, window_width=smooth_window_width,
        custom_window=None))
    smoothed_cell_ds = smoothed_cell_ds.unstack()

    vis_left_activity = smoothed_cell_ds.where(
        smoothed_cell_ds['visDiff'].isin([-0.8, -0.4, -0.2, -0.1]), drop=True
    )['smoothed_fr'].mean('Trial')

    vis_right_activity = smoothed_cell_ds.where(
        smoothed_cell_ds['visDiff'].isin([0.8, 0.4, 0.2, 0.1]), drop=True
    )['smoothed_fr'].mean('Trial')

    vis_off_activity = smoothed_cell_ds.where(
        smoothed_cell_ds['visDiff'] == 0, drop=True
    )['smoothed_fr'].mean('Trial')

    aud_left_activity = smoothed_cell_ds.where(
        smoothed_cell_ds['audDiff'].isin([-60]), drop=True
    )['smoothed_fr'].mean('Trial')

    aud_right_activity = smoothed_cell_ds.where(
        smoothed_cell_ds['audDiff'].isin([60]), drop=True
    )['smoothed_fr'].mean('Trial')

    aud_center_activity = smoothed_cell_ds.where(
        smoothed_cell_ds['audDiff'] == 0, drop=True
    )['smoothed_fr'].mean('Trial')

    choose_left_activity = smoothed_cell_ds.where(
        smoothed_cell_ds['choiceThreshDir'].isin([1]), drop=True
    )['smoothed_fr'].mean('Trial')

    choose_right_activity = smoothed_cell_ds.where(
        smoothed_cell_ds['choiceThreshDir'].isin([2]), drop=True
    )['smoothed_fr'].mean('Trial')

    peri_event_time = smoothed_cell_ds['PeriEventTime'].mean('Trial').values

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        fig.set_size_inches(9, 3)

        # Vis
        axs[0].plot(peri_event_time, vis_left_activity, color='blue')
        axs[0].plot(peri_event_time, vis_right_activity, color='red')
        axs[0].plot(peri_event_time, vis_off_activity, color='black')
        axs[0].set_xlim([-0.05, 0.23])

        # Aud

        axs[1].plot(peri_event_time, aud_left_activity, color='blue')
        axs[1].plot(peri_event_time, aud_right_activity, color='red')
        axs[1].plot(peri_event_time, aud_center_activity, color='black')

        # Choice
        axs[2].plot(peri_event_time, choose_left_activity, color='blue')
        axs[2].plot(peri_event_time, choose_right_activity, color='red')

        axs[0].set_title('Vis L/O/R', size=11)
        axs[1].set_title('Aud L/C/R', size=11)
        axs[2].set_title('Choose L/R', size=11)

        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()