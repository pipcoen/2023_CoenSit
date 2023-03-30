import matplotlib.pyplot as plt
import numpy as np
from src.visualization.useSansMaths import use_sans_maths
from src.utils import get_project_root
root = get_project_root()
import os
import glob
import pickle as pkl
import pandas as pd
import src.data.process_ephys_data as pephys
import src.data.stat as stat

import decimal  # for scientific notation printing of numbers

from tqdm import tqdm
import itertools
import src.data.analyse_spikes as analyse_spike
import src.visualization.vizbehaviour as vizbehaviour
import src.visualization.vizstat as vizstat
import src.visualization.vizactivity as vizactivity
import sciplotlib.style as splstyle
import seaborn as sns

import xarray as xr

# For merging images
import sys
from PIL import Image

# For splitting subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Trajectory plot
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection

import scipy.stats as sstats
# Debugging
import pdb


def init_plot():
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4)

    return fig, ax

def plot_raster(spike_df, fig=None, ax=None, scatter_dot_size=0.1, scatter_dot_alpha=0.1,
                dot_color='black'):
    """
    Plot raster plot with individual cells as rows, and time as columns.
    
    Arguments
    --------
    spike_df   : pandas dataframe object with the columns: spikeTime (time in seconds of spike), cluNum (integer reference number for the cluster/cell/neuron)
    """

    if (fig is None) or (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4)

    for y_plot_loc, cluNum in enumerate(np.unique(target_loc_spike_df['cluNum'])):
    
        cell_spike_times = spike_df['spikeTime'].loc[
            spike_df['cluNum'] == cluNum 
        ]
    
    ax.scatter(cell_spike_times, [y_plot_loc] * len(cell_spike_times), color=dot_color,
              s=scatter_dot_size, alpha=scatter_dot_alpha)

    
    return fig, ax


def plot_trial_raster(fig, ax, target_loc_spike_df, ephys_behave_df, trial_num, init_y_plot_loc,
                     scatter_dot_size=0.5, scatter_dot_alpha=1.0, scatter_color='black',
                     scatter_label=None, rel_time=None):
    """
    Plots the raster for a single subject, in a single trial, in one specific brain region.
    """
    
    for y_plot_loc, cluNum in enumerate(np.unique(target_loc_spike_df['cluNum'])):
        cell_spike_times = target_loc_spike_df['spikeTime'].loc[
            (target_loc_spike_df['cluNum'] == cluNum) & 
            (target_loc_spike_df['spikeTime'] >= ephys_behave_df['trialStart'][trial_num-1]) & 
            (target_loc_spike_df['spikeTime'] <= ephys_behave_df['trialEnd'][trial_num-1])
        ]
        
        if rel_time == 'trial_start':
            cell_spike_times = cell_spike_times - ephys_behave_df['trialStart'][trial_num-1]

        ax.scatter(cell_spike_times, init_y_plot + np.array(
            [y_plot_loc] * len(cell_spike_times)), 
                   color=scatter_color,
                  s=scatter_dot_size, alpha=scatter_dot_alpha, label=scatter_label)
        
    return fig, ax


def plot_brain_region_raster(exp_df, spike_df, fig=None, ax=None, cmap_name='set1', init_y_plot=0,
                             x_text_loc=-0.5, scatter_dot_size=0.1, scatter_dot_alpha=0.1):
    """
    Make raster plot with clusters grouped by the brain region they belong to, with different colors.
    """


    cell_loc_list = np.unique(exp_df['cellLoc'])

    cmap = mpl.cm.get_cmap(CMAP_NAME, len(cell_loc_list))

    for n_cell_loc, cell_loc in enumerate(cell_loc_list):
        target_cell_loc = cell_loc
        clu_ref_in_cell_loc = subject_exp_df['cluNum'].loc[
                                                           subject_exp_df['cellLoc'] == target_cell_loc]
        target_loc_spike_df = recording_spike_df.loc[
                                                     recording_spike_df['cluNum'].isin(clu_ref_in_cell_loc)
                                                     ]
        fig, ax = plot_trial_raster(fig, ax, target_loc_spike_df=target_loc_spike_df, 
                                    ephys_behave_df=ephys_behave_df,
                                    trial_num=TRIAL_NUM, init_y_plot_loc=init_y_plot, 
                                    scatter_dot_size=scatter_dot_size, 
                                    scatter_dot_alpha=scatter_dot_alpha, 
                                    scatter_color=cmap.colors[n_cell_loc],
                                    scatter_label=cell_loc, 
                                    rel_time='trial_start')
        text_y_loc = init_y_plot + len(clu_ref_in_cell_loc)/2
        ax.text(x=x_text_loc, y=text_y_loc, 
                s=cell_loc, color=cmap.colors[n_cell_loc])
        init_y_plot = init_y_plot + len(clu_ref_in_cell_loc)

        

    ax.set_yticklabels([])
    ax.set_xlabel('Time (s)')

    

    return fig, ax

    

def plot_smooth_test(spike_train=None, method='full_gaussian', sigma=0.3, fig=None, ax=None):
    """
    Function for testing the smoothing behaviour by applying the smoothing filter to a single spike train.
    --------------
    Arguments
    spike_train  : 1d numpy array of the spike train, if none-provided, then a random binary spike train will be created.
    method       : method for performing the smoothing 
    sigma        : sigma parameter for the Guassian window (kernel)
    fig, ax      : figure handles, will be generated if None provided.
    TODO: pass a kernel object with the kernel paramters, rather than specifying custom params.
    """

    if (fig is None) and (ax is None):
        fig, ax = init_plot()

    if spike_train is None:
        spike_train = np.random.randint(2, size=1000)  

    ax.plot(spike_train, label='Raw')

    smoothed_spike_train = analyse_spike.smooth_spikes(spike_train, method=method, sigma=sigma)

    ax.plot(smoothed_spike_train, label='Smoothed')

    ax.legend(frameon=False)

    ax.grid()

    return fig, ax

        
def plot_psth(trial_spike_mean, peri_stim_time, trial_spike_sem=None, fig=None, ax=None, units='rate', grid=False, fill_alpha=0.3, color='blue'):
    """
    Plots peri-stimulus time histogram (or more generally peri-event time) of spike trians for a single   neuron across trials, which is usually smoothed, but this option can be distabled.

    Arguments
    ----------
    spike_array    : 
    peri_stim_time : 1D numpy array of the corresponding times to the binned spikes
    """
    if (fig is None) and (ax is None):
        fig, ax = init_plot()
    
    if units == 'rate':
        ax.set_ylabel('Firing rate (spikes/s)')
    else:
        ax.set_ylabel('Spike count')

    if grid is True:
        ax.grid()


    ax.plot(peri_stim_time, trial_spike_mean, color=color)
    if trial_spike_sem is not None:
        ax.fill_between(peri_stim_time, 
                trial_spike_mean+trial_spike_sem, 
                trial_spike_mean-trial_spike_sem, alpha=fill_alpha,
               color=color)


    # if using peri-stim time, then 0 is the time of event 
    time_of_event = 0
    ax.axvline(time_of_event, linestyle='--', color=color)
    

    return fig, ax


def plot_aligned_raster(spike_data, event_times=None, fig=None, ax=None, color='blue',
                        vline_ymin=0, vline_ymax=1, dot_size=10, dot_alpha=1.0, show_event_vline=False,
                        cmap_name='viridis', vline_color='white', sort_coord=None,
                        vmax=None, vmin=None):
    """
    Plot event-aligned raster plot for a single neuron over multiple trials.
    y axis : trial number (integer)
    x axis : peri-stimulus time (s)
    
    Parameters
    ----------
    grouped_spike_df : (dataframe, or xarray dataset)
        if dataframe, then this will be a spike times grouped by trial number
        if xarray dataset, then this will contain a data variable with the binned firing rate,
        as well as coordinate value corresponding to the peri-event time (relative to aligned event)
    event_times : (pandas dataframe)
        pandas series object with the time of each event (seconds)
    vline_ymin : (int or float)
        minimum value of the vertical line showing the event onset
    vline_ymax : (int of float)
        maximum value of the vertical line showing the event onset
    fig, ax : (matplotlib objects)
        figure handles
    sort_coord : (bool)
        whether to sort
    Returns
    ------------
    fig, ax : matplotlib figure handles
    """

    if (fig is None) and (ax is None):
        fig, ax = init_plot()

    if type(spike_data) is pd.core.frame.DataFrame:
        assert event_times is not None, print('Event times must be provided when using pandas dataframe')
        for trial in range(len(spike_data)):
            ax.scatter(np.array(spike_data.iloc[trial]['spike_times']) -
                   event_times.iloc[trial],
                   np.repeat(trial+1, len(spike_data.iloc[trial]['spike_times'])),
                  color=color, alpha=dot_alpha, edgecolor='None',
                  s=dot_size)

        event_time = 0
        ax.axvline(event_time, vline_ymin, vline_ymax, linestyle='--', color=color,
                   clip_on=False)
        im = None
    elif type(spike_data) is xr.core.dataset.Dataset or type(spike_data) is xr.core.dataarray.DataArray:
        peri_time_start = spike_data['PeriEventTime'][0]
        peri_time_end = spike_data['PeriEventTime'][-1]
        num_cell = len(spike_data['Cell'])

        # mean_across_trial_ds = spike_data.mean('Trial')
        if type(spike_data) is xr.core.dataset.Dataset:
            if sort_coord is not None:
                spike_data = spike_data.sortby(sort_coord, ascending=False)
            activity_values = spike_data['firing_rate'].values
        else:
            activity_values = spike_data.values
        im = ax.imshow(activity_values,
                  extent=[peri_time_start, peri_time_end, num_cell, 1], aspect='auto',
                  cmap=cmap_name, vmax=vmax, vmin=vmin)

        if show_event_vline:

            ax.axvline(0, linestyle='--', color=vline_color)



    return fig, ax, im


def plot_grid_psth(exp_cell_idx, exp_combined_ds_list, fig=None, ax=None,
                   shade_significant=False, p_val_title=False, p_val_title_sig_df=None,
                   p_val_title_style='stars',
                   cell_sig_df=None, plot_type='mean-trace',
                   sort_by=None, behaviour_df=None, aud_plot_loc_dict={'left': 0, 'off': 1, 'right': 2},
                   vis_plot_loc_dict={'left': 0, 'off': 1, 'right': 2},
                   cell_coord_name='expCell', time_coord_name='PeriStimTime',
                   custom_share_y=None, verbose=True
                   ):
    """
    Plots a (3 x 3) grid of peri-stimulus time mean rate (or trial-by-trial heatmap raster).
    Each grid represent a specific audio-visual condition.
    Parameters
    ----------
    exp_cell_idx: (int)
        which cell to plot
    exp_combined_ds_list: (list of xarray datasets)
        each dataset contains the aligned spike matrix associated with a specific
        audio and visual condition pair
    sort_by (str)
        string that access the variable in neuron_xr that will be used for sorting
        currently only use case is to sort by 'timeToWheelMove'
    fig: (matplotlib class object)
        figure handle
    :param ax: (matplotlib class object)
        axis handle
    :param shade_significant: (bool)
        whether to shade the grid that contains significant response
        whether the cell is significant responding to a particular condition is provided by cell_sig_df
    :param p_value_title (boolean)
        whether to plot the p-value (pre vs. post event firing rate) as subplot title on each grid.
    :param p_val_title_style (str)
        how to represent significance in the plot
        default is 'stars', which uses '*' symbol.
        if it is anything else (currently), then it will just print out the p-value in scientific notation.
    :param p_val_title_sig_df (pandas dataframe)
        only used when p_val_title is set to True
        dataframe that contains p-values used for subplot title
        not to be confused with cell_sig_df, which is used for sorting the order of the plots
        current use case is when I want to sort the order of passive plots by sig in the active condition,
        but I still want to include the passive condition significance p-values in the subplot titles.
    cell_sig_df: (pandas dataframe)
        dataframe containing info about statistical significance of each condition
    :param behaviour_df : (pandas dataframe)
        dataframe containing trial by trial information
        only used here if plot_type is heatmap-w-psth
    :param plot_type: (str)
        if 'mean-trace', plots the mean firing rate (with SEM shaded)
        if 'heatmap-raster', plots trial-by-trial raster
        if 'heatmap-w-psth', plots trial by trial raster, with PSTH below each raster (about 75 / 25 vertical split)
    :param aud_plot_loc_dict (dict)
        dictionary where keys are the auditory condition to plot, and the number indicates which row to plot
    :param vis_plot_loc_dict (dict)
        dictionary where keys are the visual condition to plot, and the number indicates which column to plot
    time_coord_name (str)
        name of the time coordinate. Usually either PeriStimTime or PeriEventTime
    custom_share_y (bool or None)
        override my sensible defaults of whethr to sharey under different plotting conditions
        if None, then uses by defaults, if bool, then do what you say.
    -------------------------
    Output
    -----------------
    fig, ax : matplotlib figure handles
    """

    if plot_type == 'heatmap-w-psth':
        assert behaviour_df is not None, print('Behaviour df is required to do heatmap with PSTH plots.')

    # TODO: no need to repeat subplot creation so many times, just need nrow and ncolumn
    if (fig is None) and (ax is None):
        if (plot_type == 'heatmap-raster') or (plot_type == 'heatmap-w-psth') or \
                (plot_type == 'heatmap-w-psth-passive') or (plot_type == 'mean-trace'):
            if len(exp_combined_ds_list) <= 8:
                # original 9, but some experiments seem to have audio center + visual off/center,
                # so we still need an extra row...
                # TODO: have 4 rows whenever audiocenter is one of the conditions
                nrow = 3
                ncolumn = 3

                if custom_share_y is None:
                    sharey = False
                else:
                    sharey = custom_share_y

                fig, axs = plt.subplots(nrow, ncolumn, sharex=True, sharey=sharey)
                fig.set_size_inches(5, 5)
            elif len(exp_combined_ds_list) >= 11:
                nrow = 4
                ncolumn = 3
                fig, axs = plt.subplots(nrow, ncolumn, sharex=True, sharey=False)
                fig.set_size_inches(5, 8)
            else:  # Temporary backup for strange cases of audio center without audio center + visual left/right
                nrow = 4
                ncolumn = 3
                fig, axs = plt.subplots(nrow, ncolumn, sharex=True, sharey=False)
                fig.set_size_inches(5, 8)
        else:
            fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
            fig.set_size_inches(5, 5)

    if cell_sig_df is not None:
        cell_sig_df = cell_sig_df.loc[cell_sig_df['cell_idx'] == exp_cell_idx]

    # keep a list of locations that were plotted, which can be used later to remove unplotted locations
    # plotted_x_loc = list()
    # plotted_y_loc = list()
    plotted_loc = list()

    for exp_combined_ds in exp_combined_ds_list:

        # 2019-11-19: allowing for multiple variables, so we are selecting a particular one
        if exp_cell_idx is not None:
            neuron_xr = exp_combined_ds[['firing_rate']].isel({cell_coord_name: exp_cell_idx})
        else:
            neuron_xr = exp_combined_ds[['firing_rate']]

        if plot_type == 'mean-trace':
            psth_mean = neuron_xr.mean(dim='Trial')
            psth_std = neuron_xr.std(dim='Trial') / np.sqrt(
                len(neuron_xr['Trial']))

        plot_x_loc = aud_plot_loc_dict[exp_combined_ds.attrs['aud']]
        plot_y_loc = vis_plot_loc_dict[exp_combined_ds.attrs['vis']]

        # plotted_x_loc.append(plot_x_loc)
        # plotted_y_loc.append(plot_y_loc)
        plotted_loc.append((plot_x_loc, plot_y_loc))

        if (shade_significant) and (plot_type == 'mean-trace'):
            response_is_sig = cell_sig_df.loc[
                (cell_sig_df['aud_cond'] == exp_combined_ds.attrs['aud']) &
                (cell_sig_df['vis_cond'] == exp_combined_ds.attrs['vis'])
                ]['sig_response'].values

            if len(response_is_sig) > 0:
                if response_is_sig[0] == True:
                    axs[plot_x_loc, plot_y_loc].set_facecolor(
                        (0.678, 0.847, 0.902)  # light blue
                    )

        if plot_type == 'mean-trace':
            axs[plot_x_loc, plot_y_loc].plot(psth_mean[time_coord_name].values,
                                             psth_mean['firing_rate'].values,
                                             color='black')

            axs[plot_x_loc, plot_y_loc].fill_between(
                psth_mean[time_coord_name].values,
                psth_mean['firing_rate'].values - psth_std['firing_rate'].values,
                psth_mean['firing_rate'].values + psth_std['firing_rate'].values,
                alpha=0.3, color='grey'
            )

        elif (plot_type == 'heatmap-raster') and (sort_by is None):
            pst_start = neuron_xr[time_coord_name].values[0]
            pst_end = neuron_xr[time_coord_name].values[-1]
            neuron_xr_na_removed = neuron_xr.dropna(dim='Trial')
            neuron_xr_t = neuron_xr_na_removed.transpose('Trial', 'Time')
            trial_by_time_matrix = np.squeeze(neuron_xr_t.to_array().values)
            # axs[plot_x_loc, plot_y_loc].imshow(trial_by_time_matrix, cmap='binary')
            fig, axs[plot_x_loc, plot_y_loc] = vizactivity.plot_activity_heatmap(fig=fig, ax=axs[plot_x_loc, plot_y_loc],
                                                                     activity_matrix=trial_by_time_matrix,
                                                                     cmap='binary',
                                                                     x_axis_extent=[pst_start, pst_end])
        elif plot_type == 'heatmap-raster':
            # print(neuron_xr)
            pst_start = neuron_xr[time_coord_name].values[0]
            pst_end = neuron_xr[time_coord_name].values[-1]
            # print(neuron_xr['expCell'].values.tolist()[1])

            # remove 'extra' trials where there is no recording for that neuron
            # TODO: !!!: The question is whether this also removes cells that did not fire???
            # Not actually sure why there will be NaN in the trial dimension?
            neuron_xr_na_removed = neuron_xr.dropna(dim='Trial')
            neuron_xr_t = neuron_xr_na_removed.transpose('Trial', 'Time')
            trial_by_time_matrix = np.squeeze(neuron_xr_t.to_array().values)

            # TODO: will also have to remove trials where the neuron did not participate...
            sort_values_xr = exp_combined_ds[sort_by].isel(expCell=exp_cell_idx)
            # print(exp_cell_ds)
            # sort_values_xr = exp_cell_ds.stack(expTrial=('Exp', 'Trial'))
            sort_values_vector = sort_values_xr.values
            sort_values = sort_values_vector[~np.isnan(sort_values_vector)]

            # axs[plot_x_loc, plot_y_loc].imshow(trial_by_time_matrix, cmap='binary')
            fig, axs[plot_x_loc, plot_y_loc] = vizactivity.plot_activity_heatmap(fig=fig,
                                                                     ax=axs[plot_x_loc, plot_y_loc],
                                                                     activity_matrix=trial_by_time_matrix,
                                                                     cmap='binary',
                                                                     x_axis_extent=[pst_start, pst_end],
                                                                     sort_values=sort_values,
                                                                     method='sorted')

        elif plot_type == 'heatmap-w-psth':

            pst_start = neuron_xr['PeriStimTime'].values[0]
            pst_end = neuron_xr['PeriStimTime'].values[-1]

            neuron_xr_na_removed = neuron_xr.dropna(dim='Trial')
            neuron_xr_t = neuron_xr_na_removed.transpose('Trial', 'Time')
            trial_by_time_matrix = np.squeeze(neuron_xr_t.to_array().values)

            
            # Get the reaction times (these are the sort_values)
            sort_values_xr = exp_combined_ds[sort_by].isel(expCell=exp_cell_idx)
            sort_values_vector = sort_values_xr.values
            sort_values_vector[sort_values_vector == 0] = np.Inf  # Inf to prevent no-go being removed

            # I think this actually remove *TRIALS* on the grid (ie. if there are 150 trials for audLeft visRight,
            # but only 4 trials for audLeft visOff
            # sort_values = sort_values_vector[~np.isnan(sort_values_vector)]

            sort_values = sort_values_vector

            # checked that np.argsort will place nans at the end, so that is okay.
            
            # NOTE: Since we are including no-go trials, the reaction time of NaN has to 
            # be accepted somehow, but just not plotted, and needs to be sorted correctly
            # sometimes no-go mistakenly has timeToWheelMove of 0 for some rason 
            #
            
            # sort_values = sort_values_vector[~np.isnan(sort_values_vector)]
            # TODO: this currently has the problem that it only works
            # if include_trial is set to True in the alignment.

            split_vector = behaviour_df.loc[neuron_xr['Trial'].values]['responseMade'].values


            # split the ax object to allow plotting the PSTH at the bottom
            divider = make_axes_locatable(axs[plot_x_loc, plot_y_loc])
            psth_ax = divider.append_axes('bottom', size='50%', pad=0.1, sharex=axs[plot_x_loc, plot_y_loc])

            fig, axs[plot_x_loc, plot_y_loc] = vizactivity.plot_activity_heatmap(fig=fig,
                                                        ax=axs[plot_x_loc, plot_y_loc],
                                                        activity_matrix=trial_by_time_matrix,
                                                        cmap='binary',
                                                        x_axis_extent=[pst_start, pst_end],
                                                        # y_axis_extent=[0, len(split_vector)],
                                                        sort_values=sort_values,
                                                        method='split-and-sorted',
                                                        split_vector=split_vector,
                                                        custom_split_order=[1, 2, 0],
                                                        split_dot_color_dict={1: 'blue',
                                                                              2: 'red',
                                                                              0: 'black'})


            # needs to be three mean traces, sorted by choice, left (blue), right (red), no-go (black)
            left_index = np.where(split_vector == 1)[0]
            right_index = np.where(split_vector == 2)[0]
            nogo_index = np.where(split_vector == 0)[0]

            activity_type = 'firing_rate'
            psth_ax.plot(neuron_xr['PeriStimTime'].values,
                         neuron_xr[activity_type].isel(Trial=left_index).mean(dim='Trial'), color='blue')
            psth_ax.plot(neuron_xr['PeriStimTime'].values,
                         neuron_xr[activity_type].isel(Trial=right_index).mean(dim='Trial'), color='red')
            psth_ax.plot(neuron_xr['PeriStimTime'].values,
                         neuron_xr[activity_type].isel(Trial=nogo_index).mean(dim='Trial'), color='black',
                         alpha=0.5)

            # Include p-value (of pre-stimulus and pos -stimulus two sample (wilcoxon) test) in subplot title
            if p_val_title:
                response_sig_value = cell_sig_df.loc[
                    (cell_sig_df['aud_cond'] == exp_combined_ds.attrs['aud']) &
                    (cell_sig_df['vis_cond'] == exp_combined_ds.attrs['vis'])
                    ]['p_val'].values

                if p_val_title_style == 'stars':
                    if response_sig_value[0] < 0.0001:
                        response_sig_str = '***'
                    elif response_sig_value[0] < 0.001:
                        response_sig_str = '**'
                    elif response_sig_value[0] < 0.01:
                        response_sig_str = '*'
                    else:
                        response_sig_str = ''
                    axs[plot_x_loc, plot_y_loc].set_title(response_sig_str)
                else:
                    response_sig_str = "{:.2E}".format(decimal.Decimal(str(response_sig_value[0])))
                    axs[plot_x_loc, plot_y_loc].set_title('p = ' + response_sig_str)

            # vertical line to show event time
            axs[plot_x_loc, plot_y_loc].axvline(0, linestyle='--', color='gray',
                                            alpha=0.5)

        elif plot_type == 'heatmap-w-psth-passive':
            # same idea as heatmap-w-psth, but no we can't split by the choice made.
            pst_start = neuron_xr['PeriStimTime'].values[0]
            pst_end = neuron_xr['PeriStimTime'].values[-1]

            neuron_xr_na_removed = neuron_xr.dropna(dim='Trial')
            neuron_xr_t = neuron_xr_na_removed.transpose('Trial', 'Time')
            trial_by_time_matrix = np.squeeze(neuron_xr_t.to_array().values)

            # split the ax object to allow plotting the PSTH at the bottom
            divider = make_axes_locatable(axs[plot_x_loc, plot_y_loc])
            psth_ax = divider.append_axes('bottom', size='50%', pad=0.1, sharex=axs[plot_x_loc, plot_y_loc])


            fig, axs[plot_x_loc, plot_y_loc] = vizactivity.plot_activity_heatmap(fig=fig,
                                                                 ax=axs[plot_x_loc, plot_y_loc],
                                                                 activity_matrix=trial_by_time_matrix,
                                                                 cmap='binary',
                                                                 x_axis_extent=[pst_start, pst_end],
                                                                 # y_axis_extent=[0, len(split_vector)],
                                                                 sort_values=None,
                                                                 method='simple',
                                                                 split_vector=None,
                                                                 custom_split_order=None,
                                                                 split_dot_color_dict=None)

            # Include p-value (of pre-stimulus and pos -stimulus two sample (wilcoxon) test) in subplot title
            if p_val_title:
                if p_val_title_sig_df is not None:
                    p_val_title_sig_df = p_val_title_sig_df.loc[p_val_title_sig_df['cell_idx'] == exp_cell_idx]
                    response_sig_value = p_val_title_sig_df.loc[
                        (p_val_title_sig_df['aud_cond'] == exp_combined_ds.attrs['aud']) &
                        (p_val_title_sig_df['vis_cond'] == exp_combined_ds.attrs['vis'])
                        ]['p_val'].values
                else:
                    response_sig_value = cell_sig_df.loc[
                        (cell_sig_df['aud_cond'] == exp_combined_ds.attrs['aud']) &
                        (cell_sig_df['vis_cond'] == exp_combined_ds.attrs['vis'])
                        ]['p_val'].values

                if p_val_title_style == 'stars':
                    if response_sig_value[0] < 0.0001:
                        response_sig_str = '***'
                    elif response_sig_value[0] < 0.001:
                        response_sig_str = '**'
                    elif response_sig_value[0] < 0.01:
                        response_sig_str = '*'
                    else:
                        response_sig_str = ''
                    axs[plot_x_loc, plot_y_loc].set_title(response_sig_str)
                else:
                    response_sig_str = "{:.2E}".format(decimal.Decimal(str(response_sig_value[0])))
                    axs[plot_x_loc, plot_y_loc].set_title('p = ' + response_sig_str)

            activity_type = 'firing_rate'
            psth_ax.plot(neuron_xr['PeriStimTime'].values,
                         neuron_xr[activity_type].mean(dim='Trial'), color='black')

            # vertical line to show event time
            axs[plot_x_loc, plot_y_loc].axvline(0, linestyle='--', color='gray',
                                            alpha=0.5)


        if plot_type == 'heatmap-raster':
            if plot_y_loc == 0:
                axs[plot_x_loc, plot_y_loc].set_ylabel('Audio: ' +
                                                       exp_combined_ds.attrs['aud'], color='blue', size=12)
            if plot_x_loc == 2:
                axs[plot_x_loc, plot_y_loc].set_xlabel('Visual: ' +
                                                       exp_combined_ds.attrs['vis'], color='red', size=12)
        elif (plot_type == 'heatmap-w-psth') or (plot_type == 'heatmap-w-psth-passive'):
            if plot_y_loc == min(aud_plot_loc_dict.values()):
                axs[plot_x_loc, plot_y_loc].set_ylabel('Audio: ' +
                                                       exp_combined_ds.attrs['aud'], color='blue', size=12)
            if plot_x_loc == max(aud_plot_loc_dict.values()):  # x loc as in the row, not actual x-axis
                axs[plot_x_loc, plot_y_loc].tick_params(labelbottom=False)
                psth_ax.set_xlabel('Visual: ' + exp_combined_ds.attrs['vis'], color='red')
            if plot_x_loc != max(aud_plot_loc_dict.values()):
                # note that sharing x and only showing tick marks in one plot is slightly tricky
                # thus having to set tick params
                # see: https://stackoverflow.com/questions/4209467/matplotlib-share-x-axis-but-dont-show-x-axis-tick-labels-for-both-just-one
                axs[plot_x_loc, plot_y_loc].tick_params(labelbottom=False)
                psth_ax.tick_params(labelbottom=False)
        elif plot_type == 'mean-trace':
            if plot_x_loc == min(aud_plot_loc_dict.values()):  # x loc as in the row, not actual x-axis
                # axs[plot_x_loc, plot_y_loc].tick_params(labelbottom=False)
                axs[plot_x_loc, plot_y_loc].set_title('Visual: ' + exp_combined_ds.attrs['vis'], color='red',
                                                      size=12)

    # TODO: check all left corner (y_loc) plot exists
    # if not, then still print the y-axis.
    # NOTE: following code still in development
    # The current issue is that it does not guarantee the title does correspond to what is plotted, it is
    # currently just manually selected.
    # One possibility is to get the ordering information from the loop above.
    force_y_axis_title = True
    if force_y_axis_title:
        # aud_condition_list = [exp_combined_ds.attrs['aud'] for exp_combined_ds in exp_combined_ds_list]
        # unique_aud_conditions = np.unique(aud_condition_list)
        # custom_y_axis_labels = ['Audio: left', 'Audio: off', 'Audio: Center', 'Audio: right']
        # y_axis_title_list = custom_y_axis_labels
        # for row, y_axis_title in zip(np.arange(nrow), y_axis_title_list):
        #     ax[row, 0].set_title(y_axis_title)
        for aud_condition, row_number in aud_plot_loc_dict.items():
            axs[row_number, 0].set_ylabel('Audio:' + aud_condition, color='blue')

        # align the y-axis
        fig.align_ylabels(axs[:, 0])

    # For each subplot grid, check if it was plotted, if not, remove tick marks
    for plot_row, plot_column in itertools.product(np.arange(0, nrow), np.arange(0, ncolumn)):
        if (plot_row, plot_column) not in plotted_loc:
            # remove y ticks
            # axs[plot_row, plot_column].axis('off')
            # axs[plot_row, plot_column].set_yticks([])
            # plt.setp(axs[plot_row, plot_column].get_yticklabels(), visible=False)
            axs[plot_row, plot_column].xaxis.set_tick_params(labelleft=False)
            # remove x ticks as well (?)

    if plot_type == 'mean-trace':
        fig.text(-0.04, 0.5, 'Firing rate (spikes/s)', va='center', rotation='vertical')
    elif (plot_type == 'heatmap-raster') or (plot_type == 'heatmap-w-psth'):
        fig.text(0.0, 0.5, 'Trial', va='center', rotation='vertical')

    fig.text(0.5, 0.0, 'Peri-stimulus time (s)', ha='center')

    return fig, axs


def plot_passive_grid_psth(alignment_cell_ds, fig=None, axs=None,
                           activity_name='firing_rate',
                           audio_levels=[-60, np.inf, 60],
                           visual_levels=[-0.8, 0, 0.8],
                           aud_plot_loc_dict={-60: 0, np.inf: 1, 60: 2},
                           vis_plot_loc_dict={-0.8: 0, 0: 1, 0.8: 2},
                           custom_x_lim=None, shade_time_range=None, cmap='viridis'):
    """
    Plot 3 x 3 plot to show the neural activity of a single neuron for each
    possible combination fo audio and visual stimulus during passive trials.
    This is currently used to evaluate significant cells from two-way ANOVA.
    See: Notebook 14.7
    Parameters
    ----------
    alignment_cell_ds
    fig
    axs
    activity_name
    audio_levels
    visual_levels
    aud_plot_loc_dict
    vis_plot_loc_dict
    custom_x_lim

    Returns
    -------

    """

    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)

    for aud_level, vis_level in itertools.product(audio_levels, visual_levels):
        aud_vis_ds = alignment_cell_ds.where(
            (alignment_cell_ds['visDiff'] == vis_level) &
            (alignment_cell_ds['audDiff'] == aud_level), drop=True
        )

        if len(aud_vis_ds.Trial.values) == 0:
            continue

        aud_vis_psth = aud_vis_ds.mean('Trial')

        aud_vis_fr = aud_vis_psth[activity_name].values

        row_num = aud_plot_loc_dict[aud_level]
        column_num = vis_plot_loc_dict[vis_level]

        axs[row_num, column_num].plot(
            # aud_vis_ds['PeriEventTime'].isel(Trial=0).values,
            aud_vis_ds['PeriEventTime'].values,
            aud_vis_fr)

        if shade_time_range is not None:
            axs[row_num, column_num].axvspan(shade_time_range[0],
                                             shade_time_range[1], alpha=0.3, edgecolor='none')

    axs[0, 0].set_title('Vis left', size=12)
    axs[0, 1].set_title('Vis off', size=12)
    axs[0, 2].set_title('Vis right', size=12)

    if custom_x_lim is not None:
        axs[0, 0].set_xlim(custom_x_lim)

    axs[0, 0].set_ylabel('Aud left', size=12)
    axs[1, 0].set_ylabel('Aud off', size=12)
    axs[2, 0].set_ylabel('Aud right', size=12)

    return fig, axs


def plot_grid_raster(cell_ds, aud_cond_list=[-60, 0, 60],
                     vis_cond_list=[[-0.8, -0.4, -0.2, -0.1], [0], [0.1, 0.2, 0.4, 0.8]],
                     max_rt=0.4, scatter_dot_size=0.2, time_start=None, time_end=None,
                     rt_variable_name='firstTimeToWheelMove', response_variable_name='responseMade',
                     movement_scatter_dot_size=1, include_stim_on_line=False, square_axis=False,
                     plot_psth_below=False):


    aud_plot_loc_dict = {'left': 0, 'center': 1, 'right': 2}
    vis_plot_loc_dict = {'left': 0, 'off': 1, 'right': 2}
    cell_ds_subset = cell_ds.where(
        cell_ds[rt_variable_name] <= max_rt, drop=True
    )

    peri_event_time = cell_ds.PeriEventTime.values

    sort_by_movement = True
    plot_movement_times = True

    # aesthetics
    remove_left_spine = True
    keep_only_bottom_row_xaxis = True
    include_background = False
    background_color = 'white'

    if square_axis:
        fig, axs = plt.subplots(3, 3, sharey=True, sharex=True)
    else:
        fig, axs = plt.subplots(3, 3, sharey=True, sharex=True)

    if (time_start is not None) and (time_end is not None):
        cell_ds_subset = cell_ds_subset.where(
            (cell_ds_subset['PeriEventTime'] >= time_start) &
            (cell_ds_subset['PeriEventTime'] <= time_end), drop=True
        )

        cell_ds_subset[rt_variable_name] = cell_ds_subset[rt_variable_name].isel(Time=0)
        cell_ds_subset[response_variable_name] = cell_ds_subset[response_variable_name].isel(Time=0)
        cell_ds_subset['audDiff'] = cell_ds_subset['audDiff'].isel(Time=0)
        cell_ds_subset['visDiff'] = cell_ds_subset['visDiff'].isel(Time=0)
        peri_event_time = cell_ds_subset.PeriEventTime.values

    # pdb.set_trace()

    if plot_psth_below:
        psth_ax_store = []

    for aud_cond, vis_cond in itertools.product(aud_cond_list, vis_cond_list):

        stim_cond_cell_ds = cell_ds_subset.where(
            (cell_ds_subset['audDiff'].isin(aud_cond)) &
            (cell_ds_subset['visDiff'].isin(vis_cond)), drop=True)


        if len(stim_cond_cell_ds.Trial.values) == 0:
            continue

        stim_cond_respond_left_cell_ds = stim_cond_cell_ds.where(
            stim_cond_cell_ds[response_variable_name] == 1, drop=True
        )

        stim_cond_respond_right_cell_ds = stim_cond_cell_ds.where(
            stim_cond_cell_ds[response_variable_name] == 2, drop=True
        )

        # sort by reaction time
        stim_cond_respond_left_cell_ds = stim_cond_respond_left_cell_ds.sortby(
            rt_variable_name)
        stim_cond_respond_right_cell_ds = stim_cond_respond_right_cell_ds.sortby(
            rt_variable_name)

        # if len(stim_cond_cell_ds.Trial.values) == 0:
        #     print('No trials with aud cond %.f and vis cond %.f' % (aud_cond, vis_cond))

        if aud_cond < 0:
            aud_cond_str = 'left'
        elif aud_cond > 0:
            aud_cond_str = 'right'
        elif aud_cond == 0:
            aud_cond_str = 'center'

        if vis_cond[0] < 0:
            vis_cond_str = 'left'
        elif vis_cond[0] > 0:
            vis_cond_str = 'right'
        elif vis_cond[0] == 0:
            vis_cond_str = 'off'

        row_loc = aud_plot_loc_dict[aud_cond_str]
        column_loc = vis_plot_loc_dict[vis_cond_str]

        if sort_by_movement:

            # if (vis_cond_str == 'left') & (aud_cond_str == 'right'):
            #     pdb.set_trace()

            # plot spikes
            left_spike_matrix = stim_cond_respond_left_cell_ds['firing_rate'].values.T
            num_left_trial = np.shape(left_spike_matrix)[0]

            left_rt_vec = stim_cond_respond_left_cell_ds[rt_variable_name].values

            for left_trial in np.arange(num_left_trial):
                spike_times = np.where(left_spike_matrix[left_trial, :] > 0)[0]
                axs[row_loc, column_loc].scatter(peri_event_time[spike_times],
                                                 np.repeat(left_trial, len(spike_times)), color='blue',
                                                 s=scatter_dot_size, edgecolor='none')
                # plot movement times
                if plot_movement_times:
                    axs[row_loc, column_loc].scatter(left_rt_vec[left_trial], left_trial, color='black',
                                                     s=movement_scatter_dot_size, edgecolor='none')

            right_spike_matrix = stim_cond_respond_right_cell_ds['firing_rate'].values.T
            num_right_trial = np.shape(right_spike_matrix)[0]
            right_rt_vec = stim_cond_respond_right_cell_ds[rt_variable_name].values

            for right_trial in np.arange(num_right_trial):
                spike_times = np.where(right_spike_matrix[right_trial, :] > 0)[0]
                axs[row_loc, column_loc].scatter(peri_event_time[spike_times],
                                                 np.repeat(right_trial + num_left_trial + 1, len(spike_times)),
                                                 color='red',
                                                 s=scatter_dot_size, edgecolor='none')

                # plot movement times
                if plot_movement_times:
                    axs[row_loc, column_loc].scatter(right_rt_vec[right_trial],
                                                     right_trial + num_left_trial + 1,
                                                     color='black',
                                                     s=movement_scatter_dot_size, edgecolor='none')



        else:
            spike_matrix = stim_cond_cell_ds['firing_rate'].values.T
            num_trial = np.shape(spike_matrix)[0]

            for trial in np.arange(num_trial):
                spike_times = np.where(spike_matrix[trial, :] > 0)[0]
                axs[row_loc, column_loc].scatter(spike_times, np.repeat(trial, len(spike_times)), color='black',
                                                 s=1, edgecolor='none')

        if plot_psth_below:
            # split the ax object to allow plotting the PSTH at the bottom
            divider = make_axes_locatable(axs[row_loc, column_loc])
            activity_name = 'smoothed_fr'

            if len(psth_ax_store) == 0:
                psth_ax = divider.append_axes('bottom', size='50%', pad=0.1,
                                              sharex=axs[row_loc, column_loc])
            else:
                psth_ax = divider.append_axes('bottom', size='50%', pad=0.1,
                                              sharex=axs[row_loc, column_loc],
                                              sharey=psth_ax_store[-1])

            psth_ax.plot(peri_event_time, stim_cond_respond_left_cell_ds[activity_name].mean('Trial'), color='blue')
            psth_ax.plot(peri_event_time, stim_cond_respond_right_cell_ds[activity_name].mean('Trial'), color='red')
            psth_ax_store.append(psth_ax)

        if include_stim_on_line:
            axs[row_loc, column_loc].axvline(0, linestyle='--', color='black', lw=1)

        if remove_left_spine:
            axs[row_loc, column_loc].spines['left'].set_visible(False)
            axs[row_loc, column_loc].set_yticks([])

        if keep_only_bottom_row_xaxis:
            if row_loc != 2:
                axs[row_loc, column_loc].spines['bottom'].set_visible(False)
                plt.setp(axs[row_loc, column_loc].get_xticklabels(), visible=False)
                axs[row_loc, column_loc].tick_params(length=0)

                # For some reason this just removes the plot
                if plot_psth_below:
                     psth_ax.spines['bottom'].set_visible(False)
                     plt.setp(psth_ax.get_xticklabels(), visible=False)
            else:
                axs[row_loc, column_loc].set_xlim([time_start, time_end])
                axs[row_loc, column_loc].spines['bottom'].set_bounds(time_start, time_end)
                axs[row_loc, column_loc].set_xticks([0, 0.3])


        """
        if square_axis:
             x0, x1 = axs[row_loc, column_loc].get_xlim()
             y0, y1 = axs[row_loc, column_loc].get_ylim()
             axs[row_loc, column_loc].set_aspect((x1-x0)/(y1-y0))
        """

        # axs[row_loc, column_loc].set_facecolor([127/255, 127/255, 127/255, 0.3])
        # TODO: use rectangle instead.

        if include_background:
            rect = mpl.patches.Rectangle((peri_event_time[0], -1), peri_event_time[-1] - peri_event_time[0],
                                         num_left_trial + num_right_trial + 2,
                                         linewidth=0,
                                         facecolor=[127 / 255, 127 / 255, 127 / 255, 0.2])
            axs[row_loc, column_loc].add_patch(rect)

        # TODO: further sort by visual contrast?

        # Audio and visual text labels
        if column_loc == 0:
            axs[row_loc, column_loc].set_ylabel('Audio %s' % aud_cond_str, size=10)

        if row_loc == 0:
            axs[row_loc, column_loc].set_title('Visual %s' % vis_cond_str, size=10)


    fig.text(0.5, 0., 'Peri-stimulus time (s)', size=12, ha='center')


    # fig.tight_layout(pad=3)

    return fig, axs



def plot_passive_grid_raster(cell_ds, scatter_dot_size=0.2, aud_cond_list=[-60, np.inf, 60],
                             vis_cond_list=[[-0.8, -0.4, -0.2, -0.1], [0], [0.1, 0.2, 0.4, 0.8]],
                             scatter_dot_alpha=1, time_start=None, time_end=None,
                             include_stim_on_line=False):


    peri_event_time = cell_ds.PeriEventTime.values
    aud_plot_loc_dict = {'left': 0, 'off': 1, 'right': 2}
    vis_plot_loc_dict = {'left': 0, 'off': 1, 'right': 2}

    cell_ds_subset = cell_ds

    if (time_start is not None) and (time_end is not None):
        cell_ds_subset = cell_ds_subset.where(
            (cell_ds_subset['PeriEventTime'] >= time_start) &
            (cell_ds_subset['PeriEventTime'] <= time_end), drop=True
        )

        cell_ds_subset['audDiff'] = cell_ds_subset['audDiff'].isel(Time=0)
        cell_ds_subset['visDiff'] = cell_ds_subset['visDiff'].isel(Time=0)
        peri_event_time = cell_ds_subset.PeriEventTime.values

    # aesthetics
    remove_left_spine = True
    keep_only_bottom_row_xaxis = True
    include_background = False
    background_color = 'white'

    fig, axs = plt.subplots(3, 3, sharey=True, sharex=True)

    for aud_cond, vis_cond in itertools.product(aud_cond_list, vis_cond_list):

        if ~np.isfinite(aud_cond):
            stim_cond_cell_ds = cell_ds_subset.where(
                (~np.isfinite(cell_ds_subset['audDiff'])) &
                (cell_ds_subset['visDiff'].isin(vis_cond)), drop=True)
            # pdb.set_trace()
        else:
            stim_cond_cell_ds = cell_ds_subset.where(
                (cell_ds_subset['audDiff'].isin(aud_cond)) &
                (cell_ds_subset['visDiff'].isin(vis_cond)), drop=True)

        if len(stim_cond_cell_ds.Trial.values) == 0:
            continue

        if aud_cond < 0:
            aud_cond_str = 'left'
        elif (aud_cond > 0) & np.isfinite(aud_cond):
            aud_cond_str = 'right'
        elif ~np.isfinite(aud_cond):
            aud_cond_str = 'off'

        if vis_cond[0] < 0:
            vis_cond_str = 'left'
        elif vis_cond[0] > 0:
            vis_cond_str = 'right'
        elif vis_cond[0] == 0:
            vis_cond_str = 'off'


        row_loc = aud_plot_loc_dict[aud_cond_str]
        column_loc = vis_plot_loc_dict[vis_cond_str]

        spike_matrix = stim_cond_cell_ds['firing_rate'].values.T
        num_trial = np.shape(spike_matrix)[0]

        for trial in np.arange(num_trial):
            spike_times = np.where(spike_matrix[trial, :] > 0)[0]
            axs[row_loc, column_loc].scatter(peri_event_time[spike_times],
                                             np.repeat(trial, len(spike_times)),
                                             color='black',
                                             s=scatter_dot_size,
                                             alpha=scatter_dot_alpha,
                                             edgecolor='none')

        if remove_left_spine:
            axs[row_loc, column_loc].spines['left'].set_visible(False)
            axs[row_loc, column_loc].set_yticks([])

        if keep_only_bottom_row_xaxis:
            if row_loc != 2:
                axs[row_loc, column_loc].spines['bottom'].set_visible(False)
                plt.setp(axs[row_loc, column_loc].get_xticklabels(), visible=False)
                axs[row_loc, column_loc].tick_params(length=0)
                axs[row_loc, column_loc].spines['bottom'].set_bounds(time_start, time_end)
                axs[row_loc, column_loc].set_xticks([0, 0.3])
                axs[row_loc, column_loc].set_xlim([time_start, time_end])


        # axs[row_loc, column_loc].set_facecolor([127/255, 127/255, 127/255, 0.3])
        # TODO: use rectangle instead.

        if include_stim_on_line:
            axs[row_loc, column_loc].axvline(0, linestyle='--', color='black', lw=1)

        if include_background:
            rect = mpl.patches.Rectangle((peri_event_time[0], -1), peri_event_time[-1] - peri_event_time[0],
                                         num_left_trial + num_right_trial + 2,
                                         linewidth=0,
                                         facecolor=[127 / 255, 127 / 255, 127 / 255, 0.2])
            axs[row_loc, column_loc].add_patch(rect)

        # TODO: further sort by visual contrast?

        # Audio and visual text labels
        if column_loc == 0:
            axs[row_loc, column_loc].set_ylabel('Audio %s' % aud_cond_str, size=10)

        if row_loc == 0:
            axs[row_loc, column_loc].set_title('Visual %s' % vis_cond_str, size=10)

    # Remove center plot
    axs[1, 1].spines['left'].set_visible(False)
    axs[1, 1].set_yticks([])
    axs[1, 1].spines['bottom'].set_visible(False)
    axs[1, 1].tick_params(length=0)

    fig.text(0.5, 0., 'Peri-stimulus time (s)', size=12, ha='center')

    fig.tight_layout()

    return fig, axs


def batch_plot_grid_psth(exp_combined_ds_list, exp_cell_idx_list=None, fig=None, ax=None,
                   shade_significant=False, cell_sig_df=None,
                   save_folder=None,
                   include_condition_count_plot=False,
                   plot_sig_sorted=False, sig_sort_metric='p_val',
                   p_val_title=False, p_val_title_style='stars',  p_val_title_sig_df=None,
                  sig_threshold=0.01, plot_type='mean-trace',
                    sort_by=None, behaviour_df=None, aud_plot_loc_dict={'left': 0, 'off': 1, 'right': 2},
                   vis_plot_loc_dict={'left': 0, 'off': 1, 'right': 2}):
    """
    Loops through plot_grid_psth for multiple cells.
    # TODO: add subject information
    Parameters
    ------------
    :param exp_cell_idx_list:
    :param exp_combined_ds_list:
    :param fig:
    :param ax:
    :param sig_sort_metric: (string)
        name of column to sort the cells by (in terms of 'signifiance of response')
        default uses 'p_val', which is the minimum p-value for a cell in any of the audio-visual conditions
        other options:
            'overall_rank' : (equally) weighted rank between (1) minimum p value (2) firing rate (3) number of
            significant conditions
    :param p_value_title (boolean)
        whether to plot the p-value (pre vs. post event firing rate) as subplot title on each grid.
    :parma p_val_title_sig_df (pandas dataframe)
        only used when p_val_title is set to True
        dataframe that contains p-values used for subplot title
        not to be confused with cell_sig_df, which is used for sorting the order of the plots
        current use case is when I want to sort the order of passive plots by sig in the active condition,
        but I still want to include the passive condition significance p-values in the subplot titles.
    :param shade_significant:
    :param cell_sig_df: pandas dataframe containing column for neuron id, and significance score.
    :param save_folder:
    :return:
    """

    if plot_sig_sorted:

        if sig_sort_metric == 'p_val':
            cell_sig_df = cell_sig_df.loc[cell_sig_df['p_val'] < sig_threshold]
            sorted_sig_df = cell_sig_df.sort_values(by=['p_val'])
            exp_cell_idx_list = np.unique(sorted_sig_df['cell_idx'])
        else:
            # TODO: if this takes too long to run, then can save it and read instead of computing it each time.
            ranked_sig_df = analyse_spike.rank_sig_df(cell_sig_df)
            # remove cells without any significant response to any audio-visual conditions
            ranked_sig_df_thresholded = ranked_sig_df.loc[ranked_sig_df['min_p_val'] < sig_threshold]
            sorted_ranked_df = ranked_sig_df_thresholded.sort_values(by=sig_sort_metric)
            exp_cell_idx_list = sorted_ranked_df['cell_idx']

        for cell_n, exp_cell_idx in enumerate(exp_cell_idx_list):
            fig, axs = plot_grid_psth(exp_cell_idx, exp_combined_ds_list, fig=None, ax=None,
                       shade_significant=shade_significant, cell_sig_df=cell_sig_df,
                                      p_val_title=p_val_title, p_val_title_style=p_val_title_style,
                                      p_val_title_sig_df=p_val_title_sig_df,
                                      plot_type=plot_type,
                                      sort_by=sort_by, behaviour_df=behaviour_df,
                                      aud_plot_loc_dict=aud_plot_loc_dict,
                                      vis_plot_loc_dict=vis_plot_loc_dict
                                      )

            file_name = 'sig_idx_' + str(cell_n) + '_exp_cell_idx_' + str(exp_cell_idx)
            fig.savefig(os.path.join(save_folder, file_name), dpi=FIG_DPI)

            # in case you are using interactive mode or sommething, clears memory
            plt.clf()
            plt.close(fig)

    else:
        if (exp_cell_idx_list is None) or ('all' in exp_cell_idx_list):
            # TODO: this may be problematic
            num_exp_cell = len(exp_combined_ds_list[0]['expCell'])
            exp_cell_idx_list = np.arange(0, num_exp_cell)

        for exp_cell_idx in exp_cell_idx_list:
            fig, axs = plot_grid_psth(exp_cell_idx, exp_combined_ds_list, fig=None, ax=None,
                       shade_significant=shade_significant, cell_sig_df=cell_sig_df,
                                      p_val_title=p_val_title, p_val_title_style=p_val_title_style,
                                      plot_type=plot_type,
                                      sort_by=sort_by, behaviour_df=behaviour_df,
                                      aud_plot_loc_dict=aud_plot_loc_dict,
                                      vis_plot_loc_dict=vis_plot_loc_dict
                                      )

            file_name = 'exp_cell_idx_' + str(exp_cell_idx)
            fig.savefig(os.path.join(save_folder, file_name), dpi=FIG_DPI)

            # in case you are using interactive mode or something, clears memory
            plt.clf()
            plt.close(fig)

    if include_condition_count_plot:
        fig, ax = vizbehaviour.plot_aud_vis_trial_count(behaviour_df, fig=None, ax=None)


def multi_subject_batch_plot_grid_psth(plot_alignment_folder, alignment_data_folder,
                                       exp_type_to_sort_plot_order='passive',
                                       active_alignment_folder=None,
                                       query_subject_list=['all'], query_cell_loc_list=['MOs'],
                                       query_exp_ref_list=['all'],
                                       alignment_file_name='multi-condition-alignment.pkl',
                                       sig_file_name='multi-condition-sig.pkl',
                                       plot_sig_sorted=False, sig_sort_metric='p_val',
                                       p_val_title=False, p_val_title_style='stars',
                                       sig_sorted_neuron_max=None,
                                       include_sig_heatmap=False, include_conditions_count=False,
                                       subset_behaviour_df_file_name='subset-behaviour.pkl',
                                       plot_type='mean-trace', sort_by=None,
                                       aud_plot_loc_dict={'left': 0, 'off': 1, 'right': 2},
                                       vis_plot_loc_dict={'left': 0, 'off': 1, 'right': 2}
                                       ):
    """
    # TODO: allow reading of subset of data, it currently reads everything
    Applies batch_plot_grid_psth through the alignemnt data in alignment_data_folder
    :param plot_alignment_folder: (str)
        path to folder to store the plotted alignment figures
    :param alignment_data_folder: (str)
    :param exp_type_to_sort_plot_order : (str)
        How to sort the order of plotting passive plots
        Default is 'passive', which orders the plot according to the signifcance test(s) of the passive condition
        Alternative is 'active', which orders the plot according to the signficance test(s) of the active condition
    :param active_alignment_folder : (str)
        Only used when exp_type_to_sort_plot_order is set to 'active'
        This is used to read out the cell_sig_df to be used for sorted plot orders.
    :param query_subject_list   : (list of string or int)
        list of subjects you want to plot
    :param query_cell_loc_list  : (list of string)
        list of brain locations you want to plot
    :param query_exp_ref_list   : (list of string or int)
        list of experiment references you want to plot
    :param plot_sig_sorted: (bool)
        if True, creates a new folder and plot neurons sorted by their significance.
    :param sig_sorted_neuron_max: (int)
        Maximum number of neurons to plot sorted by signifiance value.
        If None, plots all neurons.
    :param p_value_title (boolean)
        whether to plot the p-value (pre vs. post event firing rate) as subplot title on each grid.
    :param subset_behaviour_df_file_name (str)
        name of file containing behaviour pandas dataframe
    :param plot_type: (str)
        type of psth plot to make
    :return: 
    """

    if not os.path.exists(plot_alignment_folder):
        os.mkdir(plot_alignment_folder)

    if type(query_subject_list[0]) is not str:
        query_subject_list = [str(x) for x in query_subject_list]

    if type(query_exp_ref_list[0]) is not str:
        query_exp_ref_list = [str(x) for x in query_exp_ref_list]

    # Read alignment folder to look at subjects to plot

    # only if 'all'
    all_subject_folders = glob.glob(os.path.join(alignment_data_folder, '*/'))
    if 'all' in query_subject_list:
        subject_level_folder_list = all_subject_folders
    else:
        subject_level_folder_list = [x for x in all_subject_folders if
                                     any(pat in x for pat in query_subject_list)]

    for subject_level_folder in subject_level_folder_list:

        # Make corresponding subject level folder in plot folder
        subject_level_folder_basename = os.path.basename(os.path.dirname(subject_level_folder))
        fig_subject_level_folder = os.path.join(plot_alignment_folder, subject_level_folder_basename)
        if not os.path.exists(fig_subject_level_folder):
            os.mkdir(fig_subject_level_folder)

        brain_region_folder_list = glob.glob(os.path.join(subject_level_folder, '*/'))

        if not ('all' in query_cell_loc_list):
            brain_region_folder_list = [x for x in brain_region_folder_list if
                                     any(pat in x for pat in query_cell_loc_list)]

        for brain_region_folder in brain_region_folder_list:

            # Make corresponding brain region level folder in plot folder
            brain_region_level_folder_basename = os.path.basename(os.path.dirname(brain_region_folder))
            fig_brain_region_level_folder = os.path.join(fig_subject_level_folder, brain_region_level_folder_basename)
            if not os.path.exists(fig_brain_region_level_folder):
                os.mkdir(fig_brain_region_level_folder)

            if 'all' in query_exp_ref_list:
                exp_level_folder_list = glob.glob(os.path.join(brain_region_folder, '*/'))
            else:
                all_exp_level_folders = glob.glob(os.path.join(brain_region_folder, '*/'))
                exp_level_folder_list = [x for x in all_exp_level_folders if
                                     any(pat in x for pat in query_exp_ref_list)]

            for exp_level_folder in exp_level_folder_list:

                # Make corresponding exp level folder in plot folder
                exp_level_folder_basename = os.path.basename(os.path.dirname(exp_level_folder))
                fig_exp_level_folder = os.path.join(fig_brain_region_level_folder, exp_level_folder_basename)
                if not os.path.exists(fig_exp_level_folder):
                    os.mkdir(fig_exp_level_folder)

                with open(os.path.join(exp_level_folder, alignment_file_name), 'rb') as handle:
                    exp_combined_ds_list = pkl.load(handle)

                if (exp_type_to_sort_plot_order == 'active') and (active_alignment_folder is not None):
                    print('Using active alignment sig to sort passive alignment plot order.')
                    active_exp_level_folder = os.path.join(active_alignment_folder,
                                                           subject_level_folder_basename,
                                                           brain_region_level_folder_basename,
                                                           exp_level_folder_basename)

                    with open(os.path.join(active_exp_level_folder, sig_file_name), 'rb') as handle:
                        cell_sig_df = pkl.load(handle)

                    # Use passive sig df as subplot titles
                    if p_val_title:
                        with open(os.path.join(exp_level_folder, sig_file_name), 'rb') as handle:
                            p_val_title_sig_df = pkl.load(handle)
                    else:
                        p_val_title_sig_df = None

                else:
                    with open(os.path.join(exp_level_folder, sig_file_name), 'rb') as handle:
                        cell_sig_df = pkl.load(handle)
                    p_val_title_sig_df = None

                # Behaviour df is only needed for certain plots (that requires trial information)
                if include_conditions_count or (plot_type == 'heatmap-w-psth'):
                    subset_behaviour_df = pd.read_pickle(os.path.join(
                        exp_level_folder, subset_behaviour_df_file_name))
                else:
                    subset_behaviour_df = None

                if plot_sig_sorted:
                    sig_sorted_folder = os.path.join(fig_exp_level_folder, 'sig-sorted-by-' + sig_sort_metric,
                                                     plot_type)
                    if not os.path.exists(sig_sorted_folder):
                        os.makedirs(sig_sorted_folder)

                    batch_plot_grid_psth(exp_combined_ds_list, exp_cell_idx_list=None, fig=None, ax=None,
                                       shade_significant=True, cell_sig_df=cell_sig_df,
                                         save_folder=sig_sorted_folder,
                                        include_condition_count_plot=False, plot_sig_sorted=plot_sig_sorted,
                                         sig_sort_metric=sig_sort_metric, p_val_title=p_val_title,
                                         p_val_title_style=p_val_title_style,
                                         p_val_title_sig_df=p_val_title_sig_df,
                                         plot_type=plot_type, sort_by=sort_by,
                                         behaviour_df=subset_behaviour_df,
                                         aud_plot_loc_dict=aud_plot_loc_dict,
                                         vis_plot_loc_dict=vis_plot_loc_dict)

                else:
                    batch_plot_grid_psth(exp_combined_ds_list, exp_cell_idx_list=None, fig=None, ax=None,
                                         shade_significant=True, cell_sig_df=cell_sig_df,
                                         p_val_title=p_val_title, p_val_title_style=p_val_title_style,
                                         save_folder=fig_exp_level_folder,
                                         plot_type=plot_type,
                                         include_condition_count_plot=False, sort_by=sort_by,
                                         behaviour_df=subset_behaviour_df,
                                         aud_plot_loc_dict=aud_plot_loc_dict,
                                         vis_plot_loc_dict=vis_plot_loc_dict
                                         )

                # Plot heatmap showing number of significant neurons for each condition
                if include_sig_heatmap:

                    if len(cell_sig_df) == 0:
                        print('No significant cells, or something is wrong. skipping.')
                        continue

                    fig, ax = vizstat.plot_multi_condition_num_sig_neuron(cell_sig_df, fig=None, ax=None,
                                        print_num_neuron=True)
                    sig_heatmap_file_name = 'sig-heatmap.png'
                    fig.savefig(os.path.join(fig_exp_level_folder, sig_heatmap_file_name),
                                dpi=FIG_DPI)

                if include_conditions_count:
                    fig, ax = vizbehaviour.plot_aud_vis_trial_count(subset_behaviour_df)

                    trial_count_heatmap_file_name = 'trial-count.png'
                    fig.savefig(os.path.join(fig_exp_level_folder, trial_count_heatmap_file_name), dpi=FIG_DPI)


def plot_multi_condition_peak_histogram(multi_condition_peak_df, fig=None, axs=None,
                                   num_bins=20):
    """
    Plots a grid of histograms showing the peak times of neural activity (firing rate).
    :param multi_condition_peak_df:
    :param fig:
    :param axs:
    :param num_bins:
    :return:
    """

    aud_condition_list = np.unique(multi_condition_peak_df['aud_cond'])
    vis_condition_list = np.unique(multi_condition_peak_df['vis_cond'])

    if (fig is None) or (ax is None):
        fig, axs = plt.subplots(len(aud_condition_list), len(vis_condition_list),
                                sharex=True, sharey=True)
        fig.set_size_inches(6, 6)

    # This is quite hard coded, perhaps can be a user input.
    aud_plot_loc_dict = {'left': 0, 'off': 1, 'right': 2}
    vis_plot_loc_dict = {'left': 0, 'off': 1, 'right': 2}

    for aud_c, visual_c in list(itertools.product(aud_condition_list,
                                                  vis_condition_list)):


        cond_df = multi_condition_peak_df.loc[
        (multi_condition_peak_df['aud_cond'] == aud_c) &
        (multi_condition_peak_df['vis_cond'] == visual_c)
        ]

        plot_x_loc = aud_plot_loc_dict[aud_c]
        plot_y_loc = vis_plot_loc_dict[visual_c]

        axs[plot_x_loc, plot_y_loc].hist(cond_df['peak_time'], bins=num_bins)

        axs[plot_x_loc, plot_y_loc].axvline(0, linestyle='--', color='gray',
                                            alpha=0.5)

        if plot_y_loc == 0:
            axs[plot_x_loc, plot_y_loc].set_ylabel('Audio: ' +
                                                   aud_c, color='blue')
        if plot_x_loc == 2:
            axs[plot_x_loc, plot_y_loc].set_xlabel('Visual: ' +
                                                   visual_c, color='red')

    fig.text(0.5, 0.0, r'Peri-stimulus \textit{peak} time (s)', ha='center')
    fig.text(0.0, 0.5, 'Number of neurons', va='center', rotation='vertical')


    return fig, axs


def combine_active_passive_psth(active_plot_folder, passive_plot_folder, merged_plot_folder, main_folder_path,
                                query_subject_list=['all'],
                                query_exp_ref_list=['all'], query_cell_loc_list=['all'],
                                active_sub_exp_folder=None, passive_sub_exp_folder=None,
                                merge_keep_rank=False):
    """
    Merges PSTH in the active and passive condition
    :param active_plot_folder:
    :param passive_plot_folder:
    :param main_folder_path:
    :param query_subject_list:
    :param query_exp_ref_list:
    :param query_cell_loc_list:
    :param active_sub_exp_folder: (str)
        name of folder within the active exp folder to extract.
        defaults to None, which gets png file from exp_folder
        example options are 'sig-sorted' or 'sig-sorted-by-overall_rank'
    :param passive_sub_exp_folder: (str)
        name of folder within the active exp folder to extract.
    :return:
    """

    if 'all' in query_subject_list:
        query_subject_list = glob.glob(os.path.join(main_folder_path, active_plot_folder, '*/'))
        query_subject_folder_list = [os.path.basename(os.path.dirname(x)) for x in query_subject_list]
    else:
        query_subject_folder_list = ['subject-' + str(x) for x in query_subject_list]

    if len(query_subject_folder_list) == 0:
        print('Warning: subject folder list is empty')

    for query_subject in query_subject_folder_list:

        if 'all' in query_cell_loc_list:
            query_cell_loc_list = glob.glob(os.path.join(main_folder_path,
                                                         active_plot_folder, query_subject, '*/'))
            query_cell_folder_list = [os.path.basename(os.path.dirname(x)) for x in query_cell_loc_list]
        else:
            query_cell_folder_list = query_cell_loc_list

        if len(query_cell_folder_list) == 0:
            print('Warning: cell folder list is empty')

        for query_cell_loc in query_cell_folder_list:

            if 'all' in query_exp_ref_list:
                query_exp_ref_list = glob.glob(os.path.join(main_folder_path,
                                                            active_plot_folder, query_subject, query_cell_loc, '*/'))
                print(os.path.join(main_folder_path,
                                                            active_plot_folder, query_subject, query_cell_loc))
                query_exp_folder_list = [os.path.basename(os.path.dirname(x)) for x in query_exp_ref_list]
            else:
                query_exp_folder_list = ['exp-' + str(x) for x in query_exp_ref_list]

            if len(query_exp_folder_list) == 0:
                print('Warning: exp folder list is empty')
                # print(query_exp_folder_list)

            for query_exp_ref in query_exp_folder_list:

                active_condition_grid_plot_folder = os.path.join(main_folder_path,
                                                                 active_plot_folder,
                                                                 query_subject,
                                                                 query_cell_loc,
                                                                 query_exp_ref)

                passive_condition_grid_plot_folder = os.path.join(main_folder_path,
                                                                  passive_plot_folder,
                                                                  query_subject,
                                                                  query_cell_loc,
                                                                  query_exp_ref)

                if active_sub_exp_folder is not None:
                    active_condition_grid_plot_folder = os.path.join(active_condition_grid_plot_folder,
                                                                     active_sub_exp_folder)
                if passive_sub_exp_folder is not None:
                    passive_condition_grid_plot_folder = os.path.join(passive_condition_grid_plot_folder,
                                                                     passive_sub_exp_folder)

                active_plot_list = glob.glob(os.path.join(active_condition_grid_plot_folder,
                                                          '*.png'))

                print('Active plot folder: ', active_condition_grid_plot_folder)
                print('Number of active plots found: ', str(len(active_plot_list)))

                passive_plot_list = glob.glob(os.path.join(passive_condition_grid_plot_folder,
                                                           '*.png'))
                print('Number of passive plots found: ', str(len(passive_plot_list)))

                output_image_folder = os.path.join(main_folder_path,
                                                   merged_plot_folder,
                                                   query_subject,
                                                   query_cell_loc,
                                                   query_exp_ref)

                if not os.path.exists(output_image_folder):
                    os.makedirs(output_image_folder)

                for active_condition_plot in active_plot_list:
                    # extract the cell index
                    num_and_ext = active_condition_plot.split('exp_cell_idx_')[1]
                    exp_cell_index = int(num_and_ext.split('.')[0])

                    # This does not deal with matching 0 and 70
                    # match_passive_condition_plot = [i for i in passive_plot_list if
                    #                                 str(exp_cell_index) in
                    #                                 i.split('exp_cell_idx_')[1]][0]

                    match_passive_condition_plot = glob.glob(os.path.join(passive_condition_grid_plot_folder,
                                                                '*exp_cell_idx_' + str(exp_cell_index) + '.png'))




                    assert len(match_passive_condition_plot) == 1, print(
                        'No match or more than one passive plot match: ', match_passive_condition_plot, '\n',
                        'String to match:', os.path.join(passive_condition_grid_plot_folder,
                                                         '*exp_cell_idx_' + str(exp_cell_index) + '.png')
                    )

                    match_passive_condition_plot = match_passive_condition_plot[0]

                    output_image_name = 'combined_exp_cell_idx_' + str(exp_cell_index) + '.png'

                    if merge_keep_rank:
                        rank_index = active_condition_plot.split('_')[3]  # assume filename is
                        # sig_idx_28_exp_cell_idx_38.png
                        output_image_name = 'sig_idx_' + rank_index + '_' + output_image_name

                    combine_images(input_image_path_list=[active_condition_plot,
                                                          match_passive_condition_plot],
                                   output_path=os.path.join(output_image_folder,
                                                            output_image_name))


def combine_images(input_image_path_list, output_path, method='PIL'):
    """
    Combine two png images.
    TODO: this should be moved to vizutil.py
    :param input_image_path_list:
    :param output_path:
    :param method:
    :return:
    """

    images = [Image.open(x) for x in input_image_path_list]

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(output_path)


# Multispaceworld population vector analysis

def make_linearity_unit_plot(aud_plus_vis_rate,
                             aud_and_vis_rate,
                             aud_dir='L', vis_dir='R',
                             offset=3,
                             fig=None, ax=None,
                             axis_label_style='long',
                             neuron_color_map=None,
                             unity_line_color='black'):
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

    if neuron_color_map is not None:
        ax.scatter(aud_plus_vis_rate, aud_and_vis_rate,
                   c=np.arange(0, len(aud_plus_vis_rate)),
                   cmap=neuron_color_map)
    else:
        ax.scatter(aud_plus_vis_rate, aud_and_vis_rate)

    all_firing_rate = np.concatenate([aud_plus_vis_rate,
                                      aud_and_vis_rate])

    ax.set_xlim([min(all_firing_rate) - offset, max(all_firing_rate) + offset])
    ax.set_ylim([min(all_firing_rate) - offset, max(all_firing_rate) + offset])
    ax.grid()

    unity_line_min = -100  # or min(all_firing_rate)
    unity_line_max = 100  # or max(all_firing_rate)
    ax.plot(np.linspace(unity_line_min, unity_line_max, 100),
            np.linspace(unity_line_min, unity_line_max, 100),
            linestyle='--', color=unity_line_color)

    if axis_label_style == 'long':
        ax.set_xlabel(r'$(r_{\text{visR, post-stim}} - r_\text{visR, pre-stim}) +$'
                      r'$(r_\text{audR, post-stim} - r_\text{audR, pre-stim})$')

        ax.set_ylabel(r'$r_\text{audR+visR, post-stim} - r_\text{audR+visR, pre-stim}$')
    else:
        ax.set_xlabel(r'$\Delta r_\text{vis} + \Delta r_\text{aud}$')
        ax.set_ylabel(r'$\Delta r_\text{(vis + aud)}$')

    return fig, ax


def make_grid_linearity_unity_plot(alignment_data,
                                   aud_conditions=['L', 'R'],
                                   vis_conditions=['L', 'R'],
                                   aud_cond_plot_loc={'L': 0, 'R': 1},
                                   vis_cond_plot_loc={'L': 0, 'R': 1},
                                   fig=None, ax=None, sharex=True,
                                   sharey=True,
                                   axis_label_style='short',
                                   neuron_color_map=None,
                                   unity_line_color='black'):
    # TODO: this needs to be separated into one function for
    # making the pairs, and the other for plotting; separate the
    # calculation from the plotting.

    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(len(aud_conditions), len(vis_conditions),
                               sharex=sharex, sharey=sharey)
        fig.set_size_inches(8, 8)

    for aud_cond, vis_cond in itertools.product(aud_conditions,
                                                vis_conditions):
        two_single_condition_rate, one_combined_condition_rate = \
            analyse_spike.make_condition_pair_ds(alignment_data=alignment_data,
                                   aud_dir=aud_cond, vis_dir=vis_cond,
                                   take_pre_post_diff=True,
                                   activity_name='firing_rate')

        plot_row_num = aud_cond_plot_loc[aud_cond]
        plot_column_num = vis_cond_plot_loc[vis_cond]

        # print('Plotting in row %.f and column %.f' % (plot_row_num, plot_column_num))

        fig, ax[plot_row_num, plot_column_num] = \
            make_linearity_unit_plot(two_single_condition_rate,
                                     one_combined_condition_rate,
                                     aud_dir=aud_cond, vis_dir=vis_cond,
                                     offset=3, axis_label_style=axis_label_style,
                                     fig=fig,
                                     ax=ax[plot_row_num, plot_column_num],
                                     neuron_color_map=neuron_color_map,
                                     unity_line_color=unity_line_color)

        ax[plot_row_num, plot_column_num].set_title('Audio: ' + aud_cond + ', '
                                                                           'Visual: ' + vis_cond)
        # TODO: add main x axis and main y axis to show audio and visual loc.
    # TODO: consider adding colors to the neuron plot to show
    # the same neuron across the 4 plots.

    return fig, ax

# Some population vector analysis plots, perhaps will create a vizvecpop.py for these...

def plot_vec_sim_trajectory(similarity_by_bin_to_left_vec_list, similarity_by_bin_to_right_vec_list,
                            time_vec=None,
                            fig=None, ax=None, plot_method='cmap-line',
                            cmap='bwr', include_colorbar=True, xlabel=None, ylabel=None,
                            include_grid=True):
    """
    Plots trajectory of similiarity with a vector.
    TODO: generalise this to any trajectory.
    :param similarity_by_bin_to_left_vec_list:
    :param similarity_by_bin_to_right_vec_list:
    :param time_vec:
    :param fig:
    :param ax:
    :param plot_method:
    :param cmap:
    :return:
    """

    if (fig is None) or (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4)

    # colors.DivergingNorm(vcenter=0)


    if time_vec is None:
        time_vec = np.arange(0, len(similarity_by_bin_to_left_vec_list))

    # naive line plot
    # ax.plot(similarity_by_bin_to_left_vec_list, similarity_by_bin_to_right_vec_list)

    # colormap through the trial duration
    # based on: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html
    if plot_method == 'cmap-line':
        points = np.array([similarity_by_bin_to_left_vec_list, similarity_by_bin_to_right_vec_list]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # color_vals = trial_pop_vector.PeriStimTime
        color_vals = time_vec
        # norm = plt.Normalize(min(trial_pop_vector.PeriStimTime), max(trial_pop_vector.PeriStimTime))
        norm = MidpointNormalize(midpoint=0, vmin=min(color_vals), vmax=max(color_vals))
        lc = LineCollection(segments, cmap=cmap, norm=norm)

        # set the values used for colormapping
        lc.set_array(time_vec)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

        if include_colorbar:
            cbar = fig.colorbar(line)
            cbar.ax.set_ylabel('Peri-event time (s)')

    # scatter method

    elif plot_method == 'cmap-scatter':

        color_vals = time_vec
        norm_func = colors.DivergingNorm(vmin=min(trial_pop_vector.PeriStimTime),
                                         vcenter=0, vmax=max(trial_pop_vector.PeriStimTime))

        norm_color_vals = norm_func(time_vec)

        scatter_points = ax.scatter(similarity_by_bin_to_left_vec_list, similarity_by_bin_to_right_vec_list,
                                    c=color_vals, cmap=cmap,
                                    norm=MidpointNormalize(midpoint=0, vmin=min(color_vals), vmax=max(color_vals)))
        # divider = make_axes_locatable(ax)
        # cax = divider.new_horizontal(size="5%", pad=0.7, pack_start=True)
        # fig.add_axes(cax)
        if include_colorbar:
            cbar = fig.colorbar(scatter_points, ax=ax)
            cbar.ax.set_ylabel('Peri-event time (s)')

    ax.plot(np.linspace(-100, 100000), np.linspace(-100, 100000), linestyle='--', color='grey')


    # TODO: assume array input
    max_val = np.max(np.concatenate([similarity_by_bin_to_left_vec_list,
                                     similarity_by_bin_to_right_vec_list]))

    min_val = np.min(np.concatenate([similarity_by_bin_to_left_vec_list,
                                     similarity_by_bin_to_right_vec_list]))

    """
    max_val = np.max(np.concatenate([np.array(similarity_by_bin_to_left_vec_list),
                                     np.array(similarity_by_bin_to_right_vec_list)]))

    min_val = np.min(np.concatenate([np.array(similarity_by_bin_to_left_vec_list),
                                     np.array(similarity_by_bin_to_right_vec_list)]))
    """

    if 'scatter' in plot_method:
        offset = 1000
        min_val = min_val - offset
        max_val = max_val + offset

    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    if xlabel is not None:
        # ax.set_xlabel(r'$\vec{x}(t) \cdot \vec{\mu}_\text{left}$')
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        # ax.set_ylabel(r'$\vec{x}(t) \cdot \vec{\mu}_\text{right}$')
        ax.set_ylabel(ylabel)

    if include_grid:
        ax.grid()

    return fig, ax


def plot_grid_trajectory(aud_vis_res_dict, time_vec,
                         aud_conditions=['left', 'off', 'center', 'right'],
                         vis_conditions=['left', 'off', 'right'],
                         response_conditions=['left', 'right'],
                         res_cond_cmap_dict={'left': 'Blues', 'right': 'Reds'},
                         aud_pos_dict={'left': 0, 'off': 1, 'center': 2, 'right': 3},
                         vis_pos_dict={'left': 0, 'off': 1, 'right': 2},
                         main_xlabel=None, main_ylabel=None):

    fig, ax = plt.subplots(len(aud_conditions), len(vis_conditions), sharex=True, sharey=True)
    # TODO: make this into the first 'object' found within the dictionary.
    # time_vec = aud_vis_res_dict['left']['left']['left'].PeriStimTime

    for aud_cond, vis_cond, res_cond in itertools.product(aud_conditions, vis_conditions, response_conditions):

        plot_row = aud_pos_dict[aud_cond]
        plot_column = vis_pos_dict[vis_cond]

        sim_to_rL = aud_vis_res_dict[aud_cond][vis_cond][res_cond]['sim_to_rL']
        sim_to_rR = aud_vis_res_dict[aud_cond][vis_cond][res_cond]['sim_to_rR']

        if (len(sim_to_rL) > 0) and (len(sim_to_rR) > 0):
            fig, ax[plot_row, plot_column] = plot_vec_sim_trajectory(
                similarity_by_bin_to_left_vec_list=sim_to_rL,
                similarity_by_bin_to_right_vec_list=sim_to_rR,
                time_vec=time_vec,
                fig=fig, ax=ax[plot_row, plot_column], cmap=res_cond_cmap_dict[res_cond],
                include_colorbar=False, xlabel=None, ylabel=None, include_grid=False)

    # condition labels
    for aud_condition, row_number in aud_pos_dict.items():
        ax[row_number, 0].set_ylabel('Audio:' + aud_condition, color='black')

    for vis_condition, column_number in vis_pos_dict.items():
        ax[-1, column_number].set_xlabel('Visual:' + vis_condition, color='black')

    fig.align_ylabels(ax[:, 0])
    fig.align_xlabels(ax[-1, :])

    # colorbar for the entire grid plot 
    norm = compute_norm_cmap_vals(time_vec)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    import matplotlib as mpl
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Blues'), cax=cbar_ax,
                 label='Peri left movement time')

    cbar_ax_bottom = fig.add_axes([0.1, 0.05, 0.7, 0.02])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Reds'), cax=cbar_ax_bottom,
                 orientation='horizontal',
                 label='Peri right movement time')

    if main_xlabel is not None:
        fig.text(0.45, -0.02, main_xlabel, ha='center', fontsize=18)
    if main_ylabel is not None:
        fig.text(0.0, 0.5, main_ylabel, va='center', rotation='vertical', fontsize=18)

    return fig, ax


def compute_norm_cmap_vals(color_vals):

    norm = MidpointNormalize(midpoint=0, vmin=min(color_vals), vmax=max(color_vals))

    return norm

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    Source: https://chris35wills.github.io/matplotlib_diverging_colorbar/
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def plot_neuron_alignment_condition_trace(alignment_ds, cellIdx=0, cond_1='audLeft', cond_2='audRight',
                                          cond_3=None,
                                          cond_1_title='Audio left', cond_2_title='Audio right',
                                          cond_3_title=None,
                                          include_grid=True, shade_error=True,
                                          fig=None, ax=None, plot_type='trace', cmap_name='viridis',
                                          dot_size=1, dot_alpha=0.3, unimodal_trials_only=True,
                                          smooth_spikes=False, cell_dim_name='Cell',
                                          sort_by=None,
                                          show_movement_time=False,
                                          movement_time_variable_name='timeToWheelMove',
                                          include_legend=True, smooth_window_width=501,
                                          smooth_sigma=51, axvline_loc=0):
    """
    TODO: scatter is extremely slow
    Parameters
    ------------
    :param alignment_ds:
    :param cellIdx:
    :param cond_1:
    :param cond_2:
    :param cond_1_title:
    :param cond_2_title:
    :param include_grid:
    :param shade_error:
    :param fig:
    :param ax:
    :param plot_type:
    :param cmap_name:
    :param dot_size:
    :param dot_alpha:
    :param unimodal_trials_only:
    axvline_loc : (float)
        x axis value to plot the vertical line (to indiciate stimulus / movement time)
        if None specified, then no line will be plot
    :return:
    """

    # TODO: include error bars
    if (fig is None) and (ax is None):

        if plot_type == 'trace':
            fig, ax = plt.subplots()
            fig.set_size_inches(6, 4)
        elif plot_type == 'heatmap':
            fig, ax = plt.subplots(2, 1, sharex=True)
            fig.set_size_inches(6, 4)
        elif plot_type == 'scatter':
            fig, ax = plt.subplots(2, 1, sharex=True)
            fig.set_size_inches(6, 4)
        elif plot_type == 'joined_scatter':
            fig, ax = plt.subplots()
            fig.set_size_inches(6, 4)

    # cell_ds = alignment_ds.isel(Cell=cellIdx)

    if type(cond_1) is str and type(cond_2) is str:
        cell_ds = alignment_ds.isel({cell_dim_name: cellIdx})
        cond_1_cell_ds, cond_2_cell_ds = analyse_spike.get_two_cond_ds(cell_ds, cond_1=cond_1, cond_2=cond_2,
                                                                       unimodal_trials_only=unimodal_trials_only)
    elif cellIdx is not None:
        cond_1_cell_ds = cond_1.isel({cell_dim_name: cellIdx})
        cond_2_cell_ds = cond_2.isel({cell_dim_name: cellIdx})

        if cond_3 is not None:
            cond_3_cell_ds = cond_3.isel({cell_dim_name: cellIdx})

    else:
        cond_1_cell_ds = cond_1
        cond_2_cell_ds = cond_2

        # Process a third condition
        if cond_3 is not None:
            cond_3_cell_ds = cond_3

    if smooth_spikes:
        if len(cond_1_cell_ds['Trial'].values) >= 1:
            if len(cond_1_cell_ds['Trial'].values) == 1:
                print('Only one trial for cond 1')
                cond_1_smoothed_firing_rate = analyse_spike.smooth_spikes(cond_1_cell_ds['firing_rate'].T,
                                                     method='half_gaussian', sigma=smooth_sigma,
                                                                            window_width=smooth_window_width)
                cond_1_cell_ds['firing_rate'] = (['Trial', 'Time'], cond_1_smoothed_firing_rate)
            else:
                cond_1_cell_ds['firing_rate'] = cond_1_cell_ds['firing_rate'].groupby('Trial').apply(
                                                      analyse_spike.smooth_spikes,
                                                      method='half_gaussian', sigma=smooth_sigma,
                                                      window_width=smooth_window_width)
        if len(cond_2_cell_ds['Trial'].values) >= 1:
            if len(cond_2_cell_ds['Trial'].values) == 1:
                print('Only one trial for cond 2')
                cond_2_smoothed_firing_rate = analyse_spike.smooth_spikes(cond_2_cell_ds['firing_rate'].T,
                                                                            method='half_gaussian',
                                                                            sigma=smooth_sigma,
                                                                            window_width=smooth_window_width)
                cond_2_cell_ds['firing_rate'] = (['Trial', 'Time'], cond_2_smoothed_firing_rate)
            else:
                cond_2_cell_ds['firing_rate'] = cond_2_cell_ds['firing_rate'].groupby('Trial').apply(
                                                        analyse_spike.smooth_spikes,
                                                        method='half_gaussian', sigma=smooth_sigma,
                                                        window_width=smooth_window_width)

        if cond_3 is not None:
            # print('Doing smoothing on condition 3')
            if len(cond_3_cell_ds['Trial'].values) >= 1:
                if len(cond_3_cell_ds['Trial'].values) == 1:
                    print('Only one trial for cond 3')
                    cond_3_smoothed_firing_rate = analyse_spike.smooth_spikes(cond_3_cell_ds['firing_rate'].T,
                                                                                method='half_gaussian',
                                                                                sigma=smooth_sigma,
                                                                                window_width=smooth_window_width)
                    cond_3_cell_ds['firing_rate'] = (['Trial', 'Time'], cond_3_smoothed_firing_rate)
                else:
                    # print('Performing smoothing')
                    cond_3_cell_ds['firing_rate'] = cond_3_cell_ds['firing_rate'].groupby('Trial').apply(
                        analyse_spike.smooth_spikes,
                        method='half_gaussian', sigma=smooth_sigma,
                        window_width=smooth_window_width)

    if len(cond_1_cell_ds.Trial.values) == 0:
        print('Condition 1 has no trials')
    if len(cond_2_cell_ds.Trial.values) == 0:
        print('Condition 2 has no trials')

    cond_1_firing_rate_mean = cond_1_cell_ds['firing_rate'].mean(dim='Trial').values
    cond_2_firing_rate_mean = cond_2_cell_ds['firing_rate'].mean(dim='Trial').values

    if cond_3 is not None:
        cond_3_firing_rate_mean = cond_3_cell_ds['firing_rate'].mean(dim='Trial').values

    """
    if smooth_spikes:

        cond_1_firing_rate_mean = analyse_spike.smooth_spikes(
            cond_1_firing_rate_mean, method='half_gaussian', sigma=51, window_width=501)
        cond_2_firing_rate_mean = analyse_spike.smooth_spikes(
            cond_2_firing_rate_mean, method='half_gaussian', sigma=51, window_width=501)
    """

    if plot_type == 'trace':
        ax.plot(cond_1_cell_ds['PeriEventTime'].values,
                cond_1_firing_rate_mean,
                color='blue', label=cond_1_title)

        ax.plot(cond_2_cell_ds['PeriEventTime'].values,
                cond_2_firing_rate_mean,
                color='red', label=cond_2_title)

        if cond_3 is not None:
            ax.plot(cond_3_cell_ds['PeriEventTime'].values,
                    cond_3_firing_rate_mean,
                    color='grey', label=cond_3_title)


        cond_1_firing_rate_std = cond_1_cell_ds['firing_rate'].std(dim='Trial').values
        cond_1_firing_rate_sem = cond_1_firing_rate_std / np.sqrt(len(cond_1_cell_ds['Trial'].values))

        cond_2_firing_rate_std = cond_2_cell_ds['firing_rate'].std(dim='Trial').values
        cond_2_firing_rate_sem = cond_2_firing_rate_std / np.sqrt(len(cond_2_cell_ds['Trial'].values))

        if cond_3 is not None:
            cond_3_firing_rate_std = cond_3_cell_ds['firing_rate'].std(dim='Trial').values
            cond_3_firing_rate_sem = cond_3_firing_rate_std / np.sqrt(len(cond_3_cell_ds['Trial'].values))

        if shade_error:
            ax.fill_between(cond_1_cell_ds['PeriEventTime'].values,
                            cond_1_firing_rate_mean + cond_1_firing_rate_sem,
                            cond_1_firing_rate_mean - cond_1_firing_rate_sem,
                            alpha=0.3, linewidth=0, color='blue')

            ax.fill_between(cond_2_cell_ds['PeriEventTime'].values,
                            cond_2_firing_rate_mean + cond_2_firing_rate_sem,
                            cond_2_firing_rate_mean - cond_2_firing_rate_sem,
                            alpha=0.3, linewidth=0, color='red')

            if cond_3 is not None:
                ax.fill_between(cond_3_cell_ds['PeriEventTime'].values,
                                cond_3_firing_rate_mean + cond_3_firing_rate_sem,
                                cond_3_firing_rate_mean - cond_3_firing_rate_sem,
                                alpha=0.3, linewidth=0, color='grey')

        if include_legend:
            ax.legend()

        ax.set_xlabel('Peri-stimulus time (s)')
        ax.set_ylabel('Firing rate (spikes / s)')

        if axvline_loc is not None:
            ax.axvline(axvline_loc, linestyle='--', color='gray')

        if include_grid:
            ax.grid()

    elif plot_type == 'heatmap':
        num_trial = len(cond_1_cell_ds['Trial'])
        peri_time_start = cond_1_cell_ds['PeriEventTime'][0].values
        peri_time_end = cond_1_cell_ds['PeriEventTime'][-1].values

        cond_1_firing_rate_matrix = cond_1_cell_ds['firing_rate'].values.T
        cond_2_firing_rate_matrix = cond_2_cell_ds['firing_rate'].values.T

        vmin = np.min(np.concatenate([cond_1_firing_rate_matrix.flatten(),
                                      cond_2_firing_rate_matrix.flatten()]))

        vmax = np.max(np.concatenate([cond_1_firing_rate_matrix.flatten(),
                                      cond_2_firing_rate_matrix.flatten()]))

        ax[0].imshow(cond_1_firing_rate_matrix, aspect='auto',
                     extent=[peri_time_start, peri_time_end, num_trial, 1],
                     cmap=cmap_name, vmin=vmin, vmax=vmax)

        im = ax[1].imshow(cond_2_firing_rate_matrix, aspect='auto',
                     extent=[peri_time_start, peri_time_end, num_trial, 1],
                     cmap=cmap_name, vmin=vmin, vmax=vmax)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        ax[0].set_title(cond_1_title)
        ax[1].set_title(cond_2_title)

        ax[0].axvline(0, linestyle='--', color='gray')
        ax[1].axvline(0, linestyle='--', color='gray')

        ax[1].set_xlabel('Peri-stimulus time (s)')
        fig.text(0.04, 0.5, 'Trial', va='center', rotation='vertical')

    elif plot_type == 'scatter':

        for trial_idx in np.arange(len(cond_1_cell_ds['Trial'])):
            cell_trial_ds = cond_1_cell_ds.isel(Trial=trial_idx)
            spike_ds = cell_trial_ds.where(cell_trial_ds['firing_rate'] > 0, drop=True)
            spike_times = spike_ds['PeriEventTime'].values
            ax[0].scatter(spike_times,
                       np.repeat(trial_idx + 1, len(spike_times)), color='blue',
                       alpha=dot_alpha, edgecolor='none', s=dot_size)

        for trial_idx in np.arange(len(cond_2_cell_ds['Trial'])):
            cell_trial_ds = cond_2_cell_ds.isel(Trial=trial_idx)
            spike_ds = cell_trial_ds.where(cell_trial_ds['firing_rate'] > 0, drop=True)
            spike_times = spike_ds['PeriEventTime'].values
            ax[1].scatter(spike_times,
                       np.repeat(trial_idx + 1, len(spike_times)), color='red',
                       alpha=dot_alpha, edgecolor='none', s=dot_size)

        ax[0].set_title(cond_1_title)
        ax[1].set_title(cond_2_title)

        ax[0].axvline(0, linestyle='--', color='gray')
        ax[1].axvline(0, linestyle='--', color='gray')

        ax[1].set_xlabel('Peri-stimulus time (s)')
        fig.text(0.04, 0.5, 'Trial', va='center', rotation='vertical')

    elif plot_type == 'joined_scatter':

        if sort_by is not None:
            cond_1_cell_ds = cond_1_cell_ds.sortby(sort_by, ascending=False)
            cond_2_cell_ds = cond_2_cell_ds.sortby(sort_by, ascending=False)

            if cond_3 is not None:
                cond_3_cell_ds = cond_3_cell_ds.sortby(sort_by, ascending=False)

        for trial_idx in np.arange(len(cond_1_cell_ds['Trial'])):
            cell_trial_ds = cond_1_cell_ds.isel(Trial=trial_idx)
            spike_ds = cell_trial_ds.where(cell_trial_ds['firing_rate'] > 0, drop=True)
            spike_times = spike_ds['PeriEventTime'].values
            ax.scatter(spike_times,
                       np.repeat(trial_idx + 1, len(spike_times)), color='blue',
                       alpha=dot_alpha, edgecolor='none', s=dot_size)

            cond_1_num_trials = len(cond_1_cell_ds['Trial'].values)

            if show_movement_time:
                ax.scatter(cell_trial_ds[movement_time_variable_name], trial_idx+1, color='black',
                           s=dot_size, marker='|')

        # TODO: a more elegant (and hopefully efficient way) by converting the array into a scatter
        # trial, bin = p.argwhere(spike_array['firing_rate'].values > 0).T
        # spike_times = spike_ds['PeriEventTime'].values[bin]
        # ax.scatter(trial, spike_times)


        for trial_idx in np.arange(len(cond_2_cell_ds['Trial'])):
            cell_trial_ds = cond_2_cell_ds.isel(Trial=trial_idx)
            spike_ds = cell_trial_ds.where(cell_trial_ds['firing_rate'] > 0, drop=True)
            spike_times = spike_ds['PeriEventTime'].values
            ax.scatter(spike_times,
                       np.repeat(cond_1_num_trials + trial_idx + 1, len(spike_times)), color='red',
                       alpha=dot_alpha, edgecolor='none', s=dot_size)

            if show_movement_time:
                ax.scatter(cell_trial_ds[movement_time_variable_name], cond_1_num_trials + trial_idx + 1, color='black',
                           s=dot_size, marker='|')

        # TODO: do the same for condition 3
        # ax[0].set_title(cond_1_title)
        # ax[1].set_title(cond_2_title)

        ax.axvline(0, linestyle='--', color='gray')
        # ax[1].axvline(0, linestyle='--', color='gray')

        ax.set_xlabel('Peri-stimulus time (s)')
        ax.set_ylabel('Trial')
        # fig.text(0.04, 0.5, 'Trial', va='center', rotation='vertical')


    return fig, ax


def plot_stim_and_movement_aligned_spikes(aligned_ds, mean_subtracted_aligned_ds=None, cell_idx=25, num_trial_to_plot=5, fig=None, ax=None,
                                          yfill_min=0, yfill_max=6, pre_movement_time=0.7,
                                          post_movement_time=0.7, ymin=0, ymax=None):

    if fig is None and ax is None:
        fig, ax = plt.subplots(num_trial_to_plot, 1, sharex=True)

    for n_trial, trial in enumerate(np.arange(num_trial_to_plot)):
        trial_ds = aligned_ds.isel(Trial=trial, CellId=cell_idx)
        if mean_subtracted_aligned_ds is not None:
            mean_subtracted_trial_ds = mean_subtracted_aligned_ds.isel(Trial=trial, CellId=cell_idx)
        ax[n_trial].plot(trial_ds['PeriStimTime'], trial_ds['SpikeRate'])
        ax[n_trial].axvline(trial_ds['timeToFirstMove'], linestyle='--')
        ax[n_trial].fill_betweenx(y=[yfill_min, yfill_max], x1=-0.1, x2=0.1, alpha=0.3, edgecolor='None', facecolor='grey')

        if mean_subtracted_aligned_ds is not None:
            ax[n_trial].plot(trial_ds['PeriStimTime'], mean_subtracted_trial_ds['SpikeRate'])

        pre_movement_window = trial_ds['timeToFirstMove'] - pre_movement_time
        post_movement_window = trial_ds['timeToFirstMove'] + post_movement_time

        ax[n_trial].fill_betweenx(y=[yfill_min, yfill_max], x1=pre_movement_window, x2=post_movement_window, alpha=0.1,
                                  edgecolor='None', facecolor='blue')
        ax[n_trial].set_ylim([ymin, ymax])

    fig.text(0.04, 0.5, va='center', rotation=90, s='Firing rate (spikes / s)')
    fig.text(0.5, -0.02, ha='center', s='Peri-stimulus time (s)')

    return fig, ax


def plot_slow_vs_fast_rt_trial_pop_raster(fast_rt_trial_ds, slow_rt_trial_ds, activity_name='SpikeRate',
                                          fig=None, ax=None):

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(10, 4)

    peri_stim_time_start = fast_rt_trial_ds['PeriStimTime'][0].values
    peri_stim_time_end = fast_rt_trial_ds['PeriStimTime'][-1].values

    fast_reaction_time = fast_rt_trial_ds['firstTimeToWheelMove']
    slow_reaction_time = slow_rt_trial_ds['firstTimeToWheelMove']

    num_cell = len(fast_rt_trial_ds['Cell'])

    global_max = np.max(np.concatenate([
        fast_rt_trial_ds[activity_name].values.flatten(),
        slow_rt_trial_ds[activity_name].values.flatten()]))

    global_min = np.min(np.concatenate([
        fast_rt_trial_ds[activity_name].values.flatten(),
        slow_rt_trial_ds[activity_name].values.flatten()]))

    ax[0].set_title('Fast reaction time (< 200 ms)', size=12)

    ax[0].imshow(fast_rt_trial_ds[activity_name].values,
                 aspect='auto',
                 extent=[peri_stim_time_start, peri_stim_time_end, num_cell, 0],
                 vmin=global_min, vmax=global_max)
    ax[0].axvline(fast_reaction_time, linestyle='--', color='white')

    ax[1].set_title('Slow reaction time (> 300 ms)', size=12)
    ax[1].imshow(slow_rt_trial_ds['SpikeRate'].values,
                 aspect='auto',
                 extent=[peri_stim_time_start, peri_stim_time_end, num_cell, 0],
                 vmin=global_min, vmax=global_max)
    ax[1].axvline(slow_reaction_time, linestyle='--', color='white')

    # stimulus line
    ax[0].axvline(0, linestyle='--', color='grey')
    ax[1].axvline(0, linestyle='--', color='grey')

    fig.text(0.05, 0.5, s='Cell', rotation=90, va='center', size=12)
    fig.text(0.5, 0.01, s='Peri-stimulus time (s)', ha='center', size=12)

    return fig, ax


def plot_grid_mean_rate_dist(cell_ds, vis_cond, aud_cond, fig=None, ax=None):

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(len(vis_cond), len(aud_cond))
        fig.set_size_inches(8, 8)




    return fig, ax


def plot_neurometric_curve(df, sem_df=None, x_var_name='visCond',
                           y_var_name='firing_rate', colors=['blue', 'red'],
                           group_name_dict={-1: 'Left', 1:'Right'},
                           groups_col_name='audCond', fig=None, ax=None):

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()

    for n_group, group_val in enumerate(np.unique(df[groups_col_name])):
        specific_group_df = df.loc[df[groups_col_name] == group_val]

        ax.scatter(specific_group_df[x_var_name], specific_group_df[y_var_name],
                   color=colors[n_group])
        if group_name_dict is not None:
            line_label = group_name_dict[group_val]
        else:
            line_label = group_val
        ax.plot(specific_group_df[x_var_name], specific_group_df[y_var_name],
                label=line_label, color=colors[n_group])

        if sem_df is not None:
            specific_group_sem_df = sem_df.loc[sem_df[groups_col_name] == group_val]

            ax.fill_between(specific_group_df[x_var_name],
                            specific_group_df[y_var_name] - specific_group_sem_df[y_var_name],
                            specific_group_df[y_var_name] + specific_group_sem_df[y_var_name],
                            alpha=0.5, color=colors[n_group], edgecolor='None')

    ax.set_xlabel('Visual contrast', size=12)
    ax.set_ylabel('Firing rate (spikes/s)', size=12)
    ax.legend(title='Audio')

    return fig, ax


def plot_active_grid_heatmap(cell_ds, max_rt=0.4, time_window_start=0.0,
                             time_window_end=0.3, vis_cond_list=[[-0.8, -0.4, -0.2, -0.1],
                                                                 [0], [0.1, 0.2, 0.4, 0.8]],
                             aud_on_xaxis=True, verbose=True,
                             pre_stim_window=None, baseline_subtraction_window=None,
                             rt_variable_name='firstTimeToWheelMove',
                             choice_variable='responseMade', custom_vmin=None,
                             round_cbar=False, min_max_ticks_only=False,
                             hide_cbar_ticks=False, cmap='viridis', plot_type='heatmap',
                             custom_vmax=None):

    cell_ds_subset = cell_ds.where(
        cell_ds[rt_variable_name] <= max_rt, drop=True
    )

    cell_ds_subset_t_sliced = cell_ds_subset.where(
        (cell_ds_subset['PeriEventTime'] >= time_window_start) &
        (cell_ds_subset['PeriEventTime'] <= time_window_end), drop=True
    )

    if baseline_subtraction_window is not None:
        cell_ds_baseline = cell_ds_subset.where(
        (cell_ds_subset['PeriEventTime'] >= baseline_subtraction_window[0]) &
        (cell_ds_subset['PeriEventTime'] <= baseline_subtraction_window[1]), drop=True
    )

    if pre_stim_window is not None:
        cell_ds_pre_stim_ds = cell_ds_subset.where(
        (cell_ds_subset['PeriEventTime'] >= pre_stim_window[0]) &
        (cell_ds_subset['PeriEventTime'] <= pre_stim_window[1]), drop=True
        )

        all_stim_cond_pre_stim_mean_fr = cell_ds_pre_stim_ds['firing_rate'].mean(
            ['Time', 'Trial']).values

    aud_cond_list = [-60, 0, 60]
    response_cond_list = [1, 2]

    stim_cond_mean_list = list()

    left_mean_fr_matrix = np.zeros((3, 3))
    right_mean_fr_matrix = np.zeros((3, 3))

    for n_aud_cond, aud_cond in enumerate(aud_cond_list):

        for n_vis_cond, vis_cond in enumerate(vis_cond_list):

            for n_response, response in enumerate(response_cond_list):

                stim_cond_response_ds = cell_ds_subset_t_sliced.where(
                    (cell_ds_subset_t_sliced['audDiff'].isin(aud_cond)) &
                    (cell_ds_subset_t_sliced['visDiff'].isin(vis_cond)) &
                    (cell_ds_subset_t_sliced[choice_variable] == response), drop=True)

                stim_cond_mean_per_trial = stim_cond_response_ds['firing_rate'].mean(['Time'])
                stim_cond_mean = stim_cond_mean_per_trial.mean('Trial')

                if baseline_subtraction_window is not None:
                    stim_cond_baseline_ds = cell_ds_baseline.where(
                        (cell_ds_baseline['audDiff'].isin(aud_cond)) &
                        (cell_ds_baseline['visDiff'].isin(vis_cond)) &
                        (cell_ds_baseline[choice_variable] == response), drop=True)
                    baseline_stim_cond_mean = stim_cond_baseline_ds['firing_rate'].mean(
                        ['Time', 'Trial'])
                    stim_cond_mean = stim_cond_mean - baseline_stim_cond_mean

                    if stim_cond_mean.size == 0:
                        stim_cond_mean = np.nan

                if response == 1:
                    if aud_on_xaxis:
                        left_mean_fr_matrix[n_vis_cond, n_aud_cond] = stim_cond_mean
                    else:
                        left_mean_fr_matrix[int(2 - n_aud_cond), n_vis_cond] = stim_cond_mean
                else:
                    if aud_on_xaxis:
                        right_mean_fr_matrix[n_vis_cond, n_aud_cond] = stim_cond_mean
                    else:
                        right_mean_fr_matrix[int(2 - n_aud_cond), n_vis_cond] = stim_cond_mean

    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)

    vmax = np.nanmax(np.stack([left_mean_fr_matrix, right_mean_fr_matrix]))
    vmin = np.nanmin(np.stack([left_mean_fr_matrix, right_mean_fr_matrix]))

    if round_cbar:
        vmax = np.floor(vmax + 0.5)
        vmin = np.floor(vmin - 0.5)



    if pre_stim_window is not None:

        if all_stim_cond_pre_stim_mean_fr.ndim == 0:
            all_stim_cond_pre_stim_mean_fr = np.array([all_stim_cond_pre_stim_mean_fr])

        vmin = np.min(np.concatenate([np.array([vmin]),
                                      all_stim_cond_pre_stim_mean_fr]))

    if custom_vmin is not None:
        vmin = custom_vmin
    if custom_vmax is not None:
        vmax = custom_vmax

    axs[0].set_title('Choose left', size=12)
    axs[1].set_title('Choose right', size=12)

    if plot_type == 'heatmap':
        im = axs[0].imshow(left_mean_fr_matrix, vmax=vmax, vmin=vmin, cmap=cmap)
        axs[1].imshow(right_mean_fr_matrix, vmax=vmax, vmin=vmin, cmap=cmap)
        axs[0].set_xticks([0, 1, 2])
        axs[0].set_ylim([-0.5, 2.5])
    elif plot_type == 'disc' or plot_type == 'disc-w-cmap':
        # See: https://stackoverflow.com/questions/59381273/heatmap-with-circles-indicating-size-of-population
        disc_x, disc_y = np.meshgrid([0, 1, 2], [0, 1, 2])
        disc_radius_max = 0.4
        disc_radius_min = 0.01

        disc_radius_left = (disc_radius_max - disc_radius_min) * (left_mean_fr_matrix - vmin) / (vmax - vmin) + disc_radius_min
        circles_left = [plt.Circle((j, i), radius=r) for r, j, i in zip(disc_radius_left.flat, disc_x.flat, disc_y.flat)]

        if plot_type == 'disc-w-cmap':
            c_left = left_mean_fr_matrix.flatten()
            col_left = PatchCollection(circles_left, array=c_left.flatten(), cmap=cmap, edgecolors='none')
            col_left.set_clim([vmin, vmax])
        else:
            col_left = PatchCollection(circles_left, facecolors='black', edgecolors='none')


        axs[0].add_collection(col_left)

        disc_radius_right = (disc_radius_max - disc_radius_min) * (right_mean_fr_matrix - vmin) / (vmax - vmin) + disc_radius_min
        circles_right = [plt.Circle((j, i), radius=r) for r, j, i in zip(disc_radius_right.flat, disc_x.flat, disc_y.flat)]

        if plot_type == 'disc-w-cmap':
            c_right = right_mean_fr_matrix.flatten()
            col_right = PatchCollection(circles_right, array=c_right.flatten(), cmap=cmap, edgecolors='none')
            col_right.set_clim([vmin, vmax])
        else:
            col_right = PatchCollection(circles_right, facecolors='black', edgecolors='none')

        axs[1].add_collection(col_right)

        axs[0].set_xticks([0, 1, 2])
        axs[0].set_ylim([-0.5, 2.5])
        axs[0].set_xlim([-0.5, 2.5])
        axs[0].set_aspect('equal')
        axs[1].set_aspect('equal')




    if aud_on_xaxis:
        axs[0].set_xticklabels([r'$A_L$', r'$A_C$', r'$A_R$'])
        axs[1].set_xticklabels([r'$A_L$', r'$A_C$', r'$A_R$'])
        axs[0].set_yticks([0, 1, 2])
        axs[0].set_yticklabels([r'$V_L$', r'$V_O$', r'$V_R$'])
    else:
        axs[0].set_yticklabels([r'$A_L$', r'$A_C$', r'$A_R$'])
        axs[0].set_xticklabels([r'$V_L$', r'$V_O$', r'$V_R$'])
        axs[0].set_yticks([2, 1, 0])
        axs[1].set_xticklabels([r'$V_L$', r'$V_O$', r'$V_R$'])

    if plot_type == 'heatmap':
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.3, 0.05, 0.4])
        cbar_obj = fig.colorbar(im, cax=cbar_ax)

        if baseline_subtraction_window is not None:
            cbar_ax.set_ylabel(r'$\Delta$ Spike/s', size=12)
        else:
            cbar_ax.set_ylabel('Spike/s', size=12)

        if min_max_ticks_only:
            cbar_obj.set_ticks([vmin, vmax])
        if hide_cbar_ticks:
            cbar_obj.ax.tick_params(size=0)

    return fig, axs


def plot_active_single_grid_heatmap(cell_ds, max_rt=0.4, time_window_start=0.0,
                             time_window_end=0.3, aud_cond_list=[-60, 0, 60],
                             vis_cond_list=[[-0.8, -0.4, -0.2, -0.1],
                                                                 [0], [0.1, 0.2, 0.4, 0.8]],
                             aud_tick_labels=[r'$A_L$', r'$A_C$', r'$A_R$'],
                             vis_tick_labels=[r'$V_L$', r'$V_O$', r'$V_R$'],
                             aud_on_xaxis=True, verbose=True,
                             pre_stim_window=None, baseline_subtraction_window=None,
                             rt_variable_name='firstTimeToWheelMove',
                             choice_variable='responseMade', custom_vmin=None, custom_vmax=None,
                             round_cbar=False, min_max_ticks_only=False,
                             hide_cbar_ticks=False, cmap='viridis', plot_type='heatmap',
                             combined_condition_method='combine-across-choice',
                             fig=None, ax=None, cbar_position='right',
                             custom_stim_vmin=None, custom_stim_vmax=None,
                             custom_choice_vmin=None, custom_choice_vmax=None,
                             circle_linewidth=0, uncertainty_metric='trial-count',
                             disc_colormap_metric='fr-diff', include_cbar=True,
                             include_title=True, print_val_on_disc=False,
                             disc_radius_max=0.4):
    """

    Parameters
    ----------
    cell_ds
    max_rt
    time_window_start
    time_window_end
    aud_cond_list
    vis_cond_list
    aud_tick_labels
    aud_on_xaxis
    verbose
    pre_stim_window
    baseline_subtraction_window
    rt_variable_name
    choice_variable
    custom_vmin
    custom_vmax
    round_cbar
    min_max_ticks_only
    hide_cbar_ticks
    cmap
    plot_type
    combined_condition_method
    fig
    ax
    cbar_position
    custom_stim_vmin
    custom_stim_vmax
    custom_choice_min
    custom_choice_vmax
    disc_colormap_metric : (str)
        'diff-over-sum-w-sig-test' : do difference over sum, but set to 0 if the mannU test is not significant

    include_cbar : (bool)
        whether to include colorbar in the plot
    Returns
    -------

    """

    supported_combined_condition_method = ['combine-across-choice', 'right-left']
    assert combined_condition_method in supported_combined_condition_method, "Supported methods are " + str(supported_combined_condition_method)


    cell_ds_subset = cell_ds.where(
        cell_ds[rt_variable_name] <= max_rt, drop=True
    )

    cell_ds_subset_t_sliced = cell_ds_subset.where(
        (cell_ds_subset['PeriEventTime'] >= time_window_start) &
        (cell_ds_subset['PeriEventTime'] <= time_window_end), drop=True
    )

    if baseline_subtraction_window is not None:
        cell_ds_baseline = cell_ds_subset.where(
        (cell_ds_subset['PeriEventTime'] >= baseline_subtraction_window[0]) &
        (cell_ds_subset['PeriEventTime'] <= baseline_subtraction_window[1]), drop=True
    )

    if pre_stim_window is not None:
        cell_ds_pre_stim_ds = cell_ds_subset.where(
        (cell_ds_subset['PeriEventTime'] >= pre_stim_window[0]) &
        (cell_ds_subset['PeriEventTime'] <= pre_stim_window[1]), drop=True
        )

        all_stim_cond_pre_stim_mean_fr = cell_ds_pre_stim_ds['firing_rate'].mean(
            ['Time', 'Trial']).values

    response_cond_list = [1, 2]

    stim_cond_mean_list = list()

    left_mean_fr_matrix = np.zeros((3, 3))
    right_mean_fr_matrix = np.zeros((3, 3))

    if plot_type == 'heatmap-w-uncertainty':
        choice_discrimination_matrix = np.zeros((3, 3))

    if disc_colormap_metric in ['fisher-diff', 'ranksum-p', 'fisher-discriminant',
                                'mannu-adjusted', 'mannu', 'diff-over-sum', 'diff-over-sum-w-sig-test']:
        choice_stat_diff_matrix = np.zeros((3, 3))

    for n_aud_cond, aud_cond in enumerate(aud_cond_list):

        for n_vis_cond, vis_cond in enumerate(vis_cond_list):

            # Store single-trial examples of left and right response to do statistics
            response_samples = list()

            for n_response, response in enumerate(response_cond_list):

                stim_cond_response_ds = cell_ds_subset_t_sliced.where(
                    (cell_ds_subset_t_sliced['audDiff'].isin(aud_cond)) &
                    (cell_ds_subset_t_sliced['visDiff'].isin(vis_cond)) &
                    (cell_ds_subset_t_sliced[choice_variable] == response), drop=True)

                stim_cond_mean_per_trial = stim_cond_response_ds['firing_rate'].mean(['Time'])
                stim_cond_mean = stim_cond_mean_per_trial.mean('Trial')

                if baseline_subtraction_window is not None:
                    stim_cond_baseline_ds = cell_ds_baseline.where(
                        (cell_ds_baseline['audDiff'].isin(aud_cond)) &
                        (cell_ds_baseline['visDiff'].isin(vis_cond)) &
                        (cell_ds_baseline[choice_variable] == response), drop=True)
                    baseline_stim_cond_mean = stim_cond_baseline_ds['firing_rate'].mean(
                        ['Time', 'Trial'])
                    stim_cond_mean = stim_cond_mean - baseline_stim_cond_mean

                    if stim_cond_mean.size == 0:
                        stim_cond_mean = np.nan

                if response == 1:
                    if aud_on_xaxis:
                        left_mean_fr_matrix[n_vis_cond, n_aud_cond] = stim_cond_mean
                    else:
                        left_mean_fr_matrix[int(2 - n_aud_cond), n_vis_cond] = stim_cond_mean
                else:
                    if aud_on_xaxis:
                        right_mean_fr_matrix[n_vis_cond, n_aud_cond] = stim_cond_mean
                    else:
                        right_mean_fr_matrix[int(2 - n_aud_cond), n_vis_cond] = stim_cond_mean

                response_samples.append(stim_cond_mean_per_trial.values)

                if plot_type == 'heatmap-w-uncertainty' and disc_colormap_metric in ['trial-count', 'fano-factor']:
                    # response_samples.append(stim_cond_mean)
                    if uncertainty_metric == 'trial-count':
                        choice_discrimination_matrix[n_vis_cond, n_aud_cond] = len(stim_cond_mean_per_trial)
                    elif uncertainty_metric == 'fano-factor':
                        choice_discrimination_matrix[n_vis_cond, n_aud_cond] = \
                            stim_cond_mean_per_trial.mean('Trial') / stim_cond_mean_per_trial.std('Trial')

            # if plot_type == 'heatmap-w-uncertainty':
            #     if uncertainty_metric == 'trial-count':
            #         choice_discrimination_matrix[n_vis_cond, n_aud_cond] = \
            #         stat.cal_auc_two_cond(cond_1=response_samples[0], cond_2=response_samples[1])
            if disc_colormap_metric == 'ranksum-p':

                # if aud_cond == -60 and vis_cond == [0]:
                #     pdb.set_trace()
                test_stat, p_val = sstats.mannwhitneyu(
                    response_samples[0], response_samples[1]
                )
                # if p_val < 0.01:
                #     pdb.set_trace()
                diff_sign = np.sign(np.mean(response_samples[0]) - np.mean(response_samples[1]))
                num_comparisons = 9
                adjusted_p_val = np.min([p_val * num_comparisons, 1])

                # if adjusted_p_val < 0.01:
                #     pdb.set_trace()

                choice_stat_diff_matrix[n_vis_cond, n_aud_cond] = np.log10(adjusted_p_val) * diff_sign

                if verbose:
                    print('Aud cond: ' + str(aud_cond))
                    print('Vis cond: ' + str(vis_cond))
                    print('Choice stat: ' + str(np.log10(adjusted_p_val) * diff_sign))

            elif disc_colormap_metric == 'mannu-adjusted':

                try:
                    test_stat, p_val = sstats.mannwhitneyu(
                        response_samples[0], response_samples[1]
                    )
                except:
                    # pdb.set_trace()
                    test_stat = 0
                    p_val = 1

                if (len(response_samples[1]) == 0) or (len(response_samples[0]) == 0):
                    p_val = 0.99

                diff_sign = np.sign(np.mean(response_samples[0]) - np.mean(response_samples[1]))

                if np.isnan(diff_sign):
                    diff_sign = 0

                num_comparisons = 9
                adjusted_p_val = np.min([p_val * num_comparisons, 1])

                log_adjusted_p_val = np.log10(adjusted_p_val)

                # if (aud_cond == -60) & (vis_cond[0] < 0):
                #       pdb.set_trace()

                choice_stat_diff_matrix[n_vis_cond, n_aud_cond] = log_adjusted_p_val * diff_sign

                if verbose:
                    print('Aud cond: ' + str(aud_cond))
                    print('Vis cond: ' + str(vis_cond))
                    print('Choice stat: ' + str(log_adjusted_p_val * diff_sign))

            elif disc_colormap_metric == 'mannu':

                try:
                    test_stat, p_val = sstats.mannwhitneyu(
                        response_samples[0], response_samples[1]
                    )
                except:
                    # pdb.set_trace()
                    test_stat = 0
                    p_val = 1

                diff_sign = np.sign(np.mean(response_samples[0]) - np.mean(response_samples[1]))
                adjusted_p_val = np.min([p_val, 1])

                choice_stat_diff_matrix[n_vis_cond, n_aud_cond] = np.log10(adjusted_p_val) * diff_sign

            elif disc_colormap_metric == 'fisher-discriminant':
                fisher_d = (np.mean(response_samples[0]) - np.mean(response_samples[1])) / \
                           (np.var(response_samples[0]) + np.var(response_samples[1]))
                choice_stat_diff_matrix[n_vis_cond, n_aud_cond] = fisher_d

            elif disc_colormap_metric == 'diff-over-sum':
                diff_over_sum = (np.mean(response_samples[1]) - np.mean(response_samples[0])) / \
                           (np.mean(response_samples[0]) + np.mean(response_samples[1]))
                choice_stat_diff_matrix[n_vis_cond, n_aud_cond] = diff_over_sum

            elif disc_colormap_metric == 'diff-over-sum-w-sig-test':
                diff_over_sum = (np.mean(response_samples[1]) - np.mean(response_samples[0])) / \
                                (np.mean(response_samples[0]) + np.mean(response_samples[1]))

                try:
                    test_stat, p_val = sstats.mannwhitneyu(
                        response_samples[0], response_samples[1]
                    )
                except:
                    # pdb.set_trace()
                    test_stat = 0
                    p_val = 1

                p_val_threshold = 0.05
                if p_val <= p_val_threshold:
                    choice_stat_diff_matrix[n_vis_cond, n_aud_cond] = diff_over_sum
                else:
                    choice_stat_diff_matrix[n_vis_cond, n_aud_cond] = 0

            # choice_stat_diff_matrix = choice_stat_diff_matrix.T

    if verbose and disc_colormap_metric in ['ranksum-p', 'mannu-adjusted', 'mannu', 'diff-over-sum']:

        if (aud_cond_list[0] == -60) & (vis_cond_list[0][0] < 0):
            choice_stat_diff_matrix = choice_stat_diff_matrix.T

        # pdb.set_trace()
        print('Choice significance matrix: ' + str(choice_stat_diff_matrix))


    if verbose:
        print('Right mean matrix: ' + str(right_mean_fr_matrix))
        print('Left mean matrix: ' + str(left_mean_fr_matrix))


    if fig is None and ax is None:
        fig, ax = plt.subplots()

    if combined_condition_method == 'combine-across-choice':
        if include_title:
            ax.set_title('Combined across choice', size=12)
        combined_mean_fr_matrix = (left_mean_fr_matrix + right_mean_fr_matrix) / 2
    elif combined_condition_method == 'right-left':
        if include_title:
            ax.set_title('Right - Left', size=12)
        combined_mean_fr_matrix = right_mean_fr_matrix - left_mean_fr_matrix

    vmax = np.nanmax(combined_mean_fr_matrix)
    vmin = np.nanmin(combined_mean_fr_matrix)

    if pre_stim_window is not None:

        if all_stim_cond_pre_stim_mean_fr.ndim == 0:
            all_stim_cond_pre_stim_mean_fr = np.array([all_stim_cond_pre_stim_mean_fr])

        vmin = np.min(np.concatenate([np.array([vmin]),
                                      all_stim_cond_pre_stim_mean_fr]))

    if round_cbar:
        vmax = np.floor(vmax + 0.5)
        vmin = np.floor(vmin - 0.5)

    if custom_vmin is not None:
        vmin = custom_vmin
    if custom_vmax is not None:
        vmax = custom_vmax

    if plot_type == 'heatmap':
        im = ax.imshow(combined_mean_fr_matrix, vmax=vmax, vmin=vmin, cmap=cmap)
        ax.set_xticks([0, 1, 2])
        ax.set_ylim([-0.5, 2.5])
    elif plot_type == 'disc' or plot_type == 'disc-w-cmap':
        # See: https://stackoverflow.com/questions/59381273/heatmap-with-circles-indicating-size-of-population
        disc_x, disc_y = np.meshgrid([0, 1, 2], [0, 1, 2])
        disc_radius_max = 0.4
        disc_radius_min = 0.01
        # pdb.set_trace()

        disc_radius_left = (disc_radius_max - disc_radius_min) * (left_mean_fr_matrix - vmin) / (vmax - vmin) + disc_radius_min
        circles_left = [plt.Circle((j, i), radius=r) for r, j, i in zip(disc_radius_left.flat, disc_x.flat, disc_y.flat)]

        if plot_type == 'disc-w-cmap':
            c_left = left_mean_fr_matrix.flatten()
            col_left = PatchCollection(circles_left, array=c_left.flatten(), cmap=cmap, edgecolors='none')
            col_left.set_clim([vmin, vmax])
        else:
            col_left = PatchCollection(circles_left, facecolors='black', edgecolors='none')


        # axs[0].add_collection(col_left)

        disc_radius_right = (disc_radius_max - disc_radius_min) * (right_mean_fr_matrix - vmin) / (vmax - vmin) + disc_radius_min
        circles_right = [plt.Circle((j, i), radius=r) for r, j, i in zip(disc_radius_right.flat, disc_x.flat, disc_y.flat)]

        if plot_type == 'disc-w-cmap':
            c_right = right_mean_fr_matrix.flatten()
            col_right = PatchCollection(circles_right, array=c_right.flatten(), cmap=cmap, edgecolors='none')
            col_right.set_clim([vmin, vmax])
        else:
            col_right = PatchCollection(circles_right, facecolors='black', edgecolors='none')

        ax.add_collection(col_right)

        ax.set_xticks([0, 1, 2])
        ax.set_ylim([-0.5, 2.5])
        ax.set_xlim([-0.5, 2.5])
        ax.set_aspect('equal')
        ax.set_aspect('equal')

    elif plot_type == 'disc-stim-and-choice':
        print('Using disc as variable, ignoring combined_condition_method input.')
        stim_mean_fr_matrix = (left_mean_fr_matrix + right_mean_fr_matrix) / 2
        choice_diff_mean_fr_matrix = right_mean_fr_matrix - left_mean_fr_matrix

        stim_vmax = np.nanmax(stim_mean_fr_matrix)
        stim_vmin = np.nanmin(stim_mean_fr_matrix)
        choice_vmax = np.nanmax(choice_diff_mean_fr_matrix)
        choice_vmin = np.nanmin(choice_diff_mean_fr_matrix)

        if custom_stim_vmin is not None:
            if custom_stim_vmin == 'baseline':
                stim_vmin = all_stim_cond_pre_stim_mean_fr
            else:
                stim_vmin = custom_stim_vmin
        if custom_stim_vmax is not None:
            stim_vmax = custom_stim_vmax
        if custom_choice_vmin is not None:
            choice_vmin = custom_choice_vmin
        if custom_choice_vmax is not None:
            choice_vmax = custom_choice_vmax

        disc_x, disc_y = np.meshgrid([0, 1, 2], [0, 1, 2])
        disc_radius_min = 0.01

        disc_radius_stim = (disc_radius_max - disc_radius_min) * (stim_mean_fr_matrix - stim_vmin) \
                           / (stim_vmax - stim_vmin) + disc_radius_min
        circles_stim = [plt.Circle((j, i), radius=r) for r, j, i in zip(disc_radius_stim.flat,
                                                                        disc_x.flat, disc_y.flat)]

        if disc_colormap_metric == 'fr-diff':
            choice_colors_vals = choice_diff_mean_fr_matrix.flatten()
            col_stim = PatchCollection(circles_stim, array=choice_colors_vals.flatten(), cmap=cmap,
                                       linewidth=circle_linewidth, edgecolors='black')
            col_stim.set_clim([choice_vmin, choice_vmax])
        elif disc_colormap_metric in ['ranksum-p', 'fisher-discriminant', 'mannu-adjusted', 'mannu', 'diff-over-sum',
                                      'diff-over-sum-w-sig-test']:
            choice_colors_vals = choice_stat_diff_matrix.flatten()
            col_stim = PatchCollection(circles_stim, array=choice_colors_vals.flatten(), cmap=cmap,
                                       linewidth=circle_linewidth, edgecolors='black')
            col_stim.set_clim([choice_vmin, choice_vmax])

        ax.add_collection(col_stim)

        ax.set_xticks([0, 1, 2])
        ax.set_ylim([-0.5, 2.5])
        ax.set_xlim([-0.5, 2.5])
        ax.set_aspect('equal')
        ax.set_aspect('equal')

        norm = colors.Normalize(vmin=choice_vmin, vmax=choice_vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        if include_cbar:
            if cbar_position == 'right':
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.3, 0.05, 0.4])
                cbar_obj = fig.colorbar(sm, cax=cbar_ax)
                if disc_colormap_metric == 'ranksum-p':
                    cbar_ax.set_ylabel(r'$\log(p)$', size=12)
                elif disc_colormap_metric == 'mannu-adjusted':
                    cbar_ax.set_ylabel(r'Adjusted $\log(p)$', size=12)
                elif disc_colormap_metric in ['diff-over-sum', 'diff-over-sum-w-sig-test']:
                    cbar_ax.set_ylabel(r'$\frac{r_1 - r_2}{r_1 + r_2}$', size=12)
                else:
                    cbar_ax.set_ylabel(r'$\Delta$ Spike/s', size=12)
            elif cbar_position == 'bottom':
                cbar_obj = fig.colorbar(sm, orientation="horizontal", ax=ax)
                cbar_ax = cbar_obj.ax
                cbar_ax.set_ylabel(r'$\Delta$ Spike/s', size=12)

    if aud_on_xaxis:
        ax.set_xticklabels(aud_tick_labels)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(vis_tick_labels)
    else:
        # ax.set_yticklabels([r'$A_L$', r'$A_C$', r'$A_R$'])
        ax.set_yticklabels(aud_tick_labels)
        ax.set_xticklabels(vis_tick_labels)
        ax.set_yticks([2, 1, 0])

    if plot_type == 'heatmap':

        if cbar_position == 'right':
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.3, 0.05, 0.4])
            cbar_obj = fig.colorbar(im, cax=cbar_ax)
        elif cbar_position == 'bottom':
            cbar_obj = fig.colorbar(im, orientation="horizontal", ax=ax)
            cbar_ax = cbar_obj.ax


        if baseline_subtraction_window is not None:
            if cbar_position == 'right':
                cbar_ax.set_ylabel(r'$\Delta$ Spike/s', size=12)
            elif cbar_position == 'bottom':
                cbar_ax.set_xlabel(r'$\Delta$ Spike/s', size=12)
        else:
            cbar_ax.set_ylabel('Spike/s', size=12)

        if min_max_ticks_only:
            cbar_obj.set_ticks([vmin, vmax])
        if hide_cbar_ticks:
            cbar_obj.ax.tick_params(size=0)

    if plot_type == 'heatmap-w-uncertainty':
        print('Plotting heatmap with uncertainity')
        choice_diff_mean_fr_matrix = right_mean_fr_matrix - left_mean_fr_matrix
        choice_vmax = np.nanmax(choice_diff_mean_fr_matrix)
        choice_vmin = np.nanmin(choice_diff_mean_fr_matrix)

        if custom_stim_vmin is not None:
            stim_vmin = custom_stim_vmin
        if custom_stim_vmax is not None:
            stim_vmax = custom_stim_vmax
        if custom_choice_vmin is not None:
            choice_vmin = custom_choice_vmin
        if custom_choice_vmax is not None:
            choice_vmax = custom_choice_vmax

        h_neg = 240
        lightness = 65
        h_pos = 0

        choice_diff_mean_fr_matrix_scaled = (choice_diff_mean_fr_matrix - choice_vmin) / (choice_vmax - choice_vmin)

        max_uncert_metric = np.max(choice_discrimination_matrix.flatten())
        min_uncert_metric = np.min(choice_discrimination_matrix.flatten())
        rgba_matrix = np.zeros((3, 3, 4))
        # Decide on the saturation
        n_row, n_col = np.shape(choice_diff_mean_fr_matrix)[0], np.shape(choice_diff_mean_fr_matrix)[1]
        for row, column in itertools.product(np.arange(n_row), np.arange(n_col)):
            sat_val = choice_diff_mean_fr_matrix_scaled[row, column] * 99
            level = (choice_diff_mean_fr_matrix[row, column] - choice_vmin) / (choice_vmax - choice_vmin)
            sat_cmap = sns.diverging_palette(h_neg=h_neg, h_pos=h_pos, s=sat_val,
                                         l=lightness, as_cmap=True)
            rgba_matrix[2-row, column, :] = sat_cmap(level)
            # ax.scatter(row, column, color=sat_cmap(level))
        # pdb.set_trace()
        ax.imshow(rgba_matrix)
        # assume heatmap is centred on zero
        ax.set_xticks([0, 1, 2])

    # pdb.set_trace()

    return fig, ax


def gen_bivariate_cmap(num_sat_values_to_sample=100, num_levels_to_sample=100):

    print("TODO")


    return cmap_list


def plot_passive_grid_heatmap(cell_ds, time_window_start=0.0,
                             time_window_end=0.3, aud_cond_list=[-60, np.inf, 60],
                             vis_cond_list=[[-0.8, -0.4, -0.2, -0.1], [0], [0.1, 0.2, 0.4, 0.8]],
                             baseline_subtraction_window=None, pre_stim_window=None,
                             aud_on_xaxis=True, round_cbar=False, verbose=True,
                             custom_vmin=None, min_max_ticks_only=False, cmap='viridis',
                             plot_type='heatmap',
                             hide_cbar_ticks=False, aud_tick_labels=[r'$A_L$', r'$A_O$', r'$A_R$'],
                             circle_linewidth=1,
                             custom_vmax=None, fig=None, ax=None):
    """

    Parameters
    ----------
    cell_ds
    time_window_start
    time_window_end
    aud_cond_list
    vis_cond_list
    baseline_subtraction_window
    pre_stim_window
    aud_on_xaxis
    round_cbar
    verbose
    custom_vmin
    min_max_ticks_only
    cmap
    plot_type : (str)
        what type of plot to make
            'heatmap' : heatmap plot (firing rate maps onto color)
            'disc' : disc plot (firing rate maps onto disc size)
    hide_cbar_ticks
    aud_tick_labels
    custom_vmax
    fig
    ax

    Returns
    -------

    """

    cell_ds_subset = cell_ds

    cell_ds_subset_t_sliced = cell_ds_subset.where(
        (cell_ds_subset['PeriEventTime'] >= time_window_start) &
        (cell_ds_subset['PeriEventTime'] <= time_window_end), drop=True
    )

    if baseline_subtraction_window is not None:
        cell_ds_baseline = cell_ds_subset.where(
        (cell_ds_subset['PeriEventTime'] >= baseline_subtraction_window[0]) &
        (cell_ds_subset['PeriEventTime'] <= baseline_subtraction_window[1]), drop=True
    )

    mean_fr_matrix = np.zeros((3, 3))

    for n_aud_cond, aud_cond in enumerate(aud_cond_list):

        for n_vis_cond, vis_cond in enumerate(vis_cond_list):

                if ~np.isfinite(aud_cond):
                    stim_cond_response_ds = cell_ds_subset_t_sliced.where(
                        ~np.isfinite(cell_ds_subset_t_sliced['audDiff']) &
                        (cell_ds_subset_t_sliced['visDiff'].isin(vis_cond)), drop=True)
                else:
                    stim_cond_response_ds = cell_ds_subset_t_sliced.where(
                        (cell_ds_subset_t_sliced['audDiff'].isin(aud_cond)) &
                        (cell_ds_subset_t_sliced['visDiff'].isin(vis_cond)), drop=True)

                stim_cond_mean_per_trial = stim_cond_response_ds['firing_rate'].mean(['Time'])
                stim_cond_mean = stim_cond_mean_per_trial.mean('Trial').values

                if baseline_subtraction_window is not None:
                    stim_cond_baseline_ds = cell_ds_baseline.where(
                        (cell_ds_baseline['audDiff'].isin(aud_cond)) &
                        (cell_ds_baseline['visDiff'].isin(vis_cond)), drop=True)
                    baseline_stim_cond_mean = stim_cond_baseline_ds['firing_rate'].mean(
                        ['Time', 'Trial'])
                    stim_cond_mean = stim_cond_mean - baseline_stim_cond_mean
                    if stim_cond_mean.size == 0:
                        stim_cond_mean = np.nan

                if verbose:
                    if vis_cond[0] < 0:
                        vis_cond_str = 'left'
                    elif vis_cond[0] == 0:
                        vis_cond_str = 'off'
                    else:
                        vis_cond_str = 'right'

                    print('Vis cond %s and aud cond %.f firing rate %.2f' % (vis_cond_str, aud_cond, stim_cond_mean))

                if aud_on_xaxis:
                    mean_fr_matrix[n_vis_cond, n_aud_cond] = stim_cond_mean
                else:
                    mean_fr_matrix[int(2-n_aud_cond), n_vis_cond] = stim_cond_mean


    if fig is None and ax is None:
        fig, ax = plt.subplots()

    if plot_type == 'heatmap':
        vmax = np.nanmax(mean_fr_matrix)
        vmin = np.nanmin(mean_fr_matrix)

        if round_cbar:
            vmax = np.floor(vmax + 0.5)
            vmin = np.floor(vmin - 0.5)

        if custom_vmin is not None:
            vmin = custom_vmin

        if aud_on_xaxis:
            im = ax.imshow(mean_fr_matrix, vmax=vmax, vmin=vmin, cmap=cmap)
            ax.set_xticklabels(aud_tick_labels)
            ax.set_yticklabels([r'$V_L$', r'$V_O$', r'$V_R$'])
            ax.set_yticks([0, 1, 2])
        else:
            im = ax.imshow(mean_fr_matrix, vmax=vmax, vmin=vmin, cmap=cmap)
            ax.set_yticklabels(aud_tick_labels)
            ax.set_xticklabels([r'$V_L$', r'$V_O$', r'$V_R$'])
            ax.set_yticks([2, 1, 0])

        ax.set_xticks([0, 1, 2])

        ax.set_ylim([-0.5, 2.5])

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.8, 0.2, 0.05, 0.6])
        cbar_obj = fig.colorbar(im, cax=cbar_ax)
        if baseline_subtraction_window is not None:
            cbar_ax.set_ylabel(r'$\Delta$ Spike/s', size=12)
        else:
            cbar_ax.set_ylabel('Spike/s', size=12)

        if min_max_ticks_only:
            cbar_obj.set_ticks([vmin, vmax])
        if hide_cbar_ticks:
            cbar_obj.ax.tick_params(size=0)

    elif plot_type == 'disc':

        vmax = np.nanmax(mean_fr_matrix)
        vmin = np.nanmin(mean_fr_matrix)

        if custom_vmin is not None:
            if custom_vmin == 'baseline':
                vmin = all_stim_cond_pre_stim_mean_fr
            else:
                vmin = custom_vmin
        if custom_vmax is not None:
            vmax = custom_vmax

        disc_x, disc_y = np.meshgrid([0, 1, 2], [0, 1, 2])
        disc_radius_min = 0.01
        disc_radius_max = 0.4
        disc_colormap_metric = None

        disc_radius_stim = (disc_radius_max - disc_radius_min) * (mean_fr_matrix - vmin) \
                           / (vmax - vmin) + disc_radius_min
        circles_stim = [plt.Circle((j, i), radius=r) for r, j, i in zip(disc_radius_stim.flat,
                                                                        disc_x.flat, disc_y.flat)]

        if disc_colormap_metric == 'fr-diff':
            choice_colors_vals = choice_diff_mean_fr_matrix.flatten()
            col_stim = PatchCollection(circles_stim, array=choice_colors_vals.flatten(), cmap=cmap,
                                       linewidth=circle_linewidth, edgecolors='black')
            col_stim.set_clim([choice_vmin, choice_vmax])
        elif disc_colormap_metric in ['ranksum-p', 'fisher-discriminant', 'mannu-adjusted', 'mannu']:
            choice_colors_vals = choice_stat_diff_matrix.flatten()
            col_stim = PatchCollection(circles_stim, array=choice_colors_vals.flatten(), cmap=cmap,
                                       linewidth=circle_linewidth, edgecolors='black')
            col_stim.set_clim([choice_vmin, choice_vmax])
        else:
            col_stim = PatchCollection(circles_stim, linewidth=circle_linewidth, edgecolors='black',
                                       facecolors='white')

        ax.add_collection(col_stim)

        ax.set_xticks([0, 1, 2])
        ax.set_ylim([-0.5, 2.5])
        ax.set_xlim([-0.5, 2.5])

        if aud_on_xaxis:
            ax.set_xticklabels(aud_tick_labels)
            ax.set_yticklabels([r'$V_L$', r'$V_O$', r'$V_R$'])
            ax.set_yticks([0, 1, 2])
        else:
            ax.set_yticklabels(aud_tick_labels)
            ax.set_xticklabels([r'$V_L$', r'$V_O$', r'$V_R$'])
            ax.set_yticks([2, 1, 0])

    return fig, ax


def plot_active_vs_passive_abs_diff_cumsum(all_responsive_cell_pre_post_fr, all_other_cell_pre_post_fr,
                                           num_bins=100):
    """

    Parameters
    ----------
    all_responsive_cell_pre_post_fr
    all_other_cell_pre_post_fr
    num_bins

    Returns
    -------

    """

    max_fr = np.max(np.concatenate([all_responsive_cell_pre_post_fr, all_other_cell_pre_post_fr]))
    min_fr = np.min(np.concatenate([all_responsive_cell_pre_post_fr, all_other_cell_pre_post_fr]))
    fr_bins = np.linspace(min_fr, max_fr, num=num_bins)

    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)

    all_other_cell_pre_post_fr_counts, other_cell_edges = np.histogram(all_other_cell_pre_post_fr, bins=fr_bins,
                                                                       density=False)
    all_responsive_cell_pre_post_fr_counts, responsive_cell_edges = np.histogram(all_responsive_cell_pre_post_fr,
                                                                                 bins=fr_bins, density=False)

    all_other_cell_pre_post_fr_counts_cumsum = np.cumsum(
        all_other_cell_pre_post_fr_counts / np.sum(all_other_cell_pre_post_fr_counts))
    all_responsive_cell_pre_post_fr_counts_cumsum = np.cumsum(
        all_responsive_cell_pre_post_fr_counts / np.sum(all_responsive_cell_pre_post_fr_counts))

    ax.plot(other_cell_edges[:-1], all_other_cell_pre_post_fr_counts_cumsum, color='gray',
            label='Neurons without passive \n stimulus response')
    ax.plot(responsive_cell_edges[:-1], all_responsive_cell_pre_post_fr_counts_cumsum, color='red',
            label='Neurons with passive \n stimulus response')

    ax.set_xlabel('Change in firing rate \n after stimulus during task (spike/s)', size=12)
    ax.set_ylabel('Cumulative probability', size=12)

    ax.legend(bbox_to_anchor=(0.4, 0.7))

    return fig, ax


def plot_vis_vs_aud_selectivity_unity(vis_neuron_vis_metric, vis_neuron_aud_metric,
                                      aud_neuron_vis_metric, aud_neuron_aud_metric,
                                      av_neuron_vis_metric=None, av_neuron_aud_metric=None,
                                      other_neuron_vis_metric=None, other_neuron_aud_metric=None,
                                      axis_in_middle=False, dot_size=10, text_only_legend=True,
                                      group_neurons=True, fig=None, ax=None):

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

    aud_neuron_color = 'purple'
    vis_neuron_color = 'orange'
    av_neuron_color = 'green'
    other_neuron_color = 'gray'

    if group_neurons:
        ax.scatter(vis_neuron_vis_metric, vis_neuron_aud_metric,
                   s=dot_size, color=vis_neuron_color, label='Visual neurons')

        ax.scatter(aud_neuron_vis_metric, aud_neuron_aud_metric,
                   s=dot_size, color=aud_neuron_color, label='Auditory neurons')

        if av_neuron_vis_metric is not None:
            ax.scatter(av_neuron_vis_metric, av_neuron_aud_metric,
                       s=dot_size, color=av_neuron_color, label='Auditory and Visual neurons')
        if other_neuron_vis_metric is not None:
            ax.scatter(other_neuron_vis_metric, other_neuron_aud_metric,
                       s=dot_size, color=other_neuron_color, label='No stimulus selectivity neurons',
                       zorder=0)  # put other neurons at the bottom

    else:
        ax.scatter(vis_neuron_vis_metric, vis_neuron_aud_metric,
                   s=dot_size, color='black')

        ax.scatter(aud_neuron_vis_metric, aud_neuron_aud_metric,
                   s=dot_size, color='black')


    if axis_in_middle:
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])
        ax.set_xlabel('Visual L/R selectivity', size=12)
        ax.set_ylabel('Auditory L/R selectivity', size=12)
        ax.xaxis.set_label_coords(0.5, -0.04)
        ax.yaxis.set_label_coords(-0.04, 0.5)
        if group_neurons:
            if text_only_legend:
                ax.text(0.5, 0.6, 'Auditory neurons', color=aud_neuron_color)
                ax.text(0.5, 0.5, 'Visual neurons', color=vis_neuron_color)
                if (av_neuron_vis_metric is not None) & (other_neuron_vis_metric is not None):
                    ax.text(0.5, 0.4, 'Audiovisual neurons', color=av_neuron_color)
                    ax.text(0.5, 0.3, 'No selectivity neurons', color=other_neuron_color)
                elif other_neuron_vis_metric is not None:
                    ax.text(0.5, 0.4, 'No selectivity neurons', color=other_neuron_color)


            else:
                ax.legend(bbox_to_anchor=(0.7, 1))
    else:
        ax.axvline(0, linestyle='--', color='gray', lw=1)
        ax.axhline(0, linestyle='--', color='gray', lw=1)
        ax.set_xlabel('Visual L/R selectivity', size=12)
        ax.set_ylabel('Auditory L/R selectivity', size=12)
        if group_neurons:
            ax.legend(bbox_to_anchor=(0.65, 1))

    return fig, ax


def plot_all_passive_sig_stim_cells(test_results_df, alignment_folder, fig_folder, effect_to_plot='visLR',
                                    fig_ext='.png', test_window_width=None, cell_indexing_method='name',
                                    aligned_event='stimOnTime'):
    """

    Parameters
    ----------
    test_results_df : (pandas dataframe)
    alignemnt_folder
    fig_folder
    effect_to_plot

    Returns
    -------

    """
    for _, cell_df in tqdm(test_results_df.iterrows()):

        subject = cell_df['subjectRef']
        exp = cell_df['expRef']
        brain_region = cell_df['brainRegion']

        if cell_indexing_method == 'name':
            cell_idx = cell_df.name
        elif cell_indexing_method == 'Cell':
            cell_idx = cell_df['Cell']

        alignment_ds = pephys.load_subject_exp_alignment_ds(
            alignment_folder=alignment_folder,
            subject_num=subject,
            exp_num=exp,
            target_brain_region=brain_region,
            aligned_event=aligned_event,
            alignment_file_ext='.nc')

        cell_ds = alignment_ds.isel(Cell=cell_idx)

        smooth_multiplier = 5
        cell_ds_stacked = cell_ds.stack(trialTime=['Trial', 'Time'])
        sigma = 3 * smooth_multiplier
        window_width = 20 * smooth_multiplier
        # also do some smoothing
        cell_ds_stacked['firing_rate'] = (['trialTime'],
                                          analyse_spike.smooth_spikes(
                                              cell_ds_stacked['firing_rate'],
                                              method='half_gaussian',
                                              sigma=sigma, window_width=window_width,
                                              custom_window=None))

        smoothed_cell_ds = cell_ds_stacked.unstack()

        peri_event_time = smoothed_cell_ds.PeriEventTime.isel(Trial=0)

        if effect_to_plot == 'audLR':

            aud_left_only_ds = smoothed_cell_ds.where(
                (smoothed_cell_ds['audDiff'] == -60) &
                (smoothed_cell_ds['visDiff'] == 0), drop=True
            )['firing_rate']

            aud_right_only_ds = smoothed_cell_ds.where(
                (smoothed_cell_ds['audDiff'] == 60) &
                (smoothed_cell_ds['visDiff'] == 0), drop=True
            )['firing_rate']

            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, ax = plt.subplots()

                ax.plot(peri_event_time, aud_left_only_ds.mean('Trial'), color='blue', label='left')
                ax.plot(peri_event_time, aud_right_only_ds.mean('Trial'), color='red', label='right')

                if 'audLRabsDiff' in cell_df.axes[0]:
                    ax.set_title('Mean absolute difference: %.1f' % cell_df['audLRabsDiff'], size=12)

                if test_window_width is not None:
                    ax.axvspan(cell_df['audLRmaxDiffTime'] - test_window_width / 2,
                               cell_df['audLRmaxDiffTime'] + test_window_width / 2,
                               alpha=0.4, lw=0, color='gray')

                ax.set_xlabel('Peri-stimulus time (s)', size=12)
                ax.set_ylabel('Spike/s', size=12)
                ax.legend()
                fig_name = 'subject_%.f_exp_%.f_cell_%.f' % (subject, exp, cell_idx)
                fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')
                plt.close(fig)

        elif effect_to_plot == 'visLR':

            vis_left_only_ds = smoothed_cell_ds.where(
                (~np.isfinite(smoothed_cell_ds['audDiff'])) &
                (smoothed_cell_ds['visDiff'] == -0.8), drop=True
            )['firing_rate']

            vis_right_only_ds = smoothed_cell_ds.where(
                (~np.isfinite(smoothed_cell_ds['audDiff'])) &
                (smoothed_cell_ds['visDiff'] == 0.8), drop=True
            )['firing_rate']

            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, ax = plt.subplots()

                ax.plot(peri_event_time.values, vis_left_only_ds.mean('Trial'), color='blue', label='left')
                ax.plot(peri_event_time.values, vis_right_only_ds.mean('Trial'), color='red', label='right')


                if 'visLRabsDiff' in cell_df.axes[0]:
                    ax.set_title('Mean absolute difference: %.1f' % cell_df['visLRabsDiff'], size=12)

                if test_window_width is not None:
                    ax.axvspan(cell_df['vis_lr_max_diff_time'] - test_window_width / 2.0,
                               cell_df['vis_lr_max_diff_time'] + test_window_width / 2.0,
                               alpha=0.4, lw=0, color='gray')

                ax.set_xlabel('Peri-stimulus time (s)', size=12)
                ax.set_ylabel('Spike/s', size=12)
                ax.legend()
                fig_name = 'subject_%.f_exp_%.f_cell_%.f' % (subject, exp, cell_idx)
                fig_ext = '.png'
                fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')
                plt.close(fig)

        elif effect_to_plot == 'audAndVisLR':

            aud_left_only_ds = smoothed_cell_ds.where(
                (smoothed_cell_ds['audDiff'] == -60) &
                (smoothed_cell_ds['visDiff'] == 0), drop=True
            )['firing_rate']

            aud_right_only_ds = smoothed_cell_ds.where(
                (smoothed_cell_ds['audDiff'] == 60) &
                (smoothed_cell_ds['visDiff'] == 0), drop=True
            )['firing_rate']

            vis_left_only_ds = smoothed_cell_ds.where(
                (~np.isfinite(smoothed_cell_ds['audDiff'])) &
                (smoothed_cell_ds['visDiff'] == -0.8), drop=True
            )['firing_rate']

            vis_right_only_ds = smoothed_cell_ds.where(
                (~np.isfinite(smoothed_cell_ds['audDiff'])) &
                (smoothed_cell_ds['visDiff'] == 0.8), drop=True
            )['firing_rate']

            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, axs = plt.subplots(1, 2)
                fig.set_size_inches(8, 4)

                axs[0].plot(peri_event_time.values, vis_left_only_ds.mean('Trial'), color='blue', label=r'$A_L$')
                axs[0].plot(peri_event_time.values, vis_right_only_ds.mean('Trial'), color='red', label=r'$A_R$')

                axs[0].set_title('Mean absolute difference: %.1f' % cell_df['visLRabsDiff'], size=12)

                axs[0].axvspan(cell_df['visLRmaxDiffTime'] - test_window_width / 2.0,
                               cell_df['visLRmaxDiffTime'] + test_window_width / 2.0,
                               alpha=0.4, lw=0, color='gray')

                axs[0].set_xlabel('Peri-stimulus time (s)', size=12)
                axs[0].set_ylabel('Spike/s', size=12)
                axs[0].legend()

                axs[1].plot(peri_event_time.values, aud_left_only_ds.mean('Trial'), color='blue', label=r'$A_L$')
                axs[1].plot(peri_event_time.values, aud_right_only_ds.mean('Trial'), color='red', label=r'$A_R$')

                if 'audLRabsDiff' in cell_df.axes[0]:
                    axs[1].set_title('Mean absolute difference: %.1f' % cell_df['audLRabsDiff'], size=12)

                if test_window_width is not None:
                    axs[1].axvspan(cell_df['audLRmaxDiffTime'] - test_window_width / 2.0,
                                   cell_df['audLRmaxDiffTime'] + test_window_width / 2.0,
                                   alpha=0.4, lw=0, color='gray')

                axs[1].set_xlabel('Peri-stimulus time (s)', size=12)
                # axs[1].set_ylabel('Spike/s', size=12)
                axs[1].legend()

                fig_name = 'subject_%.f_exp_%.f_cell_%.f' % (subject, exp, cell_idx)
                fig_ext = '.png'
                fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')
                plt.close(fig)

    return fig


def plot_active_additive_model_prediction_and_raster(movement_spike_matrix,
                                                     prediction_matrix, rt_vec, peri_event_time,
                                                     plot_prediction=True, plot_movement_times=True,
                                                     scatter_dot_size=3, spike_dot_color='black',
                                                     cmap='Reds', include_cbar=True,
                                                     movement_dot_color='red', custom_vmin=None, custom_vmax=None,
                                                     cbaraxes=None, cbar_ylabel='Predicted firing rate (spike/s)',
                                                     fig=None, ax=None):
    """

    Parameters
    ----------
    movement_spike_matrix
    prediction_matrix
    rt_vec
    peri_event_time
    plot_prediction
    plot_movement_times
    scatter_dot_size
    spike_dot_color
    cmap
    include_cbar
    movement_dot_color
    custom_vmin
    custom_vmax
    cbaraxes : colorbar axes
    fig
    ax

    Returns
    -------

    """


    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()

    num_movement_trial = np.shape(prediction_matrix)[0]

    trial_offset = 0.5  # want the point to be in the middle (to be center of heatmap)

    if plot_prediction:
        im = ax.imshow(prediction_matrix, aspect='auto', extent=[
            peri_event_time[0], peri_event_time[-1], 0, num_movement_trial,
        ], cmap=cmap, vmin=custom_vmin, vmax=custom_vmax)

        if include_cbar:
            cbar = fig.colorbar(im, cax=cbaraxes)
            cbar.ax.set_ylabel(cbar_ylabel, size=10)



    for movement_trial in np.arange(num_movement_trial):
        spike_times = np.where(movement_spike_matrix[movement_trial, :] > 0)[0]
        spike_times_peri_event = peri_event_time[spike_times] + 15/1000
        spike_times_peri_event = spike_times_peri_event[spike_times_peri_event < 0.3]
        ax.scatter(spike_times_peri_event,
                   np.repeat(movement_trial + trial_offset, len(spike_times_peri_event)),
                   color=spike_dot_color,
                   s=scatter_dot_size, edgecolor='none')

        # plot movement times
        if plot_movement_times:
            ax.scatter(rt_vec[movement_trial],
                       movement_trial + trial_offset,
                       color=movement_dot_color,
                       s=scatter_dot_size, edgecolor='none')

    ax.axvline(0, linestyle='--', color='gray', lw=1)
    # ax.set_xlim([-0.1, 0.3])
    ax.set_ylabel('Trials', size=12)
    ax.set_xlabel('Peri-stimulus time (s)', size=12)

    return fig, ax



