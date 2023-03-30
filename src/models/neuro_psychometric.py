import pdb

import pandas as pd
import pickle as pkl
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import sciplotlib.style as splstyle
from tqdm import tqdm

import time

import src.data.analyse_spikes as anaspikes
import src.data.process_ephys_data as pephys


import src.visualization.vizmodel as vizmodel
import src.visualization.vizpikes as vizpikes
import src.visualization.vizstat as vizstat
import src.visualization.vizbehaviour as vizbehaviour

from collections import defaultdict

import xarray as xr

import scipy.stats as sstats
import src.data.stat as stat

import src.models.predict_model as pmodel
import sklearn.model_selection as skselect
import sklearn
import sklearn.linear_model as sklinear
import sklearn.pipeline as sklpipeline



def bin_neural_latent_and_p_right(neural_latent_var_per_trial, y, num_vals_per_bin=25):
    """
    Bin neural_latent_var_per_trial and for each bin calculate the observed p(right)
    Can also work with behaviour_latent_var_per_trial
    Parameters
    ----------
    neural_latent_var_per_trial : numpy ndarray
    y : numpy ndarray
    num_vals_per_bin : int
        number of latent variables in each bin
        ie. if this number is 25 then I take the p(right) of every 25 latent variable after sorting them
    Returns
    -------
    latent_variable_per_bin : numpy ndarray
        vector with shape num_bins x 1
        each entry is the mean of the latent variables (number specified by num_vals_per_bin)
    actual_p_right_per_bin : numpy ndarray
        vector with shape num_bins x 1
    """

    num_trial = len(neural_latent_var_per_trial)
    sorted_neural_latent_var_per_trial = np.sort(neural_latent_var_per_trial)
    num_bins = int(np.ceil(num_trial / num_vals_per_bin))

    latent_variable_per_bin = np.zeros(num_bins, ) + np.nan
    actual_p_right_per_bin = np.zeros(num_bins, ) + np.nan

    for bin_number in np.arange(num_bins):
        bin_start = int(bin_number * num_vals_per_bin)
        bin_end = int(bin_start + num_vals_per_bin)
        bin_end = np.min([bin_end, num_trial - 1])

        lower_bound = sorted_neural_latent_var_per_trial[bin_start]
        upper_bound = sorted_neural_latent_var_per_trial[bin_end]

        subset_trial_idx = np.where(
            (neural_latent_var_per_trial >= lower_bound) &
            (neural_latent_var_per_trial <= upper_bound)
        )[0]

        latent_variable_per_bin[bin_number] = np.mean(neural_latent_var_per_trial[subset_trial_idx])
        actual_p_right_per_bin[bin_number] = np.mean(y[subset_trial_idx])

    return latent_variable_per_bin, actual_p_right_per_bin


def plot_neurometric_model(neural_latent_var_per_trial, neural_prob_per_trial,
                           latent_variable_per_bin, actual_p_right_per_bin,
                           fig=None, axs=None):
    """
    Parameters
    ----------
    neural_latent_var_per_trial : numpy ndarray
    neural_prob_per_trial : numpy ndarray
    latent_variable_per_bin : numpy ndarray
    actual_p_right_per_bin : numpy ndarray

    Returns
    -------
    fig : matplotlib figure objects
    axs : numpy ndarray of matplotlib axis objects
    """

    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        fig.set_size_inches(8, 4)

    axs[0].scatter(neural_latent_var_per_trial, neural_prob_per_trial, color='black')
    axs[0].set_xlabel(r'$Xw$', size=11)
    axs[0].set_ylabel('model p(right)', size=11)

    axs[1].scatter(latent_variable_per_bin, actual_p_right_per_bin, color='black')
    axs[1].set_xlabel(r'$Xw$', size=11)
    axs[1].set_ylabel('observed p(right)', size=11)

    return fig, axs

def plot_multiple_neurometric_models(model_results_per_brain_region, target_brain_regions=None,
                                     behaviour_results=None, n_plus_b_results_per_brain_region=None,
                                     include_num_neurons=True, color_stim_scatter_with_trial_count=True,
                                     include_binned_behaviour_plot=False, bin_behaviour_only_model=False,
                                     num_vals_per_bin=25, include_logistic_curve=False, sharex_top_and_bottom=True,
                                     model_name=None, set_to_plot=None,
                                     fig=None, axs=None):
    """
    Plots the latent variable (x-axis) vs. observed p(right) (y-axis) for a model using only the stimulus,
    models using activity from brain regions, and optionally models using both.

    Parameters
    ----------
    model_results_per_brain_region : dict
        dictionary containing the neural fitting results for each brain region
    target_brain_regions : list
        list of brain regions (str) to plot
    n_plus_b_results_per_brain_region : dict
        if None, then does include model with both neural and behaviour predictors
    color_stim_scatter_with_trial_count : bool
        whether to indicate the trial count of each stimulus condition in the stimulus plot
        using a color gradient
    include_binned_behaviour_plot : bool
        whether to include behaviour plot binned by z value rather than by trial type
    include_num_neurons : bool
        whether to include the number of neurons used
    bin_behaviour_only_model : bool
        whether to bin the output values of the model for the behaviour-only model (as is what
        is done for the neural models)
    incldue_logistic_curve : bool
        whether to also include a logistic curve
    sharex_top_and_bottom : bool
        whether to share the same x axis range for the neural-only model and the neural-plus-behaviour model
    model_name : str or None
        name of the model
    set_to_plot : str, optional
        which set - train, test or validation set to plot
    fig : matplotlib figure object
    axs : numpy ndarray of matplotlib axes objects

    Returns
    -------
    fig : matplotlib figure objects
    axs : numpy ndarray of matplotlib axes objects
    """

    logistic_curve_lw = 0.8
    logistic_curve_alpha = 0.3
    logistic_curve_color = 'gray'

    if (fig is None) and (axs is None):
        if behaviour_results is not None:
            if target_brain_regions is not None:
                num_subplots = len(target_brain_regions) + 1
            else:
                num_subplots = 3
        else:
            num_subplots = len(target_brain_regions)

        if n_plus_b_results_per_brain_region is not None:
            if target_brain_regions is not None:
                fig, axs = plt.subplots(2, num_subplots, sharey=True)
                fig.set_size_inches(15, 6)
            else:
                fig, axs = plt.subplots(1, num_subplots, sharey=True, sharex=True)
                fig.set_size_inches(9, 3)
        else:
            fig, axs = plt.subplots(1, num_subplots, sharey=True)
            fig.set_size_inches(15, 3)

    if behaviour_results is not None:

        if model_name is None:
            model_latent_per_trial_type = behaviour_results['model_latent_per_trial_type']
            actual_p_right_per_trial_type = behaviour_results['actual_p_right_per_trial_type']
        else:
            model_idx = np.where(behaviour_results['model_names'] == model_name)[0][0]

            if set_to_plot == 'train':
                model_latent_per_trial_type = behaviour_results['model_latent_per_trial_type_train_set_per_model'][model_idx, :]
                actual_p_right_per_trial_type = behaviour_results['actual_p_right_per_trial_type_per_model'][model_idx, :]
            else:
                model_latent_per_trial_type = behaviour_results['model_latent_per_trial_type_per_model'][model_idx, :]
                actual_p_right_per_trial_type = behaviour_results['actual_p_right_per_trial_type_per_model'][model_idx, :]

        if bin_behaviour_only_model:
            color_stim_scatter_with_trial_count = False

        if n_plus_b_results_per_brain_region is None:
            behaviour_only_plot_loc = 0
        else:
            if target_brain_regions is not None:
                behaviour_only_plot_loc = (0, 0)
            else:
                behaviour_only_plot_loc = 0

        if color_stim_scatter_with_trial_count:
            max_trial_count = np.max(behaviour_results['trial_num_per_trial_type'])
            cmap = mpl.cm.binary
            norm = mpl.colors.Normalize(vmin=0, vmax=max_trial_count)
            dot_colors = cmap(norm(behaviour_results['trial_num_per_trial_type']))
            axs[behaviour_only_plot_loc].scatter(model_latent_per_trial_type, actual_p_right_per_trial_type,
                                                 color=dot_colors, lw=0)
            cbar_ax = fig.add_axes([0.14, 0.6, 0.01, 0.12])
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                         cax=cbar_ax, orientation='vertical')
            cbar_ax.tick_params(labelsize=9, length=1)
            cbar_ax.set_ylabel('Trial count', size=9)
        else:
            if bin_behaviour_only_model:
                if model_name is None:
                    if set_to_plot == 'train':
                        latent_var_per_trial = behaviour_results['latent_var_per_trial_train_set']
                    else:
                        latent_var_per_trial = behaviour_results['latent_var_per_trial']
                else:
                    model_idx = np.where(behaviour_results['model_names'] == model_name)[0][0]
                    if set_to_plot == 'train':
                        latent_var_per_trial = behaviour_results['latent_var_per_trial_train_set_per_model'][model_idx, :]
                    else:
                        latent_var_per_trial = behaviour_results['latent_var_per_trial_per_model'][model_idx, :]


                b_only_model_latent_per_trial, b_only_model_p_right_per_trial = \
                    bin_neural_latent_and_p_right(latent_var_per_trial,
                                                  behaviour_results['y'], num_vals_per_bin=num_vals_per_bin)
                axs[behaviour_only_plot_loc].scatter(b_only_model_latent_per_trial,
                                                     b_only_model_p_right_per_trial,
                                                     color='black', lw=0)
            else:
                axs[behaviour_only_plot_loc].scatter(model_latent_per_trial_type,
                                                     actual_p_right_per_trial_type,
                                                     color='black', lw=0)

        if include_logistic_curve:
            if set_to_plot == 'train':
                x_vals = behaviour_results['latent_var_per_trial_train_set']
            else:
                x_vals = behaviour_results['latent_var_per_trial']

            curve_x = np.linspace(np.min(x_vals), np.max(x_vals), 100)
            curve_y = 1 / (1 + np.exp(-curve_x))
            axs[behaviour_only_plot_loc].plot(curve_x, curve_y, color=logistic_curve_color,
                                              alpha=logistic_curve_alpha,
                                              lw=logistic_curve_lw)

        axs[behaviour_only_plot_loc].set_title('Stimulus', size=11)
        axs[behaviour_only_plot_loc].set_xlabel(r'$Xw$', size=11)
        brain_region_start_counter = 1
    else:
        brain_region_start_counter = 0

    if (n_plus_b_results_per_brain_region is not None) and include_binned_behaviour_plot:
        pdb.set_trace()

    if target_brain_regions is not None:

        for n_brain_region in np.arange(len(target_brain_regions)):
            brain_region = target_brain_regions[n_brain_region]

            if n_plus_b_results_per_brain_region is None:
                brain_region_plot_loc = brain_region_start_counter + n_brain_region
            else:
                brain_region_plot_loc = (0, brain_region_start_counter + n_brain_region)

            if include_num_neurons:
                if brain_region in model_results_per_brain_region.keys():
                    num_neurons = np.shape(model_results_per_brain_region[brain_region]['X_neural'])[1] - 1
                else:
                    num_neurons = 0
                title_txt = '%s (%.f neurons)' % (brain_region, num_neurons)
            else:
                title_txt = brain_region
            axs[brain_region_plot_loc].set_title(title_txt, size=11)

            if brain_region in model_results_per_brain_region.keys():

                if model_name is None:
                    neural_latent_var_per_trial = model_results_per_brain_region[brain_region]['latent_variable_per_bin']
                    actual_p_right_per_bin = model_results_per_brain_region[brain_region]['actual_p_right_per_bin']
                else:
                    model_idx = np.where(model_results_per_brain_region[brain_region]['model_names'] == model_name)[0][0]

                    if set_to_plot == 'train':
                        neural_latent_var_per_trial = model_results_per_brain_region[brain_region][
                                                          'latent_variable_per_bin_train_set_per_model'][model_idx, :]
                        actual_p_right_per_bin = model_results_per_brain_region[brain_region][
                                                     'actual_p_right_per_bin_train_set_per_model'][model_idx, :]
                    else:
                        neural_latent_var_per_trial = model_results_per_brain_region[brain_region]['latent_variable_per_bin_per_model'][model_idx, :]
                        actual_p_right_per_bin = model_results_per_brain_region[brain_region]['actual_p_right_per_bin_per_model'][model_idx, :]

                axs[brain_region_plot_loc].scatter(neural_latent_var_per_trial, actual_p_right_per_bin, color='black')

                if include_logistic_curve:
                    if model_name is None:
                        if set_to_plot == 'train':
                            x_vals = model_results_per_brain_region[brain_region]['latent_variable_per_bin_train_set']
                        else:
                            x_vals = model_results_per_brain_region[brain_region]['latent_variable_per_bin']

                    else:
                        if set_to_plot == 'train':
                            x_vals = model_results_per_brain_region[brain_region]['latent_variable_per_bin_train_set_per_model'][model_idx, :]
                        else:
                            x_vals = model_results_per_brain_region[brain_region]['latent_variable_per_bin_per_model'][model_idx, :]

                    curve_x = np.linspace(np.min(x_vals), np.max(x_vals), 100)
                    curve_y = 1 / (1 + np.exp(-curve_x))
                    axs[brain_region_plot_loc].plot(curve_x, curve_y, color=logistic_curve_color,
                                                      alpha=logistic_curve_alpha,
                                                      lw=logistic_curve_lw)

                if n_plus_b_results_per_brain_region is None:
                    axs[brain_region_plot_loc].set_xlabel(r'$Xw$', size=11)

                if n_plus_b_results_per_brain_region is not None:
                    if model_name is None:
                        n_plus_b_neural_latent_var_per_trial = n_plus_b_results_per_brain_region[brain_region]['latent_variable_per_bin']
                        n_plus_b_actual_p_right_per_bin = n_plus_b_results_per_brain_region[brain_region]['actual_p_right_per_bin']
                    else:
                        model_idx = np.where(n_plus_b_results_per_brain_region[brain_region]['model_names'] == model_name)[0][0]

                        if set_to_plot == 'train':
                            n_plus_b_neural_latent_var_per_trial = n_plus_b_results_per_brain_region[brain_region][
                                                                       'latent_variable_per_bin_train_set_per_model'][model_idx, :]
                            n_plus_b_actual_p_right_per_bin = n_plus_b_results_per_brain_region[brain_region][
                                                                  'actual_p_right_per_bin_train_set_per_model'][model_idx, :]
                        else:
                            n_plus_b_neural_latent_var_per_trial = n_plus_b_results_per_brain_region[brain_region][
                                'latent_variable_per_bin_per_model'][model_idx, :]
                            n_plus_b_actual_p_right_per_bin = n_plus_b_results_per_brain_region[brain_region][
                                'actual_p_right_per_bin_per_model'][model_idx, :]

                    axs[1, brain_region_start_counter + n_brain_region].scatter(n_plus_b_neural_latent_var_per_trial,
                                                       n_plus_b_actual_p_right_per_bin, color='black')
                    axs[1, brain_region_start_counter + n_brain_region].set_xlabel(r'$Xw$', size=11)
                    axs[1, brain_region_start_counter + n_brain_region].set_title('%s + stimulus' % brain_region, size=11)

                    if include_logistic_curve:
                        if model_name is None:
                            if set_to_plot == 'train':
                                x_vals = n_plus_b_results_per_brain_region[brain_region]['latent_variable_per_bin_train_set']
                            else:
                                x_vals = n_plus_b_results_per_brain_region[brain_region]['latent_variable_per_bin']
                        else:
                            if set_to_plot == 'train':
                                x_vals = n_plus_b_results_per_brain_region[brain_region]['latent_variable_per_bin_train_set_per_model'][
                                       model_idx, :]
                            else:
                                x_vals = n_plus_b_results_per_brain_region[brain_region]['latent_variable_per_bin_per_model'][
                                         model_idx, :]

                        curve_x = np.linspace(np.min(x_vals), np.max(x_vals), 100)
                        curve_y = 1 / (1 + np.exp(-curve_x))
                        axs[1, brain_region_start_counter + n_brain_region].plot(curve_x, curve_y, color=logistic_curve_color,
                                                          alpha=logistic_curve_alpha,
                                                          lw=logistic_curve_lw)
                    if sharex_top_and_bottom:
                        ax1 = axs[brain_region_plot_loc]
                        ax2 = axs[1, brain_region_start_counter + n_brain_region]
                        ax1.get_shared_x_axes().join(ax1, ax2)

            else:
                # remove brain region with no neurons
                axs[0, brain_region_start_counter + n_brain_region].remove()
                if n_plus_b_results_per_brain_region is not None:
                    axs[1, brain_region_start_counter + n_brain_region].remove()
    else:
        if model_name is None:
            neural_latent_var_per_trial = model_results_per_brain_region['latent_variable_per_bin']
            actual_p_right_per_bin = model_results_per_brain_region['actual_p_right_per_bin']
        else:
            model_idx = np.where(model_results_per_brain_region['model_names'] == model_name)[0][0]

            if set_to_plot == 'train':
                neural_latent_var_per_trial = model_results_per_brain_region[
                                                  'latent_variable_per_bin_train_set_per_model'][model_idx, :]
                actual_p_right_per_bin = model_results_per_brain_region[
                                             'actual_p_right_per_bin_train_set_per_model'][model_idx, :]
            else:
                neural_latent_var_per_trial = model_results_per_brain_region[
                                                  'latent_variable_per_bin_per_model'][model_idx, :]
                actual_p_right_per_bin = model_results_per_brain_region[
                                             'actual_p_right_per_bin_per_model'][model_idx, :]

        axs[1].scatter(neural_latent_var_per_trial, actual_p_right_per_bin, color='black')

        if include_logistic_curve:
            if model_name is None:
                if set_to_plot == 'train':
                    x_vals = model_results_per_brain_region['latent_variable_per_bin_train_set']
                else:
                    x_vals = model_results_per_brain_region['latent_variable_per_bin']

            else:
                if set_to_plot == 'train':
                    x_vals = model_results_per_brain_region[
                                 'latent_variable_per_bin_train_set_per_model'][model_idx, :]
                else:
                    x_vals = model_results_per_brain_region['latent_variable_per_bin_per_model'][
                             model_idx, :]

            curve_x = np.linspace(np.min(x_vals), np.max(x_vals), 100)
            curve_y = 1 / (1 + np.exp(-curve_x))
            axs[1].plot(curve_x, curve_y, color=logistic_curve_color,
                                            alpha=logistic_curve_alpha,
                                            lw=logistic_curve_lw)

        axs[1].set_xlabel(r'$Xw$', size=11)
        axs[1].set_title('Neural', size=11)

        if n_plus_b_results_per_brain_region is not None:
            if model_name is None:
                n_plus_b_neural_latent_var_per_trial = n_plus_b_results_per_brain_region[
                    'latent_variable_per_bin']
                n_plus_b_actual_p_right_per_bin = n_plus_b_results_per_brain_region[
                    'actual_p_right_per_bin']
            else:
                model_idx = np.where(n_plus_b_results_per_brain_region['model_names'] == model_name)[0][0]

                if set_to_plot == 'train':
                    n_plus_b_neural_latent_var_per_trial = n_plus_b_results_per_brain_region[
                                                               'latent_variable_per_bin_train_set_per_model'][model_idx,
                                                           :]
                    n_plus_b_actual_p_right_per_bin = n_plus_b_results_per_brain_region[
                                                          'actual_p_right_per_bin_train_set_per_model'][model_idx, :]
                else:
                    n_plus_b_neural_latent_var_per_trial = n_plus_b_results_per_brain_region[
                                                               'latent_variable_per_bin_per_model'][model_idx, :]
                    n_plus_b_actual_p_right_per_bin = n_plus_b_results_per_brain_region[
                                                          'actual_p_right_per_bin_per_model'][model_idx, :]

            axs[2].scatter(n_plus_b_neural_latent_var_per_trial, n_plus_b_actual_p_right_per_bin, color='black')
            axs[2].set_xlabel(r'$Xw$', size=11)
            axs[2].set_title('neural + stimulus', size=11)

            if include_logistic_curve:
                if model_name is None:
                    if set_to_plot == 'train':
                        x_vals = n_plus_b_results_per_brain_region['latent_variable_per_bin_train_set']
                    else:
                        x_vals = n_plus_b_results_per_brain_region['latent_variable_per_bin']
                else:
                    if set_to_plot == 'train':
                        x_vals = n_plus_b_results_per_brain_region[
                                     'latent_variable_per_bin_train_set_per_model'][
                                 model_idx, :]
                    else:
                        x_vals = n_plus_b_results_per_brain_region['latent_variable_per_bin_per_model'][
                                 model_idx, :]

                curve_x = np.linspace(np.min(x_vals), np.max(x_vals), 100)
                curve_y = 1 / (1 + np.exp(-curve_x))
                axs[2].plot(curve_x, curve_y, color=logistic_curve_color, alpha=logistic_curve_alpha,
                            lw=logistic_curve_lw)

    axs[behaviour_only_plot_loc].set_ylim([0, 1.05])
    axs[behaviour_only_plot_loc].set_yticks([0, 0.25, 0.5, 0.75, 1])
    axs[behaviour_only_plot_loc].set_ylabel('observed p(right)', size=11)

    # remove the bottom left corner plot since there is no corresponding thing to plot
    if n_plus_b_results_per_brain_region is not None:
        if target_brain_regions is not None:
            axs[1, 0].remove()

    fig.tight_layout()

    return fig, axs


def plot_all_exp_models_summary(stim_model_lines_params, neural_only_model_line_params,
                                n_plus_b_model_line_params, target_brain_regions,
                                zmin=-1, zmax=1, fig=None, axs=None):
    """

    """

    num_subplots = len(target_brain_regions) + 1
    fig, axs = plt.subplots(2, num_subplots, sharey=True, sharex=True)
    fig.set_size_inches(15, 6)

    # Plot behaviour only model

    for b_model_params in stim_model_lines_params:
        x_vals = np.linspace(zmin, zmax, 100)
        y_vals = x_vals * b_model_params[0] + b_model_params[1]
        axs[0, 0].plot(
            x_vals, y_vals, color='gray'
        )

    for n_brain_region in np.arange(len(target_brain_regions)):
        brain_region = target_brain_regions[n_brain_region]

        neural_only_line_param_per_exp = neural_only_model_line_params[brain_region]

        for n_model_params in neural_only_line_param_per_exp:
            x_vals = np.linspace(zmin, zmax, 100)
            y_vals = x_vals * n_model_params[0] + n_model_params[1]
            axs[0, n_brain_region+1].plot(
                x_vals, y_vals, color='gray'
            )

        axs[0, n_brain_region+1].set_title('%s' % brain_region, size=11)

        n_plus_b_model_line_params_per_exp = n_plus_b_model_line_params[brain_region]
        for n_plus_b_model_params in n_plus_b_model_line_params_per_exp:
            x_vals = np.linspace(zmin, zmax, 100)
            y_vals = x_vals * n_plus_b_model_params[0] + n_plus_b_model_params[1]
            axs[1, n_brain_region+1].plot(
                x_vals, y_vals, color='gray'
            )

        axs[1, n_brain_region+1].set_title('%s + stimulus' % brain_region, size=11)

    axs[1, 0].remove()

    fig.text(0.5, 0, '$Xw$ scaled (per experiment)', size=11, ha='center')
    # fig.text(0, 0.5, r'$\log_{10}(\frac{p(r)}{1 - p(r)})$', size=11, va='center', rotation=90)
    axs[0, 0].set_ylabel(r'$\log_{10}\left(\frac{p(r) + \epsilon}{1 - p(r) + \epsilon}\right)$', size=11)

    fig.tight_layout()

    return fig, axs


def plot_behaviour_model(latent_var_per_trial, prob_per_trial,
                         model_latent_per_trial_type, actual_p_right_per_trial_type,
                         fig=None, axs=None):
    """
    Plot model output fitted only to stimulus parameters (without neural data)

    """
    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(1, 2, sharey=True)
        fig.set_size_inches(8, 4)

    axs[0].scatter(latent_var_per_trial, prob_per_trial, color='black')
    axs[0].set_xlabel(r'$Xw$', size=11)
    axs[0].set_ylabel('model p(right)', size=11)

    axs[1].scatter(model_latent_per_trial_type, actual_p_right_per_trial_type, color='black')
    axs[1].set_ylabel('observed p(right)', size=11)
    axs[1].set_xlabel(r'$Xw$', size=11)

    return fig, axs


def get_behaviour_X_and_y(alignment_ds, choice_var_name='choiceThreshDir', aud_diff_name='audDiff',
                          vis_diff_name='visDiff'):
    """
    Extracts the features X (stimulus conditions) and response variable y (left or right choice) from mouse
    behaviour dataset

    Parameters
    ----------
    alignment_ds : xarray dataset

    Returns
    -------
    X : numpy ndarray
    y : numpy ndarray
    """

    y = alignment_ds.isel(Cell=0)[choice_var_name].values - 1
    num_trial = len(y)

    if 'Cell' in alignment_ds[aud_diff_name].dims:
        aud_diff = alignment_ds[aud_diff_name].isel(Cell=0).values
    else:
        aud_diff = alignment_ds[aud_diff_name].values

    aud_on = np.isfinite(aud_diff).astype(int)
    aud_diff[~np.isfinite(aud_diff)] = 0

    if 'Cell' in alignment_ds[vis_diff_name].dims:
        vis_diff = alignment_ds[vis_diff_name].isel(Cell=0).values
    else:
        vis_diff = alignment_ds[vis_diff_name].values

    vis_on = (vis_diff != 0).astype(int)
    intercept_term = np.repeat(1, num_trial)
    X = np.stack([intercept_term, aud_on, aud_diff, vis_on, vis_diff]).T

    return X, y

def get_behaviour_X_and_y_from_df(mouse_df, aud_diff_name='audDiff', vis_diff_name='visDiff',
                                  choice_var_name='choiceThreshDir'):
    """

    """

    num_trial = len(mouse_df)
    aud_diff = mouse_df[aud_diff_name].values
    aud_on = np.isfinite(aud_diff).astype(int)
    aud_diff[~np.isfinite(aud_diff)] = 0
    vis_diff = mouse_df[vis_diff_name].values

    vis_on = (vis_diff != 0).astype(int)
    intercept_term = np.repeat(1, num_trial)

    X = np.stack([intercept_term, aud_on, aud_diff, vis_on, vis_diff]).T
    # y = mouse_df[choice_var_name].values
    y = mouse_df['goRight'].values.astype(float)

    return X, y

def get_X_neural(alignment_ds, mean_fr_window=[-0.15, 0]):
    """
    Parameters
    ----------
    alignment_ds : xarray dataset
    mean_fr_window : list
    Returns
    -------
    X_neural : numpy ndarray
    """

    neural_mean_fr = alignment_ds.where(
        (alignment_ds.PeriEventTime >= mean_fr_window[0]) &
        (alignment_ds.PeriEventTime <= mean_fr_window[1]), drop=True
    ).mean('Time')['firing_rate'].values

    num_trial = len(alignment_ds.Trial)

    intercept_term = np.repeat(1, num_trial).reshape(-1, 1)
    X_neural = np.hstack([intercept_term.reshape(-1, 1), neural_mean_fr.T])

    return X_neural


def fit_behaviour_only_model(X, y, cv_type=None, clf=None, num_folds=2, cv_random_seed=None,
                             max_iter=500):
    """
    Parameters
    ----------
    X : numpy ndarray
    y : numpy ndarray
    cv_type : str
        if None, then fits the entire dataset
    num_folds : int
        number of cross validation folds to do
    cv_random_seed: int, optional
        random seed for cross validation splits
    Returns
    -------
    latent_var_per_trial : numpy ndarray
    prob_per_trial : numpy ndarray

    """

    if clf is None:
        clf = sklinear.LogisticRegression(fit_intercept=False, max_iter=max_iter)
    elif clf == 'LR':
        clf = sklinear.LogisticRegression(fit_intercept=False, penalty='none', max_iter=max_iter)
    elif clf == 'LR-l1':
        clf = sklinear.LogisticRegression(fit_intercept=False, penalty='l1', max_iter=max_iter,
                                                 solver='liblinear')
    elif clf == 'LR-l2':
        clf = sklinear.LogisticRegression(fit_intercept=False, penalty='l2', max_iter=max_iter)
    elif clf == 'LR-l1-hyperparam-tuned':
        estimator = sklinear.LogisticRegression(fit_intercept=False, penalty='l1', max_iter=max_iter,
                                                 solver='liblinear')
        param_grid = {'C': np.logspace(-5, 5, 10, base=10)}
        clf = skselect.GridSearchCV(estimator, param_grid, cv=2)
    elif clf == 'LR-l2-hyperparam-tuned':
        estimator = sklinear.LogisticRegression(fit_intercept=False, penalty='l2', max_iter=max_iter)
        param_grid = {'C': np.logspace(-5, 5, 10, base=10)}
        clf = skselect.GridSearchCV(estimator, param_grid, cv=2)


    if cv_type is None:
        clf.fit(X, y)
        prob_per_trial = clf.predict_proba(X)[:, 1]
        model_weights = clf.coef_
        latent_var_per_trial = np.matmul(X, model_weights.T)
    else:
        num_trial = np.shape(X)[0]
        prob_per_trial = np.zeros((num_trial, )) + np.nan
        latent_var_per_trial = np.zeros((num_trial, )) + np.nan
        latent_var_per_trial_train_set = np.zeros((num_trial, )) + np.nan
        prob_per_trial_train_set = np.zeros((num_trial, )) + np.nan
        cv_splitter = skselect.KFold(n_splits=num_folds, random_state=cv_random_seed, shuffle=True)

        for train_idx, test_idx in cv_splitter.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            clf.fit(X_train, y_train)
            prob_per_trial[test_idx] = clf.predict_proba(X_test)[:, 1]

            if 'coef_' in dir(clf):
                model_weights = clf.coef_
            else:
                model_weights = clf.best_estimator_.coef_

            latent_var_per_trial[test_idx] = np.matmul(X_test, model_weights.T).flatten()

            prob_per_trial_train_set[train_idx] = clf.predict_proba(X_train)[:, 1]
            latent_var_per_trial_train_set[train_idx] =  np.matmul(X_train, model_weights.T).flatten()

    unique_rows = np.unique(X, axis=0)
    n_unique_trial_types = np.shape(unique_rows)[0]
    trial_type = np.zeros(len(y), ) + np.nan
    for unique_n in np.arange(n_unique_trial_types):
        row_to_match = tuple(unique_rows[unique_n, :])
        trial_type_loc = np.where((X == row_to_match).all(axis=1))[0]
        trial_type[trial_type_loc] = unique_n

    actual_p_right_per_trial_type = np.zeros(n_unique_trial_types, ) + np.nan
    model_latent_per_trial_type = np.zeros(n_unique_trial_types, ) + np.nan
    trial_num_per_trial_type = np.zeros(n_unique_trial_types, ) + np.nan

    model_latent_per_trial_type_train_set = np.zeros(n_unique_trial_types, ) + np.nan

    for n_id, trial_type_id in enumerate(np.sort(np.unique(trial_type))):
        subset_trial = np.where(trial_type == trial_type_id)[0]
        actual_p_right_per_trial_type[n_id] = np.mean(y[subset_trial])
        model_latent_per_trial_type[n_id] = np.mean(latent_var_per_trial[subset_trial])
        trial_num_per_trial_type[n_id] = len(subset_trial)
        model_latent_per_trial_type_train_set[n_id] = np.mean(latent_var_per_trial_train_set[subset_trial])

    return latent_var_per_trial, prob_per_trial, model_latent_per_trial_type, \
           actual_p_right_per_trial_type, trial_num_per_trial_type, \
           prob_per_trial_train_set, latent_var_per_trial_train_set, model_latent_per_trial_type_train_set


def fit_neural_only_model(X_neural, y, clf=None, cv_type=None, num_folds=2,
                          include_stimulus_predictors=False, X_stim=None, max_iter=500,
                          cv_random_seed=None):
    """
    Fit a model to predict choices (y) from neural data (X_neural)

    Parameters
    ----------
    X_neural : numpy ndarray
    y : numpy ndarray
    clf : str or sklearn classifier object
        name of the classifier to use, can also be an sklearn classifier object
    cv_type : str
        if None, then fit the entire set
        if 'cross-validate', then do n-fold cross validation
    num_folds : int
        how many cross validation folds to do
    Returns
    -------
    neural_latent_var_per_trial : numpy ndarray
    neural_prob_per_trial : numpy ndarray
    """

    if clf is None:
        neural_clf = sklinear.LogisticRegression(fit_intercept=False, max_iter=max_iter)
    elif clf == 'LR':
        neural_clf = sklinear.LogisticRegression(fit_intercept=False, penalty='none', max_iter=max_iter)
    elif clf == 'LR-l1':
        neural_clf = sklinear.LogisticRegression(fit_intercept=False, penalty='l1', max_iter=max_iter,
                                                 solver='liblinear')
    elif clf == 'LR-l2':
        neural_clf = sklinear.LogisticRegression(fit_intercept=False, penalty='l2', max_iter=max_iter)
    elif clf == 'LR-l1-hyperparam-tuned':
        estimator = sklinear.LogisticRegression(fit_intercept=False, penalty='l1', max_iter=max_iter,
                                                 solver='liblinear')
        param_grid = {'C': np.logspace(-5, 5, 10, base=10)}
        neural_clf = skselect.GridSearchCV(estimator, param_grid, cv=2)
    elif clf == 'LR-l2-hyperparam-tuned':
        estimator = sklinear.LogisticRegression(fit_intercept=False, penalty='l2', max_iter=max_iter)
        param_grid = {'C': np.logspace(-5, 5, 10, base=10)}
        neural_clf = skselect.GridSearchCV(estimator, param_grid, cv=2)
    else:
        neural_clf = clf

    if include_stimulus_predictors:
        X_all = np.concatenate([X_neural, X_stim[:, 1:]], axis=1)  # exclude intercept from X_stim, since
        # there is already an intercept term in X_neural
    else:
        X_all = X_neural

    if cv_type is None:

        neural_clf.fit(X_all, y)
        neural_prob_per_trial = neural_clf.predict_proba(X_all)[:, 1].flatten()
        neural_model_weights = neural_clf.coef_
        neural_latent_var_per_trial = np.matmul(X_all, neural_model_weights.T).flatten()

    elif cv_type == 'cross-validate':

        num_trial = np.shape(X_all)[0]
        # Output in the test set
        neural_prob_per_trial = np.zeros((num_trial, )) + np.nan
        neural_latent_var_per_trial = np.zeros((num_trial, )) + np.nan
        # Output in the train test
        model_prob_per_trial_train = np.zeros((num_trial, )) + np.nan
        latent_var_per_trial_train = np.zeros((num_trial, )) + np.nan

        # Cross validation
        cv_splitter = skselect.KFold(n_splits=num_folds, random_state=cv_random_seed, shuffle=True)

        for train_idx, test_idx in cv_splitter.split(X_all):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            neural_clf.fit(X_train, y_train)
            neural_prob_per_trial[test_idx] = neural_clf.predict_proba(X_test)[:, 1].flatten()

            if 'coef_' in dir(neural_clf):
                neural_model_weights = neural_clf.coef_
            else:
                neural_model_weights = neural_clf.best_estimator_.coef_

            neural_latent_var_per_trial[test_idx] = np.matmul(X_test, neural_model_weights.T).flatten()

            model_prob_per_trial_train[train_idx] = neural_clf.predict_proba(X_train)[:, 1].flatten()
            latent_var_per_trial_train[train_idx] = np.matmul(X_train, neural_model_weights.T).flatten()


    return neural_latent_var_per_trial, neural_prob_per_trial, model_prob_per_trial_train, latent_var_per_trial_train


def do_forward_feature_selection(X, y, clf, num_features_to_select=10,
                                 cv_type=None, num_folds=2, max_iter=500,
                                 cv_random_seed=None, selection_criteria='accuracy'):
    """
    Does forward feature selection
    """

    num_features = np.shape(X)[1]

    assert num_features >= num_features_to_select

    available_features = np.arange(num_features)
    num_trials = np.shape(X)[0]
    accuracy_per_best_n_feature = []

    best_features_selected = []
    X_current = np.repeat(1, num_trials).reshape(-1, 1)

    # Cross validation
    cv_splitter = skselect.KFold(n_splits=num_folds, random_state=cv_random_seed, shuffle=True)

    for n_feature in tqdm(np.arange(num_features_to_select)):

        if n_feature == 0:

            accuracy_per_cv = []
            for train_idx, test_idx in cv_splitter.split(X_current):
                X_train, X_test = X_current[train_idx], X_current[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                clf.fit(X_train, y_train)
                y_test_hat = clf.predict(X_test)
                # print(y_test_hat)
                # print(np.sum(y_test == 0) / len(y_test))
                accuracy = np.mean(y_test == y_test_hat)
                accuracy_per_cv.append(accuracy)

            intercept_only_accuracy = np.mean(accuracy_per_cv)
            accuracy_per_best_n_feature.append(intercept_only_accuracy)

        else:

            accuracy_per_feature_to_add = []

            if selection_criteria == 'accuracy':
                for feature_to_add in available_features:

                    X_to_try = np.concatenate([X_current, X[:, feature_to_add].reshape(-1, 1)], axis=1)

                    accuracy_per_cv = []
                    for train_idx, test_idx in cv_splitter.split(X_to_try):
                        X_train, X_test = X_to_try[train_idx], X_to_try[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        clf.fit(X_train, y_train)
                        y_test_hat = clf.predict(X_test)
                        accuracy = np.mean(y_test == y_test_hat)
                        accuracy_per_cv.append(accuracy)

                    accuracy_per_feature_to_add.append(np.mean(accuracy_per_cv))

                # Best accuracy
                best_current_accuracy = np.max(accuracy_per_feature_to_add)
                best_feature_idx = np.argmax(accuracy_per_feature_to_add)
                best_feature = available_features[best_feature_idx]
                accuracy_per_best_n_feature.append(best_current_accuracy)

            elif selection_criteria == 'random':

                feature_to_add = np.random.choice(available_features)

                X_to_try = np.concatenate([X_current, X[:, feature_to_add].reshape(-1, 1)], axis=1)

                accuracy_per_cv = []
                for train_idx, test_idx in cv_splitter.split(X_to_try):
                    X_train, X_test = X_to_try[train_idx], X_to_try[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    clf.fit(X_train, y_train)
                    y_test_hat = clf.predict(X_test)
                    accuracy = np.mean(y_test == y_test_hat)
                    accuracy_per_cv.append(accuracy)

                best_feature = feature_to_add
                best_current_accuracy = np.mean(accuracy_per_cv)
                accuracy_per_best_n_feature.append(best_current_accuracy)

            # Remove the chosen best feature from the available features
            available_features = available_features[available_features != best_feature]
            X_current = np.concatenate([X_current, X[:, best_feature].reshape(-1, 1)], axis=1)
            best_features_selected.append(best_feature)

    return best_features_selected, accuracy_per_best_n_feature


def plot_model_performance(model_performance_df, brain_regions, model_metric='log_loss',
                           single_exp_alpha=0.3, plot_brain_region_mean=False, fig=None, axs=None):

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.set_size_inches(8, 4)

    brain_region_colors = [mpl.cm.tab10(x) for x in np.arange(len(brain_regions))]

    global_min = np.min(model_performance_df[model_metric])
    global_max = np.max(model_performance_df[model_metric])
    unity_vals = np.linspace(global_min, global_max, 100)

    for exp in np.unique(model_performance_df['exp']):

        exp_df = model_performance_df.loc[
            model_performance_df['exp'] == exp
        ]

        behaviour_model_metric = exp_df.loc[
            exp_df['model'] == 'behaviour'
        ][model_metric].values

        if behaviour_model_metric.size == 0:
            print('Something is wrong with exp %.f, skipping... but please double check what is happening there' % exp)
            continue
        if behaviour_model_metric.size > 1:
            print('Seems to have repeating data points here, using temp hack to take the first one for now')
            behaviour_model_metric = behaviour_model_metric[0]

        for n_brain_region, brain_region in enumerate(brain_regions):

            if brain_region in exp_df['model'].values:

                neural_only_model_metric = exp_df.loc[
                    exp_df['model'] == '%s' % brain_region
                ][model_metric].values

                neural_plus_stim_model_metric = exp_df.loc[
                    exp_df['model'] == '%s_plus_stim' % brain_region
                ][model_metric].values

                axs[0].scatter(
                    behaviour_model_metric,
                    neural_plus_stim_model_metric,
                    color=brain_region_colors[n_brain_region],
                    alpha=single_exp_alpha, lw=0
                )

                axs[1].scatter(
                    neural_only_model_metric,
                    neural_plus_stim_model_metric,
                    color=brain_region_colors[n_brain_region],
                    alpha=single_exp_alpha, lw=0
                )

    if plot_brain_region_mean:
        for n_brain_region, brain_region in enumerate(brain_regions):
            subset_exp = model_performance_df.loc[
                model_performance_df['model'] == brain_region
            ]['exp'].values
            subset_exp_df = model_performance_df.loc[
                model_performance_df['exp'].isin(subset_exp)
            ]

            behaviour_model_metric_mean = subset_exp_df.loc[
                subset_exp_df['model'] == 'behaviour'
            ][model_metric].mean()

            neural_model_metric_mean = subset_exp_df.loc[
                subset_exp_df['model'] == brain_region
            ][model_metric].mean()

            neural_plus_stim_model_metric_mean = subset_exp_df.loc[
                subset_exp_df['model'] == '%s_plus_stim' % brain_region
                ][model_metric].mean()

            axs[0].scatter(
                behaviour_model_metric_mean,
                neural_plus_stim_model_metric_mean,
                color=brain_region_colors[n_brain_region],
                alpha=1, lw=0
            )

            axs[1].scatter(
                neural_model_metric_mean,
                neural_plus_stim_model_metric_mean,
                color=brain_region_colors[n_brain_region],
                alpha=1, lw=0
            )

    label_size = 11
    axs[0].set_xlabel('Stim', size=label_size)
    axs[0].set_ylabel('Neural + Stim', size=label_size)
    axs[1].set_xlabel('Neural', size=label_size)
    axs[1].set_ylabel('Neural + Stim', size=label_size)

    axs[0].set_xlim([global_min, global_max + 0.1])
    axs[0].set_ylim([global_min, global_max + 0.1])
    axs[1].set_xlim([global_min, global_max + 0.1])
    axs[1].set_ylim([global_min, global_max + 0.1])

    axs[0].plot(unity_vals, unity_vals, color='gray', alpha=0.3, linestyle='--')
    axs[1].plot(unity_vals, unity_vals, color='gray', alpha=0.3, linestyle='--')

    # Make legend
    legend_elements = [mpl.lines.Line2D([0], [0], marker='o', markeredgecolor=brain_region_colors[x], lw=0, label='%s' % brain_regions[x],
                              markerfacecolor=brain_region_colors[x], markersize=10) for x in
                     np.arange(len(brain_regions))
                      ]
    axs[1].legend(handles=legend_elements, bbox_to_anchor=(1.04, 0.5))

    fig.suptitle('%s' % model_metric, size=11)
    fig.tight_layout()

    return fig, axs


def add_to_model_performance_dict(model_performance_dict, model_results_per_brain_region, n_plus_b_results_per_brain_region,
                                  interim_data_folder, subject, exp, target_brain_region,
                                  include_stimulus_predictors=True, include_behaviour_only_model=True,
                                  num_vals_per_bin=25, eps=10**(-3), clf_name=None):
    """

    """


    # Neural only model
    neural_results_fpath_search_result = glob.glob(os.path.join(interim_data_folder,
                                                                's%.fe%.f_%s_neural_model_results.npz' %
                                                                (
                                                                    subject, exp,
                                                                    target_brain_region)))

    if len(neural_results_fpath_search_result) != 1:
        return model_performance_dict, model_results_per_brain_region, n_plus_b_results_per_brain_region

    neural_results = np.load(neural_results_fpath_search_result[0])

    if clf_name is None:
        neural_latent_var_per_trial = neural_results['neural_latent_var_per_trial']
    else:
        model_idx = np.where(neural_results['model_names'] == clf_name)[0][0]
        neural_latent_var_per_trial = neural_results['neural_latent_var_per_trial_per_model'][model_idx, :]

    # neural_prob_per_trial = neural_results['neural_prob_per_trial']
    y = neural_results['y']
    latent_variable_per_bin, actual_p_right_per_bin = \
        bin_neural_latent_and_p_right(neural_latent_var_per_trial, y,
                                      num_vals_per_bin=num_vals_per_bin)

    neural_results_keys = list(neural_results.keys())
    neural_results_dict = dict()
    for key in neural_results_keys:
        neural_results_dict[key] = neural_results[key]

    neural_results_dict['latent_variable_per_bin'] = latent_variable_per_bin
    neural_results_dict['actual_p_right_per_bin'] = actual_p_right_per_bin

    model_results_per_brain_region[target_brain_region] = neural_results_dict

    if clf_name is None:
        y_pred_prob = neural_results_dict['neural_prob_per_trial']
    else:
        model_idx = np.where(neural_results['model_names'] == clf_name)[0][0]
        y_pred_prob = neural_results_dict['neural_prob_per_trial_per_model'][model_idx, :]

    y_actual = neural_results_dict['y']
    y_pred_prob = [max(eps, min(1 - eps, x)) for x in y_pred_prob]
    y_pred_prob = np.array(y_pred_prob)
    neural_model_log_loss = \
        -np.mean(y_actual * np.log2(y_pred_prob) + (1 - y_actual) * np.log2(1 - y_pred_prob))

    y_pred_binary = (y_pred_prob > 0.5).astype(float)
    neural_model_accuracy = np.mean(y_pred_binary == y_actual)

    model_performance_dict['subject'].append(subject)
    model_performance_dict['exp'].append(exp)
    model_performance_dict['log_loss'].append(neural_model_log_loss)
    model_performance_dict['accuracy'].append(neural_model_accuracy)
    model_performance_dict['model'].append(target_brain_region)
    if clf_name is not None:
        model_performance_dict['clf'].append(clf_name)

    if include_stimulus_predictors:
        neural_plus_behaviour_results_fpath = os.path.join(interim_data_folder,
                                                           's%.fe%.f_%s_neural_plus_behaviour_model_results.npz' %
                                                           (subject, exp,
                                                            target_brain_region))
        neural_plus_behaviour_results = np.load(neural_plus_behaviour_results_fpath)
        neural_plus_behaviour_latent_var_per_trial = neural_plus_behaviour_results[
            'neural_latent_var_per_trial']
        y = neural_plus_behaviour_results['y']
        n_plus_b_latent_variable_per_bin, n_plus_b_actual_p_right_per_bin = \
            bin_neural_latent_and_p_right(neural_plus_behaviour_latent_var_per_trial, y,
                                          num_vals_per_bin=num_vals_per_bin)

        neural_plus_behaviour_results_keys = list(neural_plus_behaviour_results.keys())
        n_plus_b_results_dict = dict()
        for key in neural_plus_behaviour_results_keys:
            n_plus_b_results_dict[key] = neural_plus_behaviour_results[key]

        n_plus_b_results_dict['latent_variable_per_bin'] = n_plus_b_latent_variable_per_bin
        n_plus_b_results_dict['actual_p_right_per_bin'] = n_plus_b_actual_p_right_per_bin

        n_plus_b_results_per_brain_region[target_brain_region] = n_plus_b_results_dict

        if clf_name is None:
            y_pred_prob = n_plus_b_results_dict['neural_prob_per_trial']
        else:
            model_idx = np.where(n_plus_b_results_dict['model_names'] == clf_name)[0][0]
            y_pred_prob = n_plus_b_results_dict['neural_prob_per_trial_per_model'][model_idx, :]

        y_actual = n_plus_b_results_dict['y']

        y_pred_prob = [max(eps, min(1 - eps, x)) for x in y_pred_prob]
        y_pred_prob = np.array(y_pred_prob)

        n_plus_b_model_log_loss = \
            -np.mean(y_actual * np.log2(y_pred_prob) + (1 - y_actual) * np.log2(1 - y_pred_prob))

        y_pred_binary = (y_pred_prob > 0.5).astype(float)
        n_plus_b_model_accuracy = np.mean(y_pred_binary == y_actual)

        model_performance_dict['subject'].append(subject)
        model_performance_dict['exp'].append(exp)
        model_performance_dict['log_loss'].append(n_plus_b_model_log_loss)
        model_performance_dict['accuracy'].append(n_plus_b_model_accuracy)
        model_performance_dict['model'].append('%s_plus_stim' % target_brain_region)
        if clf_name is not None:
            model_performance_dict['clf'].append(clf_name)

    if include_behaviour_only_model:
        behaviour_results_fpath = os.path.join(interim_data_folder,
                                               's%.fe%.f_behaviour_model_results.npz' % (
                                                   subject, exp))

        if not os.path.exists(behaviour_results_fpath):
            print('%s does not exist, skipping' % behaviour_results_fpath)
            return model_performance_dict, model_results_per_brain_region, n_plus_b_results_per_brain_region

        behaviour_results = np.load(behaviour_results_fpath)
        model_latent_per_trial_type = behaviour_results['model_latent_per_trial_type'],
        actual_p_right_per_trial_type = behaviour_results['actual_p_right_per_trial_type']

        if clf_name is None:
            y_pred_prob = behaviour_results['prob_per_trial']
        else:
            model_idx = np.where(behaviour_results['model_names'] == clf_name)[0][0]
            y_pred_prob = behaviour_results['prob_per_trial_per_model'][model_idx, :]

        y_actual = behaviour_results['y']
        y_pred_prob = [max(eps, min(1 - eps, x)) for x in y_pred_prob]
        y_pred_prob = np.array(y_pred_prob)

        behaviour_model_log_loss = \
            -np.mean(y_actual * np.log2(y_pred_prob) + (1 - y_actual) * np.log2(1 - y_pred_prob))

        y_pred_binary = (y_pred_prob > 0.5).astype(float)
        behaviour_model_accuracy = np.mean(y_pred_binary == y_actual)

        model_performance_dict['subject'].append(subject)
        model_performance_dict['exp'].append(exp)
        model_performance_dict['log_loss'].append(behaviour_model_log_loss)
        model_performance_dict['accuracy'].append(behaviour_model_accuracy)
        model_performance_dict['model'].append('behaviour')
        if clf_name is not None:
            model_performance_dict['clf'].append(clf_name)

    return model_performance_dict, model_results_per_brain_region, n_plus_b_results_per_brain_region


def get_subset_behaviour_df(mouse_df, s_condition='all', exclude_no_go=False, exclude_invalid_trials=False):
    """
    Subsets behaviour data based on previous trial condition
    Parameters
    ----------
    mouse_df : pandas dataframe
    Returns
    -------
    subset_df : pandas dataframe
    """

    if s_condition == 'all':

        subset_df = mouse_df

    elif s_condition == 'after_left_choice':

        subset_index = mouse_df.loc[
            mouse_df['goLeft'] == True
            ].index.values

    elif s_condition == 'after_right_choice':

        subset_index = mouse_df.loc[
            mouse_df['goRight'] == True
            ].index.values

    elif s_condition == 'after_rewarded_choice':

        mouse_rewarded = mouse_df['feedbackGiven'].values == 1
        subset_index = mouse_df.iloc[mouse_rewarded].index.values
        # subset_index = mouse_df.loc[
        #    ~mouse_df['rewardTimes'].isna()
        # ].index.values

    elif s_condition == 'after_rewarded_left_choice':

        mouse_rewarded = mouse_df['feedbackGiven'].values == 1
        mouse_go_left = (mouse_df['goLeft'] == True).values
        subset_index = mouse_df.iloc[
            mouse_rewarded & mouse_go_left
        ].index.values

    elif s_condition == 'after_rewarded_right_choice':

        mouse_rewarded = mouse_df['feedbackGiven'].values == 1
        mouse_go_right = (mouse_df['goRight'] == True).values
        subset_index = mouse_df.iloc[
            mouse_rewarded & mouse_go_right
            ].index.values

    elif s_condition == 'after_rewarded_low_vis':

        mouse_rewarded = mouse_df['feedbackGiven'].values == 1

        subset_index = mouse_df.loc[
            mouse_rewarded &
            (np.abs(mouse_df['visDiff']) > 0) &
            (np.abs(mouse_df['visDiff']) < 0.2)
            ].index.values


    elif s_condition == 'after_rewarded_low_vis_left_choice':

        mouse_rewarded = mouse_df['feedbackGiven'].values == 1
        mouse_go_left = (mouse_df['goLeft'] == True).values
        subset_index = mouse_df.loc[
            mouse_rewarded &
            mouse_go_left &
            (np.abs(mouse_df['visDiff']) > 0) &
            (np.abs(mouse_df['visDiff']) < 0.2)
            ].index.values

    elif s_condition == 'after_rewarded_low_vis_right_choice':

        mouse_rewarded = mouse_df['feedbackGiven'].values == 1
        mouse_go_right = (mouse_df['goRight'] == True).values
        subset_index = mouse_df.loc[
            mouse_rewarded &
            mouse_go_right &
            (np.abs(mouse_df['visDiff']) > 0) &
            (np.abs(mouse_df['visDiff']) < 0.2)
            ].index.values

    elif s_condition == 'after_rewarded_conflict':

        mouse_rewarded = mouse_df['feedbackGiven'].values == 1
        subset_index = mouse_df.loc[
            # (~mouse_df['rewardTimes'].isna()) &
            mouse_rewarded &
            (mouse_df['bimodal'] == True) &
            (mouse_df['coherent'] == False)
            ].index.values

    elif s_condition == 'after_rewarded_conflict_left_choice':

        mouse_rewarded = mouse_df['feedbackGiven'].values == 1
        mouse_go_left = (mouse_df['goLeft'] == True).values
        # mouse_rewarded = ~mouse_df['rewardTimes'].isna()
        subset_index = mouse_df.loc[
            mouse_go_left &
            mouse_rewarded &
            (mouse_df['bimodal'] == True) &
            (mouse_df['coherent'] == False)
            ].index.values

    elif s_condition == 'after_rewarded_conflict_right_choice':

        mouse_rewarded = mouse_df['feedbackGiven'].values == 1
        mouse_go_right = (mouse_df['goRight'] == True).values
        # mouse_rewarded = ~mouse_df['rewardTimes'].isna()
        subset_index = mouse_df.loc[
            mouse_go_right &
            mouse_rewarded &
            (mouse_df['bimodal'] == True) &
            (mouse_df['coherent'] == False)
            ].index.values

    elif s_condition == 'after_rewarded_conflict_aud_choice':

        mouse_rewarded = mouse_df['feedbackGiven'].values == 1
        aud_sign = np.sign(mouse_df['audDiff'].values)
        # Map 1 : left  2: right to -1 and +1, 0 is still no-go
        choice_sign = mouse_df['responseRecorded'].values
        choice_sign[choice_sign == 1] = -1
        choice_sign[choice_sign == 2] = 1

        subset_index = mouse_df.loc[
            mouse_rewarded &
            (mouse_df['coherent'] == False) &
            (choice_sign == aud_sign) &
            (choice_sign != 0)
            ].index.values

    elif s_condition == 'after_rewarded_conflict_vis_choice':

        mouse_rewarded = mouse_df['feedbackGiven'].values == 1
        vis_sign = np.sign(mouse_df['visDiff'].values)
        # Map 1 : left  2: right to -1 and +1, 0 is still no-go
        choice_sign = mouse_df['responseRecorded'].values
        choice_sign[choice_sign == 1] = -1
        choice_sign[choice_sign == 2] = 1

        subset_index = mouse_df.loc[
            # (~mouse_df['rewardTimes'].isna()) &
            mouse_rewarded &
            (mouse_df['coherent'] == False) &
            (choice_sign == vis_sign) &
            (choice_sign != 0)
            ].index.values

    else:
        print('WARNING: subset condition %s not supported' % s_condition)

    if s_condition != 'all':
        min_index_value = np.min(mouse_df.index.values)
        max_index_value = np.max(mouse_df.index.values)
        # subset_index = subset_index[subset_index > min_index_value]
        subset_index = subset_index[subset_index < max_index_value]
        try:
            subset_df = mouse_df.loc[subset_index + 1]
        except:
            pdb.set_trace()

    if exclude_no_go:
        subset_df = subset_df.loc[
            subset_df['noGo'] == False
            ]

    if exclude_invalid_trials:
        subset_df = subset_df.loc[
            subset_df['validTrial'] == 1
            ]

    return subset_df

def main():

    supported_processes = ['fit_neural_model', 'fit_behaviour_model',
                           'plot_single_neural_model_output', 'plot_single_behaviour_model_output',
                           'plot_multiple_neural_model_output', 'plot_all_exp_models_summary',
                           'get_model_performance', 'plot_model_performance',
                           'fit_neural_model_pinkrigs', 'fit_behaviour_model_pinkrigs',
                           'plot_multiple_neural_model_output_pinkrigs',
                           'plot_decoding_over_days_pinkrigs', 'plot_conditional_psychometric_curves',
                           'fit_subset_cond_psychometric_models']

    processes_to_run = ['plot_conditional_psychometric_curves']

    interim_data_folder = '/Volumes/Partition 1/data/interim/neuro-psychometric-model'
    fig_folder = '/Volumes/Partition 1/data/interim/neuro-psychometric-model/plots'

    # TODO: this should be named choice alignment folder
    stim_alignment_folder = '/Volumes/Partition 1/data/interim/active-m2-choice-init-reliable-alignment'
    subject = 3
    exp = 21
    target_brain_region = 'MOs'

    process_params = {
        'fit_neural_model': dict(
            cv_type='cross-validate',
            subjects=[1, 2, 3, 4, 5, 6],
            exps={
                1: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                2: [15, 16, 17, 18, 19],
                3: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                4: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
                5: [42, 43, 44, 45, 46, 47],
                6: [48, 49, 50, 51, 52, 53, 54, 55, 56],
                },
            brain_regions=['MOs', 'OLF', 'ILA', 'ACA', 'ORB', 'PL'],
            clfs_to_fit=['LR-l1', 'LR-l2', 'LR', 'LR-l1-hyperparam-tuned', 'LR-l2-hyperparam-tuned'],
            include_stimulus_predictors=False,
            run_forward_feature_selection=True,
            min_least_common_choice=20,
            min_neurons=30,
            cv_random_seed=1,
            max_iter=2000,
        ),
        'fit_neural_model_pinkrigs': dict(
            pink_rig_data_folder='/Volumes/Partition 1/data/interim/neuro-psychometric-model/pinkRigsData',
            pink_rig_model_folder='/Volumes/Partition 1/data/interim/neuro-psychometric-model/pinkRigsModelResults',
            cv_type='cross-validate',
            clfs_to_fit=['LR-l1', 'LR-l2', 'LR', 'LR-l1-hyperparam-tuned', 'LR-l2-hyperparam-tuned'],
            include_stimulus_predictors=False,
            min_least_common_choice=20,
            min_neurons=30,
            cv_random_seed=1,
            max_iter=2000,
        ),
        'fit_behaviour_model_pinkrigs': dict(
            pink_rig_data_folder='/Volumes/Partition 1/data/interim/neuro-psychometric-model/pinkRigsData',
            pink_rig_model_folder='/Volumes/Partition 1/data/interim/neuro-psychometric-model/pinkRigsModelResults',
            cv_type='cross-validate',
            clfs_to_fit=['LR-l1', 'LR-l2', 'LR', 'LR-l1-hyperparam-tuned', 'LR-l2-hyperparam-tuned'],
            min_least_common_choice=20,
            min_neurons=30,
            cv_random_seed=1,
            max_iter=2000,
        ),
        'fit_behaviour_model': dict(
            subjects=[1, 2, 3, 4, 5, 6],
            exps={
                1: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                2: [15, 16, 17, 18, 19],
                3: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                4: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
                5: [42, 43, 44, 45, 46, 47],
                6: [48, 49, 50, 51, 52, 53, 54, 55, 56],
            },
            cv_type='cross-validate',
            min_least_common_choice=20,
            clfs_to_fit=['LR-l1', 'LR-l2', 'LR', 'LR-l1-hyperparam-tuned', 'LR-l2-hyperparam-tuned'],
            cv_random_seed=1,
            max_iter=2000,
        ),
        'plot_multiple_neural_model_output': dict(
            subjects=[3],
            exps={
                1: [3, 4, 5, 7, 8, 9, 10],
                2: [15, 16, 17, 18, 19],
                3: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                4: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
                5: [42, 43, 44, 45, 46, 47],
                6: [48, 49, 50, 51, 52, 53, 54, 55, 56],
            },
            brain_regions=['MOs', 'OLF', 'ILA', 'ACA', 'ORB', 'PL'],
            clfs_to_plot=['LR-l1', 'LR-l2', 'LR', 'LR-l1-hyperparam-tuned', 'LR-l2-hyperparam-tuned'],
            num_vals_per_bin=25,
            include_stimulus_predictors=True,
            include_behaviour_only_model=True,
            include_num_neurons=True,
            bin_behaviour_only_model=True,
            include_logistic_curve=True,
            sets_to_plot=['train', 'test'],
        ),
        'plot_multiple_neural_model_output_pinkrigs': dict(
            pink_rig_model_folder='/Volumes/Partition 1/data/interim/neuro-psychometric-model/pinkRigsModelResults',
            pink_rig_plot_folder='/Volumes/Partition 1/data/interim/neuro-psychometric-model/plotsPinkRigs',
            clfs_to_plot=['LR-l1', 'LR-l2', 'LR', 'LR-l1-hyperparam-tuned', 'LR-l2-hyperparam-tuned'],
            num_vals_per_bin=25,
            include_stimulus_predictors=True,
            include_behaviour_only_model=True,
            include_num_neurons=True,
            bin_behaviour_only_model=True,
            include_logistic_curve=True,
            sets_to_plot=['train', 'test'],
        ),
        'plot_all_exp_models_summary': dict(
            subjects=[1, 2, 3, 4, 5, 6],
            exps={
                1: [3, 4, 5, 7, 8, 9, 10],
                2: [15, 16, 17, 18, 19],
                3: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                4: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
                5: [42, 43, 44, 45, 46, 47],
                6: [48, 49, 50, 51, 52, 53, 54, 55, 56],
            },
            brain_regions=['MOs', 'OLF', 'ILA', 'ACA', 'ORB', 'PL'],
            num_vals_per_bin=10,
            include_stimulus_predictors=True,
            include_behaviour_only_model=True,
            scale_z=True,
        ),
        'get_model_performance': dict(
            subjects=[1, 2, 3, 4, 5, 6],
            exps={
                1: [3, 4, 5, 7, 8, 9, 10],
                2: [15, 16, 17, 18, 19],
                3: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                4: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
                5: [42, 43, 44, 45, 46, 47],
                6: [48, 49, 50, 51, 52, 53, 54, 55, 56],
            },
            brain_regions=['MOs', 'OLF', 'ILA', 'ACA', 'ORB', 'PL'],
            clf_names=['LR-l1', 'LR-l2', 'LR', 'LR-l1-hyperparam-tuned', 'LR-l2-hyperparam-tuned'],
            num_vals_per_bin=10,
            include_stimulus_predictors=True,
            include_behaviour_only_model=True,
            eps=10 ** (-4)
        ),
        'plot_model_performance': dict(
            brain_regions=['MOs', 'OLF', 'ILA', 'ACA', 'ORB', 'PL'],
            model_metric='accuracy',  # accuracy or log_loss
            single_exp_alpha=0.3,
            plot_brain_region_mean=True,
            clf_names=['LR-l1', 'LR-l2', 'LR', 'LR-l1-hyperparam-tuned', 'LR-l2-hyperparam-tuned'],
        ),
        'plot_decoding_over_days_pinkrigs': dict(
            pink_rig_data_folder='/Volumes/Partition 1/data/interim/neuro-psychometric-model/pinkRigsData',
            pink_rig_model_folder='/Volumes/Partition 1/data/interim/neuro-psychometric-model/pinkRigsModelResults',
            pink_rig_plot_folder='/Volumes/Partition 1/data/interim/neuro-psychometric-model/plotsPinkRigs',
            clfs_to_fit=['LR-l1', 'LR-l2', 'LR', 'LR-l1-hyperparam-tuned', 'LR-l2-hyperparam-tuned'],
        ),
        'fit_neural_model_forward_feature_selection': dict(
            cv_type='cross-validate',
            subjects=[1, 2, 3, 4, 5, 6],
            exps={
                1: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                2: [15, 16, 17, 18, 19],
                3: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                4: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
                5: [42, 43, 44, 45, 46, 47],
                6: [48, 49, 50, 51, 52, 53, 54, 55, 56],
            },
            brain_regions=['MOs', 'OLF', 'ILA', 'ACA', 'ORB', 'PL'],
            clfs_to_fit=['LR-l1', 'LR-l2', 'LR', 'LR-l1-hyperparam-tuned', 'LR-l2-hyperparam-tuned'],
            include_stimulus_predictors=False,
            min_least_common_choice=20,
            min_neurons=30,
            cv_random_seed=1,
            max_iter=2000,
        ),
        'plot_conditional_psychometric_curves': dict(
            # behaviour_df_fpath='/Volumes/T7/multisensory-integration-local/multisensory-integration/data/interim/active-m2-good-w-nogo/subset/ephys_behaviour_df.pkl',
            # behaviour_df_fpath='/Volumes/Partition 1/data/interim/active-m2-w-date/subset/ephys_behaviour_df.pkl',
            behaviour_df_fpath='/Volumes/Partition 1/data/interim/multispaceworld-all-active-behaviour/ephys_behaviour_df.pkl',
            fig_folder='/Users/timothysit/multisensory-integration/reports/figures/psychometric',
            exclude_laser_experiments=True,
            exclude_no_go=True,
            exclude_invalid_trials=False,
            subplot_grid=[5, 3],
            fig_size=[8, 8],
            subset_conditions=['all', 'after_left_choice', 'after_right_choice',
                                 'after_rewarded_choice', 'after_rewarded_left_choice', 'after_rewarded_right_choice',
                                 'after_rewarded_low_vis', 'after_rewarded_low_vis_left_choice', 'after_rewarded_low_vis_right_choice',
                                 'after_rewarded_conflict', 'after_rewarded_conflict_left_choice', 'after_rewarded_conflict_right_choice',
                                 'after_rewarded_conflict_aud_choice', 'after_rewarded_conflict_vis_choice'],
            subjects=[1, 2, 3, 4, 5, 6],  # note this currently does nothing
            exps={
                1: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                2: [15, 16, 17, 18, 19],
                3: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                4: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
                5: [42, 43, 44, 45, 46, 47],
                6: [48, 49, 50, 51, 52, 53, 54, 55, 56],
            },
        ),
        'fit_subset_cond_psychometric_models': dict(
            behaviour_df_fpath='/Volumes/Partition 1/data/interim/multispaceworld-all-active-behaviour/ephys_behaviour_df.pkl',
            fig_folder='/Users/timothysit/multisensory-integration/reports/figures/psychometric',
            exclude_laser_experiments=True,
            exclude_no_go=True,
            subset_conditions=['all', 'after_left_choice', 'after_right_choice',
                               'after_rewarded_choice', 'after_rewarded_low_vis',
                               'after_rewarded_conflict', 'after_rewarded_conflict_aud_choice',
                               'after_rewarded_conflict_vis_choice'],
        )
    }

    for process in processes_to_run:

        assert process in supported_processes

        if process == 'fit_behaviour_model':
            print('Fitting behaviour model')
            min_least_common_choice = process_params[process]['min_least_common_choice']
            clfs_to_fit = process_params[process]['clfs_to_fit']
            max_iter = process_params[process]['max_iter']

            if not os.path.exists(stim_alignment_folder):
                print('Cannot find stimulus alignment folder: %s' % stim_alignment_folder)

            for subject in process_params[process]['subjects']:
                for exp in process_params[process]['exps'][subject]:
                    alignment_ds = pephys.load_subject_exp_alignment_ds(alignment_folder=stim_alignment_folder,
                                                                        subject_num=subject, exp_num=exp,
                                                                        target_brain_region='any',
                                                                        aligned_event='choiceInitTime',
                                                                        alignment_file_ext='.nc')
                    if alignment_ds is None:
                        continue
                    if len(alignment_ds.Cell) == 0:
                        continue

                    X, y = get_behaviour_X_and_y(alignment_ds)

                    unique_y, unique_counts = np.unique(y, return_counts=True)
                    if (len(unique_y) == 1) or (np.min(unique_counts) < min_least_common_choice):
                        continue
                    if len(alignment_ds.Cell) == 0:
                        continue

                    print('Fitting behaviour model to subject %.f experiment %.f' % (subject, exp))

                    latent_var_per_trial_per_model = []
                    prob_per_trial_per_model = []
                    model_latent_per_trial_type_per_model = []
                    actual_p_right_per_trial_type_per_model = []
                    # Train set
                    model_latent_per_trial_type_train_set_per_model = []
                    latent_var_per_trial_train_set_per_model = []

                    for clf in clfs_to_fit:
                        latent_var_per_trial, prob_per_trial, model_latent_per_trial_type, actual_p_right_per_trial_type,\
                        trial_num_per_trial_type, prob_per_trial_train_set, latent_var_per_trial_train_set, \
                        model_latent_per_trial_type_train_set = \
                            fit_behaviour_only_model(X, y, clf=clf,
                                                     cv_type=process_params[process]['cv_type'], num_folds=2,
                                                     cv_random_seed=1, max_iter=max_iter)

                        latent_var_per_trial_per_model.append(latent_var_per_trial)
                        prob_per_trial_per_model.append(prob_per_trial)
                        model_latent_per_trial_type_per_model.append(model_latent_per_trial_type)
                        actual_p_right_per_trial_type_per_model.append(actual_p_right_per_trial_type)
                        model_latent_per_trial_type_train_set_per_model.append(model_latent_per_trial_type_train_set)
                        latent_var_per_trial_train_set_per_model.append(latent_var_per_trial_train_set)

                    latent_var_per_trial_per_model = np.stack(latent_var_per_trial_per_model)
                    model_latent_per_trial_type_train_set_per_model = np.stack(model_latent_per_trial_type_train_set_per_model)
                    latent_var_per_trial_train_set_per_model = np.stack(latent_var_per_trial_train_set_per_model)

                    # Save results
                    save_name = 's%.fe%.f_behaviour_model_results.npz' % (subject, exp)
                    np.savez(os.path.join(interim_data_folder, save_name),
                             X=X, y=y, prob_per_trial=prob_per_trial,
                             latent_var_per_trial=latent_var_per_trial,
                             model_latent_per_trial_type=model_latent_per_trial_type,
                             actual_p_right_per_trial_type=actual_p_right_per_trial_type,
                             trial_num_per_trial_type=trial_num_per_trial_type,
                             model_names=clfs_to_fit,
                             latent_var_per_trial_per_model=latent_var_per_trial_per_model,
                             prob_per_trial_per_model=prob_per_trial_per_model,
                             model_latent_per_trial_type_per_model=model_latent_per_trial_type_per_model,
                             actual_p_right_per_trial_type_per_model=actual_p_right_per_trial_type_per_model,
                             prob_per_trial_train_set=prob_per_trial_train_set,
                             latent_var_per_trial_train_set=latent_var_per_trial_train_set,
                             model_latent_per_trial_type_train_set_per_model=model_latent_per_trial_type_train_set_per_model,
                             latent_var_per_trial_train_set_per_model=latent_var_per_trial_train_set_per_model)

        if process == 'fit_neural_model':

            min_least_common_choice = process_params[process]['min_least_common_choice']
            min_neurons = process_params[process]['min_neurons']
            clfs_to_fit = process_params[process]['clfs_to_fit']
            cv_random_seed = process_params[process]['cv_random_seed']
            max_iter = process_params[process]['max_iter']
            run_forward_feature_selection = process_params[process]['run_forward_feature_selection']

            for subject in process_params[process]['subjects']:
                for exp in process_params[process]['exps'][subject]:
                    if process_params[process]['brain_regions'] == 'all':
                        available_files = glob.glob(
                            os.path.join(
                                stim_alignment_folder, 'subject_%.f_exp_%.f_*.nc' % (subject, exp)))
                        brain_regions_to_analyse = [x.split('_')[4] for x in available_files]
                    else:
                        brain_regions_to_analyse = process_params[process]['brain_regions']

                    for target_brain_region in brain_regions_to_analyse:

                        alignment_ds = pephys.load_subject_exp_alignment_ds(alignment_folder=stim_alignment_folder,
                                                                            subject_num=subject, exp_num=exp,
                                                                            target_brain_region=target_brain_region,
                                                                            aligned_event='choiceInitTime',
                                                                        alignment_file_ext='.nc')
                        if alignment_ds is None:
                            continue
                        num_cell = len(alignment_ds.Cell)
                        if num_cell == 0:
                            continue

                        if min_neurons is not None:
                            if num_cell < min_neurons:
                                print('%.f neurons found, which is fewer than specified %.f, skipping...'
                                       % (num_cell, min_neurons))
                                continue

                        include_stimulus_predictors = process_params[process]['include_stimulus_predictors']
                        print('Fitting neural model to subject %.f experiment %.f in %s' % (
                        subject, exp, target_brain_region))
                        X, y = get_behaviour_X_and_y(alignment_ds)

                        unique_y, unique_counts = np.unique(y, return_counts=True)
                        if (len(np.unique(y)) == 1) or (np.min(unique_counts) < min_least_common_choice):
                            continue

                        X_neural = get_X_neural(alignment_ds, mean_fr_window=[-0.15, 0])

                        neural_latent_var_per_trial_per_model = []
                        neural_prob_per_trial_per_model = []
                        neural_latent_var_per_trial_train_set_per_model = []
                        neural_prob_per_trial_train_set_per_model = []

                        if run_forward_feature_selection:

                            forward_clf = sklinear.LogisticRegression(fit_intercept=False, penalty='l1', max_iter=max_iter, solver='liblinear')
                            num_features_to_select = np.shape(X_neural[:, 1:])[1]
                            _, accuracy_per_best_n_feature = do_forward_feature_selection(X_neural[:, 1:], y, forward_clf,
                                                                                          selection_criteria='accuracy',
                                                                                          num_features_to_select=num_features_to_select)
                            _, accuracy_per_random_n_feature = do_forward_feature_selection(X_neural[:, 1:], y, forward_clf,
                                                                                            selection_criteria='random',
                                                                                            num_features_to_select=num_features_to_select)

                        for clf in clfs_to_fit:

                            neural_latent_var_per_trial, neural_prob_per_trial, model_prob_per_trial_train, \
                            latent_var_per_trial_train = fit_neural_only_model(
                                X_neural, y,  cv_type=process_params[process]['cv_type'], clf=clf,
                                include_stimulus_predictors=include_stimulus_predictors,
                                X_stim=X, cv_random_seed=cv_random_seed, max_iter=max_iter)

                            # Test set
                            neural_latent_var_per_trial_per_model.append(
                                neural_latent_var_per_trial
                            )
                            neural_prob_per_trial_per_model.append(
                                neural_prob_per_trial
                            )

                            # Train set
                            neural_latent_var_per_trial_train_set_per_model.append(
                                latent_var_per_trial_train
                            )
                            neural_prob_per_trial_train_set_per_model.append(
                                model_prob_per_trial_train
                            )

                        neural_latent_var_per_trial_per_model = np.stack(neural_latent_var_per_trial_per_model)
                        neural_prob_per_trial_per_model = np.stack(neural_prob_per_trial_per_model)
                        neural_latent_var_per_trial_train_set_per_model = np.stack(neural_latent_var_per_trial_train_set_per_model)
                        neural_prob_per_trial_train_set_per_model = np.stack(neural_prob_per_trial_train_set_per_model)

                        # Save results
                        if include_stimulus_predictors:
                            # TODO: note neural_latent_var_per_trial will be phased out after
                            # testing neural_latent_var_per_trial_per_model works
                            save_name = 's%.fe%.f_%s_neural_plus_behaviour_model_results.npz' % (subject, exp, target_brain_region)
                            X_all = np.concatenate([X_neural, X[:, 1:]], axis=1)  # exclude the intercept from X

                            # TODO: use dict
                            if run_forward_feature_selection:
                                np.savez(os.path.join(interim_data_folder, save_name),
                                         X_neural=X_neural, X=X, y=y, neural_prob_per_trial=neural_prob_per_trial,
                                         neural_latent_var_per_trial=neural_latent_var_per_trial, X_all=X_all,
                                         model_names=clfs_to_fit,
                                         neural_latent_var_per_trial_per_model=neural_latent_var_per_trial_per_model,
                                         neural_prob_per_trial_per_model=neural_prob_per_trial_per_model,
                                         neural_latent_var_per_trial_train_set_per_model=neural_latent_var_per_trial_train_set_per_model,
                                         neural_prob_per_trial_train_set_per_model=neural_prob_per_trial_train_set_per_model,
                                         accuracy_per_best_n_feature=accuracy_per_best_n_feature,
                                         accuracy_per_random_n_feature=accuracy_per_random_n_feature)
                            else:
                                np.savez(os.path.join(interim_data_folder, save_name),
                                         X_neural=X_neural, X=X, y=y, neural_prob_per_trial=neural_prob_per_trial,
                                         neural_latent_var_per_trial=neural_latent_var_per_trial, X_all=X_all,
                                         model_names=clfs_to_fit,
                                         neural_latent_var_per_trial_per_model=neural_latent_var_per_trial_per_model,
                                         neural_prob_per_trial_per_model=neural_prob_per_trial_per_model,
                                         neural_latent_var_per_trial_train_set_per_model=neural_latent_var_per_trial_train_set_per_model,
                                         neural_prob_per_trial_train_set_per_model=neural_prob_per_trial_train_set_per_model)

                        else:
                            save_name = 's%.fe%.f_%s_neural_model_results.npz' % (subject, exp, target_brain_region)
                            np.savez(os.path.join(interim_data_folder, save_name),
                                 X_neural=X_neural, X=X, y=y, neural_prob_per_trial=neural_prob_per_trial,
                                 neural_latent_var_per_trial=neural_latent_var_per_trial,
                                 model_names=clfs_to_fit,
                                 neural_latent_var_per_trial_per_model=neural_latent_var_per_trial_per_model,
                                 neural_prob_per_trial_per_model=neural_prob_per_trial_per_model,
                                 neural_latent_var_per_trial_train_set_per_model=neural_latent_var_per_trial_train_set_per_model,
                                 neural_prob_per_trial_train_set_per_model=neural_prob_per_trial_train_set_per_model)

        if process == 'fit_behaviour_model_pinkrigs':

            print('Fitting behaviour model')
            min_least_common_choice = process_params[process]['min_least_common_choice']
            clfs_to_fit = process_params[process]['clfs_to_fit']
            max_iter = process_params[process]['max_iter']

            if not os.path.exists(stim_alignment_folder):
                print('Cannot find stimulus alignment folder: %s' % stim_alignment_folder)

            alignment_ds_paths = glob.glob(os.path.join(process_params[process]['pink_rig_data_folder'], '*.nc'))
            choice_var_name = 'timeline_choiceMoveDir'
            aud_diff_name = 'audDiff'
            vis_diff_name = 'visDiff'

            for path in alignment_ds_paths:

                fname = os.path.basename(path)
                subject = fname.split('_')[0]
                exp_date = fname.split('_')[1]
                exp_num = fname.split('_')[2].split('.')[0]
                alignment_ds = xr.open_dataset(path)

                if alignment_ds is None:
                    continue
                if len(alignment_ds.Cell) == 0:
                    continue

                vis_azimuth = alignment_ds['stim_visAzimuth']
                vis_azimuth[np.isnan(vis_azimuth)] = 0
                alignment_ds['visDiff'] = alignment_ds['stim_visContrast'] * np.sign(vis_azimuth)
                alignment_ds['audDiff'] = alignment_ds['stim_audAmplitude'] * np.sign(alignment_ds['stim_audAzimuth'])
                X, y = get_behaviour_X_and_y(alignment_ds, choice_var_name=choice_var_name,
                                             aud_diff_name=aud_diff_name, vis_diff_name=vis_diff_name)

                unique_y, unique_counts = np.unique(y, return_counts=True)
                if (len(unique_y) == 1) or (np.min(unique_counts) < min_least_common_choice):
                    continue
                if len(alignment_ds.Cell) == 0:
                    continue

                print('Fitting behaviour model to %s %s experiment %s' % (subject, exp_date, exp_num))

                latent_var_per_trial_per_model = []
                prob_per_trial_per_model = []
                model_latent_per_trial_type_per_model = []
                actual_p_right_per_trial_type_per_model = []
                # Train set
                model_latent_per_trial_type_train_set_per_model = []
                latent_var_per_trial_train_set_per_model = []

                for clf in clfs_to_fit:
                    latent_var_per_trial, prob_per_trial, model_latent_per_trial_type, actual_p_right_per_trial_type, \
                    trial_num_per_trial_type, prob_per_trial_train_set, latent_var_per_trial_train_set, \
                    model_latent_per_trial_type_train_set = \
                        fit_behaviour_only_model(X, y, clf=clf,
                                                 cv_type=process_params[process]['cv_type'], num_folds=2,
                                                 cv_random_seed=1, max_iter=max_iter)

                    latent_var_per_trial_per_model.append(latent_var_per_trial)
                    prob_per_trial_per_model.append(prob_per_trial)
                    model_latent_per_trial_type_per_model.append(model_latent_per_trial_type)
                    actual_p_right_per_trial_type_per_model.append(actual_p_right_per_trial_type)
                    model_latent_per_trial_type_train_set_per_model.append(model_latent_per_trial_type_train_set)
                    latent_var_per_trial_train_set_per_model.append(latent_var_per_trial_train_set)

                latent_var_per_trial_per_model = np.stack(latent_var_per_trial_per_model)
                model_latent_per_trial_type_train_set_per_model = np.stack(
                    model_latent_per_trial_type_train_set_per_model)
                latent_var_per_trial_train_set_per_model = np.stack(latent_var_per_trial_train_set_per_model)

                # Save results
                save_name = '%s_%s_%s_behaviour_model_results.npz' % (
                    subject, exp_date, exp_num)
                np.savez(os.path.join(process_params[process]['pink_rig_model_folder'], save_name),
                         X=X, y=y, prob_per_trial=prob_per_trial,
                         latent_var_per_trial=latent_var_per_trial,
                         model_latent_per_trial_type=model_latent_per_trial_type,
                         actual_p_right_per_trial_type=actual_p_right_per_trial_type,
                         trial_num_per_trial_type=trial_num_per_trial_type,
                         model_names=clfs_to_fit,
                         latent_var_per_trial_per_model=latent_var_per_trial_per_model,
                         prob_per_trial_per_model=prob_per_trial_per_model,
                         model_latent_per_trial_type_per_model=model_latent_per_trial_type_per_model,
                         actual_p_right_per_trial_type_per_model=actual_p_right_per_trial_type_per_model,
                         prob_per_trial_train_set=prob_per_trial_train_set,
                         latent_var_per_trial_train_set=latent_var_per_trial_train_set,
                         model_latent_per_trial_type_train_set_per_model=model_latent_per_trial_type_train_set_per_model,
                         latent_var_per_trial_train_set_per_model=latent_var_per_trial_train_set_per_model)

        if process == 'fit_neural_model_pinkrigs':

            min_least_common_choice = process_params[process]['min_least_common_choice']
            min_neurons = process_params[process]['min_neurons']
            clfs_to_fit = process_params[process]['clfs_to_fit']
            cv_random_seed = process_params[process]['cv_random_seed']
            max_iter = process_params[process]['max_iter']
            alignment_ds_paths = glob.glob(os.path.join(process_params[process]['pink_rig_data_folder'], '*.nc'))
            choice_var_name = 'timeline_choiceMoveDir'
            aud_diff_name = 'audDiff'
            vis_diff_name = 'visDiff'

            for path in alignment_ds_paths:

                fname = os.path.basename(path)
                subject = fname.split('_')[0]
                exp_date = fname.split('_')[1]
                exp_num = fname.split('_')[2].split('.')[0]
                alignment_ds = xr.open_dataset(path)

                num_cell = len(alignment_ds.Cell)
                if num_cell == 0:
                    continue

                if min_neurons is not None:
                    if num_cell < min_neurons:
                        print('%.f neurons found, which is fewer than specified %.f, skipping...'
                              % (num_cell, min_neurons))
                        continue

                include_stimulus_predictors = process_params[process]['include_stimulus_predictors']
                print('Fitting neural model to subject %s %s exp %s' % (
                    subject, exp_date, exp_num))

                vis_azimuth = alignment_ds['stim_visAzimuth']
                vis_azimuth[np.isnan(vis_azimuth)] = 0
                alignment_ds['visDiff'] = alignment_ds['stim_visContrast'] * np.sign(vis_azimuth)
                alignment_ds['audDiff'] = alignment_ds['stim_audAmplitude'] * np.sign(alignment_ds['stim_audAzimuth'])

                X, y = get_behaviour_X_and_y(alignment_ds, choice_var_name=choice_var_name,
                                             aud_diff_name=aud_diff_name, vis_diff_name=vis_diff_name)

                if np.sum(~np.isfinite(X.flatten())) > 0:
                    pdb.set_trace()

                unique_y, unique_counts = np.unique(y, return_counts=True)
                if (len(np.unique(y)) == 1) or (np.min(unique_counts) < min_least_common_choice):
                    continue

                neural_activity = alignment_ds['activity'].values
                intercept_term = np.repeat(1, np.shape(neural_activity)[0]).reshape(-1, 1)
                X_neural = np.concatenate([intercept_term, neural_activity], axis=1)

                neural_latent_var_per_trial_per_model = []
                neural_prob_per_trial_per_model = []
                neural_latent_var_per_trial_train_set_per_model = []
                neural_prob_per_trial_train_set_per_model = []



                for clf in clfs_to_fit:
                    neural_latent_var_per_trial, neural_prob_per_trial, model_prob_per_trial_train, \
                    latent_var_per_trial_train = fit_neural_only_model(
                        X_neural, y, cv_type=process_params[process]['cv_type'], clf=clf,
                        include_stimulus_predictors=include_stimulus_predictors,
                        X_stim=X, cv_random_seed=cv_random_seed, max_iter=max_iter)

                    # Test set
                    neural_latent_var_per_trial_per_model.append(
                        neural_latent_var_per_trial
                    )
                    neural_prob_per_trial_per_model.append(
                        neural_prob_per_trial
                    )

                    # Train set
                    neural_latent_var_per_trial_train_set_per_model.append(
                        latent_var_per_trial_train
                    )
                    neural_prob_per_trial_train_set_per_model.append(
                        model_prob_per_trial_train
                    )

                neural_latent_var_per_trial_per_model = np.stack(neural_latent_var_per_trial_per_model)
                neural_prob_per_trial_per_model = np.stack(neural_prob_per_trial_per_model)
                neural_latent_var_per_trial_train_set_per_model = np.stack(
                    neural_latent_var_per_trial_train_set_per_model)
                neural_prob_per_trial_train_set_per_model = np.stack(neural_prob_per_trial_train_set_per_model)

                # Save results
                if include_stimulus_predictors:
                    # TODO: note neural_latent_var_per_trial will be phased out after
                    # testing neural_latent_var_per_trial_per_model works
                    save_name = '%s_%s_%s_neural_plus_behaviour_model_results.npz' % (
                    subject, exp_date, exp_num)
                    X_all = np.concatenate([X_neural, X[:, 1:]], axis=1)  # exclude the intercept from X
                    np.savez(os.path.join(process_params[process]['pink_rig_model_folder'], save_name),
                             X_neural=X_neural, X=X, y=y, neural_prob_per_trial=neural_prob_per_trial,
                             neural_latent_var_per_trial=neural_latent_var_per_trial, X_all=X_all,
                             model_names=clfs_to_fit,
                             neural_latent_var_per_trial_per_model=neural_latent_var_per_trial_per_model,
                             neural_prob_per_trial_per_model=neural_prob_per_trial_per_model,
                             neural_latent_var_per_trial_train_set_per_model=neural_latent_var_per_trial_train_set_per_model,
                             neural_prob_per_trial_train_set_per_model=neural_prob_per_trial_train_set_per_model)

                else:
                    save_name = '%s_%s_%s_neural_model_results.npz' % (subject, exp_date, exp_num)
                    np.savez(os.path.join(process_params[process]['pink_rig_model_folder'], save_name),
                             X_neural=X_neural, X=X, y=y, neural_prob_per_trial=neural_prob_per_trial,
                             neural_latent_var_per_trial=neural_latent_var_per_trial,
                             model_names=clfs_to_fit,
                             neural_latent_var_per_trial_per_model=neural_latent_var_per_trial_per_model,
                             neural_prob_per_trial_per_model=neural_prob_per_trial_per_model,
                             neural_latent_var_per_trial_train_set_per_model=neural_latent_var_per_trial_train_set_per_model,
                             neural_prob_per_trial_train_set_per_model=neural_prob_per_trial_train_set_per_model)


        if process == 'plot_single_neural_model_output':

            print('Plotting single neural model output')
            neural_results_fpath = os.path.join(interim_data_folder, 's%.fe%.f_%s_neural_model_results.npz' %
                                                (subject, exp, target_brain_region))
            neural_results = np.load(neural_results_fpath)

            neural_latent_var_per_trial = neural_results['neural_latent_var_per_trial']
            neural_prob_per_trial = neural_results['neural_prob_per_trial']
            y = neural_results['y']
            latent_variable_per_bin, actual_p_right_per_bin = \
                bin_neural_latent_and_p_right(neural_latent_var_per_trial, y, num_vals_per_bin=25)

            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, axs = plot_neurometric_model(neural_latent_var_per_trial, neural_prob_per_trial,
                               latent_variable_per_bin, actual_p_right_per_bin,
                               fig=None, axs=None)

                fig_name = 's%.fe%.f_%s_neural_model_results' % (subject, exp, target_brain_region)
                fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

        if process == 'plot_single_behaviour_model_output':

            print('Plotting single behaviour model output')

            behaviour_results_fpath = os.path.join(interim_data_folder,
                                's%.fe%.f_%s_behaviour_model_results.npz' % (subject, exp, target_brain_region))

            behaviour_results = np.load(behaviour_results_fpath)

            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, axs = plot_behaviour_model(latent_var_per_trial=behaviour_results['latent_var_per_trial'],
                                                prob_per_trial=behaviour_results['prob_per_trial'],
                                                model_latent_per_trial_type=behaviour_results['model_latent_per_trial_type'],
                                                actual_p_right_per_trial_type=behaviour_results['actual_p_right_per_trial_type'],
                                                fig=None, axs=None)

                fig_name = 's%.fe%.f_%s_behaviour_model_results' % (subject, exp, target_brain_region)
                fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

        if process == 'plot_multiple_neural_model_output':
            print('Running process %s' % process)
            num_vals_per_bin = process_params[process]['num_vals_per_bin']
            include_behaviour_only_model = process_params[process]['include_behaviour_only_model']
            include_stimulus_predictors = process_params[process]['include_stimulus_predictors']
            include_num_neurons = process_params[process]['include_num_neurons']
            bin_behaviour_only_model = process_params[process]['bin_behaviour_only_model']
            include_logistic_curve = process_params[process]['include_logistic_curve']
            clfs_to_plot = process_params[process]['clfs_to_plot']
            sets_to_plot = process_params[process]['sets_to_plot']

            for subject in process_params[process]['subjects']:
                for exp in process_params[process]['exps'][subject]:
                    if process_params[process]['brain_regions'] == 'all':
                        # TODO: this does not work yet
                        available_files = glob.glob(
                            os.path.join(
                                stim_alignment_folder, 'subject_%.f_exp_%.f_*.nc' % (subject, exp)))
                        brain_regions_to_analyse = [x.split('_')[4] for x in available_files]
                    else:
                        brain_regions_to_analyse = process_params[process]['brain_regions']

                    model_results_per_brain_region = dict()
                    if include_stimulus_predictors:
                        n_plus_b_results_per_brain_region = dict()
                    else:
                        n_plus_b_results_per_brain_region = None

                    for target_brain_region in brain_regions_to_analyse:

                        neural_results_fpath_search_result = glob.glob(os.path.join(interim_data_folder,
                                                            's%.fe%.f_%s_neural_model_results.npz' %
                                                            (subject, exp, target_brain_region)))

                        if len(neural_results_fpath_search_result) != 1:
                            continue

                        neural_results = np.load(neural_results_fpath_search_result[0])

                        if 'model_names' in neural_results.files:
                            latent_variable_per_bin_per_model = []
                            actual_p_right_per_bin_per_model = []

                            latent_variable_per_bin_train_set_per_model = []
                            actual_p_right_per_bin_train_set_per_model = []

                            for n_model in np.arange(len(neural_results['model_names'])):
                                y = neural_results['y']
                                neural_latent_var_per_trial = neural_results['neural_latent_var_per_trial_per_model'][n_model, :]
                                latent_variable_per_bin, actual_p_right_per_bin = \
                                    bin_neural_latent_and_p_right(neural_latent_var_per_trial, y,
                                                                  num_vals_per_bin=num_vals_per_bin)
                                latent_variable_per_bin_per_model.append(latent_variable_per_bin)
                                actual_p_right_per_bin_per_model.append(actual_p_right_per_bin)

                                # Train set
                                neural_latent_var_per_trial_train_set = neural_results['neural_latent_var_per_trial_train_set_per_model'][n_model, :]
                                latent_variable_per_bin_train, actual_p_right_per_bin_train = \
                                    bin_neural_latent_and_p_right(neural_latent_var_per_trial_train_set, y,
                                                                  num_vals_per_bin=num_vals_per_bin)
                                latent_variable_per_bin_train_set_per_model.append(latent_variable_per_bin_train)
                                actual_p_right_per_bin_train_set_per_model.append(actual_p_right_per_bin_train)


                            neural_results_keys = list(neural_results.keys())
                            neural_results_dict = dict()
                            for key in neural_results_keys:
                                neural_results_dict[key] = neural_results[key]

                            neural_results_dict['latent_variable_per_bin_per_model'] = np.stack(latent_variable_per_bin_per_model)
                            neural_results_dict['actual_p_right_per_bin_per_model'] = np.stack(actual_p_right_per_bin_per_model)
                            neural_results_dict['latent_variable_per_bin_train_set_per_model'] = np.stack(
                                latent_variable_per_bin_train_set_per_model)
                            neural_results_dict['actual_p_right_per_bin_train_set_per_model'] = np.stack(
                                actual_p_right_per_bin_train_set_per_model)

                        else:
                            neural_latent_var_per_trial = neural_results['neural_latent_var_per_trial']
                            # neural_prob_per_trial = neural_results['neural_prob_per_trial']
                            y = neural_results['y']
                            latent_variable_per_bin, actual_p_right_per_bin = \
                                bin_neural_latent_and_p_right(neural_latent_var_per_trial, y,
                                                              num_vals_per_bin=num_vals_per_bin)

                            neural_results_keys = list(neural_results.keys())
                            neural_results_dict = dict()
                            for key in neural_results_keys:
                                neural_results_dict[key] = neural_results[key]

                            neural_results_dict['latent_variable_per_bin'] = latent_variable_per_bin
                            neural_results_dict['actual_p_right_per_bin'] = actual_p_right_per_bin

                        model_results_per_brain_region[target_brain_region] = neural_results_dict

                        if include_stimulus_predictors:
                            neural_plus_behaviour_results_fpath = os.path.join(interim_data_folder,
                                                                's%.fe%.f_%s_neural_plus_behaviour_model_results.npz' %
                                                                (subject, exp,
                                                                 target_brain_region))
                            neural_plus_behaviour_results = np.load(neural_plus_behaviour_results_fpath)

                            if 'model_names' in neural_plus_behaviour_results.files:

                                latent_variable_per_bin_per_model = []
                                actual_p_right_per_bin_per_model = []
                                latent_variable_per_bin_train_set_per_model = []
                                actual_p_right_per_bin_train_set_per_model = []
                                for n_model in np.arange(len(neural_plus_behaviour_results['model_names'])):
                                    y = neural_plus_behaviour_results['y']
                                    neural_latent_var_per_trial = neural_plus_behaviour_results[
                                                                      'neural_latent_var_per_trial_per_model'][n_model,
                                                                  :]
                                    latent_variable_per_bin, actual_p_right_per_bin = \
                                        bin_neural_latent_and_p_right(neural_latent_var_per_trial, y,
                                                                      num_vals_per_bin=num_vals_per_bin)


                                    latent_variable_per_bin_per_model.append(latent_variable_per_bin)
                                    actual_p_right_per_bin_per_model.append(actual_p_right_per_bin)

                                    # Train set
                                    neural_latent_var_per_trial_train_set = neural_plus_behaviour_results[
                                                                      'neural_latent_var_per_trial_train_set_per_model'][n_model,
                                                                  :]
                                    latent_variable_per_bin_train_set, actual_p_right_per_bin_train_set = \
                                        bin_neural_latent_and_p_right(neural_latent_var_per_trial_train_set, y,
                                                                      num_vals_per_bin=num_vals_per_bin)

                                    latent_variable_per_bin_train_set_per_model.append(
                                        latent_variable_per_bin_train_set
                                    )
                                    actual_p_right_per_bin_train_set_per_model.append(
                                        actual_p_right_per_bin_train_set
                                    )


                                neural_plus_behaviour_results_keys = list(neural_plus_behaviour_results.keys())
                                n_plus_b_results_dict = dict()
                                for key in neural_plus_behaviour_results_keys:
                                    n_plus_b_results_dict[key] = neural_plus_behaviour_results[key]

                                n_plus_b_results_dict['latent_variable_per_bin_per_model'] = np.stack(
                                    latent_variable_per_bin_per_model)
                                n_plus_b_results_dict['actual_p_right_per_bin_per_model'] = np.stack(
                                    actual_p_right_per_bin_per_model)

                                n_plus_b_results_dict['latent_variable_per_bin_train_set_per_model'] = np.stack(
                                    latent_variable_per_bin_train_set_per_model)
                                n_plus_b_results_dict['actual_p_right_per_bin_train_set_per_model'] = np.stack(
                                    actual_p_right_per_bin_train_set_per_model)

                                n_plus_b_results_per_brain_region[target_brain_region] = n_plus_b_results_dict

                            else:

                                neural_plus_behaviour_latent_var_per_trial = neural_plus_behaviour_results['neural_latent_var_per_trial']
                                y = neural_plus_behaviour_results['y']
                                n_plus_b_latent_variable_per_bin, n_plus_b_actual_p_right_per_bin = \
                                    bin_neural_latent_and_p_right(neural_plus_behaviour_latent_var_per_trial, y,
                                                                  num_vals_per_bin=num_vals_per_bin)

                                neural_plus_behaviour_results_keys = list(neural_plus_behaviour_results.keys())
                                n_plus_b_results_dict = dict()
                                for key in neural_plus_behaviour_results_keys:
                                    n_plus_b_results_dict[key] = neural_plus_behaviour_results[key]

                                n_plus_b_results_dict['latent_variable_per_bin'] = n_plus_b_latent_variable_per_bin
                                n_plus_b_results_dict['actual_p_right_per_bin'] = n_plus_b_actual_p_right_per_bin

                                n_plus_b_results_per_brain_region[target_brain_region] = n_plus_b_results_dict

                    if include_behaviour_only_model:
                        behaviour_results_fpath = os.path.join(interim_data_folder,
                                                               's%.fe%.f_behaviour_model_results.npz' % (
                                                                   subject, exp))

                        behaviour_results = np.load(behaviour_results_fpath)
                    else:
                        behaviour_results = None

                    for model_name in clfs_to_plot:
                        with plt.style.context(splstyle.get_style('nature-reviews')):

                            for set_to_plot in sets_to_plot:
                                fig, axs = plot_multiple_neurometric_models(model_results_per_brain_region,
                                                            target_brain_regions=brain_regions_to_analyse,
                                                            behaviour_results=behaviour_results,
                                                            n_plus_b_results_per_brain_region=n_plus_b_results_per_brain_region,
                                                            include_num_neurons=include_num_neurons,
                                                            bin_behaviour_only_model=bin_behaviour_only_model,
                                                            num_vals_per_bin=num_vals_per_bin,
                                                            include_logistic_curve=include_logistic_curve,
                                                            model_name=model_name, set_to_plot=set_to_plot,
                                                            fig=None, axs=None)

                                if include_stimulus_predictors:
                                    fig_name = 's%.fe%.f_%s_multiple_brain_region_plus_stim_neurometric_model_comparison' % (subject, exp, model_name)
                                else:
                                    fig_name = 's%.fe%.f_%s_multiple_brain_region_neurometric_model_comparison' % (subject, exp, model_name)

                                if bin_behaviour_only_model:
                                    fig_name += '_behave_model_binned'

                                fig_name += '_%s_set' % set_to_plot

                                fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

        if process == 'plot_multiple_neural_model_output_pinkrigs':

            print('Running process %s' % process)
            num_vals_per_bin = process_params[process]['num_vals_per_bin']
            include_behaviour_only_model = process_params[process]['include_behaviour_only_model']
            include_stimulus_predictors = process_params[process]['include_stimulus_predictors']
            include_num_neurons = process_params[process]['include_num_neurons']
            bin_behaviour_only_model = process_params[process]['bin_behaviour_only_model']
            include_logistic_curve = process_params[process]['include_logistic_curve']
            clfs_to_plot = process_params[process]['clfs_to_plot']
            sets_to_plot = process_params[process]['sets_to_plot']

            neural_results_fpath_search_result = glob.glob(os.path.join(
                process_params[process]['pink_rig_model_folder'], '*neural_model_results.npz'
            ))

            for neural_results_path in neural_results_fpath_search_result:

                neural_results_basename = os.path.basename(neural_results_path)
                subject = neural_results_basename.split('_')[0]
                exp_date = neural_results_basename.split('_')[1]
                exp_num = neural_results_basename.split('_')[2].split('.')[0]

                neural_results = np.load(neural_results_path)

                if 'model_names' in neural_results.files:
                    latent_variable_per_bin_per_model = []
                    actual_p_right_per_bin_per_model = []

                    latent_variable_per_bin_train_set_per_model = []
                    actual_p_right_per_bin_train_set_per_model = []

                    for n_model in np.arange(len(neural_results['model_names'])):
                        y = neural_results['y']
                        neural_latent_var_per_trial = neural_results['neural_latent_var_per_trial_per_model'][
                                                      n_model, :]
                        latent_variable_per_bin, actual_p_right_per_bin = \
                            bin_neural_latent_and_p_right(neural_latent_var_per_trial, y,
                                                          num_vals_per_bin=num_vals_per_bin)
                        latent_variable_per_bin_per_model.append(latent_variable_per_bin)
                        actual_p_right_per_bin_per_model.append(actual_p_right_per_bin)

                        # Train set
                        neural_latent_var_per_trial_train_set = neural_results[
                                                                    'neural_latent_var_per_trial_train_set_per_model'][
                                                                n_model, :]
                        latent_variable_per_bin_train, actual_p_right_per_bin_train = \
                            bin_neural_latent_and_p_right(neural_latent_var_per_trial_train_set, y,
                                                          num_vals_per_bin=num_vals_per_bin)
                        latent_variable_per_bin_train_set_per_model.append(latent_variable_per_bin_train)
                        actual_p_right_per_bin_train_set_per_model.append(actual_p_right_per_bin_train)

                    neural_results_keys = list(neural_results.keys())
                    neural_results_dict = dict()
                    for key in neural_results_keys:
                        neural_results_dict[key] = neural_results[key]

                    neural_results_dict['latent_variable_per_bin_per_model'] = np.stack(
                        latent_variable_per_bin_per_model)
                    neural_results_dict['actual_p_right_per_bin_per_model'] = np.stack(
                        actual_p_right_per_bin_per_model)
                    neural_results_dict['latent_variable_per_bin_train_set_per_model'] = np.stack(
                        latent_variable_per_bin_train_set_per_model)
                    neural_results_dict['actual_p_right_per_bin_train_set_per_model'] = np.stack(
                        actual_p_right_per_bin_train_set_per_model)

                else:
                    neural_latent_var_per_trial = neural_results['neural_latent_var_per_trial']
                    # neural_prob_per_trial = neural_results['neural_prob_per_trial']
                    y = neural_results['y']
                    latent_variable_per_bin, actual_p_right_per_bin = \
                        bin_neural_latent_and_p_right(neural_latent_var_per_trial, y,
                                                      num_vals_per_bin=num_vals_per_bin)

                    neural_results_keys = list(neural_results.keys())
                    neural_results_dict = dict()
                    for key in neural_results_keys:
                        neural_results_dict[key] = neural_results[key]

                    neural_results_dict['latent_variable_per_bin'] = latent_variable_per_bin
                    neural_results_dict['actual_p_right_per_bin'] = actual_p_right_per_bin

                model_results_per_brain_region = neural_results_dict

                if include_stimulus_predictors:
                    neural_plus_behaviour_results_fpath = os.path.join(process_params[process]['pink_rig_model_folder'],
                                                                       '%s_%s_%s_neural_plus_behaviour_model_results.npz' %
                                                                       (subject, exp_date, exp_num))

                    neural_plus_behaviour_results = np.load(neural_plus_behaviour_results_fpath)

                    if 'model_names' in neural_plus_behaviour_results.files:

                        latent_variable_per_bin_per_model = []
                        actual_p_right_per_bin_per_model = []
                        latent_variable_per_bin_train_set_per_model = []
                        actual_p_right_per_bin_train_set_per_model = []
                        for n_model in np.arange(len(neural_plus_behaviour_results['model_names'])):
                            y = neural_plus_behaviour_results['y']
                            neural_latent_var_per_trial = neural_plus_behaviour_results[
                                                              'neural_latent_var_per_trial_per_model'][n_model,
                                                          :]
                            latent_variable_per_bin, actual_p_right_per_bin = \
                                bin_neural_latent_and_p_right(neural_latent_var_per_trial, y,
                                                              num_vals_per_bin=num_vals_per_bin)

                            latent_variable_per_bin_per_model.append(latent_variable_per_bin)
                            actual_p_right_per_bin_per_model.append(actual_p_right_per_bin)

                            # Train set
                            neural_latent_var_per_trial_train_set = neural_plus_behaviour_results[
                                                                        'neural_latent_var_per_trial_train_set_per_model'][
                                                                    n_model,
                                                                    :]
                            latent_variable_per_bin_train_set, actual_p_right_per_bin_train_set = \
                                bin_neural_latent_and_p_right(neural_latent_var_per_trial_train_set, y,
                                                              num_vals_per_bin=num_vals_per_bin)

                            latent_variable_per_bin_train_set_per_model.append(
                                latent_variable_per_bin_train_set
                            )
                            actual_p_right_per_bin_train_set_per_model.append(
                                actual_p_right_per_bin_train_set
                            )

                        neural_plus_behaviour_results_keys = list(neural_plus_behaviour_results.keys())
                        n_plus_b_results_dict = dict()
                        for key in neural_plus_behaviour_results_keys:
                            n_plus_b_results_dict[key] = neural_plus_behaviour_results[key]

                        n_plus_b_results_dict['latent_variable_per_bin_per_model'] = np.stack(
                            latent_variable_per_bin_per_model)
                        n_plus_b_results_dict['actual_p_right_per_bin_per_model'] = np.stack(
                            actual_p_right_per_bin_per_model)

                        n_plus_b_results_dict['latent_variable_per_bin_train_set_per_model'] = np.stack(
                            latent_variable_per_bin_train_set_per_model)
                        n_plus_b_results_dict['actual_p_right_per_bin_train_set_per_model'] = np.stack(
                            actual_p_right_per_bin_train_set_per_model)

                        n_plus_b_results_per_brain_region = n_plus_b_results_dict

                    else:

                        neural_plus_behaviour_latent_var_per_trial = neural_plus_behaviour_results[
                            'neural_latent_var_per_trial']
                        y = neural_plus_behaviour_results['y']
                        n_plus_b_latent_variable_per_bin, n_plus_b_actual_p_right_per_bin = \
                            bin_neural_latent_and_p_right(neural_plus_behaviour_latent_var_per_trial, y,
                                                          num_vals_per_bin=num_vals_per_bin)

                        neural_plus_behaviour_results_keys = list(neural_plus_behaviour_results.keys())
                        n_plus_b_results_dict = dict()
                        for key in neural_plus_behaviour_results_keys:
                            n_plus_b_results_dict[key] = neural_plus_behaviour_results[key]

                        n_plus_b_results_dict['latent_variable_per_bin'] = n_plus_b_latent_variable_per_bin
                        n_plus_b_results_dict['actual_p_right_per_bin'] = n_plus_b_actual_p_right_per_bin

                        n_plus_b_results_per_brain_region = n_plus_b_results_dict

                if include_behaviour_only_model:
                    behaviour_results_fpath = os.path.join(process_params[process]['pink_rig_model_folder'],
                                                           '%s_%s_%s_behaviour_model_results.npz' % (subject, exp_date, exp_num))

                    behaviour_results = np.load(behaviour_results_fpath)
                else:
                    behaviour_results = None

                for model_name in clfs_to_plot:
                    with plt.style.context(splstyle.get_style('nature-reviews')):

                        for set_to_plot in sets_to_plot:
                            fig, axs = plot_multiple_neurometric_models(model_results_per_brain_region,
                                                                        target_brain_regions=None,
                                                                        behaviour_results=behaviour_results,
                                                                        n_plus_b_results_per_brain_region=n_plus_b_results_per_brain_region,
                                                                        include_num_neurons=include_num_neurons,
                                                                        bin_behaviour_only_model=bin_behaviour_only_model,
                                                                        num_vals_per_bin=num_vals_per_bin,
                                                                        include_logistic_curve=include_logistic_curve,
                                                                        model_name=model_name,
                                                                        set_to_plot=set_to_plot,
                                                                        fig=None, axs=None)

                            num_trial = np.shape(model_results_per_brain_region['X_neural'])[0]
                            num_neuron = np.shape(model_results_per_brain_region['X_neural'])[1] - 1

                            fig.suptitle('%s %s %s %s set, %.f trials, %.f neurons' % (subject, exp_date, exp_num, set_to_plot,
                                                                                        num_trial, num_neuron), size=11, y=1.06)

                            if include_stimulus_predictors:
                                fig_name = '%s_%s_%s_multiple_brain_region_plus_stim_neurometric_model_comparison' % (
                                subject, exp_date, exp_num)
                            else:
                                fig_name = '%s_%s_%s_multiple_brain_region_neurometric_model_comparison' % (
                                subject, exp_date, exp_num)

                            if bin_behaviour_only_model:
                                fig_name += '_behave_model_binned'

                            fig_name += '_%s_set' % set_to_plot

                            fig.savefig(os.path.join(process_params[process]['pink_rig_plot_folder'], fig_name), dpi=300, bbox_inches='tight')


        if process == 'plot_all_exp_models_summary':
            print('Running process %s' % process)
            num_vals_per_bin = process_params[process]['num_vals_per_bin']
            include_behaviour_only_model = process_params[process]['include_behaviour_only_model']
            include_stimulus_predictors = process_params[process]['include_stimulus_predictors']
            target_brain_regions = process_params[process]['brain_regions']
            scale_z = process_params[process]['scale_z']

            stim_model_lines_params = []
            neural_only_model_line_params = defaultdict(list)
            n_plus_b_model_line_params = defaultdict(list)

            # Loop through each subject, experiment, to get the line of best fit
            for subject in process_params[process]['subjects']:
                for exp in process_params[process]['exps'][subject]:
                    if process_params[process]['brain_regions'] == 'all':
                        # TODO: this does not work yet
                        available_files = glob.glob(
                            os.path.join(
                                stim_alignment_folder, 'subject_%.f_exp_%.f_*.nc' % (subject, exp)))
                        brain_regions_to_analyse = [x.split('_')[4] for x in available_files]
                    else:
                        brain_regions_to_analyse = process_params[process]['brain_regions']

                    model_results_per_brain_region = dict()
                    if include_stimulus_predictors:
                        n_plus_b_results_per_brain_region = dict()
                    else:
                        n_plus_b_results_per_brain_region = None

                    for target_brain_region in brain_regions_to_analyse:

                        neural_results_fpath_search_result = glob.glob(os.path.join(interim_data_folder,
                                                                                    's%.fe%.f_%s_neural_model_results.npz' %
                                                                                    (
                                                                                    subject, exp, target_brain_region)))

                        if len(neural_results_fpath_search_result) != 1:
                            continue

                        neural_results = np.load(neural_results_fpath_search_result[0])
                        neural_latent_var_per_trial = neural_results['neural_latent_var_per_trial']
                        # neural_prob_per_trial = neural_results['neural_prob_per_trial']
                        y = neural_results['y']
                        latent_variable_per_bin, actual_p_right_per_bin = \
                            bin_neural_latent_and_p_right(neural_latent_var_per_trial, y,
                                                          num_vals_per_bin=num_vals_per_bin)

                        neural_results_keys = list(neural_results.keys())
                        neural_results_dict = dict()
                        for key in neural_results_keys:
                            neural_results_dict[key] = neural_results[key]

                        neural_results_dict['latent_variable_per_bin'] = latent_variable_per_bin
                        neural_results_dict['actual_p_right_per_bin'] = actual_p_right_per_bin

                        model_results_per_brain_region[target_brain_region] = neural_results_dict

                        # Fit linear line with z = Xw on the x axis and log10(p(right) / (1 - p(right))) on the x axis
                        # small_norm_term = 1 / len(model_results_per_brain_region[target_brain_region]['y'])
                        small_norm_term = 1 / num_vals_per_bin
                        # prevents devision by zero (effectively add one trial of each stim)
                        log_odds_per_bin = np.log10(
                            (actual_p_right_per_bin + small_norm_term) / (1 - actual_p_right_per_bin + small_norm_term)
                        )

                        if scale_z:
                            z_max = 1
                            z_min = -1
                            new_z_range = z_max - z_min
                            old_z_range = np.max(latent_variable_per_bin) - np.min(latent_variable_per_bin)

                            latent_variable_per_bin = (
                                ((latent_variable_per_bin - np.min(latent_variable_per_bin)) * new_z_range) / old_z_range
                            ) + z_min

                        try:
                            neural_model_m, neural_model_b = np.polyfit(latent_variable_per_bin, log_odds_per_bin, deg=1)
                        except:
                            pdb.set_trace()

                        neural_only_model_line_params[target_brain_region].append(np.array([neural_model_m, neural_model_b]))

                        if include_stimulus_predictors:
                            neural_plus_behaviour_results_fpath = os.path.join(interim_data_folder,
                                                                               's%.fe%.f_%s_neural_plus_behaviour_model_results.npz' %
                                                                               (subject, exp,
                                                                                target_brain_region))
                            neural_plus_behaviour_results = np.load(neural_plus_behaviour_results_fpath)
                            neural_plus_behaviour_latent_var_per_trial = neural_plus_behaviour_results[
                                'neural_latent_var_per_trial']
                            y = neural_plus_behaviour_results['y']
                            n_plus_b_latent_variable_per_bin, n_plus_b_actual_p_right_per_bin = \
                                bin_neural_latent_and_p_right(neural_plus_behaviour_latent_var_per_trial, y,
                                                              num_vals_per_bin=num_vals_per_bin)

                            neural_plus_behaviour_results_keys = list(neural_plus_behaviour_results.keys())
                            n_plus_b_results_dict = dict()
                            for key in neural_plus_behaviour_results_keys:
                                n_plus_b_results_dict[key] = neural_plus_behaviour_results[key]

                            n_plus_b_results_dict['latent_variable_per_bin'] = n_plus_b_latent_variable_per_bin
                            n_plus_b_results_dict['actual_p_right_per_bin'] = n_plus_b_actual_p_right_per_bin

                            n_plus_b_results_per_brain_region[target_brain_region] = n_plus_b_results_dict

                            # Fit linear line with z = Xw on the x axis and log10(p(right) / (1 - p(right))) on the x axis
                            # small_norm_term = 1 / len(n_plus_b_results_per_brain_region[target_brain_region]['y'])
                            small_norm_term = 1 / num_vals_per_bin
                            # prevents devision by zero (effectively add one trial of each stim)

                            n_plus_b_log_odds_per_bin = np.log10(
                                (n_plus_b_actual_p_right_per_bin + small_norm_term) / (1 - n_plus_b_actual_p_right_per_bin + small_norm_term)
                            )

                            if scale_z:
                                z_max = 1
                                z_min = -1
                                new_z_range = z_max - z_min
                                old_z_range = np.max(n_plus_b_latent_variable_per_bin) - np.min(n_plus_b_latent_variable_per_bin)
                                n_plus_b_latent_variable_per_bin = (
                                                                  ((n_plus_b_latent_variable_per_bin - np.min(
                                                                      n_plus_b_latent_variable_per_bin)) * new_z_range) / old_z_range
                                                          ) + z_min

                            n_plus_b_model_m, n_plus_b_model_b = np.polyfit(n_plus_b_latent_variable_per_bin, n_plus_b_log_odds_per_bin, deg=1)
                            n_plus_b_model_line_params[target_brain_region].append(
                                np.array([n_plus_b_model_m, n_plus_b_model_b]))

                    if include_behaviour_only_model:
                        behaviour_results_fpath = os.path.join(interim_data_folder,
                                                               's%.fe%.f_behaviour_model_results.npz' % (
                                                                   subject, exp))

                        if not os.path.exists(behaviour_results_fpath):
                            print('%s does not exist, skipping' % behaviour_results_fpath)
                            continue

                        behaviour_results = np.load(behaviour_results_fpath)
                        model_latent_per_trial_type = behaviour_results['model_latent_per_trial_type'],
                        actual_p_right_per_trial_type = behaviour_results['actual_p_right_per_trial_type']

                        # Fit linear line with z = Xw on the x axis and log10(p(right) / (1 - p(right))) on the x axis
                        # small_norm_term = 1 / len(behaviour_results['y']) # prevents devision by zero (effectively add one trial of each stim)
                        small_norm_term = 1 / behaviour_results['trial_num_per_trial_type']
                        bhvr_model_log_odds_per_bin = np.log10(
                            (actual_p_right_per_trial_type + small_norm_term)
                             / (1 - actual_p_right_per_trial_type + small_norm_term))

                        if scale_z:
                            z_max = 1
                            z_min = -1
                            new_z_range = z_max - z_min
                            old_z_range = np.max(model_latent_per_trial_type) - np.min(
                                model_latent_per_trial_type)
                            model_latent_per_trial_type = (
                                                                       ((model_latent_per_trial_type - np.min(
                                                                           model_latent_per_trial_type)) * new_z_range) / old_z_range
                                                               ) + z_min

                        bhvr_model_m, bhvr_model_b = np.polyfit(model_latent_per_trial_type.flatten(),
                                                                        bhvr_model_log_odds_per_bin, deg=1)
                        stim_model_lines_params.append(
                            np.array([bhvr_model_m, bhvr_model_b])
                        )
                    else:
                        behaviour_results = None

            # Plot results
            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, axs = plot_all_exp_models_summary(stim_model_lines_params, neural_only_model_line_params,
                                                   n_plus_b_model_line_params, target_brain_regions,
                                                   zmin=-1, zmax=1)

                fig_name = 'all_exp_line_of_best_fit_log_odds'
                fig_path = os.path.join(fig_folder, fig_name)
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')

                print('Saved figure to %s' % fig_path)
        if process == 'get_model_performance':

            print('Running process %s' % process)
            num_vals_per_bin = process_params[process]['num_vals_per_bin']
            include_behaviour_only_model = process_params[process]['include_behaviour_only_model']
            include_stimulus_predictors = process_params[process]['include_stimulus_predictors']
            target_brain_regions = process_params[process]['brain_regions']
            eps = process_params[process]['eps']
            clf_names = process_params[process]['clf_names']
            model_performance_dict = defaultdict(list)


            # Loop through each subject, experiment, to get the line of best fit
            for subject in process_params[process]['subjects']:
                for exp in process_params[process]['exps'][subject]:
                    if process_params[process]['brain_regions'] == 'all':
                        # TODO: this does not work yet
                        available_files = glob.glob(
                            os.path.join(
                                stim_alignment_folder, 'subject_%.f_exp_%.f_*.nc' % (subject, exp)))
                        brain_regions_to_analyse = [x.split('_')[4] for x in available_files]
                    else:
                        brain_regions_to_analyse = process_params[process]['brain_regions']

                    model_results_per_brain_region = dict()
                    if include_stimulus_predictors:
                        n_plus_b_results_per_brain_region = dict()
                    else:
                        n_plus_b_results_per_brain_region = None

                    for target_brain_region in brain_regions_to_analyse:

                        for clf_name in clf_names:
                            model_performance_dict, model_results_per_brain_region, n_plus_b_results_per_brain_region \
                                = add_to_model_performance_dict(model_performance_dict, model_results_per_brain_region,
                                                                n_plus_b_results_per_brain_region,
                                                               interim_data_folder=interim_data_folder,
                                                               subject=subject, exp=exp,
                                                               target_brain_region=target_brain_region,
                                                               include_stimulus_predictors=include_stimulus_predictors,
                                                               include_behaviour_only_model=include_behaviour_only_model,
                                                               num_vals_per_bin=num_vals_per_bin, eps=eps,
                                                               clf_name=clf_name)

                    else:
                        behaviour_results = None

            model_performance_df = pd.DataFrame.from_dict(model_performance_dict)
            model_performance_df.to_csv(os.path.join(interim_data_folder, 'model_fitting_comparison.csv'))

        if process == 'plot_model_performance':
            print('Plotting model performance')

            brain_regions = process_params[process]['brain_regions']
            model_metric = process_params[process]['model_metric']
            single_exp_alpha = process_params[process]['single_exp_alpha']
            plot_brain_region_mean = process_params[process]['plot_brain_region_mean']

            model_performance_df = pd.read_csv(os.path.join(interim_data_folder, 'model_fitting_comparison.csv'))

            for clf_name in process_params[process]['clf_names']:

                subset_model_performance_df = model_performance_df.loc[
                    model_performance_df['clf'] == clf_name
                ]

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, axs = plot_model_performance(subset_model_performance_df, brain_regions=brain_regions,
                                                      model_metric=model_metric,
                                                      single_exp_alpha=single_exp_alpha,
                                                      plot_brain_region_mean=plot_brain_region_mean)
                    fig_name = 'all_exp_model_comparison_scatter_%s_%s' % (model_metric, clf_name)
                    fig_full_path = os.path.join(fig_folder, fig_name)
                    fig.savefig(fig_full_path, dpi=300, bbox_inches='tight')
        if process == 'plot_decoding_over_days_pinkrigs':

            # First get unique subjects
            alignment_ds_paths = glob.glob(os.path.join(process_params[process]['pink_rig_data_folder'], '*.nc'))
            choice_var_name = 'timeline_choiceMoveDir'
            aud_diff_name = 'audDiff'
            vis_diff_name = 'visDiff'
            subject_list = []
            for path in alignment_ds_paths:
                fname = os.path.basename(path)
                subject = fname.split('_')[0]
                subject_list.append(subject)


            for subject in np.unique(subject_list):

                subject_alignment_paths = glob.glob(os.path.join(process_params[process]['pink_rig_data_folder'], '%s*.nc' % subject))
                subject_alignment_paths = np.sort(subject_alignment_paths)

                behaviour_performance_list = []
                neural_decoding_performance_list = []
                stim_decoding_performance_list = []
                num_neuron_list = []
                xticklabel_list = []

                for alignment_path in subject_alignment_paths:

                    fname = os.path.basename(alignment_path)
                    exp_date = fname.split('_')[1]
                    exp_num = fname.split('_')[2].split('.')[0]
                    alignment_ds = xr.open_dataset(alignment_path)



                    # Get model result to get decoding performance
                    behaviour_model_path = os.path.join(process_params[process]['pink_rig_model_folder'], '%s_%s_%s_behaviour_model_results.npz' % (subject, exp_date, exp_num))
                    neural_model_path = os.path.join(process_params[process]['pink_rig_model_folder'], '%s_%s_%s_neural_model_results.npz' % (subject, exp_date, exp_num))

                    try:
                        behaviour_results = np.load(behaviour_model_path)
                        neural_results = np.load(neural_model_path)
                    except:
                        continue

                    # number of neurons
                    num_neuron = len(alignment_ds.Cell)
                    num_neuron_list.append(num_neuron)

                    # Get behaviour performance
                    num_trial = len(alignment_ds.Trial)
                    prop_correct = np.sum(alignment_ds['stim_correctResponse'].values == alignment_ds[
                        'timeline_choiceMoveDir'].values) / num_trial
                    behaviour_performance_list.append(prop_correct)

                    clf_name = 'LR-l1'
                    num_vals_per_bin = 25
                    eps = 0.00001
                    # Get neural model accuracy
                    if clf_name is None:
                        neural_latent_var_per_trial = neural_results['neural_latent_var_per_trial']
                    else:
                        model_idx = np.where(neural_results['model_names'] == clf_name)[0][0]
                        neural_latent_var_per_trial = neural_results['neural_latent_var_per_trial_per_model'][model_idx,
                                                      :]

                    # neural_prob_per_trial = neural_results['neural_prob_per_trial']
                    y = neural_results['y']
                    latent_variable_per_bin, actual_p_right_per_bin = \
                        bin_neural_latent_and_p_right(neural_latent_var_per_trial, y,
                                                      num_vals_per_bin=num_vals_per_bin)

                    neural_results_keys = list(neural_results.keys())
                    neural_results_dict = dict()
                    for key in neural_results_keys:
                        neural_results_dict[key] = neural_results[key]

                    neural_results_dict['latent_variable_per_bin'] = latent_variable_per_bin
                    neural_results_dict['actual_p_right_per_bin'] = actual_p_right_per_bin

                    if clf_name is None:
                        y_pred_prob = neural_results_dict['neural_prob_per_trial']
                    else:
                        model_idx = np.where(neural_results['model_names'] == clf_name)[0][0]
                        y_pred_prob = neural_results_dict['neural_prob_per_trial_per_model'][model_idx, :]

                    y_actual = neural_results_dict['y']
                    y_pred_prob = [max(eps, min(1 - eps, x)) for x in y_pred_prob]
                    y_pred_prob = np.array(y_pred_prob)
                    neural_model_log_loss = \
                        -np.mean(y_actual * np.log2(y_pred_prob) + (1 - y_actual) * np.log2(1 - y_pred_prob))

                    y_pred_binary = (y_pred_prob > 0.5).astype(float)
                    neural_model_accuracy = np.mean(y_pred_binary == y_actual)

                    neural_decoding_performance_list.append(neural_model_accuracy)


                    # Behaviour model accuracy
                    model_latent_per_trial_type = behaviour_results['model_latent_per_trial_type'],
                    actual_p_right_per_trial_type = behaviour_results['actual_p_right_per_trial_type']

                    if clf_name is None:
                        y_pred_prob = behaviour_results['prob_per_trial']
                    else:
                        model_idx = np.where(behaviour_results['model_names'] == clf_name)[0][0]
                        y_pred_prob = behaviour_results['prob_per_trial_per_model'][model_idx, :]

                    y_actual = behaviour_results['y']
                    y_pred_prob = [max(eps, min(1 - eps, x)) for x in y_pred_prob]
                    y_pred_prob = np.array(y_pred_prob)
                    y_pred_binary = (y_pred_prob > 0.5).astype(float)
                    behaviour_model_accuracy = np.mean(y_pred_binary == y_actual)

                    stim_decoding_performance_list.append(behaviour_model_accuracy)

                    xticklabel_list.append('%s %s' % (exp_date, exp_num))



                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, axs = plt.subplots(1, 3, sharex=True)
                    fig.set_size_inches(9, 3)

                    exp_idx = np.arange(len(behaviour_performance_list))

                    axs[0].plot(exp_idx, behaviour_performance_list)
                    axs[1].plot(exp_idx, stim_decoding_performance_list, label='Stim')
                    axs[1].plot(exp_idx, neural_decoding_performance_list, label='Neural')
                    axs[2].plot(exp_idx, num_neuron_list)

                    axs[1].legend()
                    axs[0].set_ylim([0, 1])
                    axs[1].set_ylim([0, 1])


                    axs[0].set_ylabel('Behaviour performance', size=11)
                    axs[1].set_ylabel('Decoding performance', size=11)
                    axs[2].set_ylabel('Number of neurons', size=11)

                    axs[0].set_xticks(exp_idx)
                    axs[0].set_xticklabels(xticklabel_list, rotation=45, size=9)
                    axs[1].set_xticks(exp_idx)
                    axs[1].set_xticklabels(xticklabel_list, rotation=45, size=9)
                    axs[2].set_xticks(exp_idx)
                    axs[2].set_xticklabels(xticklabel_list, rotation=45, size=9)

                    fig.tight_layout()
                    fig_name = '%s_performance_over_days' % subject
                    fig_folder = '/Volumes/Partition 1/data/interim/neuro-psychometric-model/plotsPinkRigs/overDays'
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

        if process == 'fit_neural_model_forward_feature_selection':

            min_least_common_choice = process_params[process]['min_least_common_choice']
            min_neurons = process_params[process]['min_neurons']
            clfs_to_fit = process_params[process]['clfs_to_fit']
            cv_random_seed = process_params[process]['cv_random_seed']
            max_iter = process_params[process]['max_iter']

            clf = sklinear.LogisticRegression(fit_intercept=False, penalty='l1', max_iter=max_iter,
                                                     solver='liblinear')

            for subject in process_params[process]['subjects']:
                for exp in process_params[process]['exps'][subject]:
                    if process_params[process]['brain_regions'] == 'all':
                        available_files = glob.glob(
                            os.path.join(
                                stim_alignment_folder, 'subject_%.f_exp_%.f_*.nc' % (subject, exp)))
                        brain_regions_to_analyse = [x.split('_')[4] for x in available_files]
                    else:
                        brain_regions_to_analyse = process_params[process]['brain_regions']

                    for target_brain_region in brain_regions_to_analyse:

                        alignment_ds = pephys.load_subject_exp_alignment_ds(alignment_folder=stim_alignment_folder,
                                                                            subject_num=subject, exp_num=exp,
                                                                            target_brain_region=target_brain_region,
                                                                            aligned_event='choiceInitTime',
                                                                            alignment_file_ext='.nc')
                        if alignment_ds is None:
                            continue
                        num_cell = len(alignment_ds.Cell)
                        if num_cell == 0:
                            continue

                        if min_neurons is not None:
                            if num_cell < min_neurons:
                                print('%.f neurons found, which is fewer than specified %.f, skipping...'
                                      % (num_cell, min_neurons))
                                continue

                        include_stimulus_predictors = process_params[process]['include_stimulus_predictors']
                        print('Fitting neural model to subject %.f experiment %.f in %s' % (
                            subject, exp, target_brain_region))
                        X, y = get_behaviour_X_and_y(alignment_ds)

                        unique_y, unique_counts = np.unique(y, return_counts=True)
                        if (len(np.unique(y)) == 1) or (np.min(unique_counts) < min_least_common_choice):
                            continue

                        X_neural = get_X_neural(alignment_ds, mean_fr_window=[-0.15, 0])

                        num_features_to_select = np.shape(X_neural[:, 1:])[1]
                        _, accuracy_per_best_n_feature = do_forward_feature_selection(X_neural[:, 1:], y, clf,
                                                                                                         selection_criteria='accuracy',
                                                                                                         num_features_to_select=num_features_to_select)
                        _, accuracy_per_random_n_feature = do_forward_feature_selection(X_neural[:, 1:], y, clf,
                                                                                                           selection_criteria='random',
                                                                                                           num_features_to_select=num_features_to_select)
        if process == 'plot_conditional_psychometric_curves':

            df_path = process_params[process]['behaviour_df_fpath']
            fig_folder = process_params[process]['fig_folder']
            subplot_grid = process_params[process]['subplot_grid']
            fig_size = process_params[process]['fig_size']
            exclude_invalid_trials = process_params[process]['exclude_invalid_trials']
            exclude_laser_experiments = process_params[process]['exclude_laser_experiments']
            behaviour_df = pd.read_pickle(df_path)
            behaviour_df['noGo'] = np.isnan(behaviour_df['reactionTime'])

            exclude_no_go = process_params[process]['exclude_no_go']
            subset_conditions = process_params[process]['subset_conditions']

            if exclude_laser_experiments:
                exp_laser_sum = behaviour_df.groupby('expRef').agg('sum')['laserPower']
                subset_exp = exp_laser_sum.loc[exp_laser_sum == 0]
                behaviour_df = behaviour_df.loc[behaviour_df['expRef'].isin(subset_exp.index)]


            for subject in np.unique(behaviour_df['subjectId']):

                print('Getting psychometric curves for %s' % subject)

                mouse_df = behaviour_df.loc[
                    behaviour_df['subjectId'] == subject
                ]


                with plt.style.context(splstyle.get_style('nature-reviews')):
                    all_cond_fig, all_cond_ax = plt.subplots(subplot_grid[0], subplot_grid[1], sharex=True, sharey=True)
                    all_cond_fig.set_size_inches(fig_size[0], fig_size[1])

                for n_condition, s_condition in enumerate(subset_conditions):

                    subset_df = get_subset_behaviour_df(mouse_df, s_condition=s_condition, exclude_no_go=True,
                                                        exclude_invalid_trials=exclude_invalid_trials)

                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, ax = plt.subplots()
                        fig.set_size_inches(5, 4)
                        fig, ax = vizbehaviour.plot_psychometric(behaviour_data=subset_df,
                          include_legend=False, fig=fig, ax=ax,
                          custom_aud_conds=[-60, 0, 60],
                          aud_cond_labels={-60: 'left', 0:'center', 60: 'right', np.inf: 'off'},
                          aud_cond_colors={-60: 'blue', 0: 'gray', 60: 'red'})

                        ax.set_ylim([-0.025, 1.025])
                        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                        ax.spines['left'].set_bounds([0, 1])

                        if exclude_invalid_trials:
                            exclude_invalid_trials_str = '_exclude_invalid'
                        else:
                            exclude_invalid_trials_str = ''

                        if exclude_no_go:
                            fig_name = '%s_%s_psychometric_exclude_no_go%s' % (subject, s_condition, exclude_invalid_trials_str)
                        else:
                            fig_name = '%s_%s_psychometric%s' % (subject, s_condition, exclude_invalid_trials_str)
                        fig.savefig(os.path.join(fig_folder, fig_name), bbox_inches='tight')
                        plt.close(fig)

                        all_cond_fig, all_cond_ax.flatten()[n_condition] = vizbehaviour.plot_psychometric(
                          behaviour_data=subset_df,
                          include_legend=False, fig=all_cond_fig, ax=all_cond_ax.flatten()[n_condition],
                          custom_aud_conds=[-60, 0, 60],
                          aud_cond_labels={-60: 'left', 0:'center', 60: 'right', np.inf: 'off'},
                          aud_cond_colors={-60: 'blue', 0: 'gray', 60: 'red'})

                        all_cond_ax.flatten()[n_condition].set_xlabel('')
                        all_cond_ax.flatten()[n_condition].set_ylabel('')
                        all_cond_ax.flatten()[n_condition].set_ylim([-0.025, 1.025])
                        all_cond_ax.flatten()[n_condition].set_title(s_condition, size=8)

                if exclude_no_go:
                    all_cond_fig_name = '%s_all_subset_conds_psychometric_exclude_no_go%s' % (subject, exclude_invalid_trials_str)
                else:
                    all_cond_fig_name = '%s_all_subset_conds_psychometric%s' % (subject, exclude_invalid_trials_str)

                all_cond_fig.suptitle('%s' % subject, size=11, y=0.94)
                all_cond_fig.text(0.5, 0.04, 'Visual contrast', size=11, ha='center')
                all_cond_fig.text(0.04, 0.5, 'P(right)', size=11, va='center', rotation=90)
                all_cond_fig.savefig(os.path.join(fig_folder, all_cond_fig_name), bbox_inches='tight')

        if process == 'fit_subset_cond_psychometric_models':

            df_path = process_params[process]['behaviour_df_fpath']
            fig_folder = process_params[process]['fig_folder']
            exclude_laser_experiments = process_params[process]['exclude_laser_experiments']
            behaviour_df = pd.read_pickle(df_path)
            exclude_no_go = process_params[process]['exclude_no_go']
            subset_conditions = process_params[process]['subset_conditions']
            choice_var_name = 'responseRecorded'
            cv_random_seed = 1
            num_resamples = 10
            num_folds = 2
            max_iter = 1000

            behaviour_df['noGo'] = np.isnan(behaviour_df['reactionTime'])

            if exclude_laser_experiments:
                exp_laser_sum = behaviour_df.groupby('expRef').agg('sum')['laserPower']
                subset_exp = exp_laser_sum.loc[exp_laser_sum == 0]
                behaviour_df = behaviour_df.loc[behaviour_df['expRef'].isin(subset_exp.index)]

            for subject in np.unique(behaviour_df['subjectId']):
                print('Fitting psychometric curves for %s' % subject)

                mouse_df = behaviour_df.loc[
                    behaviour_df['subjectId'] == subject
                    ]

                subset_conditions = [x for x in subset_conditions if x != 'all']  # all can be removed

                if exclude_no_go:
                    full_df = mouse_df.loc[
                        mouse_df['noGo'] == False
                        ]
                else:
                    full_df = mouse_df

                X_full, y_full = get_behaviour_X_and_y_from_df(full_df, choice_var_name=choice_var_name)

                model_losses_per_s_condition = defaultdict(dict)

                for n_condition, s_condition in enumerate(subset_conditions):

                    subset_df = get_subset_behaviour_df(mouse_df, s_condition=s_condition, exclude_no_go=True)

                    # get train and test indices
                    subset_clf = sklinear.LogisticRegression(fit_intercept=False, penalty='none', max_iter=max_iter)
                    full_clf = sklinear.LogisticRegression(fit_intercept=False, penalty='none', max_iter=max_iter)

                    X_subset, y_subset = get_behaviour_X_and_y_from_df(subset_df, choice_var_name=choice_var_name)

                    train_full_test_full_loss = np.zeros((num_resamples, num_folds)) + np.nan
                    train_full_test_subset_loss = np.zeros((num_resamples, num_folds)) + np.nan
                    train_subset_test_subset_loss = np.zeros((num_resamples, num_folds)) + np.nan

                    for resample_n in np.arange(num_resamples):
                        resample_idx = np.random.choice(np.arange(len(y_full)), len(y_subset), replace=False)
                        X_full_resampled = X_full[resample_idx, :]
                        y_full_resampled = y_full[resample_idx]

                        cv_splitter = skselect.KFold(n_splits=num_folds, random_state=cv_random_seed, shuffle=True)

                        for n_fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X_subset)):
                            X_subset_train, X_subset_test = X_subset[train_idx], X_subset[test_idx]
                            y_subset_train, y_subset_test = y_subset[train_idx], y_subset[test_idx]

                            X_full_resampled_train, X_full_resampled_test = X_full_resampled[train_idx], X_full_resampled[test_idx]
                            y_full_resampled_train, y_full_resampled_test = y_full_resampled[train_idx], y_full_resampled[test_idx]

                            # Fit subset condition model
                            subset_clf.fit(X_subset_train, y_subset_train)

                            # Fit full_clf model to all conditions
                            full_clf.fit(X_full_resampled_train, y_full_resampled_train)

                            # Test full_clf model on all conditions
                            full_clf_y_full_prob_hat = full_clf.predict_proba(X_full_resampled_test)[:, 1]

                            # Test both models on subset condition
                            full_clf_y_subset_prob_hat = full_clf.predict_proba(X_subset_test)[:, 1]
                            subset_clf_y_subset_prob_hat = subset_clf.predict_proba(X_subset_test)[:, 1]

                            # Convert probabilities to log loss
                            train_full_test_full_loss[resample_n, n_fold] = -np.mean(y_full_resampled_test * np.log2(full_clf_y_full_prob_hat) + (1 - y_full_resampled_test) * np.log2(1 - full_clf_y_full_prob_hat))
                            train_full_test_subset_loss[resample_n, n_fold] = -np.mean(y_subset_test * np.log2(full_clf_y_subset_prob_hat) + (1 - y_subset_test) * np.log2(1 - full_clf_y_subset_prob_hat))
                            train_subset_test_subset_loss[resample_n, n_fold] = -np.mean(y_subset_test * np.log2(subset_clf_y_subset_prob_hat) + (1 - y_subset_test) * np.log2(1 - subset_clf_y_subset_prob_hat))

                    # Mean across the cross-validation folds
                    model_losses_per_s_condition[s_condition]['train_full_test_full_loss'] = np.mean(train_full_test_full_loss, axis=1)
                    model_losses_per_s_condition[s_condition]['train_full_test_subset_loss'] = np.mean(train_full_test_subset_loss, axis=1)
                    model_losses_per_s_condition[s_condition]['train_subset_test_subset_loss'] = np.mean(train_subset_test_subset_loss, axis=1)

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(7, 4)

                    for n_condition, s_condition in enumerate(subset_conditions):

                        x_middle_loc = n_condition + 1
                        x_locs = np.random.normal(x_middle_loc, 0.1, len(model_losses_per_s_condition[s_condition]['train_full_test_subset_loss']))

                        # include fit full test full
                        # x0_loc = np.random.normal(0, 0.1, len(model_losses_per_s_condition[s_condition]['train_full_test_subset_loss']))
                        # ax.scatter(x0_loc, model_losses_per_s_condition[s_condition]['train_full_test_full_loss'], color='gray')
                        ax.scatter(x_locs, model_losses_per_s_condition[s_condition]['train_full_test_subset_loss'], color='gray')
                        ax.scatter(x_middle_loc, model_losses_per_s_condition[s_condition]['train_subset_test_subset_loss'][0], color='red')

                    # custom legend
                    legend_elements = [mpl.lines.Line2D([0], [0], marker='o', color='none', label='All train set -> Condition test set',
                                              markerfacecolor='gray', markersize=12, lw=0, markeredgecolor='none'),
                                       mpl.lines.Line2D([0], [0], marker='o', color='none', label='Condition train set -> Condition test set',
                                                        markerfacecolor='red', markersize=12, lw=0, markeredgecolor='none')
                                       ]

                    # Create the figure
                    ax.legend(handles=legend_elements, bbox_to_anchor=(1.04, 0.5))
                    ax.set_xticks(np.arange(1, len(subset_conditions)+1))
                    ax.set_xticklabels(subset_conditions, size=9, rotation=30)
                    ax.set_ylabel('Log_2 loss', size=11)
                    ax.set_xlabel('Condition', size=11)
                    ax.set_title('%s' % subject, size=11)

                    if exclude_no_go:
                        fig_name = '%s_all_subset_conds_psychometric_fit_comparison_exclude_no_go' % subject
                    else:
                        fig_name = '%s_all_subset_conds_psychometric_fit_comparison' % subject
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()



