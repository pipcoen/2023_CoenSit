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
from matplotlib.colors import LinearSegmentedColormap


def scale_model_weights(model_best_params_weight_sorted):
    """
    Scale model weights from -1 to 1 (just to color them using cmap)
    Parameters
    ----------
    model_best_params_weight_sorted

    Returns
    -------

    """

    positive_model_best_params_weight_sorted = model_best_params_weight_sorted[
        model_best_params_weight_sorted >= 0
        ]

    negative_model_best_params_weight_sorted = model_best_params_weight_sorted[
        model_best_params_weight_sorted < 0
        ]

    positive_model_best_params_weight_sorted_scaled = (positive_model_best_params_weight_sorted - np.min(
        positive_model_best_params_weight_sorted)) \
                                                      / (np.max(positive_model_best_params_weight_sorted) - np.min(
        positive_model_best_params_weight_sorted))

    negative_model_best_params_weight_sorted_abs = np.abs(negative_model_best_params_weight_sorted)
    negative_model_best_params_weight_sorted_scaled = (negative_model_best_params_weight_sorted_abs - np.min(
        negative_model_best_params_weight_sorted_abs)) \
                                                      / (np.max(negative_model_best_params_weight_sorted_abs) - np.min(
        negative_model_best_params_weight_sorted_abs))

    negative_model_best_params_weight_sorted_scaled = -negative_model_best_params_weight_sorted_scaled

    model_best_params_weight_sorted_scaled = np.concatenate([
        negative_model_best_params_weight_sorted_scaled,
        positive_model_best_params_weight_sorted_scaled
    ])

    return model_best_params_weight_sorted_scaled


def plot_model_single_trial_scheme(model_result, trials_to_plot=[1],
                                   fig_ext='.pdf', stim_cond_name='arv0', fig_folder=None,
                                   plot_every_n_neuron=5):
    """

    Parameters
    ----------
    model_result
    trials_to_plot
    fig_ext
    stim_cond_name
    fig_folder
    plot_every_n_neuron

    Returns
    -------

    """

    pre_preprocessed_alignment_ds_test = model_result['pre_preprocessed_alignment_ds_test']
    y_test_pred_da = model_result['y_test_pred_da']

    for trial_idx in trials_to_plot:

        fig_name = '5_model_single_trial_example_w_stimOnLine_trial_%s_%.f' % (stim_cond_name, trial_idx)

        if stim_cond_name == 'arv0':
            example_trial_ds = pre_preprocessed_alignment_ds_test.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == 60) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == 0), drop=True
            ).isel(Trial=trial_idx)

            example_trial_output_ds = y_test_pred_da.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == 60) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == 0), drop=True
            ).isel(Trial=trial_idx)

        elif stim_cond_name == 'a0v0p8':
            example_trial_ds = pre_preprocessed_alignment_ds_test.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == np.inf) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == 0.8), drop=True
            ).isel(Trial=trial_idx)

            example_trial_output_ds = y_test_pred_da.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == np.inf) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == 0.8), drop=True
            ).isel(Trial=trial_idx)
        elif stim_cond_name == 'arv0p8':
            example_trial_ds = pre_preprocessed_alignment_ds_test.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == 60) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == 0.8), drop=True
            ).isel(Trial=trial_idx)

            example_trial_output_ds = y_test_pred_da.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == 60) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == 0.8), drop=True
            ).isel(Trial=trial_idx)

        elif stim_cond_name == 'a0v0p4':
            example_trial_ds = pre_preprocessed_alignment_ds_test.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == np.inf) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == 0.4), drop=True
            ).isel(Trial=trial_idx)

            example_trial_output_ds = y_test_pred_da.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == np.inf) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == 0.4), drop=True
            ).isel(Trial=trial_idx)
        elif stim_cond_name == 'a0v0p2':
            example_trial_ds = pre_preprocessed_alignment_ds_test.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == np.inf) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == 0.2), drop=True
            ).isel(Trial=trial_idx)

            example_trial_output_ds = y_test_pred_da.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == np.inf) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == 0.2), drop=True
            ).isel(Trial=trial_idx)
        elif stim_cond_name == 'a0v0p1':
            example_trial_ds = pre_preprocessed_alignment_ds_test.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == np.inf) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == 0.1), drop=True
            ).isel(Trial=trial_idx)

            example_trial_output_ds = y_test_pred_da.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == np.inf) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == 0.1), drop=True
            ).isel(Trial=trial_idx)
        elif stim_cond_name == 'alv0p8':
            example_trial_ds = pre_preprocessed_alignment_ds_test.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == -60) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == 0.8), drop=True
            ).isel(Trial=trial_idx)

            example_trial_output_ds = y_test_pred_da.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == -60) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == 0.8), drop=True
            ).isel(Trial=trial_idx)
        elif stim_cond_name == 'arv-0p8':
            example_trial_ds = pre_preprocessed_alignment_ds_test.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == 60) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == -0.8), drop=True
            ).isel(Trial=trial_idx)

            example_trial_output_ds = y_test_pred_da.where(
                (pre_preprocessed_alignment_ds_test['audDiff'] == 60) &
                (pre_preprocessed_alignment_ds_test['visDiff'] == -0.8), drop=True
            ).isel(Trial=trial_idx)
        elif stim_cond_name == 'all':
            example_trial_ds = pre_preprocessed_alignment_ds_test.isel(Trial=trial_idx)
            example_trial_output_ds = y_test_pred_da.where(
                pre_preprocessed_alignment_ds_test['audDiff'].isin([-60, 60, np.inf]), drop=True).isel(Trial=trial_idx)

        # load model weights
        model_params = np.stack(model_result['param_history'])
        epoch_min_loss = np.argmin(model_result['test_loss'])
        model_best_params = model_params[epoch_min_loss, :]

        # TODO: is this 140 thing valid anymore??? Ah actually ths answer seem to be yes.
        model_best_params_weights = model_best_params[140:]
        model_best_params_weight_sorted = np.sort(model_best_params_weights)
        model_best_params_weight_sort_idx = np.argsort(model_best_params_weights)

        # TODO: this sacling is a bit weird, if you look at the neurons then it should be about half/half.
        # I think I need to subset the positive and negative ones, then scale them individuall to be between -1 and 1
        # scaled model weights to between -1 and 1 (to plot alpha values)
        """
        model_best_params_weight_sorted_scaled = 2 * (
                    model_best_params_weight_sorted - np.min(model_best_params_weight_sorted)) \
                                                 / (np.max(model_best_params_weight_sorted) - np.min(
            model_best_params_weight_sorted)) - 1
        """
        model_best_params_weight_sorted_scaled = scale_model_weights(model_best_params_weight_sorted)

        black_to_red_cmap = LinearSegmentedColormap.from_list(
            'blackToRed', ['Black', 'Red'], N=100)
        black_to_blue_cmap = LinearSegmentedColormap.from_list(
            'blackToBlue', ['Black', 'Blue'], N=100)

        red_to_black_to_blue = True
        cmap_idx_offset = 30
        num_neuron = len(example_trial_ds.Cell)

        include_stim_on_line = True

        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig = plt.figure()
            fig.set_size_inches(8, 4)
            gs = fig.add_gridspec(4, 10)

            ax1 = fig.add_subplot(gs[0:2, 1:5])

            example_trial_raster_sorted = example_trial_ds['firing_rate'].isel(
                Cell=model_best_params_weight_sort_idx)

            for n_cell, cell in enumerate(example_trial_raster_sorted.Cell.values):

                cell_fr = example_trial_raster_sorted.sel(Cell=cell)
                spike_times_idx = np.where(cell_fr > 0)[0]
                spike_times_sec = example_trial_raster_sorted.PeriEventTime.values[spike_times_idx]
                cell_y_loc = np.repeat(n_cell, len(spike_times_idx))

                cell_scaled_weight = model_best_params_weight_sorted_scaled[n_cell]

                if cell_scaled_weight > 0:
                    if red_to_black_to_blue:
                        cmap_idx = int(np.abs(cell_scaled_weight) * 100) + cmap_idx_offset
                        ax1.scatter(spike_times_sec, cell_y_loc, s=3, color=black_to_red_cmap(cmap_idx), alpha=1)
                    else:
                        ax1.scatter(spike_times_sec, cell_y_loc, s=3, color='red', alpha=np.abs(cell_scaled_weight))
                elif cell_scaled_weight < 0:
                    if red_to_black_to_blue:
                        cmap_idx = int(np.abs(cell_scaled_weight) * 100) + cmap_idx_offset
                        ax1.scatter(spike_times_sec, cell_y_loc, s=3, color=black_to_blue_cmap(cmap_idx), alpha=1)
                    else:
                        ax1.scatter(spike_times_sec, cell_y_loc, s=3, color='blue', alpha=np.abs(cell_scaled_weight))
                # cell_fired = np.where()
                # ax.scatter()

            ax1.set_ylabel('Neuron weight', size=12)
            ax1.spines['bottom'].set_visible(False)
            ax1.set_xticks([])

            # y-axis of ax1
            # ax1.spines['left'].set_bounds(0, num_neuron)
            # ax1.spines['left'].set_visible(True)
            ax1.set_yticks([0, 70, num_neuron])
            ax1.set_yticklabels([-0.03, 0, 0.03])

            # ax1.imshow(, cmap='binary', aspect='auto')

            ax2 = fig.add_subplot(gs[2:, 1:5])
            ax2.plot(example_trial_output_ds.PeriEventTime, example_trial_output_ds, color='black')
            ax2.axhline(1, linestyle='--', color='red')
            ax2.axhline(-1, linestyle='--', color='blue')

            # text on decision thresholds to indicate choice
            ax2.text(0.17, 1.2, 'Choose right', size=8, color='red')
            ax2.text(0.17, -1.6, 'Choose left', size=8, color='blue')

            ax2.set_ylim([-3.5, 3.5])

            ax2.set_ylabel('Model output', size=12)
            ax2.set_xlabel('Peri-stimulus time (s)', size=12)
            ax2.set_xticks([-0.1, 0, 0.1, 0.2, 0.3])
            # ax2.spines['bottom'].set_bounds(-0.1, 0.3)
            ax2.get_shared_x_axes().join(ax1, ax2)

            ax3 = fig.add_subplot(gs[0:2, 5:7])

            linewidth = 1
            decision_variable_dot_y_loc = 62

            cell_scale_weight_threshold = 0  # classify between left and right neurons

            for n_cell, cell in enumerate(example_trial_raster_sorted.Cell.values):
                cell_scaled_weight = model_best_params_weight_sorted_scaled[n_cell]

                if n_cell % plot_every_n_neuron == 0:
                    if cell_scaled_weight > cell_scale_weight_threshold:
                        if red_to_black_to_blue:
                            cmap_idx = int(np.abs(cell_scaled_weight) * 100) + cmap_idx_offset
                            ax3.plot([0, 50], [n_cell, decision_variable_dot_y_loc], color=black_to_red_cmap(cmap_idx),
                                     linewidth=linewidth, alpha=1)
                        else:
                            ax3.plot([0, 50], [n_cell, decision_variable_dot_y_loc], color='red', linewidth=linewidth,
                                     alpha=np.abs(cell_scaled_weight))
                    elif cell_scaled_weight < cell_scale_weight_threshold:
                        if red_to_black_to_blue:
                            cmap_idx = int(np.abs(cell_scaled_weight) * 100) + cmap_idx_offset
                            ax3.plot([0, 50], [n_cell, decision_variable_dot_y_loc], color=black_to_blue_cmap(cmap_idx),
                                     linewidth=linewidth, alpha=1)
                        else:
                            ax3.plot([0, 50], [n_cell, decision_variable_dot_y_loc], color='blue', linewidth=linewidth,
                                     alpha=np.abs(cell_scaled_weight))

            ax3.set_yticks([])
            ax3.spines['left'].set_visible(False)
            ax3.set_xticks([])
            ax3.spines['bottom'].set_visible(False)

            ax3.text(50, 75, r'$w > 0$', size=12, color='red')
            ax3.text(50, 25, r'$w < 0$', size=12, color='blue')
            ax3.scatter(50, decision_variable_dot_y_loc, color='black', zorder=10)

            # add in the equation
            ax3.text(58, 50, r'$d_t$')

            # recurring loop

            ax3.set_xlim([0, 120])
            ax3.set_ylim([0, n_cell])

            # ax4 = fig.add_subplot(gs[1:, 6:])

            if include_stim_on_line:
                ax2.axvline(linestyle='-', color='gray', alpha=0.5)

            fig.tight_layout()

            if fig_folder is not None:
                fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')

            plt.close(fig)

    return fig


def main():
    fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'

    alignment_ds_path = '/Volumes/Partition 1/data/interim/multispaceworld-rnn/new_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials.nc'
    model_number = 18
    # model_result_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/drift-model-%.f/'% model_number
    model_result_folder = '/Volumes/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f' % model_number
    target_vis_cond_list = [-0.8, -0.4, -0.2, 0.2, 0.4, 0.8, 0.0, 0.0, 0.8, 0.8, -0.8, -0.8, -0.1, 0.1]
    target_aud_cond_list = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                            -60, 60, 60, -60, 60, -60, np.inf, np.inf]
    all_stim_cond_pred_matrix_dict, pre_preprocessed_alignment_ds_dev = jaxdmodel.load_model_outputs(
        model_result_folder, alignment_ds_path, target_random_seed=None,
        target_vis_cond_list=target_vis_cond_list,
        target_aud_cond_list=target_aud_cond_list, drift_param_N=1)

    model_result_folder = '/Volumes/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f' % model_number

    model_result = pd.read_pickle(
        '/Volumes/Ultra Touch/multispaceworld-rnn-models/drift-model-18/driftParam_c_1_shuffle_0_cv0.pkl')
    target_aud_cond = [60]
    target_vis_cond = [0]
    trials_to_plot = [16]  # np.arange(0, 1008)  # only 72 trials in the test set.
    # stim_cond_name = 'arv0'
    # stim_cond_name = 'all'
    stim_cond_name = 'a0v0p8'
    plot_every_n_neuron = 4
    fig_ext = '.png'
    # fig_folder = '/Volumes/Partition 1/reports/figures/figure-6-for-pip/example-trials-from-drift-model'
    # fig_folder = None
    fig_name = 'figure-6a-top.pdf'
    fig = plot_model_single_trial_scheme(model_result, trials_to_plot=trials_to_plot,
                                         fig_folder=None, fig_ext=fig_ext,
                                         stim_cond_name=stim_cond_name,
                                         plot_every_n_neuron=plot_every_n_neuron)

    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

    fig_name = 'figure-6a-bottom.pdf'

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()

        peri_event_time = np.linspace(-0.1, 0.3, 143)

        vis_right_traces = all_stim_cond_pred_matrix_dict[(np.inf, 0.8)].T

        ax.plot(peri_event_time, np.mean(vis_right_traces, axis=1), color='red', alpha=1, lw=3)
        ax.plot(peri_event_time, vis_right_traces[:, 0:50], color='red', alpha=0.1)

        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()