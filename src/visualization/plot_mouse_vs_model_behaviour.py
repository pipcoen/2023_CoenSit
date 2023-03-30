import pandas as pd
import pickle as pkl
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

import src.data.struct_to_dataframe as stdf

# Plotting
import sciplotlib.polish as splpolish
import sciplotlib.style as splstyle


import src.models.predict_model as pmodel
import src.models.jax_decision_model as jaxdmodel
import time

import src.data.analyse_spikes as anaspikes
import src.data.analyse_behaviour as anabehave
import src.data.process_ephys_data as pephys

import src.visualization.vizmodel as vizmodel
import src.visualization.vizpikes as vizpikes
import src.visualization.vizstat as vizstat
import src.visualization.vizbehaviour as vizbehave
import src.models.psychometric_model as psychmodel

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

fig_ext = '.pdf'
offset_cal_method = 'group_mice'
plots_to_make = ['plot_four_cond_rt']
fig_folder = '/media/timsit/Partition 1/reports/figures/figure-6-for-pip/'

model_number = 20
model_result_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/drift-model-%.f/'% model_number
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


# model_combined_df_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/drift-model-22/model_output_and_behaviour/model_behaviour_df_test_seed_0_1_2.pkl'
# mouse_combined_df_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/drift-model-22/model_output_and_behaviour/matching_mouse_behaviour_df.pkl'

model_combined_df_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/drift-model-23/model_output_and_behaviour/model_behaviour_df_test_seed_0.pkl'
mouse_combined_df_path = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/drift-model-23/model_output_and_behaviour/matching_mouse_behaviour_df.pkl'


model_conflict_combined_df_path = \
    '/media/timsit/Partition 1/reports/figures/figure-6-for-pip/model_conflict_combined_df.pkl'
mouse_conflict_combined_df_path = \
    '/media/timsit/Partition 1/reports/figures/figure-6-for-pip/mouse_conflict_combined_df.pkl'


def plot_subject_level_diff_cond_rt(model_combined_df, mouse_combined_df):


    cond_types_to_plot = ['aud_only', 'vis_only', 'coherent', 'conflict']
    cond_colors = ['purple', 'orange', 'blue', 'brown']
    include_time_shift = True
    rt_var_name = 'choiceInitTimeRelStim'
    center_estimation_method = 'median'
    error_bar_metric = 'sem'
    rt_units = 'ms'
    scatter_points = False
    split_connection_lines = True
    offset_cal_method = 'group_mice'  # either all_mice or group_mice

    # Calculate offset
    # Method 1: combine all mice together
    if offset_cal_method == 'all_mice':
        model_median_rt = np.median(model_combined_df['reactionTime'])
        mouse_median_rt = np.median(mouse_combined_df['choiceInitTimeRelStim'])
        mouse_minus_model_median_rt = mouse_median_rt - model_median_rt
    elif offset_cal_method == 'group_mice':
        model_median_rt = np.median(model_combined_df['reactionTime'])
        mouse_gruoped_median_rt = mouse_combined_df.groupby('subjectRef').agg('median')
        mouse_median_rt = np.median(mouse_gruoped_median_rt[rt_var_name])
        mouse_minus_model_median_rt = mouse_median_rt - model_median_rt

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 5)
        num_subject = len(np.unique(mouse_combined_df['subjectRef']))
        num_cond_type = len(np.unique(mouse_combined_df['condType']))
        subject_cond_type_matrix = np.zeros((num_subject, num_cond_type))

        for n_subject, subjectRef in enumerate(np.unique(mouse_combined_df['subjectRef'])):

            single_mouse_df = mouse_combined_df.loc[
                mouse_combined_df['subjectRef'] == subjectRef
                ]

            cond_type_grouped_rt = single_mouse_df.groupby('condType').agg('median')
            cond_type_grouped_rt = cond_type_grouped_rt.loc[cond_types_to_plot]

            if rt_units == 'ms':
                cond_type_grouped_rt[rt_var_name] = cond_type_grouped_rt[rt_var_name] * 1000

            if scatter_points:
                ax.scatter(np.arange(len(cond_type_grouped_rt)), cond_type_grouped_rt[rt_var_name], color='gray',
                           edgecolor='none')

            if split_connection_lines:
                ax.plot(np.arange(len(cond_type_grouped_rt))[0:2],
                        cond_type_grouped_rt[rt_var_name][0:2], color='gray',
                        zorder=0)
                ax.plot(np.arange(len(cond_type_grouped_rt))[2:],
                        cond_type_grouped_rt[rt_var_name][2:], color='gray',
                        zorder=0)
            else:
                ax.plot(np.arange(len(cond_type_grouped_rt)), cond_type_grouped_rt[rt_var_name], color='gray',
                        zorder=0)

            subject_cond_type_matrix[n_subject, :] = cond_type_grouped_rt[rt_var_name].values

        # include median across subjects?
        # ax.scatter(np.arange(num_cond_type), np.median(subject_cond_type_matrix, 0), color='black')

        # Plot model
        xloc_offset = 0
        model_cond_type_grouped_rt = model_combined_df.groupby('condType').agg('median')
        model_cond_type_grouped_rt['reactionTime'] = model_cond_type_grouped_rt['reactionTime'] + \
                                                     mouse_minus_model_median_rt

        if rt_units == 'ms':
            model_cond_type_grouped_rt['reactionTime'] = model_cond_type_grouped_rt['reactionTime'] * 1000

        model_cond_type_grouped_rt = model_cond_type_grouped_rt.loc[cond_types_to_plot]
        model_x_loc = np.arange(len(model_cond_type_grouped_rt)) + xloc_offset

        # ax.scatter(model_x_loc,
        #            model_cond_type_grouped_rt['reactionTime'],
        #            color='red')
        ax.plot(model_x_loc[0:2],
                model_cond_type_grouped_rt['reactionTime'][0:2],
                color='red')
        ax.plot(model_x_loc[2:],
                model_cond_type_grouped_rt['reactionTime'][2:],
                color='red')

        # Plot model error bars
        for n_cond, cond_type in enumerate(cond_types_to_plot):
            model_cond_rt = model_combined_df.loc[
                model_combined_df['condType'] == cond_type
                ]['reactionTime']

            # Add offset
            model_cond_rt = model_cond_rt + mouse_minus_model_median_rt

            if rt_units == 'ms':
                model_cond_rt = model_cond_rt * 1000

            if error_bar_metric == 'sem':
                # mouse_rt_sem = np.std(mouse_cond_rt) / np.sqrt(len(mouse_cond_rt)) * 2
                model_rt_sem = np.std(model_cond_rt) / np.sqrt(len(model_cond_rt)) * 3
                # mouse_rt_lines = [mouse_mean_rt - mouse_rt_sem,
                #                               mouse_mean_rt + mouse_rt_sem]
                model_mean_rt = np.median(model_cond_rt)
                model_rt_lines = [model_mean_rt - model_rt_sem,
                                  model_mean_rt + model_rt_sem]

                ax.plot([model_x_loc[n_cond], model_x_loc[n_cond]],
                        model_rt_lines, color='red')

        if rt_units == 'ms':
            ax.set_ylabel('Reaction time (ms)', size=12)
        else:
            ax.set_ylabel('Reaction time (s)', size=12)

        ax.set_xticks(np.arange(len(cond_type_grouped_rt)))
        ax.set_xticklabels(['Audio only', 'Visual only', 'Coherent', 'Conflict'], size=12)

    return fig, ax


def plot_conflict_choose_aud_vs_choose_vis(mouse_conflict_combined_df,
                                           model_conflict_combined_df,
                                           mouse_and_model_offset=None,
                                           plot_type='across_mice'):

    cond_types_to_plot = ['conflictChooseAud', 'conflictChooseVis']
    xticklabels = ['Choose audio', 'Choose visual']
    cond_colors = ['purple', 'orange']

    center_estimation_method = 'median'
    error_bar_metric = 'sem'
    rt_units = 'ms'
    rt_var_name = 'choiceInitTimeRelStim'
    between_group_gap = 0.1
    x_center_loc = [0, 1]

    if plot_type == 'across_mice':

        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig, ax = plt.subplots()
            fig.set_size_inches(5, 4)

            num_subject = len(np.unique(mouse_conflict_combined_df['subjectRef']))
            num_cond_type = len(np.unique(mouse_conflict_combined_df['condType']))

            all_y_val_list = list()

            for n_subject, subjectRef in enumerate(np.unique(mouse_conflict_combined_df['subjectRef'])):

                single_mouse_df = mouse_conflict_combined_df.loc[
                    mouse_conflict_combined_df['subjectRef'] == subjectRef]

                y_vals = np.zeros((len(cond_types_to_plot), 1))
                for n_cond_type, cond_type in enumerate(cond_types_to_plot):
                    cond_type_df = single_mouse_df.loc[
                        single_mouse_df['condType'] == cond_type
                    ]
                    if center_estimation_method == 'median':
                        y_vals[n_cond_type] = np.median(cond_type_df[rt_var_name])

                if rt_units == 'ms':
                    y_vals = y_vals * 1000

                all_y_val_list.append(y_vals)

                ax.plot(x_center_loc, y_vals, color='gray')

            all_y_vals = np.squeeze(np.stack(all_y_val_list))

        pdb.set_trace()
        # PLot model


        model_y_vals = np.zeros((len(cond_types_to_plot), 1))
        for n_cond_type, cond_type in enumerate(cond_types_to_plot):

            model_cond_type_df = model_conflict_combined_df.loc[
                model_conflict_combined_df['condType'] == cond_type
            ]
            model_y_vals[n_cond_type] = np.median(model_cond_type_df['reactionTime'])

        if rt_units == 'ms':
            model_y_vals = model_y_vals * 1000

        if mouse_and_model_offset is not None:
            model_y_vals = model_y_vals + mouse_and_model_offset

        ax.plot(x_center_loc, model_y_vals, color='red')


        # Labels
        ax.set_xticks(x_center_loc)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel('Reaction time (ms)')


    elif plot_type == 'all_mice_combined':

        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig, ax = plt.subplots()

            for n_cond, cond_type in enumerate(cond_types_to_plot):

                n_cond_x = x_center_loc[n_cond]

                mouse_cond_rt = mouse_conflict_combined_df.loc[
                    mouse_conflict_combined_df['condType'] == cond_type
                    ]['choiceInitTimeRelStim']

                model_cond_rt = model_conflict_combined_df.loc[
                    model_conflict_combined_df['condType'] == cond_type
                    ]['reactionTime']

                if rt_units == 'ms':
                    mouse_cond_rt = mouse_cond_rt * 1000
                    model_cond_rt = model_cond_rt * 1000

                if center_estimation_method == 'mean':
                    mouse_mean_rt = np.mean(mouse_cond_rt)
                    model_mean_rt = np.mean(model_cond_rt) + mouse_minus_model_mean_rt * 1000
                elif center_estimation_method == 'median':
                    mouse_mean_rt = np.median(mouse_cond_rt)
                    model_mean_rt = np.median(model_cond_rt) + mouse_minus_model_median_rt * 1000

                if error_bar_metric == 'sem':
                    mouse_rt_sem = np.std(mouse_cond_rt) / np.sqrt(len(mouse_cond_rt)) * 2
                    model_rt_sem = np.std(model_cond_rt) / np.sqrt(len(model_cond_rt)) * 2
                    mouse_rt_lines = [mouse_mean_rt - mouse_rt_sem,
                                      mouse_mean_rt + mouse_rt_sem]
                    model_rt_lines = [model_mean_rt - model_rt_sem,
                                      model_mean_rt + model_rt_sem]
                elif error_bar_metric == '95percentile':
                    mouse_rt_lines = [np.percentile(mouse_cond_rt, 5),
                                      np.percentile(mouse_cond_rt, 95)]
                    model_rt_lines = [np.percentile(model_cond_rt, 5),
                                      np.percentile(model_cond_rt, 95)]
                elif error_bar_metric == 'std':
                    mouse_rt_sem = np.std(mouse_cond_rt)
                    model_rt_sem = np.std(model_cond_rt)
                    mouse_rt_lines = [mouse_mean_rt - mouse_rt_sem,
                                      mouse_mean_rt + mouse_rt_sem]
                    model_rt_lines = [model_mean_rt - model_rt_sem,
                                      model_mean_rt + model_rt_sem]

                ax.scatter(n_cond_x - between_group_gap, mouse_mean_rt, facecolor=cond_colors[n_cond],
                           edgecolor=cond_colors[n_cond])
                ax.scatter(n_cond_x + between_group_gap, model_mean_rt,
                           facecolor='white', edgecolor=cond_colors[n_cond], zorder=1)

                ax.plot([n_cond_x - between_group_gap, n_cond_x - between_group_gap], mouse_rt_lines,
                        color=cond_colors[n_cond])

                ax.plot([n_cond_x + between_group_gap, n_cond_x + between_group_gap], model_rt_lines,
                        color=cond_colors[n_cond], zorder=0)

            # ax.set_ylim([210, 280])
            # ax.set_yticks([220, 280])

            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])

            if rt_units == 'ms':
                ylabel_text = 'Reaction time (ms)'
            elif rt_units == 's':
                ylabel_text = 'Reaction time (s)'

            ax.set_ylabel(ylabel_text, size=12)
            text_y_loc = 150
            ax.text(x_center_loc[0] + 0., text_y_loc, 'Conflict choose audio', size=12, color='purple', ha='center')
            ax.text(x_center_loc[1], text_y_loc, 'Conflict choose visual', size=12, color='orange', ha='center')

            ax.set_xlim([-0.4, 1.5])

            ax.set_ylim([150, 230])

            # include legend
            dot_legend_x_loc = 1.4
            legend_yloc = 220
            legend_text_distance = 5
            text_legend_x_loc = dot_legend_x_loc + 0.1
            ax.scatter(dot_legend_x_loc, legend_yloc, color='black', edgecolor='none')
            ax.text(text_legend_x_loc, legend_yloc, 'Mouse', color='black', va='center')
            ax.scatter(dot_legend_x_loc, legend_yloc - legend_text_distance, color='none', edgecolor='black')
            ax.text(text_legend_x_loc, legend_yloc - legend_text_distance, 'Model', color='black', va='center')

    return fig, ax


def plot_model_mean_output(all_stim_cond_pred_matrix_dict, peri_stim_time):

    # peri_stim_time = pre_preprocessed_alignment_ds_dev.PeriEventTime.values
    include_decision_threshold_line = True

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
        fig.set_size_inches(4, 6)

        axs[0].plot(peri_stim_time, np.mean(all_stim_cond_pred_matrix_dict[np.inf, -0.8], axis=0),
                    color='blue', alpha=0.8)
        axs[0].plot(peri_stim_time, np.mean(all_stim_cond_pred_matrix_dict[np.inf, -0.4], axis=0),
                    color='blue', alpha=0.4)
        axs[0].plot(peri_stim_time, np.mean(all_stim_cond_pred_matrix_dict[np.inf, -0.2], axis=0),
                    color='blue', alpha=0.2)
        axs[0].plot(peri_stim_time, np.mean(all_stim_cond_pred_matrix_dict[np.inf, 0.2], axis=0), color='red',
                    alpha=0.2)
        axs[0].plot(peri_stim_time, np.mean(all_stim_cond_pred_matrix_dict[np.inf, 0.4], axis=0), color='red',
                    alpha=0.4)
        axs[0].plot(peri_stim_time, np.mean(all_stim_cond_pred_matrix_dict[np.inf, 0.8], axis=0), color='red',
                    alpha=0.8)
        axs[0].set_title('Visual only', size=12)

        axs[1].plot(peri_stim_time, np.mean(all_stim_cond_pred_matrix_dict[-60, 0], axis=0), color='blue', alpha=0.8)
        axs[1].plot(peri_stim_time, np.mean(all_stim_cond_pred_matrix_dict[60, 0], axis=0), color='red', alpha=0.8)
        axs[1].set_title('Audio only', size=12)

        axs[2].plot(peri_stim_time, np.mean(all_stim_cond_pred_matrix_dict[60, 0.8], axis=0),
                    color='red', alpha=0.8)
        axs[2].plot(peri_stim_time, np.mean(all_stim_cond_pred_matrix_dict[-60, 0.8], axis=0),
                    color='black', alpha=0.8, label='Audio left, visual right')
        axs[2].plot(peri_stim_time, np.mean(all_stim_cond_pred_matrix_dict[60, -0.8], axis=0),
                    color='gray', alpha=0.8, label='Audio right, visual left')
        axs[2].plot(peri_stim_time, np.mean(all_stim_cond_pred_matrix_dict[-60, -0.8], axis=0), color='blue', alpha=0.8)

        axs[2].set_title('Audio + visual', size=12)

        axs[2].set_xlabel('Peri-stimulus time (s)', size=12)

        # axs[2].legend(frameon=False, prop={'size': 6})

        fig.tight_layout()
        fig.text(s='Model output', x=0.0, y=0.5, rotation=90, size=12, va='center')

        # axs[0].set_ylim(-10, 10)
        # axs[0].set_yticks([-1, 0, 1])
        # axs[0].set_yticks([-3, -2, -1, 0, 1, 2, 3])

        # also include line to show the decision threshold
        if include_decision_threshold_line:
            axs[0].axhline(1, linestyle='--', color='grey', alpha=0.5, linewidth=1)
            axs[0].axhline(-1, linestyle='--', color='grey', alpha=0.5, linewidth=1)

            axs[1].axhline(1, linestyle='--', color='grey', alpha=0.5, linewidth=1)
            axs[1].axhline(-1, linestyle='--', color='grey', alpha=0.5, linewidth=1)

            axs[2].axhline(1, linestyle='--', color='grey', alpha=0.5, linewidth=1)
            axs[2].axhline(-1, linestyle='--', color='grey', alpha=0.5, linewidth=1)

        # Some text
        vis_x_loc = 0.1  # 0.7

        axs[0].text(vis_x_loc, 0.9, 'Visual right (80%)', size=8, color='red', alpha=0.8, transform=axs[0].transAxes)
        axs[0].text(vis_x_loc, 0.8, 'Visual right (40%)', size=8, color='red', alpha=0.4, transform=axs[0].transAxes)
        axs[0].text(vis_x_loc, 0.7, 'Visual right (20%)', size=8, color='red', alpha=0.2, transform=axs[0].transAxes)

        axs[0].text(vis_x_loc, 0.1, 'Visual left (80%)', size=8, color='blue', alpha=0.8, transform=axs[0].transAxes)
        axs[0].text(vis_x_loc, 0.2, 'Visual left (40%)', size=8, color='blue', alpha=0.4, transform=axs[0].transAxes)
        axs[0].text(vis_x_loc, 0.3, 'Visual left (20%)', size=8, color='blue', alpha=0.2, transform=axs[0].transAxes)

        axs[1].text(vis_x_loc, 0.2, 'Audio left', size=8, color='blue', alpha=0.8, transform=axs[1].transAxes)
        axs[1].text(vis_x_loc, 0.8, 'Audio right', size=8, color='red', alpha=0.8, transform=axs[1].transAxes)

        arvr_text_x_loc = 0.1  # 0.5
        arvr_text_y_loc = 0.9
        alvr_text_x_loc = 0.1  # 0.5
        alvr_text_y_loc = 0.8

        arvl_text_x_loc = 0.1  # 0.2
        arvl_text_y_loc = 0.2
        alvl_text_x_loc = 0.1  # 0.2
        alvl_text_y_loc = 0.1

        axs[2].text(arvr_text_x_loc, arvr_text_y_loc, 'Audio right + Visual right', size=8, color='red', alpha=0.8,
                    transform=axs[2].transAxes)
        axs[2].text(alvr_text_x_loc, alvr_text_y_loc, 'Audio left + Visual right', size=8, color='black', alpha=0.8,
                    transform=axs[2].transAxes)

        axs[2].text(arvl_text_x_loc, arvl_text_y_loc, 'Audio right + Visual left', size=8, color='gray', alpha=0.8,
                    transform=axs[2].transAxes)
        axs[2].text(alvl_text_x_loc, alvl_text_y_loc, 'Audio left + Visual left', size=8, color='blue', alpha=0.8,
                    transform=axs[2].transAxes)


    return fig, axs


def get_ephys_experiments_used_in_model(subset_neurons_df, mouse_behaviour_df):




    return subset_mouse_df


def main():

    print('Figures are going to save in %s' % fig_folder)

    if extract_model_behaviour_from_output:
        print('Extracting model behaviour from model output')
        if target_random_seed is not None:
            search_str = '*shuffle_%.f*.pkl' % target_random_seed

        else:
            search_str = '*.pkl'

        alignment_ds = xr.open_dataset(alignment_ds_path)

        start_time = -0.1
        end_time = 0.3

        # re-index trial values
        alignment_ds = alignment_ds.assign_coords({'Trial': np.arange(0, len(alignment_ds.Trial.values))})

        subset_neuron_idx = [0, 1]
        subset_alignment_ds = alignment_ds.isel(Cell=subset_neuron_idx)

        pre_preprocessed_alignment_ds_dev = subset_alignment_ds.where(
            (subset_alignment_ds['PeriEventTime'] >= start_time) &
            (subset_alignment_ds['PeriEventTime'] <= end_time), drop=True
        )

        target_vis_cond_list = [-0.8, -0.4, -0.2, 0.2, 0.4, 0.8, 0.0, 0.0, 0.8, 0.8, -0.8, -0.8,
                                -0.1, 0.1]
        target_aud_cond_list = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                -60, 60, 60, -60, 60, -60, np.inf, np.inf]

        model_results = [pd.read_pickle(x) for x in glob.glob(os.path.join(model_result_folder, search_str))]

        model_type = 'drift'

        y_test_pred_da_list = [m_result['y_test_pred_da'] for m_result in model_results]

        all_stim_cond_pred_matrix_dict = jax_dmodel.get_stim_cond_response(
            alignment_ds=subset_alignment_ds, y_test_pred_da_list=y_test_pred_da_list,
            target_vis_cond_list=target_vis_cond_list, target_aud_cond_list=target_aud_cond_list
        )

        model_behaviour_df = jax_dmodel.all_stim_cond_pred_matrix_dict_to_model_behaviour_df(
            all_stim_cond_pred_matrix_dict=all_stim_cond_pred_matrix_dict,
            alignment_ds=pre_preprocessed_alignment_ds_dev,
            right_decision_threshold_val=decision_threshold_val,
            left_decision_threshold_val=-decision_threshold_val,
            model_type=model_type,
            left_choice_val=0, right_choice_val=1,
            target_vis_cond_list=target_vis_cond_list, target_aud_cond_list=target_aud_cond_list
        )

        # remove early and no response trials
        go_model_behaviour_df = model_behaviour_df.loc[
            model_behaviour_df['reactionTime'] >= 0
            ]

        model_behaviour_save_folder = os.path.join(model_result_folder, 'model_output_and_behaviour')

        if not os.path.exists(model_behaviour_save_folder):
            os.mkdir(model_behaviour_save_folder)

        with open(os.path.join(model_behaviour_save_folder, model_output_fname), 'wb') as handle:
            pkl.dump(all_stim_cond_pred_matrix_dict, handle)

        model_behaviour_df.to_pickle(os.path.join(model_behaviour_save_folder, model_behaviour_output_fname))


    if get_mouse_combined_df or get_all_mouse_combined_df:

        active_behave_df = pd.read_pickle(mouse_active_behave_df_path)

        if get_mouse_combined_df: 
            print('Using model subset neurons to get mouse behaviuor df with matching neurons')
            stim_selective_cells = pd.read_pickle(subset_neurons_df_path)
            subset_active_behaviour_df = active_behave_df.loc[
                active_behave_df['expRef'].isin(np.unique(stim_selective_cells['Exp']))
            ]
        else:
            subset_active_behaviour_df = active_behave_df

        # only include valid trials
        valid_active_behave_df = subset_active_behaviour_df.loc[
            (subset_active_behaviour_df['validTrial'] == 1)
        ]
        # remove no-go trials
        go_subset_active_behaviour_df = valid_active_behave_df.loc[
            valid_active_behave_df['noGo'] == False
            ]
        # remove trials where audio is off
        no_aud_off_subset_active_behaviour_df = go_subset_active_behaviour_df.loc[
            np.isfinite(go_subset_active_behaviour_df['audDiff'])
        ]

        no_aud_off_subset_active_behaviour_df.to_pickle(mouse_combined_df_path)

    model_combined_df = pd.read_pickle(model_combined_df_path)
    mouse_combined_df = pd.read_pickle(mouse_combined_df_path)

    if 'plot_model_convergence' in plots_to_make:
        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig, axs = plt.subplots(2, 1, sharex=True)
            fig.set_size_inches(5, 5)

            line_alpha = 0.1

            for model_result in model_results:
                axs[0].plot(model_result['epoch'], model_result['dev_loss'], color='black', alpha=line_alpha)
                axs[1].plot(model_result['epoch'], model_result['test_loss'], color='black', alpha=line_alpha)

            axs[0].set_ylabel('Training set loss $L_{\mathrm{dev}}$', size=12)
            axs[1].set_xlabel('Epochs', size=12)

            axs[1].set_ylabel('Test set loss $L_{\mathrm{test}}$', size=12)

            axs[0].set_yscale('log')
            axs[1].set_yscale('log')

            fig_name = 'drift_model_9_train_and_test_loss'
            fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')

    if 'plot_model_mean_output' in plots_to_make:
        # TODO: this can all be simplified by saving the peri-event time somewhere
        alignment_ds = xr.open_dataset(alignment_ds_path)

        start_time = -0.1
        end_time = 0.3

        # re-index trial values
        alignment_ds = alignment_ds.assign_coords({'Trial': np.arange(0, len(alignment_ds.Trial.values))})

        subset_neuron_idx = [0, 1]
        subset_alignment_ds = alignment_ds.isel(Cell=subset_neuron_idx)

        pre_preprocessed_alignment_ds_dev = subset_alignment_ds.where(
            (subset_alignment_ds['PeriEventTime'] >= start_time) &
            (subset_alignment_ds['PeriEventTime'] <= end_time), drop=True
        )

        peri_stim_time = pre_preprocessed_alignment_ds_dev.PeriEventTime.values
        fig, axs = plot_model_mean_output(all_stim_cond_pred_matrix_dict, peri_stim_time)
        fig_name = 'drift_model_%.f_output_in_each_stim_condition_test_random_seed_0' % model_number
        fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')

    if 'plot_model_only_psychometric' in plots_to_make:

        # Plot model psychometric without fitting
        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig, ax = vizmodel.plot_nn_model_psychometric(nn_behaviour_df=model_behaviour_df, connect_dots=False,
                                                          plot_log_scale=True, remove_no_go=True,
                                                          remove_early_choice=True,
                                                          )
            fig_name = 'model_psychometric_dots'
            fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')

        subject_p_right, stim_cond_val, model_prediction_val, explained_var, popt = \
            psychmodel.fit_and_predict_psych_model(subject_behave_df=go_model_behaviour_df, small_norm_term=0)

        vis_exponent = popt[3]
        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig, ax = vizbehave.plot_psychometric_model_fit(subject_p_right, stim_cond=stim_cond_val,
                                                            model_prediction=model_prediction_val,
                                                            vis_exponent=vis_exponent,
                                                            aud_center_to_off=True, fig=None, ax=None)
            fig_name = 'model_psychometric_dots_with_fits'
            fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')

    if 'plot_model_vs_mouse_psychometric' in plots_to_make:
        print('Plotting model versus mouse psychometric')
        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig, ax = plt.subplots()
            fig.set_size_inches(4, 4)

            # Mouse first
            go_subset_active_behaviour_df = mouse_combined_df.loc[
                mouse_combined_df['noGo'] == False
                ]

            no_aud_off_subset_active_behaviour_df = go_subset_active_behaviour_df.loc[
                np.isfinite(go_subset_active_behaviour_df['audDiff'])
            ]

            subject_p_right, mouse_stim_cond_val, mouse_model_prediction_val, explained_var, mouse_popt = psychmodel.fit_and_predict_psych_model(
                subject_behave_df=no_aud_off_subset_active_behaviour_df, small_norm_term=0)

            mouse_vis_exponent = mouse_popt[3]
            fig, ax = vizbehave.plot_psychometric_model_fit(subject_p_right,
                                                            stim_cond=mouse_stim_cond_val,
                                                            model_prediction=mouse_model_prediction_val,
                                                            vis_exponent=mouse_vis_exponent, include_scatter=True,
                                                            linestyle='--', include_legend=False,
                                                            aud_center_to_off=True, fig=fig, ax=ax)


            # Model fit
            vis_exp_lower_bound = 0.6
            vis_exp_init_guess = 0.6
            vis_exp_upper_bound = 2
            small_norm_term = 0

            model_behaviour_df['chooseRight'] = model_behaviour_df['choice']
            model_p_right, stim_cond_val, model_prediction_val, explained_var, model_popt = psychmodel.fit_and_predict_psych_model(
                subject_behave_df=model_behaviour_df, small_norm_term=small_norm_term,
                vis_exp_lower_bound=vis_exp_lower_bound, vis_exp_init_guess=vis_exp_init_guess,
                vis_exp_upper_bound=vis_exp_upper_bound)
            model_vis_exponent = model_popt[3]
            fig, ax = vizbehave.plot_psychometric_model_fit(model_p_right, stim_cond=stim_cond_val,
                                                            model_prediction=model_prediction_val,
                                                            vis_exponent=model_vis_exponent,
                                                            aud_center_to_off=True, fig=fig, ax=ax,
                                                            open_circ_for_sim=False,
                                                            open_circ_for_multimodal=False,
                                                            removed_highest_coherent=False, include_scatter=False,
                                                            include_legend=False)

            custom_legend_objs = [mpl.lines.Line2D([0], [0], color='black', lw=2),
                                  mpl.lines.Line2D([0], [0], color='black', linestyle='--', lw=2)
                                  ]
            ax.text(0.6, 4, r'$A_R$', size=12, color='red')
            ax.text(0.6, -0.8, r'$A_L$', size=12, color='blue')

            ax.legend(custom_legend_objs, ['Drift model', 'Mice'])

            ax.set_ylim([-4, 4])
            fig.tight_layout()


    if 'plot_mouse_vs_model_inactivation' in plots_to_make:

        print('Plotting mouse vs model inactivation psychometric curve')


    if 'plot_four_cond_rt' in plots_to_make:
        print('Plotting reaction time comparison between four stimulus conditions')

        mouse_rt_var_name = 'choiceInitTimeRelStim'

        # Calculate model and mouse offset
        mouse_combined_df = anabehave.get_df_stim_cond(
                     subset_active_behaviour_df=mouse_combined_df,
                     include_mouse_number=True,
                     include_exp_number=False,
                     mouse_rt_var_name='choiceInitTimeRelStim',
                     mouse_choice_var_name='choiceThreshDir',
                     fields_to_include=['audDiff', 'visDiff'],
                     vis_cond_name='visDiff', aud_cond_name='audDiff',
                     verbose=False)

        # pdb.set_trace()

        model_combined_df = anabehave.get_df_stim_cond(
            subset_active_behaviour_df=model_combined_df,
            include_mouse_number=False,
            include_exp_number=False,
            mouse_rt_var_name=None,
            mouse_choice_var_name=None,
            fields_to_include=['audCond', 'visCond', 'reactionTime'],
            vis_cond_name='visCond', aud_cond_name='audCond',
            verbose=False)

        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig, ax = vizbehave.plot_model_vs_mouse_four_cond_plot(model_combined_df, mouse_combined_df)
            fig_name = 'model_%.f_mouse_vs_model_rt_four_cond' % model_number
            fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')

        print('Figure saved')

    if 'plot_conflict_choose_aud_vs_choose_vis' in plots_to_make:

        mouse_conflict_combined_df = pd.read_pickle(mouse_conflict_combined_df_path)
        model_conflict_combined_df = pd.read_pickle(model_conflict_combined_df_path)



        bo_mouse_conflict_combined_df = anabehave.get_conflict_choice_rt(active_mouse_behaviour_only_df,
                                                                         mouse_choice_name='responseRecorded',
                                                                         data_from='mouse')

        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig, ax = vizbehave.plot_mouse_and_model_conflict_choose_av(
                mouse_conflict_combined_df=bo_mouse_conflict_combined_df,
                mouse_combined_df=active_mouse_behaviour_only_df_processed,
                model_conflict_combined_df=model_conflict_combined_df,
                model_combined_df=model_combined_df,
                rt_var_name='timeToResponseThresh',
                fig=None, ax=None)

        fig_name = 'conflict_choose_aud_vs_choose_vis.pdf'
        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()