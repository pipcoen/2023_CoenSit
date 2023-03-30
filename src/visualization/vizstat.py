from src.visualization.useSansMaths import use_sans_maths
# use_sans_maths()
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib_venn import venn3
import glob
import os
import pickle as pkl
import numpy as np
import pandas as pd
import pdb   # debugging
FIG_DPI = 300
import itertools

# Distribution plot
from sklearn.neighbors import KernelDensity

# Circle significance plots
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib import colors

def batch_plot_selectivity_pie(suite2p_data_folder_path, raw_data_folder_path, subject_name, exp_date, exp_num, plane_num, fig_folder, sig_df_file_name='sig_neuron_results.pkl'):

    
    if type(subject_name) is not list:
        subject_name = [subject_name]
    if type(exp_date) is not list:
        exp_date = [exp_date]
    if type(exp_num) is not list:
        exp_num = [exp_num]
    if type(plane_num) is not list:
        plane_num = [plane_num]

    # specify format of subject name to be ssffff so that unrelated folders don't get read out...

    if 'all' in subject_name:
        subject_name = glob.glob(os.path.join(suite2p_data_folder_path, '*/'))
        subject_name = os.path.basename(os.path.dirname(subject_name))

    for s_name in subject_name:


        if 'all' in exp_date:
            # print(os.path.join(suite2p_data_folder_path, s_name, '*/'))
            exp_date = glob.glob(os.path.join(suite2p_data_folder_path, s_name, '*/'))
            exp_date = [os.path.basename(os.path.dirname(x)) for x in exp_date]

        for e_date in exp_date:

            if 'all' in exp_num:
                exp_num = glob.glob(os.path.join(suite2p_data_folder_path, s_name, e_date, '*/'))
                exp_num = [os.path.basename(os.path.dirname(x)) for x in exp_num]

            for e_num in exp_num:

                if 'all' in plane_num:
                    plane_num = glob.glob(os.path.join(suite2p_data_folder_path, s_name, e_date, e_num, 'suite2p',  '*/'))
                    plane_num = [os.path.basename(os.path.dirname(x)) for x in plane_num]

                for p_num in plane_num:

                    sig_result_file_name = os.path.join(suite2p_data_folder_path, s_name, e_date, e_num,
                                           'suite2p', 'plane' + str(p_num), sig_df_file_name)

                    with open(sig_result_file_name, 'rb') as handle:
                        test_df = pkl.load(handle)

                    fig, ax = plot_selectivity_pie(test_df, fig_background_color=[0.8, 0.8, 0.8])
                    fig_name = 'sig_neuron_pie_' + s_name + '_' + e_date + '_exp_' + e_num + '_' +  'plane' + str(p_num)
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=FIG_DPI,
                                facecolor=fig.get_facecolor())


def plot_selectivity_pie(all_test_df, fig=None, ax=None,
                         fig_background_color=[0.8, 0.8, 0.8], sig_threshold=0.05,
                         direction=None):
    """
    (Very hard coded function to) Plot the number of neurons responding significant to each trial type.

    Arugments

    all_test_df   : dataframe containing statistical test result for each cell across all stimuli conditions 
    fig, ax       : figure handles
    direction     : whether to further subset into cells that increase in firing rate vs. decrease in firing rate, options (1) None - both increase and decrease (2) 'Increase' (3) 'Decrease' 
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(facecolor=(fig_background_color))
        fig.set_size_inches(4, 4)
    
    # merge, audio only and visual only test results (TODO: do audio-visual as well)
    audio_only_sig_df = all_test_df.loc[
                                        (all_test_df['P_val_aud'] <= sig_threshold) &
                                        (all_test_df['P_val_vis'] > sig_threshold) & 
                                        (all_test_df['P_val_audvis'] > sig_threshold)
                                        ]
    visual_only_sig_df = all_test_df.loc[
                                         (all_test_df['P_val_aud'] > sig_threshold) &
                                         (all_test_df['P_val_vis'] <= sig_threshold) & 
                                         (all_test_df['P_val_audvis'] > sig_threshold)
                                         ]
    audio_visual_only_sig_df = all_test_df.loc[
                                               (all_test_df['P_val_aud'] > sig_threshold) &
                                               (all_test_df['P_val_vis'] > sig_threshold) & 
                                               (all_test_df['P_val_audvis'] <= sig_threshold)
                                               ]
    audio_only_AND_visual_only_df = all_test_df.loc[
                                                    (all_test_df['P_val_aud'] <= sig_threshold) &
                                                    (all_test_df['P_val_vis'] <= sig_threshold) & 
                                                    (all_test_df['P_val_audvis'] > sig_threshold)
                                                    ]
    audio_only_AND_audio_visual_only_sig_df = all_test_df.loc[
                                                              (all_test_df['P_val_aud'] <= sig_threshold) &
                                                              (all_test_df['P_val_vis'] > sig_threshold) & 
                                                              (all_test_df['P_val_audvis'] <= sig_threshold)
                                                              ]
    visual_only_AND_audio_visual_only_sig_df = all_test_df.loc[
                                                               (all_test_df['P_val_aud'] > sig_threshold) &
                                                               (all_test_df['P_val_vis'] <= sig_threshold) & 
                                                               (all_test_df['P_val_audvis'] <= sig_threshold)
                                                               ]
    all_sig_df =  all_test_df.loc[
                                  (all_test_df['P_val_aud'] <= sig_threshold) &
                                  (all_test_df['P_val_vis'] <= sig_threshold) & 
                                  (all_test_df['P_val_audvis'] <= sig_threshold)
                                  ]
    no_response_df =  all_test_df.loc[
                                      (all_test_df['P_val_aud'] > sig_threshold) &
                                      (all_test_df['P_val_vis'] > sig_threshold) & 
                                      (all_test_df['P_val_audvis'] > sig_threshold)
                                      ]

    if direction is None:
        v = venn3(subsets=(len(audio_only_sig_df), # Abc
         len(visual_only_sig_df), # aBc
        len(audio_only_AND_visual_only_df),  # ABc
        len(audio_visual_only_sig_df), # abC 
        len(audio_only_AND_audio_visual_only_sig_df), # AbC
        len(visual_only_AND_audio_visual_only_sig_df),  # aBC
        len(all_sig_df)),  # ABC
        set_labels= ('Audio trials', 'Visual trials',
                    'Audio-visual trials'), 
        ax=ax)


    ax.text(-0.5, -0.5, s=str(len(no_response_df)))

    return fig, ax 


def plot_multi_condition_num_sig_neuron(cell_sig_df, fig=None, ax=None,
                                        print_num_neuron=True):
    """
    Plot heatmap showing number of significant neurons for each pair of trial condition,
    with respect to auditory and visual stimulus.
    :param cell_sig_df:
    :param fig:
    :param ax:
    :param print_num_neuron:
    :return:
    """

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)



    cell_sig_df = cell_sig_df.rename(columns={'vis_cond': 'vis-cond',
                                              'aud_cond': 'aud-cond'})

    sum_sig_neuron_df = pd.pivot_table(cell_sig_df, values='sig_response',
                                       index=['aud-cond'], columns=['vis-cond'],
                                       aggfunc=np.sum)

    sns.heatmap(sum_sig_neuron_df, ax=ax, annot=True, square=True,
                cbar=True)



    if print_num_neuron == True:
        num_neuron = len(np.unique(cell_sig_df['cell_idx']))
        ax.set_title('Total number of neurons: ' + str(num_neuron))

    # some strange bug with seaborn
    ax.set_ylim(3, 0)

    return fig, ax


def plot_single_condition_sig_neuron(stat_test_df, fig=None, ax=None, level='exp',
                                     unit='count', brain_region='all',
                                     sig_label='Significant auditory left/right selectivity',
                                     p_val=0.05):

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)

    if (brain_region is not None) and (brain_region != 'all'):
        stat_test_df = stat_test_df.loc[
            stat_test_df['cellLoc'] == brain_region
            ]
    else:
        stat_test_df['cellLoc'] = stat_test_df['cellLoc'].astype(str)

    sig_test_df = stat_test_df.loc[
            stat_test_df['pVal'] <= p_val
            ]

    if level == 'exp':
        cell_counts = stat_test_df.groupby('expRef').agg('count')['cellLoc']
        sig_cell_counts = sig_test_df.groupby('expRef').agg('count')['cellLoc']
        ax.set_xlabel('Experiment', size=12)


    elif level == 'cellLoc':
        cell_counts = stat_test_df.groupby('cellLoc').agg('count')['cellIdx']
        sig_cell_counts = sig_test_df.groupby('cellLoc').agg('count')['cellIdx']
        ax.set_xlabel('Brain region', size=12)
        plt.setp(ax.get_xticklabels(), ha="center", rotation=45)


    if unit == 'count':
        ax.bar(cell_counts.index, cell_counts, label='Recorded')

        ax.bar(sig_cell_counts.index, sig_cell_counts, label=sig_label)

    elif unit == 'prop':
        prop_sig_cell_per_exp = sig_cell_counts / cell_counts
        ax.bar(prop_sig_cell_per_exp.index, prop_sig_cell_per_exp)
        ax.axhline(0.05, linestyle='--', color='grey')


    return fig, ax


def plot_unpaired_scatter(df, subset_condition='subset_alignment_condition', groupby='exp', agg_metric='mean',
                          group_name_order=None,
                          y_metric='relative_score', fig=None, ax=None, scatter_alpha=0.3,
                          group_colors=['red', 'blue', 'green'], dot_size=3, mean_dot_size=30, infer_xticklabels=True,
                          jitter_val=0.1, plot_mean=True, xlabel_size=12, plot_spread=True, spread_metric='2sem',
                          highlight_neuron_idx_dict=None):

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 6)

    if group_name_order is not None:
        unique_subset_conditions = group_name_order
    else:
        unique_subset_conditions = np.unique(df[subset_condition])
    x_locs = np.arange(len(unique_subset_conditions))
    x_val_list = list()
    y_val_list = list()
    for n, subset_df_condition in enumerate(unique_subset_conditions):
        df_subset = df.loc[(df[subset_condition] == subset_df_condition)]
        if groupby is not None:
            df_subset_grouped = df_subset.groupby(groupby).agg(agg_metric)
            df_subset_grouped = df_subset_grouped.sort_index()
        else:
            df_subset_grouped = df_subset
        df_subset_x_vals = np.random.normal(x_locs[n], jitter_val, len(df_subset_grouped))
        ax.scatter(df_subset_x_vals, (df_subset_grouped[y_metric]), color=(group_colors[n]),
          s=dot_size,
          alpha=scatter_alpha,
          lw=0)
        if highlight_neuron_idx_dict is not None:
            highlight_neuron_idx = highlight_neuron_idx_dict[n]
            if highlight_neuron_idx is not None:
                ax.scatter((df_subset_x_vals[highlight_neuron_idx]), (df_subset_grouped[y_metric].iloc[highlight_neuron_idx]), color='red',
                  s=dot_size,
                  alpha=1,
                  lw=0)
        x_val_list.append(df_subset_x_vals)
        y_val_list.append(df_subset_grouped[y_metric])
        if plot_mean:
            ax.scatter((x_locs[n]), (np.mean(df_subset_grouped[y_metric])), color=(group_colors[n]),
              s=mean_dot_size,
              alpha=1)
        if plot_spread:
            if spread_metric == '2sem':
                spread_val = 2 * np.std(df_subset_grouped[y_metric]) / np.sqrt(len(df_subset_grouped[y_metric]))
            else:
                if spread_metric == 'std':
                    spread_val = np.std(df_subset_grouped[y_metric])
                ax.plot([x_locs[n], x_locs[n]], [
                 np.mean(df_subset_grouped[y_metric]) - spread_val,
                 np.mean(df_subset_grouped[y_metric]) + spread_val],
                  color=(group_colors[n]),
                  lw=2)

    ax.set_xticks(x_locs)
    if infer_xticklabels:
        ax.set_xticklabels(unique_subset_conditions, size=xlabel_size)
    return fig, ax


def plot_box_w_scatter(fig, ax):


    return fig, ax


def plot_paired_scatter(df, subset_condition='subset_alignment_condition',
                        groupby='exp', agg_metric='mean',
                        y_metric='relative_score', fig=None, ax=None,
                        infer_xticklabels=True, jitter_val=0.1):
    """

    :param df:
    :param subset_condition:
    :param groupby:
    :param agg_metric:
    :param y_metric:
    :param fig:
    :param ax:
    :param infer_xticklabels:
    :param jitter_val:
    :return:
    """
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 6)

    unique_subset_conditions = np.unique(df[subset_condition])
    x_locs = np.arange(len(unique_subset_conditions))

    x_val_list = list()
    y_val_list = list()

    for n, subset_df_condition in enumerate(unique_subset_conditions):
        df_subset = df.loc[df[subset_condition] == subset_df_condition]
        df_subset_grouped = df_subset.groupby(groupby).agg(agg_metric)
        df_subset_grouped = df_subset_grouped.sort_index()

        df_subset_x_vals = np.random.normal(
            x_locs[n], jitter_val, len(df_subset_grouped)
        )

        ax.scatter(df_subset_x_vals, df_subset_grouped[y_metric])

        # save values for later connecting the dots
        x_val_list.append(df_subset_x_vals)
        y_val_list.append(df_subset_grouped[y_metric])

    # connect the dots
    ax.plot(x_val_list, y_val_list, color='black', alpha=0.3)

    ax.set_xticks(x_locs)

    if infer_xticklabels:
        ax.set_xticklabels(unique_subset_conditions)

    return fig, ax


def plot_boxplot(df, group_cond_name='vis',
                 variable_names=['choose_vis_rt', 'choose_aud_rtl'],
                 colors=['#91D1C2FF', '#DC0000'], horizontal_offset=0.2, fig=None, ax=None,
                 drop_na=False, verbose=True, box_colors=['orange', 'purple'],
                 group_name_order=None, show_outlier=True, include_kde=False,
                 kde_range=[0, 0.3], kde_bandwidth=0.02, log_dens_min=0, log_dens_max=10,
                 vert_boxplot=False):
    """
    Boxplot with some nice defaults; mainly from setting all whisker, cap etc. colors to the same.
    Parameters
    ------------
    df : (pandas dataframe)
        pandas dataframe where each row is a sample
    group_cond_name : (str)
    variable_names : (list)
    colors : (list)
    :param horizontal_offset:
    :param fig:
    :param ax:
    :return:
    """

    if fig is None and ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4)

    if group_name_order is None:
        group_name_order = np.unique(df[group_cond_name])

    for n_group, group_cond in enumerate(group_name_order):

        if verbose:
            print('Group: ', group_cond)

        group_subset_df = df.loc[df[group_cond_name] == group_cond]

        if drop_na:
            group_subset_df = group_subset_df.dropna()

        if len(group_subset_df) == 0:
            print('Warning: group_subset_df is empty')
            continue

        if len(variable_names) == 2:
            # currently assume only two groups with x_pos
            variable_pos = [n_group - horizontal_offset, n_group + horizontal_offset]
        elif len(variable_names) == 1:
            variable_pos = [n_group]
        else:
            print('Invalid variable_names length')

        for variable, x_pos, color in zip(variable_names, variable_pos, colors):

            cond_boxplot = ax.boxplot(
                    group_subset_df[variable],
                    positions=[x_pos],
                    patch_artist=True,
                    showfliers=show_outlier,
                vert=vert_boxplot)

            if box_colors is not None and len(variable_names) == 1:
                color = box_colors[n_group]
            else:
                color = color

            for item in ['boxes', 'whiskers', 'fliers', 'caps']:
                plt.setp(cond_boxplot[item], color=color)

            plt.setp(cond_boxplot['medians'], color='white')

            if include_kde:
                cond_values = group_subset_df[variable].values
                cond_kde = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(cond_values.reshape(-1, 1))
                kde_range_values = np.linspace(kde_range[0], kde_range[1], 1000)
                cond_log_dens = cond_kde.score_samples(kde_range_values.reshape(-1, 1))
                # if log_dens_min is not None and log_dens_max is not None:
                #     cond_log_dens = (cond_log_dens - log_dens_min) / (log_dens_max - log_dens_min)
                # cond_log_dens = cond_log_dens * horizontal_offset
                # pdb.set_trace()
                y_vals = np.exp(cond_log_dens)
                if log_dens_min is not None and log_dens_max is not None:
                    y_vals = (y_vals - log_dens_min) / (log_dens_max - log_dens_min)
                baseline = np.zeros(len(y_vals)) #  + x_pos
                # ax.fill_between(y_vals, kde_range_values, 0, color=color)
                if vert_boxplot:
                    ax.fill_betweenx(kde_range_values, y_vals+x_pos, baseline + x_pos,
                                     alpha=0.7,
                                     color=color)
                else:
                    ax.fill_between(kde_range_values, y_vals+x_pos, baseline + x_pos,
                                     alpha=0.7,
                                     color=color)


    return fig, ax


def plot_two_modality_metric(modality_test_df, x_metric='resLeftRightSSMD',
                             y_metric='audLeftRightSSMD', offset=0.05,
                             include_zero_axis_line=True, fig=None, ax=None,
                             custom_x_lim=None, custom_y_lim=None, include_moving_average=False,
                             num_moving_ave_bins=50):
    """
    Plots some metric that indicates preference for a particular direction
    in a modality for each cell. Modality can refer to
    (1) Visual stimuli (2) auditory stimuli or (3) choice
    and usually we plot left/right distribution differences

    Parameters
    -----------
    :param modality_test_df:
    :param x_metric:
    :param y_metric:
    :param offset:
    :param include_zero_axis_line:
    :param fig:
    :param ax:
    :param custom_x_lim:
    :param custom_y_lim:
    :return:
    """

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

    global_max = np.nanmax(np.concatenate([modality_test_df[x_metric],
                                           modality_test_df[y_metric]]))

    global_min = np.nanmin(np.concatenate([modality_test_df[x_metric],
                                           modality_test_df[y_metric]]))

    global_abs_max = np.max([global_max, abs(global_min)])

    ax.scatter(modality_test_df[x_metric],
               modality_test_df[y_metric],
              edgecolor='none', alpha=0.7, s=4)

    if include_moving_average:
        x_metric_bin = np.linspace(np.min(modality_test_df[x_metric]),
                                   np.max(modality_test_df[x_metric]),
                                   num_moving_ave_bins)
        y_metric_ave_list = list()
        for bin_idx in range(len(x_metric_bin) - 1):
            x_metric_left_edge = x_metric_bin[bin_idx]
            x_metric_right_edge = x_metric_bin[bin_idx + 1]

            modality_test_df_subset = modality_test_df.loc[
                (modality_test_df[x_metric] >= x_metric_left_edge) &
                (modality_test_df[x_metric] < x_metric_right_edge)
            ]

            y_metric_ave_list.append(np.mean(modality_test_df_subset[y_metric]))

        # pdb.set_trace()
        ax.plot(x_metric_bin[:-1], y_metric_ave_list, color='black', linewidth=1)

    if custom_x_lim is None:
        ax.set_xlim([-global_abs_max - offset, global_abs_max + offset])
    else:
        ax.set_xlim(custom_x_lim)

    if custom_y_lim is None:
        ax.set_ylim([-global_abs_max - offset, global_abs_max + offset])
    else:
        ax.set_ylim(custom_y_lim)

    if include_zero_axis_line:
        ax.axvline(0, linestyle='--', color='gray', linewidth=1, alpha=0.5)
        ax.axhline(0, linestyle='--', color='gray', linewidth=1, alpha=0.5)



    return fig, ax


def plot_permutation_test_result(true_sample, shuffle_samples,
                                 num_bins=100, percentile_score=None,
                                 fig=None, ax=None):

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4)

    ax.hist(shuffle_samples, bins=num_bins, alpha=0.4)
    ax.axvline(true_sample, color='black', linewidth=1.5)
    ax.set_xlabel('Test statistic', size=12)
    ax.set_ylabel('Counts', size=12)

    if percentile_score is not None:
        ax.set_title('p = %.3f' % percentile_score, size=12)

    return fig, ax


def plot_windowed_decoding_threshold_crossing(cond_1_ds, cond_2_ds, decoding_score_per_window, peri_event_time,
                                              peri_event_time_window_end, decoding_threshold=0.55):

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(peri_event_time, cond_1_ds.mean('Trial'), color='blue')
    axs[0].plot(peri_event_time, cond_2_ds.mean('Trial'), color='red')
    axs[1].plot(peri_event_time_window_end, decoding_score_per_window)
    axs[1].axhline(decoding_threshold, color='gray', linestyle='--')

    threshold_crossing_frame = np.where(decoding_score_per_window >= decoding_threshold)[0]

    if len(threshold_crossing_frame) > 0:
        threshold_crossing_time = peri_event_time_window_end[threshold_crossing_frame]
        threshold_crossing_time_post_stim = threshold_crossing_time[threshold_crossing_time >= 0]

        if len(threshold_crossing_time_post_stim) > 0:
            axs[1].set_title('Time of threshold crossing: %.2f' % threshold_crossing_time_post_stim[0], size=12)

    fig.tight_layout()

    return fig, axs


def plot_pairwise_test(exp_and_brain_region_grouped_df, brain_region_var_name, m_comp,
                       brain_order_to_compare, p_value_metric_name='adjusted_pval',
                       discrete_p_val_sizes=False, custom_p_value_matrix=None, transpose_results=False, flip_sign=False,
                       sig_threshold=0.05,
                       fig=None, ax=None):
    """
    Plots the pairwise mean difference in relative decoding accuracy between brain regions, such that the color
    denotes the mean difference and the size of the circles denote the statistical significance.

    exp_and_brain_region_grouped_df : (pandas dataframe)
    brain_region_var_name : (list)
    m_comp : (stats models object)
    discrete_p_val_sizes : (bool)
    flip_sign : (bool)

    """

    color_as_size_effect = True
    effect_sizes = -m_comp.meandiffs

    num_brain_regions = len(np.unique(exp_and_brain_region_grouped_df[brain_region_var_name]))
    brain_regions = np.unique(exp_and_brain_region_grouped_df[brain_region_var_name])
    significance_matrix = np.zeros((num_brain_regions, num_brain_regions)) + np.nan
    effect_size_matrix = np.zeros((num_brain_regions, num_brain_regions))
    p_value_matrix = np.zeros((num_brain_regions, num_brain_regions))

    is_sig_matrix = np.zeros((num_brain_regions, num_brain_regions))

    n_comparison = 0
    brain_region_idx = np.arange(num_brain_regions)

    if custom_p_value_matrix is None:
        if type(m_comp) == pd.core.frame.DataFrame:
            p_values = m_comp[p_value_metric_name]
        else:
            p_values = m_comp.pvalues


    if discrete_p_val_sizes:
        if custom_p_value_matrix is None:
            p_values[p_values > sig_threshold] = 0.5
            p_values[(p_values < sig_threshold) & (p_values > 0.001)] = sig_threshold
            p_values[(p_values < 0.001)] = 0.001
        else:
            custom_p_value_matrix[custom_p_value_matrix > sig_threshold] = 0.5
            custom_p_value_matrix[(custom_p_value_matrix < sig_threshold) & (custom_p_value_matrix > 0.001)] = sig_threshold
            custom_p_value_matrix[(custom_p_value_matrix < 0.001)] = 0.001

    if custom_p_value_matrix is None:
        for group1_loc, group2_loc in itertools.combinations(brain_region_idx, 2):

            if m_comp.reject[n_comparison]:
                # Flip the sign, so we are doing group 2 vs group 1
                # significance_matrix[group1_loc, group2_loc] = - np.sign(m_comp.meandiffs)[n_comparison]
                significance_matrix[group1_loc, group2_loc] = - np.log10(p_values)[n_comparison]
                effect_size_matrix[group1_loc, group2_loc] = m_comp.meandiffs[n_comparison]
                is_sig_matrix[group1_loc, group2_loc] = 1
            else:
                # set them to some default value
                # significance_matrix[group1_loc, group2_loc] = 0
                # effect_size_matrix[group1_loc, group2_loc] = 0
                significance_matrix[group1_loc, group2_loc] = - np.log10(p_values)[n_comparison]
                effect_size_matrix[group1_loc, group2_loc] = m_comp.meandiffs[n_comparison]

            n_comparison += 1
    else:
        custom_p_value_matrix = np.flipud(np.fliplr(custom_p_value_matrix)[:, ::-1])
        for group1_loc, group2_loc in itertools.combinations(brain_region_idx, 2):

            if custom_p_value_matrix[group1_loc, group2_loc] < sig_threshold:
                # Flip the sign, so we are doing group 2 vs group 1
                # significance_matrix[group1_loc, group2_loc] = - np.sign(m_comp.meandiffs)[n_comparison]
                significance_matrix[group1_loc, group2_loc] = - np.log10(custom_p_value_matrix[group1_loc, group2_loc])
                effect_size_matrix[group1_loc, group2_loc] = m_comp.meandiffs[n_comparison]
                is_sig_matrix[group1_loc, group2_loc] = 1
            else:
                # set them to some default value
                # significance_matrix[group1_loc, group2_loc] = 0
                # effect_size_matrix[group1_loc, group2_loc] = 0
                significance_matrix[group1_loc, group2_loc] = - np.log10(custom_p_value_matrix[group1_loc, group2_loc])
                effect_size_matrix[group1_loc, group2_loc] = m_comp.meandiffs[n_comparison]

            n_comparison += 1
        # pdb.set_trace()
    # cmap = mpl.colors.LinearSegmentedColormap.from_list(
    #     'Custom cmap', cmaplist, 3)

    cmap = 'bwr'

    # use discs
    # the radius is some version of the significance

    color_vmin = -0.2
    color_vmax = 0.2

    disc_vmin = 0
    disc_vmax = 3
    disc_radius_max = 0.4
    disc_radius_min = 0.05
    disc_x, disc_y = np.meshgrid(np.arange(len(brain_order_to_compare)),
                                 np.arange(len(brain_order_to_compare)))

    disc_radius_left = (disc_radius_max - disc_radius_min) * (significance_matrix - disc_vmin) / (
                disc_vmax - disc_vmin) + disc_radius_min

    circles_left = [plt.Circle((j, i), radius=r) for r, j, i in zip(disc_radius_left.flat, disc_x.flat, disc_y.flat)]

    if transpose_results:
        num_disc = len(brain_order_to_compare)
        circles_left = [plt.Circle((i, num_disc-j-1), radius=r) for r, j, i in zip(disc_radius_left.flat, disc_x.flat, disc_y.flat)]

    circle_outline_color = is_sig_matrix.flatten().tolist()

    for n, element in enumerate(circle_outline_color):
        if element > 0.5:
            circle_outline_color[n] = 'black'
        else:
            circle_outline_color[n] = 'none'

    if flip_sign:
        effect_size_matrix = -effect_size_matrix

    discs = PatchCollection(circles_left, array=effect_size_matrix.flatten(), cmap=cmap,
                            edgecolors=circle_outline_color)
    discs.set_clim([color_vmin, color_vmax])

    # from matplotlib.patches import Patch
    # cmaplist = ['Blue', 'Gray', 'Red']

    if discrete_p_val_sizes:
        circle_size_1 = (disc_radius_max - disc_radius_min) * (-np.log10(0.001) - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min
        circle_size_2 = (disc_radius_max - disc_radius_min) * (-np.log10(0.05) - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min
        circle_size_3 = (disc_radius_max - disc_radius_min) * (-np.log10(0.5) - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min
        label_1 = r'$p < 10^{-3}$'
        label_2 = r'$p < %s$' % sig_threshold
        label_3 = r'$p > %s$' % sig_threshold
    else:
        circle_size_1 = (disc_radius_max - disc_radius_min) * (3 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min
        circle_size_2 = (disc_radius_max - disc_radius_min) * (2 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min
        circle_size_3 = (disc_radius_max - disc_radius_min) * (1 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min
        label_1 = r'$p < 10^{-3}$'
        label_2 = r'$p = 10^{-1}$'
        label_3 = r'$p = 10^{-1}$'
    marker_size_1 = 30
    marker_size_2 = marker_size_1 * (circle_size_2 / circle_size_1)
    marker_size_3 = marker_size_1 * (circle_size_3 / circle_size_1)
    legend_elements = [mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_1,
                                        color='pink', label=label_1, markeredgecolor='black', lw=0),
                       mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_2,
                                        color='pink', label=label_2, markeredgecolor='black', lw=0),
                       mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_3,
                                        color='pink', label=label_3, markeredgecolor='none', lw=0)]

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 4)

    ax.set_xticks(brain_region_idx)
    ax.set_yticks(brain_region_idx)
    ax.set_xticklabels(brain_order_to_compare)

    if transpose_results:
        ax.set_yticklabels(brain_order_to_compare[::-1])
    else:
        ax.set_yticklabels(brain_order_to_compare)

    ax.set_ylabel('Group 1', size=12)
    ax.set_xlabel('Group 2', size=12)

    ax.add_collection(discs)

    ax.set_xlim([-0.5, 5.5])
    ax.set_ylim([5.5, -0.5])

    norm = colors.Normalize(vmin=color_vmin, vmax=color_vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm).set_label(label='Mean difference (Group 1 - Group 2)', size=11)

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.8, 1), labelspacing=2.5)

    return fig, ax



def compile_selectivity_index_result(all_dprime_df_trained, all_dprime_df_naive,
                                     all_dprime_df_naive_shuffled):
    all_dprime_df_trained_aud_left = all_dprime_df_trained.loc[
        all_dprime_df_trained['comparison_type'] == 'audLeftRight'
        ]

    all_dprime_df_naive_aud_left = all_dprime_df_naive.loc[
        all_dprime_df_naive['comparison_type'] == 'audLeftRight'
        ]

    # Trained
    res = sstats.cumfreq(np.abs(all_dprime_df_trained_aud_left['d_prime']).dropna(),
                         numbins=400)
    x = res.lowerlimit + np.linspace(0, res.binsize * res.cumcount.size,
                                     res.cumcount.size)

    # Naive
    res = sstats.cumfreq(np.abs(all_dprime_df_naive_aud_left['d_prime']).dropna(),
                         numbins=400)
    x = res.lowerlimit + np.linspace(0, res.binsize * res.cumcount.size,
                                     res.cumcount.size)

    # Naive shuffled
    naive_aud_on_shuffled = all_dprime_df_naive_shuffled.loc[
        ~np.isnan(all_dprime_df_naive_shuffled['shuffle']) &
        (all_dprime_df_naive_shuffled['comparison_type'] == 'audLeftRight')
        ]

    naive_aud_on_shuffled['d_prime_abs'] = np.abs(naive_aud_on_shuffled['d_prime'])

    shuffled_y = []
    shuffled_x = []

    for shuffle in tqdm(np.unique(all_dprime_df_naive_shuffled['shuffle'].dropna())):
        shuffled_df = all_dprime_df_naive_shuffled.loc[
            all_dprime_df_naive_shuffled['shuffle'] == shuffle
            ]

        shuffled_df['d_prime_abs'] = np.abs(shuffled_df['d_prime'])

        res = sstats.cumfreq(np.abs(shuffled_df['d_prime_abs']).dropna(),
                             numbins=400,
                             defaultreallimits=(0, np.max(np.abs(all_dprime_df_trained_aud_on['d_prime']))))

        x = res.lowerlimit + np.linspace(0, res.binsize * res.cumcount.size,
                                         res.cumcount.size)
        shuffled_x.append(x)
        shuffled_y.append(res.cumcount / np.max(res.cumcount))
        # ax.plot(x, res.cumcount / np.max(res.cumcount), color='gray', label='Shuffled')

    shuffled_y_mean = np.mean(shuffled_y, axis=0)
    shuffled_y_lower = sstats.scoreatpercentile(shuffled_y, 1, axis=0)
    shuffled_y_upper = sstats.scoreatpercentile(shuffled_y, 99, axis=0)
    shuffled_y_std = np.std(shuffled_y, axis=0)

    return None

def plot_decoding_variability_across_experiments(brain_region_stats, jitter_level=0.1,
                                                 fig=None, ax=None):
    """
    brain_region_stats : (dict)
    jitter_level : (float)
    fig : (matplotlib figure object)
    ax : (matplotlib axes object)
    """

    fig, ax = plt.subplots()

    for n_brain_region, brain_region in enumerate(brain_region_stats.keys()):

        n_subject = len(brain_region_stats[brain_region]['within_subject_std'])
        x_loc_jittered = np.random.normal(n_brain_region, jitter_level, n_subject)

        if n_brain_region == len(brain_region_stats.keys()) - 1:
            labels = ['Within subject standard deviation',
                      'All experiment standard deviation',
                      'Standard deviation across mean of subjects']
        else:
            labels = [None, None, None]

        ax.scatter(x_loc_jittered, brain_region_stats[brain_region]['within_subject_std'],
                   color='black', label=labels[0])
        ax.scatter(n_brain_region, brain_region_stats[brain_region]['all_experiment_std'], color='red',
                   label=labels[1])
        ax.scatter(n_brain_region, brain_region_stats[brain_region]['std_across_subject_mean'], color='gray',
                   label=labels[2])

    ax.legend(bbox_to_anchor=(1.04, 0.7))
    ax.set_xticks(np.arange(0, n_brain_region + 1))
    ax.set_xticklabels(brain_region_stats.keys())

    ax.set_ylabel('Standard deviation', size=11)

    return fig, ax


def get_axes_object_max(ax, x_loc=1, object_type='line', verbose=False):
    """
    Obtains the maximum height of any matplotlib object given a specific x location.
    Parameters
    ----------
    ax
    x_loc
    object_type
    verbose
    Returns
    -------
    """
    if object_type == 'line':
        axes_objects = ax.lines
    else:
        axes_objects = ax.get_children()
    y_data_store = list()
    for ax_obj in axes_objects:
        if x_loc in ax_obj.get_xdata():
            if verbose:
                print(ax_obj.get_xdata())
                print(ax_obj.get_ydata())
            y_data_store.extend(ax_obj.get_ydata())

    return np.max(y_data_store)

def add_stat_annot(fig, ax, x_start_list, x_end_list, y_start_list=None, y_end_list=None, line_height=2, stat_list=['*'],
                   text_list=None, text_y_offset=0.2, text_x_offset=-0.01, text_size=12):
    """
    Add annotation indicating statistical significance (mainly to be used in boxplots, but can be generalised
    to any boxplot-like figurse, such as stripplots)
    Note that this is temporarily here, should be a feature of sciplotlib.text
    Parameters
    -----------

    """
    if type(text_x_offset) is not list:
        text_x_offset = np.repeat(text_x_offset, len(x_start_list))
    if type(x_start_list) is not list:
        x_start_list = [x_start_list]
    n_stat = 0
    for x_start, x_end, y_start, y_end, stat in zip(x_start_list, x_end_list, y_start_list, y_end_list, stat_list):
        if y_start is None:
            y_start = get_axes_object_max(ax, x_loc=x_start, object_type='line') + line_height
        if y_end is None:
            max_at_x_end = get_axes_object_max(ax, x_loc=x_end, object_type='line')
            print(max_at_x_end)
            y_end = max_at_x_end + line_height
        y_start_end_max = np.max([y_start, y_end])
        sig_line = ax.plot([x_start, x_start, x_end, x_end], [
         y_start, y_start_end_max + line_height, y_start_end_max + line_height, y_end],
          linewidth=1,
          color='k')
        ax.text(x=((x_start + x_end) / 2 + text_x_offset[n_stat]), y=(y_start_end_max + line_height + text_y_offset),
          s=stat,
          horizontalalignment='center',
          size=text_size)
        n_stat += 1

    return (fig, ax)