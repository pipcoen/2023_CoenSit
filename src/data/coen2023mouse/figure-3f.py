"""
This scripts generates figure 3f : Auditory stimulus decoding

Internal notes:
top panel is from : figure-4-paper-w-pip (note to use the newest decoding result!)
bottom panel is from: figure-4-paper-w-pip-reviewer-updates-checkpoint-2021-11-30
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

from matplotlib.collections import PatchCollection
from matplotlib import colors
import string

def main():

    # TOP PANEL
    # all_classification_results_df = pd.read_pickle('/media/timsit/Partition 1/data/interim/active-decode-stim/decode-multimodal-aud-lr/aud_lr_subset_30_neurons_no_balancing_100_to_300ms.pkl')

    # March 3 2021: Use 0 - 300 ms window
    # all_classification_results_df = pd.read_pickle('/media/timsit/Partition 1/data/interim/active-decode-stim/aud_lr_subset_30_neurons_no_balancing_0_to_300ms.pkl')

    # March 23 2023
    all_classification_results_df = pd.read_pickle(
        '/Volumes/Partition 1/data/interim/active-decode-stim/aud_lr_subset_30_neurons_no_balancing_0_to_300ms_5_subsamples_min_25_labels.pkl')

    # Maker the ordering the same as vis LR
    custom_brain_region_order = ['MOs', 'ACA', 'PL', 'ORB', 'ILA', 'OLF']

    fig_name = '4_decode_aud_LR_subset_30_neuron_no_balancing_0_to_300ms_custom_ylim_w_pval'
    fig_ext = '.pdf'

    custom_subset_brain_regions = ['MOs', 'ACA', 'PL', 'OLF', 'ORB', 'ILA']
    all_classification_results_df = all_classification_results_df.loc[
        all_classification_results_df['brainRegion'].isin(custom_subset_brain_regions)
    ]

    # Calculate relative accuracy
    all_classification_results_df['accuracyRelBaseline'] = (all_classification_results_df['mean_classifier_score'] -
                                                            all_classification_results_df['mean_control_score']) / \
                                                           (1 - all_classification_results_df['mean_control_score'])

    all_classification_results_df = all_classification_results_df.reset_index()
    all_classification_results_df['exp'] = all_classification_results_df['exp'].astype(int)

    mean_accuracy_df = all_classification_results_df.groupby('brainRegion').agg('mean').reset_index()
    mean_accuracy_df = mean_accuracy_df.loc[
        mean_accuracy_df['brainRegion'].isin(custom_subset_brain_regions)
    ]

    plot_sig_star = True
    plot_p_val = True
    verbose = True

    jitter_level = 0.05
    show_sem = True
    verbose = False
    y_metric = 'accuracyRelBaseline'

    lower_limit = -0.3
    custom_ylim = [-0.3, 1]

    mean_accuracy_df_sorted = mean_accuracy_df.sort_values(by=y_metric, ascending=False)

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4)

        fig, ax = vizmodel.plot_brain_region_decoding_acc_comparison(all_classification_results_df,
                                                                     exp_var_name='exp',
                                                                     acc_metric='accuracyRelBaseline',
                                                                     jitter_level=0.05,
                                                                     dot_size=5, lower_limit=lower_limit,
                                                                     plot_sig_star=plot_sig_star, plot_p_val=plot_p_val,
                                                                     verbose=verbose,
                                                                     custom_brain_region_order=custom_brain_region_order,
                                                                     fig=fig, ax=ax)
        ax.set_ylabel('Accuracy relative to baseline', size=12)

        ax.set_title('Decoding audio left/right', size=12)
        if custom_ylim is not None:
            ax.set_ylim(custom_ylim)

        # fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')

    # BOTTOM PANEL
    # decoding_result_path = '/home/timothysit/Documents/msi-key-data/active-decode-stim/aud_lr_subset_30_neurons_no_balancing_0_to_300ms_5_subsamples_min_25_labels.pkl'
    # decoding_result_path = '/media/timsit/Partition 1/data/interim/active-decode-stim/aud_lr_subset_30_neurons_no_balancing_0_to_300ms_5_subsamples_min_25_labels.pkl'
    decoding_result_path = '/Volumes/Partition 1/data/interim/active-decode-stim/aud_lr_subset_30_neurons_no_balancing_0_to_300ms_5_subsamples_min_25_labels.pkl'

    all_classification_results_df = pd.read_pickle(decoding_result_path)
    brain_region_var_name = 'brainRegion'
    acc_metric = 'accuracyRelBaseline'
    custom_subset_brain_regions = ['MOs', 'ACA', 'PL', 'OLF', 'ORB', 'ILA']
    all_classification_results_df = all_classification_results_df.loc[
        all_classification_results_df[brain_region_var_name].isin(custom_subset_brain_regions)
    ]

    # Calculate relative accuracy
    all_classification_results_df['accuracyRelBaseline'] = (all_classification_results_df['mean_classifier_score'] -
                                                            all_classification_results_df['mean_control_score']) / \
                                                           (1 - all_classification_results_df['mean_control_score'])
    all_classification_results_df = all_classification_results_df.reset_index()
    all_classification_results_df['exp'] = all_classification_results_df['exp'].astype(int)

    mean_accuracy_df = all_classification_results_df.groupby('brainRegion').agg('mean').reset_index()
    mean_accuracy_df = mean_accuracy_df.loc[
        mean_accuracy_df['brainRegion'].isin(custom_subset_brain_regions)
    ]


    brain_grouped_dprime = []
    acc_metric = 'accuracyRelBaseline'
    exp_var_name = 'exp'
    brain_region_var_name = 'brainRegion'
    cv_ave_method = 'mean'
    exp_and_brain_region_grouped_df = all_classification_results_df.groupby(
        [exp_var_name, brain_region_var_name]).agg(
        cv_ave_method).reset_index()

    all_exp_grouped_accuracy = exp_and_brain_region_grouped_df.groupby(brain_region_var_name).agg(cv_ave_method)

    brain_region_order = all_exp_grouped_accuracy.sort_values(acc_metric, ascending=False).reset_index()[
        'brainRegion'].values

    # subset brain region used in paper
    exp_and_brain_region_grouped_df = exp_and_brain_region_grouped_df.loc[
        exp_and_brain_region_grouped_df[brain_region_var_name].isin(['MOs', 'PL', 'ORB', 'ACA', 'ILA', 'OLF'])
    ]

    # brain_order_to_compare = np.unique(comparision_type_df['brain_region'])
    # brain_order_to_compare = ['MOs', 'ACA', 'ORB', 'ILA', 'PL', 'OLF']
    brain_order_to_compare = ['MOs', 'PL', 'ORB', 'ACA', 'ILA', 'OLF']

    brain_new_name = list(string.ascii_lowercase)[0:len(brain_order_to_compare)]
    # exp_and_brain_region_grouped_df[brain_region_var_name] = exp_and_brain_region_grouped_df[brain_region_var_name].astype(str)
    map_to_new_name = dict(zip(brain_order_to_compare, brain_new_name))

    # Map to new name to reinforce order of comparison I want
    exp_and_brain_region_grouped_df[brain_region_var_name] = [map_to_new_name[x] for x in
                                                              exp_and_brain_region_grouped_df[
                                                                  brain_region_var_name].values]

    # metric_name = 'abs_d_prime'
    # brain_region_var_name = 'brain_region'

    new_brain_order = [map_to_new_name[x] for x in brain_order_to_compare]
    for brain_region in new_brain_order:
        brain_region_df = exp_and_brain_region_grouped_df.loc[
            exp_and_brain_region_grouped_df[brain_region_var_name] == brain_region
            ]

        brain_region_dprime = brain_region_df[acc_metric]

        brain_grouped_dprime.append(brain_region_dprime.values)

    stats, p_val = sstats.f_oneway(*brain_grouped_dprime)

    print('One way anova p-value: %.4f' % p_val)
    print(p_val)

    print('F = %.4f' % stats)
    print('One way anova p-value: %.4f' % p_val)
    print(p_val)

    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # perform multiple pairwise comparison (Tukey HSD)
    m_comp = pairwise_tukeyhsd(endog=exp_and_brain_region_grouped_df[acc_metric],
                               groups=exp_and_brain_region_grouped_df[brain_region_var_name], alpha=0.05)
    print(m_comp)


    transpose = True
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

    for group1_loc, group2_loc in itertools.combinations(brain_region_idx, 2):

        if m_comp.reject[n_comparison]:
            # Flip the sign, so we are doing group 2 vs group 1
            # significance_matrix[group1_loc, group2_loc] = - np.sign(m_comp.meandiffs)[n_comparison]
            significance_matrix[group1_loc, group2_loc] = - np.log10(m_comp.pvalues)[n_comparison]
            effect_size_matrix[group1_loc, group2_loc] = -m_comp.meandiffs[n_comparison]
            is_sig_matrix[group1_loc, group2_loc] = 1
        else:
            # set them to some default value
            # significance_matrix[group1_loc, group2_loc] = 0
            # effect_size_matrix[group1_loc, group2_loc] = 0
            significance_matrix[group1_loc, group2_loc] = - np.log10(m_comp.pvalues)[n_comparison]
            effect_size_matrix[group1_loc, group2_loc] = -m_comp.meandiffs[n_comparison]

        n_comparison += 1

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

    if transpose:
        circles_left = [plt.Circle((i, len(brain_order_to_compare) - j - 1), radius=r) for r, j, i in
                        zip(disc_radius_left.flat, disc_x.flat, disc_y.flat)]

    circle_outline_color = is_sig_matrix.flatten().tolist()

    for n, element in enumerate(circle_outline_color):
        if element > 0.5:
            circle_outline_color[n] = 'black'
        else:
            circle_outline_color[n] = 'none'

    discs = PatchCollection(circles_left, array=effect_size_matrix.flatten(), cmap=cmap,
                            edgecolors=circle_outline_color)
    discs.set_clim([color_vmin, color_vmax])

    # from matplotlib.patches import Patch
    # cmaplist = ['Blue', 'Gray', 'Red']

    circle_size_1 = (disc_radius_max - disc_radius_min) * (3 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min
    circle_size_2 = (disc_radius_max - disc_radius_min) * (2 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min
    circle_size_3 = (disc_radius_max - disc_radius_min) * (1 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min

    marker_size_1 = 30
    marker_size_2 = marker_size_1 * (circle_size_2 / circle_size_1)
    marker_size_3 = marker_size_1 * (circle_size_3 / circle_size_1)
    legend_elements = [mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_1,
                                        color='pink', label=r'$p < 10^{-3}$', markeredgecolor='black', lw=0),
                       mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_2,
                                        color='pink', label=r'$p = 10^{-2}$', markeredgecolor='black', lw=0),
                       mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_3,
                                        color='pink', label=r'$p = 10^{-1}$', markeredgecolor='none', lw=0)]

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 4)
        # ax.imshow(significance_matrix, cmap=cmap)
        ax.set_xticks(brain_region_idx)
        ax.set_yticks(brain_region_idx)
        ax.set_xticklabels(brain_order_to_compare)
        ax.set_yticklabels(brain_order_to_compare)

        if transpose:
            ax.set_yticklabels(brain_order_to_compare[::-1])

        if transpose:
            ax.set_xlabel('Group 1', size=12)
            ax.set_ylabel('Group 2', size=12)
        else:
            ax.set_ylabel('Group 1', size=12)
            ax.set_xlabel('Group 2', size=12)

        ax.add_collection(discs)

        ax.set_xlim([-0.5, 5.5])
        ax.set_ylim([5.5, -0.5])

        # TODO: cbar should be group 1 - group 2
        norm = colors.Normalize(vmin=color_vmin, vmax=color_vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm).set_label(label='Mean difference (Group 1 - Group 2)', size=11)

        ax.legend(handles=legend_elements, bbox_to_anchor=(1.8, 1), labelspacing=2.5)

        fig_folder = '/media/timsit/Partition 1/reports/figures/revision-figs-for-pip/'
        fig_name = 'active_audLeftRight_decoding_all_brain_region_stats_new_order.pdf'
        # fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()