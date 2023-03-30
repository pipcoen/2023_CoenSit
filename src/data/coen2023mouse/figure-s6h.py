"""
This script generates figure S6h from the paper
They are the disc plots comparing selectivity index for visual (left panel), auditory (middle panel) and
choice decoding (right panel)

Internal notes:
This is from notebook : 20.5-multispaceworld-quantify-selectivity-index-in-active-condition
"""
from matplotlib.collections import PatchCollection
from matplotlib import colors
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
import string
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib.collections import PatchCollection
from matplotlib import colors

def main():

    # LEFT PANEL : VISUAL STIMULUS D PRIME DISC PLOT
    save_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-trained-neurons-selectivity/'
    save_name = 'all_exp_stim_dprime_0to300ms_df.csv'
    all_exp_dprime_df = pd.read_csv(os.path.join(save_folder, save_name))
    all_exp_dprime_df = all_exp_dprime_df.reset_index()
    all_exp_dprime_df['d_prime'] = all_exp_dprime_df['d_prime'].astype(float)
    subset_all_exp_dprime_df = all_exp_dprime_df.loc[
        all_exp_dprime_df['d_prime'].apply(lambda x: type(x) in [np.ndarray, int, np.int64, float, np.float64])
    ].dropna()
    subset_all_exp_dprime_df = subset_all_exp_dprime_df.replace('FRP', 'MOs')

    group_by_exp = True

    brain_grouped_dprime = []
    target_comparison_type = 'visLeftRight'
    comparision_type_df = subset_all_exp_dprime_df.loc[
        subset_all_exp_dprime_df['comparison_type'] == target_comparison_type
        ]

    comparision_type_df['abs_d_prime'] = np.abs(comparision_type_df['d_prime'])

    if group_by_exp:
        comparision_type_df = comparision_type_df.groupby(['Exp', 'brain_region']).agg('mean').reset_index()

    # brain_order_to_compare = np.unique(comparision_type_df['brain_region'])
    # brain_order_to_compare = ['MOs', 'ACA', 'ORB', 'ILA', 'PL', 'OLF']

    brain_order_to_compare = ['MOs', 'PL', 'ORB', 'ACA', 'ILA', 'OLF']

    brain_new_name = list(string.ascii_lowercase)[0:len(brain_order_to_compare)]
    comparision_type_df['brain_region'] = comparision_type_df['brain_region'].astype(str)
    map_to_new_name = dict(zip(brain_order_to_compare, brain_new_name))

    # Map to new name to reinforce order of comparison I want
    comparision_type_df['brain_region'] = [map_to_new_name[x] for x in comparision_type_df['brain_region'].values]

    metric_name = 'abs_d_prime'
    brain_region_var_name = 'brain_region'

    new_brain_order = [map_to_new_name[x] for x in brain_order_to_compare]
    for brain_region in new_brain_order:
        brain_region_df = comparision_type_df.loc[
            comparision_type_df['brain_region'] == brain_region
            ]

        brain_region_dprime = brain_region_df['abs_d_prime']

        brain_grouped_dprime.append(brain_region_dprime.values)

    stats, p_val = sstats.f_oneway(*brain_grouped_dprime)

    print('One way anova p-value: %.4f' % p_val)
    print(p_val)

    print('F = %.4f' % stats)
    print('One way anova p-value: %.4f' % p_val)
    print(p_val)

    # perform multiple pairwise comparison (Tukey HSD)
    m_comp = pairwise_tukeyhsd(endog=comparision_type_df[metric_name],
                               groups=comparision_type_df[brain_region_var_name], alpha=0.05)
    print(m_comp)

    transpose = True
    color_as_size_effect = True
    effect_sizes = -m_comp.meandiffs

    num_brain_regions = len(np.unique(subset_all_exp_dprime_df[brain_region_var_name]))
    brain_regions = np.unique(subset_all_exp_dprime_df[brain_region_var_name])
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
    # circle_size_2 = (disc_radius_max - disc_radius_min) * (2 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min
    circle_size_2 = (disc_radius_max - disc_radius_min) * (1.3 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min

    # circle_size_3 = (disc_radius_max - disc_radius_min) * (1 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min

    marker_size_1 = 30
    marker_size_2 = marker_size_1 * (circle_size_2 / circle_size_1)
    # marker_size_3 = marker_size_1 * (circle_size_3 / circle_size_1)

    # legend_elements = [mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_1,
    #                                    color='pink', label=r'$p < 10^{-3}$', markeredgecolor='black', lw=0),
    #                  mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_2,
    #                                    color='pink', label=r'$p = 10^{-2}$', markeredgecolor='black', lw=0),
    #                  mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_3,
    #                                    color='pink', label=r'$p = 10^{-1}$', markeredgecolor='none', lw=0)]

    legend_elements = [mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_1,
                                        color='pink', label=r'$p < 0.001$', markeredgecolor='black', lw=0),
                       mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_2,
                                        color='pink', label=r'$p = 0.05$', markeredgecolor='black', lw=0)]

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
            ax.set_ylabel('Group 2', size=12)
            ax.set_xlabel('Group 1', size=12)
        else:
            ax.set_ylabel('Group 1', size=12)
            ax.set_xlabel('Group 2', size=12)

        ax.add_collection(discs)

        ax.set_xlim([-0.5, 5.5])
        ax.set_ylim([5.5, -0.5])

        # TODO: cbar should be group 1 - group 2
        norm = colors.Normalize(vmin=color_vmin, vmax=color_vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, label='Mean difference')

        ax.legend(handles=legend_elements, bbox_to_anchor=(1.8, 1), labelspacing=2.5)

        fig_folder = '/media/timsit/Partition 1/reports/figures/revision-figs-for-pip/'
        # fig_name = 'active_visLeftRight_selectivity_all_brain_region_stats_indv_neurons.pdf'
        fig_name = 'active_visLeftRight_selectivity_all_brain_region_stats_exp_grouped_v2_new_order.pdf'
        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')



    # MIDDLE PANEL: AUDITORY STIMULUS D PRIME DISC PLOT

    # load aud left/right results
    save_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-trained-neurons-selectivity/'
    save_name = 'all_exp_stim_dprime_0to300ms_df.csv'
    all_exp_dprime_df = pd.read_csv(os.path.join(save_folder, save_name))
    all_exp_dprime_df = all_exp_dprime_df.reset_index()
    all_exp_dprime_df['d_prime'] = all_exp_dprime_df['d_prime'].astype(float)
    subset_all_exp_dprime_df = all_exp_dprime_df.loc[
        all_exp_dprime_df['d_prime'].apply(lambda x: type(x) in [np.ndarray, int, np.int64, float, np.float64])
    ].dropna()
    subset_all_exp_dprime_df = subset_all_exp_dprime_df.replace('FRP', 'MOs')

    group_by_exp = True

    brain_grouped_dprime = []
    target_comparison_type = 'audLeftRight'
    comparision_type_df = subset_all_exp_dprime_df.loc[
        subset_all_exp_dprime_df['comparison_type'] == target_comparison_type
        ]

    comparision_type_df['abs_d_prime'] = np.abs(comparision_type_df['d_prime'])
    if group_by_exp:
        comparision_type_df = comparision_type_df.groupby(['Exp', 'brain_region']).agg('mean').reset_index()

    # brain_order_to_compare = np.unique(comparision_type_df['brain_region'])
    # brain_order_to_compare = ['MOs', 'ACA', 'ORB', 'ILA', 'PL', 'OLF']

    # 2021-11-26: New brain region order
    brain_order_to_compare = ['MOs', 'PL', 'ORB', 'ACA', 'ILA', 'OLF']

    brain_new_name = list(string.ascii_lowercase)[0:len(brain_order_to_compare)]

    # brain_new_name = ['MOs', 'ACA', 'ORB', 'ILA', 'PL', 'OLF']

    comparision_type_df['brain_region'] = comparision_type_df['brain_region'].astype(str)
    map_to_new_name = dict(zip(brain_order_to_compare, brain_new_name))

    # Map to new name to reinforce order of comparison I want
    comparision_type_df['brain_region'] = [map_to_new_name[x] for x in comparision_type_df['brain_region'].values]

    # subset_all_exp_dprime_df['abs_d_prime'] = np.abs(subset_all_exp_dprime_df['d_prime'])

    metric_name = 'abs_d_prime'
    brain_region_var_name = 'brain_region'

    new_brain_order = [map_to_new_name[x] for x in brain_order_to_compare]
    for brain_region in new_brain_order:
        brain_region_df = comparision_type_df.loc[
            comparision_type_df['brain_region'] == brain_region
            ]

        brain_region_dprime = brain_region_df['abs_d_prime']

        brain_grouped_dprime.append(brain_region_dprime.values)

    stats, p_val = sstats.f_oneway(*brain_grouped_dprime)

    print('One way anova p-value: %.4f' % p_val)
    print(p_val)

    print('F = %.4f' % stats)
    print('One way anova p-value: %.4f' % p_val)
    print(p_val)



    # perform multiple pairwise comparison (Tukey HSD)
    m_comp = pairwise_tukeyhsd(endog=comparision_type_df[metric_name],
                               groups=comparision_type_df[brain_region_var_name], alpha=0.05)
    print(m_comp)



    transpose = True

    color_as_size_effect = True
    effect_sizes = -m_comp.meandiffs

    num_brain_regions = len(np.unique(subset_all_exp_dprime_df[brain_region_var_name]))
    brain_regions = np.unique(subset_all_exp_dprime_df[brain_region_var_name])
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
    # circle_size_2 = (disc_radius_max - disc_radius_min) * (2 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min
    circle_size_2 = (disc_radius_max - disc_radius_min) * (1.3 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min

    # circle_size_3 = (disc_radius_max - disc_radius_min) * (1 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min

    marker_size_1 = 30
    marker_size_2 = marker_size_1 * (circle_size_2 / circle_size_1)
    # marker_size_3 = marker_size_1 * (circle_size_3 / circle_size_1)

    # legend_elements = [mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_1,
    #                                    color='pink', label=r'$p < 10^{-3}$', markeredgecolor='black', lw=0),
    #                  mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_2,
    #                                    color='pink', label=r'$p = 10^{-2}$', markeredgecolor='black', lw=0),
    #                  mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_3,
    #                                    color='pink', label=r'$p = 10^{-1}$', markeredgecolor='none', lw=0)]

    legend_elements = [mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_1,
                                        color='pink', label=r'$p < 0.001$', markeredgecolor='black', lw=0),
                       mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_2,
                                        color='pink', label=r'$p = 0.05$', markeredgecolor='black', lw=0)]

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 4)
        # ax.imshow(significance_matrix, cmap=cmap)
        ax.set_xticks(brain_region_idx)
        ax.set_yticks(brain_region_idx)
        ax.set_xticklabels(brain_order_to_compare)

        if transpose:
            ax.set_yticklabels(brain_order_to_compare[::-1])
        else:
            ax.set_yticklabels(brain_order_to_compare)

        if transpose:
            ax.set_ylabel('Group 2', size=12)
            ax.set_xlabel('Group 1', size=12)
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
        # cbar.ax.tick_params(labelsize=8)

        ax.legend(handles=legend_elements, bbox_to_anchor=(1.8, 1), labelspacing=2.5)

        fig_folder = '/media/timsit/Partition 1/reports/figures/revision-figs-for-pip/'
        fig_name = 'active_audLeftRight_selectivity_all_brain_region_stats_exp_grouped_v2_new_order.pdf'
        # fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


    # RIGHT PANEL : CHOICE D PRIME DISC PLOT
    # load data
    active_moveLeftRight_result_save_path = '/media/timsit/Partition 1/data/interim/multispaceworld-trained-neurons-selectivity/all_exp_moveLeftRight_dprime_-130msto0ms.csv'
    all_exp_dprime_df = pd.read_csv(active_moveLeftRight_result_save_path)
    # all_exp_dprime_df = pd.concat(all_dprime_df)
    all_exp_dprime_df = all_exp_dprime_df.reset_index()
    all_exp_dprime_df['d_prime'] = all_exp_dprime_df['d_prime'].astype(float)
    subset_all_exp_dprime_df = all_exp_dprime_df.loc[
        all_exp_dprime_df['d_prime'].apply(lambda x: type(x) in [np.ndarray, int, np.int64, float, np.float64])
    ].dropna()
    subset_all_exp_dprime_df = subset_all_exp_dprime_df.replace('FRP', 'MOs')

    # Do the stats

    group_by_exp = True

    brain_grouped_dprime = []
    target_comparison_type = 'moveLeftRight'
    comparision_type_df = subset_all_exp_dprime_df.loc[
        subset_all_exp_dprime_df['comparison_type'] == target_comparison_type
        ]

    comparision_type_df['abs_d_prime'] = np.abs(comparision_type_df['d_prime'])

    if group_by_exp:
        comparision_type_df = comparision_type_df.groupby(['Exp', 'brain_region']).agg('mean').reset_index()

    # brain_order_to_compare = np.unique(comparision_type_df['brain_region'])
    # brain_order_to_compare = ['MOs', 'ACA', 'ORB', 'ILA', 'PL', 'OLF']

    # new order
    brain_order_to_compare = ['MOs', 'PL', 'ORB', 'ACA', 'ILA', 'OLF']

    brain_new_name = list(string.ascii_lowercase)[0:len(brain_order_to_compare)]
    comparision_type_df['brain_region'] = comparision_type_df['brain_region'].astype(str)
    map_to_new_name = dict(zip(brain_order_to_compare, brain_new_name))

    # Map to new name to reinforce order of comparison I want
    comparision_type_df['brain_region'] = [map_to_new_name[x] for x in comparision_type_df['brain_region'].values]

    # subset_all_exp_dprime_df['abs_d_prime'] = np.abs(subset_all_exp_dprime_df['d_prime'])

    metric_name = 'abs_d_prime'
    brain_region_var_name = 'brain_region'

    new_brain_order = [map_to_new_name[x] for x in brain_order_to_compare]
    for brain_region in new_brain_order:
        brain_region_df = comparision_type_df.loc[
            comparision_type_df['brain_region'] == brain_region
            ]

        brain_region_dprime = brain_region_df['abs_d_prime']

        brain_grouped_dprime.append(brain_region_dprime.values)

    stats, p_val = sstats.f_oneway(*brain_grouped_dprime)

    print('One way anova p-value: %.4f' % p_val)
    print(p_val)

    print('F = %.4f' % stats)
    print('One way anova p-value: %.4f' % p_val)
    print(p_val)

    # perform multiple pairwise comparison (Tukey HSD)
    m_comp = pairwise_tukeyhsd(endog=comparision_type_df[metric_name],
                               groups=comparision_type_df[brain_region_var_name], alpha=0.05)
    print(m_comp)


    transpose = True
    color_as_size_effect = True
    effect_sizes = -m_comp.meandiffs

    num_brain_regions = len(np.unique(subset_all_exp_dprime_df[brain_region_var_name]))
    brain_regions = np.unique(subset_all_exp_dprime_df[brain_region_var_name])
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
    # circle_size_2 = (disc_radius_max - disc_radius_min) * (2 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min
    circle_size_2 = (disc_radius_max - disc_radius_min) * (1.3 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min
    # circle_size_3 = (disc_radius_max - disc_radius_min) * (1 - disc_vmin) / (disc_vmax - disc_vmin) + disc_radius_min

    marker_size_1 = 30
    marker_size_2 = marker_size_1 * (circle_size_2 / circle_size_1)
    # marker_size_3 = marker_size_1 * (circle_size_3 / circle_size_1)
    # legend_elements = [mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_1,
    #                                    color='pink', label=r'$p < 10^{-3}$', markeredgecolor='black', lw=0),
    #                  mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_2,
    #                                    color='pink', label=r'$p = 10^{-2}$', markeredgecolor='black', lw=0),
    #                  mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_3,
    #                                    color='pink', label=r'$p = 10^{-1}$', markeredgecolor='none', lw=0)]

    legend_elements = [mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_1,
                                        color='pink', label=r'$p < 0.001$', markeredgecolor='black', lw=0),
                       mpl.lines.Line2D([0], [0], marker='o', markersize=marker_size_2,
                                        color='pink', label=r'$p = 0.05$', markeredgecolor='black', lw=0)]

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

        # fig_folder = '/media/timsit/Partition 1/reports/figures/revision-figs-for-pip/'
        # fig_name = 'active_choiceLeftRight_selectivity_all_brain_region_stats_indv_neurons.pdf'
        # fig_name = 'active_choiceLeftRight_selectivity_all_brain_region_exp_grouped_v2_new_order.pdf'
        # fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()