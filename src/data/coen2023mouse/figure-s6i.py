"""
This scripts generate figure S6i : visual ccSP and auditory ccSP

Internal notes:
This is from the notebook: figure-4-paper-with-pip
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import sciplotlib.style as splstyle
import glob


def main():
    custom_ylim = [0, 4.5]

    # LEFT PANEL: VISUAL CCSP
    fig_folder = '/media/timsit/Partition 1/reports/figures/figure-4-for-pip/'
    fig_name = '4_visual_left_right_CCCP_custom_ylim'
    fig_ext = '.pdf'

    all_exp_stat_test_df = pd.read_pickle(
        '/media/timsit/Partition 1/data/interim/multispaceworld-CCCP/new-parent-name-vis-lr-second-try/vis_lr_0p4_cccp.pkl')
    p_val = 0.01
    min_fr = 1

    subset_all_exp_stat_test_df = all_exp_stat_test_df.loc[
        all_exp_stat_test_df['meanFr'] >= min_fr
        ]

    subset_all_exp_stat_test_sig_df = subset_all_exp_stat_test_df.loc[
        (subset_all_exp_stat_test_df['pVal'] >= (1 - p_val / 2)) |
        (subset_all_exp_stat_test_df['pVal'] <= (p_val / 2))
        ]

    sig_cell_counts = subset_all_exp_stat_test_sig_df.groupby('CellLoc').agg('count')
    total_cell_counts = subset_all_exp_stat_test_df.groupby('CellLoc').agg('count')

    ### Make FRP into MOs as well
    FRPMOs_total_cell_counts = total_cell_counts.loc['MOs'] + total_cell_counts.loc['FRP']
    FRPMOs_total_cell_counts.name = 'FRPMOs'
    total_cell_counts = total_cell_counts.append([FRPMOs_total_cell_counts])

    FRPMOs_sig_cell_counts = sig_cell_counts.loc['MOs'] + sig_cell_counts.loc['FRP']
    FRPMOs_sig_cell_counts.name = 'FRPMOs'
    sig_cell_counts = sig_cell_counts.append([FRPMOs_sig_cell_counts])

    prop_sig_cell = sig_cell_counts['Cell'] / total_cell_counts['Cell']

    # Remove FRP and MOs, and use FRPMOs and MOs
    prop_sig_cell = prop_sig_cell.drop(['FRP', 'MOs'])

    prop_sig_cell_sorted = prop_sig_cell.sort_values(ascending=False)
    prop_sig_cell_sorted = prop_sig_cell_sorted.rename({'FRPMOs': 'MOs'})

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4)

        # proportion
        # ax.bar(above_min_neuron_count_prop_sig_cccp_sorted.index,
        #        above_min_neuron_count_prop_sig_cccp_sorted)

        # plot percentage instead
        ax.bar(prop_sig_cell_sorted.index,
               prop_sig_cell_sorted * 100)

        # ax.set_xticks(np.arange(len(sorted_mean_relative_score_by_brain_region)))
        ax.set_xticklabels(prop_sig_cell_sorted.index,
                           rotation=30)

        ax.set_ylabel('Neurons with \n significant visual "CCCP" (%)', size=12)
        ax.set_xlabel('Brain region', size=12)

        ax.axhline(1, linestyle='--', color='gray')

        fig.tight_layout()

        if custom_ylim is not None:
            ax.set_ylim(custom_ylim)
        fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')



    # RIGHT PANEL: AUDITORY CCSP
    fig_folder = '/media/timsit/Partition 1/reports/figures/figure-4-for-pip/'
    fig_name = '4_audio_left_right_CCCP_0ms_to_300ms_custom_ylim'
    fig_ext = '.pdf'


    all_exp_stat_test_df = pd.read_pickle(
        '/media/timsit/Partition 1/data/interim/multispaceworld-CCCP/new-parent-names-aud-lr/aud_lr_cccp.pkl')
    p_val = 0.01
    min_fr = 1

    subset_all_exp_stat_test_df = all_exp_stat_test_df.loc[
        all_exp_stat_test_df['meanFr'] >= min_fr
        ]

    subset_all_exp_stat_test_sig_df = subset_all_exp_stat_test_df.loc[
        (subset_all_exp_stat_test_df['pVal'] >= (1 - p_val / 2)) |
        (subset_all_exp_stat_test_df['pVal'] <= (p_val / 2))
        ]

    sig_cell_counts = subset_all_exp_stat_test_sig_df.groupby('CellLoc').agg('count')
    total_cell_counts = subset_all_exp_stat_test_df.groupby('CellLoc').agg('count')

    ### Make FRP into MOs as well
    FRPMOs_total_cell_counts = total_cell_counts.loc['MOs'] + total_cell_counts.loc['FRP']
    FRPMOs_total_cell_counts.name = 'FRPMOs'
    total_cell_counts = total_cell_counts.append([FRPMOs_total_cell_counts])

    FRPMOs_sig_cell_counts = sig_cell_counts.loc['MOs'] + sig_cell_counts.loc['FRP']
    FRPMOs_sig_cell_counts.name = 'FRPMOs'
    sig_cell_counts = sig_cell_counts.append([FRPMOs_sig_cell_counts])

    prop_sig_cell = sig_cell_counts['Cell'] / total_cell_counts['Cell']

    # Remove FRP and MOs, and use FRPMOs and MOs
    prop_sig_cell = prop_sig_cell.drop(['FRP', 'MOs'])

    prop_sig_cell_sorted = prop_sig_cell.sort_values(ascending=False)
    prop_sig_cell_sorted = prop_sig_cell_sorted.rename({'FRPMOs': 'MOs'})

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4)

        # proportion
        # ax.bar(above_min_neuron_count_prop_sig_cccp_sorted.index,
        #        above_min_neuron_count_prop_sig_cccp_sorted)

        # plot percentage instead
        ax.bar(prop_sig_cell_sorted.index,
               prop_sig_cell_sorted * 100)

        # ax.set_xticks(np.arange(len(sorted_mean_relative_score_by_brain_region)))
        ax.set_xticklabels(prop_sig_cell_sorted.index,
                           rotation=30)

        ax.set_ylabel('Neurons with \n significant auditory "CCCP" (%)', size=12)
        ax.set_xlabel('Brain region', size=12)

        ax.axhline(1, linestyle='--', color='gray')

        fig.tight_layout()

        if custom_ylim is not None:
            ax.set_ylim(custom_ylim)
        fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')
