"""
This scripts generate figure S6j : percentage of CCCP neurons

Internal notes:
This is from the notebook: figure-4-paper-with-pip
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import sciplotlib.style as splstyle
import glob


def main():
    fig_folder = ''  # where to save the figure
    fig_name = '4_CCCP_barchart_choiceThreshDirFromChoiceInit_custom_ylim_130ms_to_0ms'
    fig_ext = '.pdf'

    # cccp_results_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-CCCP/new-parent-names'
    # cccp_results_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-CCCP/choiceThreshDirFromChoiceInit/'

    # 2021-03-21, Taking 130 ms - 0 ms aligned to stimulus onset
    cccp_results_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-CCCP/choiceThreshDirFromChoiceInit_130ms_before_movement/'

    cccp_results_df = pd.concat(
        pd.read_pickle(fname) for fname in glob.glob(os.path.join(
            cccp_results_folder, '*.pkl'
        ))
    )

    alpha = 0.005
    min_neuron_count = 400
    brain_region_to_exclude = ['Nan']

    # Merge FRP and MOs
    cccp_results_df['CellLoc'] = cccp_results_df['CellLoc'].replace({'FRPMOs': 'MOs'})

    # Get significant CCCP neurons (either side)
    sig_cccp_results_df = cccp_results_df.loc[
        (cccp_results_df['pVal'] <= alpha) |
        (cccp_results_df['pVal'] >= (1 - alpha))
        ]

    sig_cccp_cell_counts = sig_cccp_results_df.groupby('CellLoc').agg('count')['Cell']
    cell_counts = cccp_results_df.groupby('CellLoc').agg('count')['Cell']

    prop_sig_cccp_cell_counts = sig_cccp_cell_counts / cell_counts
    percentage_sig_cccp_cell_counts = prop_sig_cccp_cell_counts * 100

    above_min_neuron_count_brain_regions = cell_counts.loc[
        cell_counts >= min_neuron_count
        ].index

    above_min_neuron_count_prop_sig_cccp = prop_sig_cccp_cell_counts[
        above_min_neuron_count_brain_regions
    ]

    above_min_neuron_count_prop_sig_cccp_sorted = above_min_neuron_count_prop_sig_cccp.sort_values(ascending=False)
    above_min_neuron_count_prop_sig_cccp_sorted = above_min_neuron_count_prop_sig_cccp_sorted.loc[
        ~above_min_neuron_count_prop_sig_cccp_sorted.index.isin(brain_region_to_exclude)
    ]

    custom_ylim = [0, 4.5]

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4)

        # proportion
        # ax.bar(above_min_neuron_count_prop_sig_cccp_sorted.index,
        #        above_min_neuron_count_prop_sig_cccp_sorted)

        # plot percentage instead
        ax.bar(above_min_neuron_count_prop_sig_cccp_sorted.index,
               above_min_neuron_count_prop_sig_cccp_sorted * 100)

        # ax.set_xticks(np.arange(len(sorted_mean_relative_score_by_brain_region)))
        ax.set_xticklabels(above_min_neuron_count_prop_sig_cccp_sorted.index,
                           rotation=30)

        ax.set_ylabel('Neurons with \n significant CCCP (%)', size=12)
        ax.set_xlabel('Brain region', size=12)

        ax.axhline(1, linestyle='--', color='gray')

        if custom_ylim is not None:
            ax.set_ylim(custom_ylim)

        fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), bbox_inches='tight')

if __name__ == '__main__':
    main()