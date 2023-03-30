"""
This script plots figure S7t of the paper.
This is the motion energy in passive vs. active.

Internal notes
 - this is from the script batch_multispaceworld_compare_passive_active.py
 - TODO: convert pickle file to csv
"""

import numpy as np, xarray as xr, argparse, logging, os, glob, pandas as pd, pims
import src.data.process_facecam as pface
import re, pdb
import matplotlib.pyplot as plt
import sciplotlib.style as splstyle
from collections import defaultdict
import matplotlib as mpl
import scipy.stats as sstats



def plot_paired_exp_comparison(active_vs_passive_df, fig=None, ax=None, print_test_stat=True,
                               p_val_threshold=0.0001, test_stat_y_loc=0.5):
    """
    active_vs_passive_df: panda dataframe
    """
    with plt.style.context(splstyle.get_style('nature-reviews')):
        cmap_name = 'tab10'
        cmap = plt.get_cmap(cmap_name).colors
        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
                fig.set_size_inches(4, 4)
        xtick_locs = [
         0, 1]
        subjects = np.unique(active_vs_passive_df['subject-num'])
        for n_subject, subject in enumerate(subjects):
            subject_df = active_vs_passive_df.loc[(active_vs_passive_df['subject-num'] == subject)]
            num_exp = len(subject_df)
            y_vals = [
             subject_df['active-motion-energy'], subject_df['passive-motion-energy']]
            y_vals = [np.abs(y) for y in y_vals]
            y_vals = np.array(y_vals)
            x_vals = np.tile(xtick_locs, (num_exp, 1))
            if cmap_name is None:
                color = 'gray'
            else:
                color = cmap[n_subject]
            ax.plot((x_vals.T), y_vals, color=color)

        custom_legend = [mpl.lines.Line2D([0], [0], color=(cmap[x]), lw=2, label=('%s' % (x + 1))) for x in np.arange(len(subjects))]
        ax.legend(handles=custom_legend, title='Mouse')
        if print_test_stat:
            test_stat, p_val = sstats.ttest_rel(active_vs_passive_df['active-motion-energy'], active_vs_passive_df['passive-motion-energy'])
            ax.text(0.8, test_stat_y_loc, ('$t=%.3f$' % test_stat), transform=(ax.transAxes))
            if p_val < p_val_threshold:
                ax.text(0.8, (test_stat_y_loc - 0.05), ('$p < 10^{%.f}$' % np.ceil(np.log10(p_val))), transform=(ax.transAxes))
            else:
                ax.text(0.8, (test_stat_y_loc - 0.05), ('p=%.4f' % p_val), transform=(ax.transAxes))
        ax.set_xticklabels(['Active', 'Passive'], size=12)
        ax.set_xticks(xtick_locs)
        ax.set_xlim([-0.2, 1.2])
        ax.set_ylabel('Motion energy', size=12)
    return fig, ax



def main():

    fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
    fig_name = 'fig-s7t.pdf'

    active_vs_passive_df_path = '/Volumes/Partition 1/reports/figures/active-vs-passive-facemap/active_vs_passive_df.pkl'

    active_vs_passive_df = pd.read_pickle(active_vs_passive_df_path)


    fig, ax = plot_paired_exp_comparison(active_vs_passive_df,
                                         fig=None, ax=None, print_test_stat=True,
                                         p_val_threshold=0.0001, test_stat_y_loc=0.5)
    fig.savefig((os.path.join(fig_folder, fig_name)), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()