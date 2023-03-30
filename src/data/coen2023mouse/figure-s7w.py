"""
This scripts plots figure S7w of the paper.
This is the single neuron decoding plot.
"""

import pandas as pd, pickle as pkl, numpy as np, os, glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
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
import src.data.cal_time_of_separation as caltos
import itertools
import sklearn.linear_model as sklinear
import sklearn.model_selection as sklselect
import sklearn, pdb, inspect




def make_group_comparisons(df, variable_names, group_cond_name='decoding_type', comparison_group_1=[
        'audOn', 'audOn', 'audLR'], comparison_group_2=[
        'audLR', 'visLR', 'visLR'], sig_log_threshold=0.001):
    """
    Compare multiple group conditions, used in plot_single_neuron_decoding_results
    Parameters
    ----------
    df
    variable_names
    comparison_group_1
    comparison_group_2
    sig_log_threshold

    Returns
    -------

    """
    test_stat_list = list()
    p_val_list = list()
    for cg1, cg2 in zip(comparison_group_1, comparison_group_2):
        group_1_vals = df.loc[(df[group_cond_name] == cg1)][variable_names[0]].values
        group_2_vals = df.loc[(df[group_cond_name] == cg2)][variable_names[0]].values
        test_stat, p_val = sstats.ttest_ind(group_1_vals, group_2_vals)
        test_stat_list.append(test_stat)
        p_val_list.append(p_val)

    comparison_stat_results = pd.DataFrame.from_dict({'group1':comparison_group_1,
     'group2':comparison_group_2,
     'testStat':test_stat_list,
     'pVal':p_val_list})
    stat_list = list()
    for test_stat, p_val in zip(comparison_stat_results['testStat'], comparison_stat_results['pVal']):
        if p_val < sig_log_threshold:
            stat_text = '$t=%.3f$ \n $p < 10^{%.f}$' % (test_stat, np.ceil(np.log10(p_val)))
        else:
            stat_text = '$t=%.3f$ \n $p = %.4f$' % (test_stat, p_val)
        stat_list.append(stat_text)

    return stat_list



def plot_single_neuron_decoding_results(single_neuron_decoding_df_path, y_metric='decoding_score', spread_metric='2sem',
                                        custom_ylim=[0, 1], hline_loc=0.5, fig_folder=None,
                                        fig_name='single_neuron_decoding_results', fig_ext='.pdf',
                                        y_label='Decoding accuracy', fig=None, ax=None):

    single_neuron_decoding_df = pd.read_pickle(single_neuron_decoding_df_path)
    min_accuracy = None
    if min_accuracy is not None:
        single_neuron_decoding_df = single_neuron_decoding_df.loc[(single_neuron_decoding_df['decoding_score'] > min_accuracy)]
    group_cond_name = 'decoding_type'
    group_name_order = ['Audio on',
     'Audio left/right',
     'Visual on',
     'Visual left/right']
    box_colors = [
     'Gray',
     'Purple',
     'Black',
     'Orange']
    highlight_neuron_idx_dict = None
    comparison_group_1 = [
     'Audio on', 'Audio on', 'Visual on',
     'Audio left/right']
    comparison_group_2 = ['Audio left/right', 'Visual on', 'Visual left/right',
     'Visual left/right']
    variable_names = [y_metric]
    stat_list = make_group_comparisons(df=single_neuron_decoding_df, variable_names=variable_names,
      comparison_group_1=comparison_group_1,
      comparison_group_2=comparison_group_2)

    with plt.style.context(splstyle.get_style('nature-reviews')):
        if (fig is None) and (ax is None):
            fig, ax = plt.subplots()
            fig.set_size_inches(4, 4)
        fig, ax = vizstat.plot_unpaired_scatter(df=single_neuron_decoding_df, subset_condition=group_cond_name,
          groupby=None,
          agg_metric='mean',
          group_name_order=group_name_order,
          group_colors=box_colors,
          scatter_alpha=0.5,
          y_metric=y_metric,
          fig=fig,
          ax=ax,
          plot_spread=True,
          infer_xticklabels=True,
          jitter_val=0.05,
          dot_size=4,
          highlight_neuron_idx_dict=highlight_neuron_idx_dict,
          spread_metric=spread_metric,
          xlabel_size=8)
        x_start_list = [
         0, 0, 2, 1]
        x_end_list = [1, 2, 3, 3]
        y_start_list = [0.8, 0.9, 0.7, 0.85]
        y_end_list = [0.7, 0.8, 0.7, 0.85]
        fig, ax = vizstat.add_stat_annot(fig=fig, ax=ax, x_start_list=x_start_list, x_end_list=x_end_list, y_start_list=y_start_list,
          y_end_list=y_end_list,
          line_height=0.025,
          stat_list=stat_list,
          text_x_offset=[0, 0, 0, 0.5],
          text_y_offset=0.01,
          text_size=8)
        if hline_loc is not None:
            ax.axhline(hline_loc, linestyle='--', color='gray', lw=1)
        if custom_ylim is not None:
            ax.set_ylim(custom_ylim)
        ax.set_ylabel(y_label, size=12)
    fig.savefig((os.path.join(fig_folder, fig_name + fig_ext)), dpi=300, bbox_inches='tight')


def main():
    fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
    fig_name = 'fig-s7w'
    fig_ext = '.pdf'


    single_neuron_decoding_df_path = '/Volumes/Partition 1/data/interim/discrimination-time-interim-data/decoding/single_neuron_decoding.pkl'
    y_metric = 'decoding_score_rel_baseline'
    spread_metric = '2sem'
    custom_ylim = [-0.2, 1]
    hline_loc = 0
    y_label = 'Decoding accuracy'

    plot_single_neuron_decoding_results(
        single_neuron_decoding_df_path=single_neuron_decoding_df_path,
        y_metric=y_metric,
        spread_metric=spread_metric,
        custom_ylim=custom_ylim,
        hline_loc=hline_loc,
        fig_folder=fig_folder,
        y_label=y_label,
        fig_name=fig_name,
        fig_ext=fig_ext,
    )


if __name__ == '__main__':
    main()
