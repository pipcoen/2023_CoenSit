"""
This script plots figure 4h of the paper.
This is the discrimination time of neurons.

Internal notes:
The code is from the decompiled version of figure_for_paper_discrimination_time.py
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

main_data_folder = '/Volumes/Partition 1/data/interim'


aud_on_cells_aud_on_repsonse_time_fpath = os.path.join(main_data_folder, 'discrimination-time-interim-data/not-smoothed/aud_on_cells_aud_on_threshold_crossing_time.pkl')
aud_lr_cells_aud_lr_response_time_fpath = os.path.join(main_data_folder, 'discrimination-time-interim-data/not-smoothed/aud_lr_cells_aud_lr_threshold_crossing_time.pkl')
vis_lr_cells_vis_lr_response_time_fpath = os.path.join(main_data_folder, 'discrimination-time-interim-data/not-smoothed/vis_lr_cells_vis_lr_threshold_crossing_time.pkl')
aud_lr_cells_aud_on_response_time_fpath = os.path.join(main_data_folder, 'discrimination-time-interim-data/not-smoothed/aud_lr_cells_aud_on_threshold_crossing_time.pkl')
vis_lr_cells_vis_on_response_time_fpath = os.path.join(main_data_folder, 'discrimination-time-interim-data/not-smoothed/vis_lr_cells_vis_on_threshold_crossing_time.pkl')

# fig_folder='/media/timsit/Partition 1/reports/figures/figure-5-for-pip/discrimination-time/'
# fig_name='5_passive_neuron_time_of_mannU_sig_diff_w_pval_dots_and_2sem_w_highlight_neuron_w_visonset_double_check'
fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
fig_name = 'fig-4h.pdf'

include_aud_lr_and_vis_lr_cells_onset=True


def plot_discrimination_time(aud_on_cells_aud_on_repsonse_time_fpath,
                             aud_lr_cells_aud_lr_response_time_fpath,
                             vis_lr_cells_vis_lr_response_time_fpath,
                             aud_lr_cells_aud_on_response_time_fpath=None,
                             vis_lr_cells_vis_on_response_time_fpath=None, fig_folder=None,
                             subset_procedure=None, fig_name=None, include_aud_lr_and_vis_lr_cells_onset=False,
                             use_same_vis_lr_and_vis_on_cells=False, min_time=0.01, max_time=0.3):
    """

    Parameters
    ----------
    aud_on_cells_aud_on_repsonse_time_fpath
    aud_lr_cells_aud_lr_response_time_fpath
    vis_lr_cells_vis_lr_response_time_fpath
    fig_folder
    subset_procedure
    fig_name
    include_aud_lr_and_vis_lr_cells_onset : (bool)
        whether to include audio left/right cells in audio onset time calculation
        and whether to include vis left/right cells in visual onset time calculation
    min_time : (float)
        minimum discrimination time
    max_time : (float)
        maximum discrimination time
    Returns
    -------

    """
    aud_on_threshold_crossing_time = pd.read_pickle(aud_on_cells_aud_on_repsonse_time_fpath)
    aud_lr_threshold_crossing_time = pd.read_pickle(aud_lr_cells_aud_lr_response_time_fpath)
    vis_lr_threshold_crossing_time_subset = np.array(pd.read_pickle(vis_lr_cells_vis_lr_response_time_fpath))
    if include_aud_lr_and_vis_lr_cells_onset:
        aud_lr_cells_aud_on_response_time = np.array(pd.read_pickle(aud_lr_cells_aud_on_response_time_fpath))
        vis_lr_cells_vis_on_response_time = np.array(pd.read_pickle(vis_lr_cells_vis_on_response_time_fpath))
    else:
        aud_lr_cells_aud_on_response_time = None
        vis_lr_cells_vis_on_response_time = None
    if subset_procedure is not None:
        vis_threshold_crossing_time_np = np.array(vis_response_time)
        aud_on_threshold_crossing_time_np = np.array(aud_on_response_time)
        aud_threshold_crossing_time_np = np.array(aud_response_time)
        vis_response_time = vis_threshold_crossing_time_np[((vis_threshold_crossing_time_np <= 0.3) & (vis_threshold_crossing_time_np >= 0.01))]
        aud_on_response_time = aud_on_threshold_crossing_time_np[((aud_on_threshold_crossing_time_np <= 0.3) & (aud_on_threshold_crossing_time_np >= 0.01))]
        aud_response_time = aud_threshold_crossing_time_np[((aud_threshold_crossing_time_np <= 0.3) & (aud_threshold_crossing_time_np >= 0.01))]
        discriminationTime = np.concatenate([aud_on_response_time, aud_response_time, vis_response_time])
        neuronType = np.concatenate([np.repeat('audOn', len(aud_on_response_time)),
         np.repeat('audLR', len(aud_response_time)),
         np.repeat('visLR', len(vis_response_time))])
        time_of_discrim_df = pd.DataFrame.from_dict({'discriminationTime':discriminationTime,  'neuronType':neuronType})
    elif use_same_vis_lr_and_vis_on_cells:
        vis_lr_cell_subset_index = np.where((vis_lr_threshold_crossing_time_subset > 0.05) & (vis_lr_threshold_crossing_time_subset >= min_time) & (vis_lr_threshold_crossing_time_subset <= max_time))[0]
        vis_lr_threshold_crossing_time_subset = vis_lr_threshold_crossing_time_subset[vis_lr_cell_subset_index]
        vis_lr_cells_vis_on_response_time = vis_lr_cells_vis_on_response_time[vis_lr_cell_subset_index]
    else:
        vis_lr_threshold_crossing_time_subset = vis_lr_threshold_crossing_time_subset[(vis_lr_threshold_crossing_time_subset > 0.05)]
        vis_lr_cells_vis_on_response_time = vis_lr_cells_vis_on_response_time[(vis_lr_cells_vis_on_response_time > 0.05)]
    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = caltos.plot_discrimination_time(vis_response_time=vis_lr_threshold_crossing_time_subset,
          aud_on_response_time=aud_on_threshold_crossing_time,
          aud_response_time=aud_lr_threshold_crossing_time,
          aud_lr_aud_on_response_time=aud_lr_cells_aud_on_response_time,
          vis_lr_vis_on_response_time=vis_lr_cells_vis_on_response_time,
          min_time=min_time,
          highlight_neuron_idx=0,
          max_time=max_time)

        fig.savefig(os.path.join(fig_folder, fig_name), bbox_inches='tight')


def main():


    plot_discrimination_time(aud_on_cells_aud_on_repsonse_time_fpath=aud_on_cells_aud_on_repsonse_time_fpath,
                             aud_lr_cells_aud_lr_response_time_fpath=aud_lr_cells_aud_lr_response_time_fpath,
                             vis_lr_cells_vis_lr_response_time_fpath=vis_lr_cells_vis_lr_response_time_fpath,
                             aud_lr_cells_aud_on_response_time_fpath=aud_lr_cells_aud_on_response_time_fpath,
                             vis_lr_cells_vis_on_response_time_fpath=vis_lr_cells_vis_on_response_time_fpath,
                             include_aud_lr_and_vis_lr_cells_onset=include_aud_lr_and_vis_lr_cells_onset,
                             fig_folder=fig_folder, fig_name=fig_name)

if __name__ == '__main__':
    main()