"""
This script generates figure S6g from the paper :
They are the cumulative histogram plots comparing selectivity index for visual (left panel), auditory (middle panel) and
choice decoding (right panel) across brain regions.
In these plots the brain regions are labelled and coloured differently.
In the publication version we have set the colour of MOs to red and other brain regions to black.

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

    # LEFT PANEL : VISUAL STIMULUS CUMULATIVE HISTOGRAM
    save_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-trained-neurons-selectivity/'
    save_name = 'all_exp_stim_dprime_0to300ms_df.csv'
    all_exp_dprime_df = pd.read_csv(os.path.join(save_folder, save_name))
    all_exp_dprime_df = all_exp_dprime_df.reset_index()
    all_exp_dprime_df['d_prime'] = all_exp_dprime_df['d_prime'].astype(float)
    subset_all_exp_dprime_df = all_exp_dprime_df.loc[
        all_exp_dprime_df['d_prime'].apply(lambda x: type(x) in [np.ndarray, int, np.int64, float, np.float64])
    ].dropna()
    subset_all_exp_dprime_df = subset_all_exp_dprime_df.replace('FRP', 'MOs')

    # target_comparison_type = 'audLeftRight'
    target_comparison_type = 'visLeftRight'
    # target_comparison_type = 'audOnOff'
    comparision_type_df = subset_all_exp_dprime_df.loc[
        subset_all_exp_dprime_df['comparison_type'] == target_comparison_type
        ]
    numbins = 1000
    # linewidth = 1.5

    # Example visual left/right: s3e31c6
    # Example auditory left/right: s3e21c18
    plot_example_neuron = True
    target_exp = 31
    target_brain_region = 'MOs'
    target_cell_idx = 6

    # target_exp = 21
    # target_brain_region = 'MOs'
    # target_cell_idx = 18

    fig_ext = '.pdf'

    cell_df = subset_all_exp_dprime_df.loc[
        (subset_all_exp_dprime_df['Exp'] == target_exp) &
        (subset_all_exp_dprime_df['brain_region'] == target_brain_region) &
        (subset_all_exp_dprime_df['Cell_idx'] == target_cell_idx) &
        (subset_all_exp_dprime_df['comparison_type'] == target_comparison_type)
        ]

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

        for brain_region in np.unique(comparision_type_df['brain_region']):
            brain_region_df = comparision_type_df.loc[
                comparision_type_df['brain_region'] == brain_region
                ]

            brain_region_dprime = brain_region_df['d_prime']

            res = sstats.cumfreq(np.abs(brain_region_dprime), numbins=numbins)
            x = res.lowerlimit + np.linspace(0, res.binsize * res.cumcount.size,
                                             res.cumcount.size)

            ax.plot(x, res.cumcount / np.max(res.cumcount), label=brain_region)

        ax.legend()

        if plot_example_neuron:
            ax.scatter(np.abs(cell_df['d_prime'].values), 1.04, marker='v', color='black')
            ax.spines['left'].set_bounds(0, 1)

        ax.set_xlabel(r'D prime: $\frac{\mu_1 - \mu_2}{(\sigma_1 + \sigma_2)/2}$', size=12)
        ax.set_ylabel('Proportion of neurons', size=12)

        if target_comparison_type == 'visLeftRight':
            title_txt = 'Visual left/right'
        elif target_comparison_type == 'audLeftRight':
            title_txt = 'Audio left/right'
        elif target_comparison_type == 'audOnOff':
            title_txt = 'Audio on/off'

        ax.set_title(title_txt, size=12)

        print('D prime: %.4f' % cell_df['d_prime'].values)

        fig_folder = '/media/timsit/Partition 1/reports/figures/revision-figs-for-pip/'
        fig_name = 'active_%s_selectivity_all_brain_regions_0to100ms' % target_comparison_type
        # fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')


    # MIDDLE PANEL : AUDITORY STIMULUS CUMULATIVE HISTOGRAM
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

    # target_comparison_type = 'audLeftRight'
    target_comparison_type = 'audLeftRight'
    # target_comparison_type = 'audOnOff'
    comparision_type_df = subset_all_exp_dprime_df.loc[
        subset_all_exp_dprime_df['comparison_type'] == target_comparison_type
        ]
    numbins = 1000
    # linewidth = 1.5

    # Example visual left/right: s3e31c6
    # Example auditory left/right: s3e21c18
    plot_example_neuron = True
    target_exp = 31
    target_brain_region = 'MOs'
    target_cell_idx = 6

    target_exp = 21
    target_brain_region = 'MOs'
    target_cell_idx = 18

    fig_ext = '.pdf'

    cell_df = subset_all_exp_dprime_df.loc[
        (subset_all_exp_dprime_df['Exp'] == target_exp) &
        (subset_all_exp_dprime_df['brain_region'] == target_brain_region) &
        (subset_all_exp_dprime_df['Cell_idx'] == target_cell_idx) &
        (subset_all_exp_dprime_df['comparison_type'] == target_comparison_type)
        ]

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

        for brain_region in np.unique(comparision_type_df['brain_region']):
            brain_region_df = comparision_type_df.loc[
                comparision_type_df['brain_region'] == brain_region
                ]

            brain_region_dprime = brain_region_df['d_prime']

            res = sstats.cumfreq(np.abs(brain_region_dprime), numbins=numbins)
            x = res.lowerlimit + np.linspace(0, res.binsize * res.cumcount.size,
                                             res.cumcount.size)

            ax.plot(x, res.cumcount / np.max(res.cumcount), label=brain_region)

        ax.legend()

        if plot_example_neuron:
            ax.scatter(np.abs(cell_df['d_prime'].values), 1.04, marker='v', color='black')
            ax.spines['left'].set_bounds(0, 1)

        ax.set_xlabel(r'D prime: $\frac{\mu_1 - \mu_2}{(\sigma_1 + \sigma_2)/2}$', size=12)
        ax.set_ylabel('Proportion of neurons', size=12)

        if target_comparison_type == 'visLeftRight':
            title_txt = 'Visual left/right'
        elif target_comparison_type == 'audLeftRight':
            title_txt = 'Audio left/right'
        elif target_comparison_type == 'audOnOff':
            title_txt = 'Audio on/off'

        ax.set_title(title_txt, size=12)

        print('D prime: %.4f' % cell_df['d_prime'].values)

        fig_folder = '/media/timsit/Partition 1/reports/figures/revision-figs-for-pip/'
        fig_name = 'active_%s_selectivity_all_brain_regions_0to100ms' % target_comparison_type
        # fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')

    # RIGHT PANEL : CHOICE CUMULATIVE HISTOGRAM
    # load results back
    active_moveLeftRight_result_save_path = '/media/timsit/Partition 1/data/interim/multispaceworld-trained-neurons-selectivity/all_exp_moveLeftRight_dprime_-130msto0ms.csv'
    all_exp_dprime_df = pd.read_csv(active_moveLeftRight_result_save_path)
    # all_exp_dprime_df = pd.concat(all_dprime_df)
    all_exp_dprime_df = all_exp_dprime_df.reset_index()
    all_exp_dprime_df['d_prime'] = all_exp_dprime_df['d_prime'].astype(float)
    subset_all_exp_dprime_df = all_exp_dprime_df.loc[
        all_exp_dprime_df['d_prime'].apply(lambda x: type(x) in [np.ndarray, int, np.int64, float, np.float64])
    ].dropna()
    subset_all_exp_dprime_df = subset_all_exp_dprime_df.replace('FRP', 'MOs')

    target_comparison_type = 'moveLeftRight'

    comparision_type_df = subset_all_exp_dprime_df.loc[
        subset_all_exp_dprime_df['comparison_type'] == target_comparison_type
        ]
    numbins = 1000
    # linewidth = 1.5

    # Example choose left/right: s3e21c95
    target_exp = 21
    target_brain_region = 'MOs'
    target_cell_idx = 95

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

        for brain_region in np.unique(comparision_type_df['brain_region']):
            brain_region_df = comparision_type_df.loc[
                comparision_type_df['brain_region'] == brain_region
                ]

            brain_region_dprime = brain_region_df['d_prime']

            res = sstats.cumfreq(np.abs(brain_region_dprime), numbins=numbins)
            x = res.lowerlimit + np.linspace(0, res.binsize * res.cumcount.size,
                                             res.cumcount.size)

            ax.plot(x, res.cumcount / np.max(res.cumcount), label=brain_region)

        ax.legend()

        ax.set_xlabel(r'D prime: $\frac{\mu_1 - \mu_2}{(\sigma_1 + \sigma_2)/2}$', size=12)
        ax.set_ylabel('Proportion of neurons', size=12)

        if plot_example_neuron:
            cell_df = subset_all_exp_dprime_df.loc[
                (subset_all_exp_dprime_df['Exp'] == target_exp) &
                (subset_all_exp_dprime_df['brain_region'] == target_brain_region) &
                (subset_all_exp_dprime_df['Cell_idx'] == target_cell_idx) &
                (subset_all_exp_dprime_df['comparison_type'] == target_comparison_type)
                ]

            ax.scatter(np.abs(cell_df['d_prime'].values), 1.04, marker='v', color='black')
            ax.spines['left'].set_bounds(0, 1)

        if target_comparison_type == 'visLeftRight':
            title_txt = 'Visual left/right'
        elif target_comparison_type == 'audLefRight':
            title_txt = 'Audio left/right'
        elif target_comparison_type == 'audOnOff':
            title_txt = 'Audio on/off'
        elif target_comparison_type == 'moveLeftRight':
            title_txt = 'Choose left/right'

        ax.set_title(title_txt, size=12)

        print('D prime: %.4f' % cell_df['d_prime'].values)

        fig_ext = '.pdf'
        fig_folder = '/media/timsit/Partition 1/reports/figures/revision-figs-for-pip/'
        fig_name = 'active_%s_selectivity_all_brain_regions_-130ms_to_0ms%s' % (target_comparison_type, fig_ext)
        # fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()