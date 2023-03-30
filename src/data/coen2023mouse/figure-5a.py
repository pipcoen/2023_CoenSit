"""
This scripts generates figure 5a : Cumulative histogram of visual left/right selectivity
Internal notes:
from the notebook : naive-vs-proficient-mice-passive-prop-sig-neurons

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

data_main_folder = '/Volumes/Partition 1/data/interim'
fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
fig_name = 'fig5a'
fig_ext = '.pdf'
custom_xlim = [0, 0.5]  # this is to match the paper xlim

def main():
    # all_dprime_df_trained = pd.read_csv('/media/timsit/Partition 1/data/interim/trained-vs-naive-stim-response/trained_mice_aud_left_right_vis_left_right_dprime_at_max_window.csv')
    # all_dprime_df_naive = pd.read_csv('/media/timsit/Partition 1/data/interim/trained-vs-naive-stim-response/naive_mice_aud_left_right_vis_left_right_dprime_at_max_window.csv')
    # all_dprime_df_naive = pd.read_csv('/media/timsit/Partition 1/data/interim/trained-vs-naive-stim-response/naive_mice_aud_left_right_vis_left_right_dprime_at_max_window_subset_stim_cond.csv')


    all_dprime_df_trained = pd.read_csv(
        os.path.join(data_main_folder,
                     'trained-vs-naive-stim-response/trained_mice_aud_left_right_vis_left_right_dprime_at_max_window_subset_stim_cond_w_aud_on_off.csv'))
    # all_dprime_df_naive = pd.read_csv('/media/timsit/Partition 1/data/interim/trained-vs-naive-stim-response/naive_mice_aud_left_right_vis_left_right_dprime_at_max_window.csv')
    all_dprime_df_naive = pd.read_csv(
        os.path.join(data_main_folder, 'trained-vs-naive-stim-response/naive_mice_aud_left_right_vis_left_right_dprime_at_max_window_subset_stim_cond_w_aud_on_off.csv'))

    all_dprime_df_naive_shuffled = pd.read_csv(
        os.path.join(data_main_folder, 'trained-vs-naive-stim-response/naive_mice_aud_left_right_vis_left_right_dprime_at_max_window_subset_stim_cond_w_aud_on_off_w_shuffles.csv'))

    include_shuffle = True

    all_dprime_df_naive_shuffled_visLR = all_dprime_df_naive_shuffled.loc[
        all_dprime_df_naive_shuffled['comparison_type'] == 'visLeftRight'
        ]

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

        all_dprime_df_trained_aud_left = all_dprime_df_trained.loc[
            all_dprime_df_trained['comparison_type'] == 'visLeftRight'
            ]

        all_dprime_df_naive_aud_left = all_dprime_df_naive.loc[
            all_dprime_df_naive['comparison_type'] == 'visLeftRight'
            ]

        all_dprime_df_trained_aud_on = all_dprime_df_trained.loc[
            all_dprime_df_trained['comparison_type'] == 'audOnOff'
            ]

        # Trained
        res = sstats.cumfreq(np.abs(all_dprime_df_trained_aud_left['d_prime']).dropna(),
                             numbins=400)
        x = res.lowerlimit + np.linspace(0, res.binsize * res.cumcount.size,
                                         res.cumcount.size)

        ax.plot(x, res.cumcount / np.max(res.cumcount), color='red', label='Trained')

        # Naive
        res = sstats.cumfreq(np.abs(all_dprime_df_naive_aud_left['d_prime']).dropna(),
                             numbins=400)
        x = res.lowerlimit + np.linspace(0, res.binsize * res.cumcount.size,
                                         res.cumcount.size)

        ax.plot(x, res.cumcount / np.max(res.cumcount), color='black', label='Naive')

        if include_shuffle:
            # Naive shuffled
            shuffled_y = []
            shuffled_x = []

            # all_dprime_df_naive_shuffled
            # TODO: this should be specifically vis left/right right?

            for shuffle in tqdm(np.unique(all_dprime_df_naive_shuffled['shuffle'].dropna())):
                # shuffled_df = all_dprime_df_naive_shuffled.loc[
                #     all_dprime_df_naive_shuffled['shuffle'] == shuffle
                #  ]
                shuffled_df = all_dprime_df_naive_shuffled_visLR.loc[
                    all_dprime_df_naive_shuffled_visLR['shuffle'] == shuffle
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
            ax.plot(np.mean(shuffled_x, axis=0), shuffled_y_mean, color='gray', label='Shuffled', zorder=-1)
            ax.fill_between(np.mean(shuffled_x, axis=0), shuffled_y_lower, shuffled_y_upper, color='gray', alpha=0.5)

            ax.set_xlim([0, 1.5])

        ax.legend()

        ax.set_xlabel(r'D prime: $\frac{\mu_1 - \mu_2}{(\sigma_1 + \sigma_2)/2}$', size=12)
        ax.set_ylabel('Proportion of neurons', size=12)
        ax.set_title('Visual left/right', size=12)

        # Do statistical test
        test_stat, p_val = sstats.ttest_ind(np.abs(all_dprime_df_trained_aud_left['d_prime']).dropna(),
                                            np.abs(all_dprime_df_naive_aud_left['d_prime']).dropna())
        print('Test stat: %.4f' % test_stat)

        if p_val < 0.0001:
            print(r'$p < 10^{%.f}$' % np.floor(np.log10(p_val)))
        else:
            print('P val: %.4f' % p_val)

        ax.set_xlim(custom_xlim)

        fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()