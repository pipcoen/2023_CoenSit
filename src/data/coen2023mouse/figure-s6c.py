"""
This script generates figure S6c from the paper :
This is the decoding accuracy before movement onset, removing unreliable trials (movement before main movement onset)

Internal notes:
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

fig_folder = ''
fig_name = ''

def main():
    files_folder_path = '/Volumes/Partition 1/data/interim/active-m2-choice-init-v2-decoding-reliable/decodeChoiceThreshDir/window20'

    all_exp_classification_results_df = pd.concat([
        pd.read_pickle(x) for x in glob.glob(os.path.join(files_folder_path, '*.pkl'))
    ])

    # Do some re-naming
    all_exp_classification_results_df['brainRegion'] = all_exp_classification_results_df['brain_region']
    all_exp_classification_results_df['Exp'] = all_exp_classification_results_df['exp_num']
    all_exp_classification_results_df['Subject'] = all_exp_classification_results_df['subject_num']
    all_exp_classification_results_df['rel_score'] = (all_exp_classification_results_df['classifier_score'] -
                                                      all_exp_classification_results_df['control_score']) / \
                                                     (1 - all_exp_classification_results_df['baseline_hit_prop'])

    all_exp_classification_results_df['brainRegion'] = all_exp_classification_results_df['brainRegion'].replace(
        {'FRPMOs': 'MOs'})

    all_exp_classification_results_df['rel_score'] = (all_exp_classification_results_df['classifier_score'] -
                                                      all_exp_classification_results_df['control_score']) / \
                                                     (1 - all_exp_classification_results_df['control_score'])

    # temp measure to deal with FRP
    all_exp_classification_results_df['brainRegion'] = all_exp_classification_results_df['brainRegion'].replace(
        {'FRP': 'MOs'})

    subset_brain_regions = ['MOs', 'ACA', 'PL', 'OLF', 'ORB', 'ILA']
    subset_all_exp_classification_results_df = all_exp_classification_results_df.loc[
        all_exp_classification_results_df['brainRegion'].isin(subset_brain_regions)
    ]

    plot_sig_star = True
    plot_p_val = True
    verbose = True
    custom_ylim = None
    with plt.style.context(splstyle.get_style('nature-reviews')):
        lower_limit = -0.3
        custom_ylim = [-0.3, 1]

        fig, ax = vizmodel.plot_brain_region_decoding_acc_comparison(subset_all_exp_classification_results_df,
                                                                     acc_metric='rel_score', jitter_level=0.05,
                                                                     dot_size=5, lower_limit=lower_limit,
                                                                     plot_sig_star=plot_sig_star, plot_p_val=plot_p_val,
                                                                     verbose=verbose,
                                                                     fig=None, ax=None)

        if custom_ylim is not None:
            ax.set_ylim(custom_ylim)