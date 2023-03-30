"""
This script generates figure S6f from the paper :

They are
LEFT PANEL: AP-Position vs choice prediction accuracy
MIDDLE PANEL: ML Position vs choice prediction accuracy
RIGHT PANEL : Mouse performance vs choice prediction accuracy

Internal notes:
This is from 3.20d-decoding-performance-and-behaviour-and-brain-location
"""
import pandas as pd
import pickle as pkl
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import sciplotlib.style as splstyle
import sciplotlib.polish as splpolish
# plt.style.use(os.path.join(root, 'src/visualization/ts.mplstyle'))  # TODO: specify relative path...
# FIG_DPI = 300



import time

import src.data.analyse_spikes as anaspikes
import src.data.analyse_behaviour as anabehave
import src.data.process_ephys_data as pephys


import src.visualization.vizmodel as vizmodel
import src.visualization.vizpikes as vizpikes
import src.visualization.vizstat as vizstat
import src.visualization.vizbehaviour as vizbehave

from collections import defaultdict

import xarray as xr

import scipy.stats as sstats
import src.data.stat as stat

import src.models.predict_model as pmodel
import sklearn as skl
import sklearn.model_selection as skselect
import sklearn

from pymer4.models import Lmer


def main():

    # Load the data and do the stats

    # This is for reliable movement
    decoding_file_folder = '/Volumes/Partition 1/data/interim/active-m2-good-reliable-decode-left-right-shuffle-window-20'
    decoding_df_paths = glob.glob(os.path.join(decoding_file_folder, '*classification*.pkl'))
    decoding_df = pd.concat([pd.read_pickle(x) for x in decoding_df_paths])

    decoding_df = decoding_df.loc[decoding_df['brain_region'] == 'MOs']

    # data_folder = '/media/timsit/Partition 1/data/interim/active-m2-good-w-two-movement/subset/'
    data_folder = '/Volumes/Partition 1/data/interim/active-m2-good-reliable-movement/subset'
    behave_df_file_name = 'ephys_behaviour_df.pkl'

    active_behave_df = pd.read_pickle(os.path.join(data_folder, behave_df_file_name))

    # This is with newer folders
    subset_active_behave_df = pephys.subset_behaviour_df(
        behaviour_df=active_behave_df, remove_invalid=True,
        remove_no_go=True, remove_no_stim=True,
        min_reaction_time=0.1, reaction_time_variable_name='firstTimeToWheelMove',
        remove_rt_mismatch=False,
        remove_reverse_wheel_move_trials=False,
        time_range_to_cal_gradient=[-0.2, 0.2]
    )

    # window_start_loc = 25
    window_start_loc = 20

    performance_df = anabehave.compute_behaviour_performance(subset_active_behave_df,
                                                             performance_metric='correctResponse')

    decoding_df = pmodel.compute_relative_score(decoding_df)

    specific_window_decoding_df = pmodel.compute_window_mean_accuracy(decoding_df,
                                                                      window_units='bins',
                                                                      target_window_start_loc=window_start_loc)

    decoding_and_performance_df = specific_window_decoding_df.set_index(
        ['subjectRef', 'expRef']).join(
        performance_df.set_index(['subjectRef', 'expRef']))

    decoding_and_performance_df = decoding_and_performance_df.reset_index()


    # Do the stats for the choice vs. decoding performance
    # see this: https://stats.stackexchange.com/questions/13166/rs-lmer-cheat-sheet
    # correct_response_random_intercept_md = Lmer("relativeScore ~ correctResponse + (1 | subjectRef)",
    #           data=decoding_and_performance_df)
    correct_response_random_intercept_md = Lmer("relativeScore ~ correctResponse + (1 + correctResponse | subjectRef)",
                                                data=decoding_and_performance_df)
    print(correct_response_random_intercept_md.fit())

    correct_response_random_slope_and_intercept_param_df = correct_response_random_intercept_md.fixef


    # Load data and do stats for AP and ML vs decoding performance
    active_choice_decoding_df = pd.read_csv(
        '/Volumes/Partition 1/data/interim/decoding-per-probe/active_decoding_chooseLeftRight.csv')

    # Flip ML
    ml_values = active_choice_decoding_df['ML']
    neg_values_idx = np.where(ml_values < 0)[0]
    ml_values[neg_values_idx] = -ml_values[neg_values_idx]

    # Not necessary but just to make it clear
    active_choice_decoding_df['ML'] = ml_values

    random_slope_and_intercept_md = Lmer("relativeScore ~ AP + (1 + AP | subjectRef) ",
                                         data=active_choice_decoding_df)

    # random_slope_and_intercept_md = Lmer("relativeScore ~ correctResponse + (correctResponse | subjectRef) + (0 + correctResponse | subjectRef)",
    #            data=decoding_and_performance_df)
    print(random_slope_and_intercept_md.fit())
    AP_response_random_slope_and_intercept_param_df = random_slope_and_intercept_md.fixef

    random_slope_and_intercept_md = Lmer("relativeScore ~ ML + (1 + ML | subjectRef) ",
                                         data=active_choice_decoding_df)

    # random_slope_and_intercept_md = Lmer("relativeScore ~ correctResponse + (correctResponse | subjectRef) + (0 + correctResponse | subjectRef)",
    #            data=decoding_and_performance_df)
    print(random_slope_and_intercept_md.fit())
    ML_response_random_slope_and_intercept_param_df = random_slope_and_intercept_md.fixef

    # Plot all three panels together
    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, axs = plt.subplots(1, 3, sharey=True)
        fig.set_size_inches(8, 3)

        fig, axs[2] = vizmodel.plot_decoding_and_performance(decoding_and_performance_df,
                                                             accuracy_metric='relativeScore',
                                                             plot_mean_lme_model_line=True,
                                                             lme_model_fitted_param_df=correct_response_random_slope_and_intercept_param_df,
                                                             fig=fig, ax=axs[2])

        # AP
        allow_repeats = True
        connect_dots = False
        plot_mean_lme_model_line = True
        interpolation_range = [140, 300]

        fig, axs[0] = vizmodel.plot_decoding_and_performance(active_choice_decoding_df,
                                                             accuracy_metric='relativeScore',
                                                             behaviour_metric='AP',
                                                             lme_model_fitted_param_df=AP_response_random_slope_and_intercept_param_df,
                                                             connect_dots=connect_dots,
                                                             plot_mean_lme_model_line=plot_mean_lme_model_line,
                                                             plot_lme_model_line_per_subject=True,
                                                             subject_lme_line_alpha=0.5, allow_repeats=allow_repeats,
                                                             interpolation_range=interpolation_range, fig=fig,
                                                             ax=axs[0])

        # ML

        allow_repeats = True
        connect_dots = False
        plot_mean_lme_model_line = True
        interpolation_range = [50, 200]

        fig, axs[1] = vizmodel.plot_decoding_and_performance(active_choice_decoding_df,
                                                             accuracy_metric='relativeScore',
                                                             behaviour_metric='ML',
                                                             lme_model_fitted_param_df=ML_response_random_slope_and_intercept_param_df,
                                                             connect_dots=connect_dots,
                                                             plot_mean_lme_model_line=plot_mean_lme_model_line,
                                                             plot_lme_model_line_per_subject=True,
                                                             subject_lme_line_alpha=0.5, allow_repeats=allow_repeats,
                                                             interpolation_range=interpolation_range, fig=fig,
                                                             ax=axs[1])

        # ax.set_ylabel('Classifier score relative to baseline', size=10)
        # ax.legend(title='Mouse', fontsize=10, bbox_to_anchor=(1.04, 1.04))
        # ax.set_title('Random interval, fixed slope model', fontsize=10, weight='bold')
        # axs[0].set_xlim([0.5, 1])

        # axs[0].set_ylim([-0.1, 1])
        axs[0].set_ylabel('Choice prediction accuracy', size=10)
        axs[1].set_xlabel('Lateral distance from bregma', size=10)
        axs[0].set_xlabel('AP position relative to bregma', size=10)

        fig_folder = '/Volumes/Partition 1/reports/figures/multispaceworld-decoding-and-behaviour-performance/'
        fig_name = 'AP_ML_pCorrect_vs_decoding'
        fig_ext = ['.pdf', '.png']

        for ext in fig_ext:
            fig.savefig(os.path.join(fig_folder, fig_name + ext), bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()