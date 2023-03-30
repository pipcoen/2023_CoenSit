"""
This plots figure 6h of the paper.
This is the psychometric fit of the behaviour output of the trained model after inactivating MOs neurons


Internal notes:
This is from notebook 21.21c-simulate-inactivation-MOs-hemisphere-unilateral-model
Note that it currently requires xarray == 0.16.2
TODO: there are some pickle files that should be converted to something more readable and independent of xarray version

"""
import src.models.decision_model_simulate_inactivation as dmodel_inactivate


import numpy as np

import xarray as xr
import pandas as pd

import os
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import sciplotlib.style as splstyle

import src.data.analyse_spikes as anaspikes
import src.data.process_ephys_data as pephys
import src.models.network_model as nmodel
import src.data.stat as stat
import scipy.stats as sstats

import src.visualization.vizmodel as vizmodel
import src.models.psychometric_model as psychmodel
import src.visualization.vizbehaviour as vizbehave

import sklearn.model_selection as sklselection
import sklearn.linear_model as sklinear

from tqdm import tqdm

import itertools
import pickle as pkl
import pdb


# Model fitting dependencies
import src.models.jax_decision_model as jax_dmodel
import src.models.decision_model_sample_and_fit as dmodel_sample_and_fit
import src.models.decision_model_simulate_inactivation as sim_inact

import functools
from jax.experimental import optimizers

import src.visualization.plot_mouse_vs_model_behaviour as plot_mVm
import src.visualization.report_plot_model_vs_mouse_behaviour as report_plot_mVm



fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
fig_name = 'fig-6h.pdf'

def main():

    left_scaling = 1
    right_scaling = 0.6
    right_scaling_str = str(right_scaling).replace('.', 'p')
    left_scaling_str = str(left_scaling).replace('.', 'p')


    # model_output_save_name = 'leftHem%s_rightHem%s_allTime_testSet_modelOutput.pkl' % (left_scaling_str, right_scaling_str)
    model_output_save_name = 'leftHem%s_rightHem%s_allTime_testSet_modelOutput.pkl' % (
    left_scaling_str, right_scaling_str)

    # model_inactivation_output_path = os.path.join('/media/timsit/T7/drift-model-24/inactivation_model_behaviour',
    #                                               model_output_save_name)
    # model_inactivation_output_path = os.path.join('/Volumes/Ultra Touch/multispaceworld-rnn-models/drift-model-24/inactivation_model_behaviour',
    #                                               model_output_save_name)

    model_inactivation_output_path = '/Volumes/Ultra Touch/multispaceworld-rnn-models/drift-model-24/inactivation_model_behaviour/random-seed-1/leftHem1_rightHem0p6_allTime_testSet_modelOutput.pkl'

    model_inactivation_output = pd.read_pickle(model_inactivation_output_path)

    # left_decision_threshold_val = -1.34
    # right_decision_threshold_val = 0.890

    # No specified random seed
    # left_decision_threshold_val = -1.687755
    # right_decision_threshold_val = 1.618367

    # Random seed 1 (obtained via fit_decision_threshold2)
    left_decision_threshold_val = -1.734694
    right_decision_threshold_val = 1.244898

    # Original
    # left_decision_threshold_val = -1
    # right_decision_threshold_val = 1

    model_type = 'drift'
    target_vis_cond_list = [-0.8, -0.4, -0.2, 0.2, 0.4, 0.8,
                            0.0, 0.0, 0.8, 0.8, -0.8, -0.8,
                            -0.1, 0.1]
    target_aud_cond_list = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                            -60, 60, 60, -60, 60, -60, np.inf, np.inf]

    all_cv_model_inactivation_behaviour_df = list()

    for n_cv, model_output in model_inactivation_output.items():
        cv_model_inactivation_behaviour_df = jax_dmodel.all_stim_cond_pred_matrix_dict_to_model_behaviour_df(
            all_stim_cond_pred_matrix_dict=model_output['model_output'],
            alignment_ds=None,
            left_decision_threshold_val=left_decision_threshold_val,
            right_decision_threshold_val=right_decision_threshold_val,
            model_type=model_type,
            left_choice_val=0, right_choice_val=1,
            target_vis_cond_list=target_vis_cond_list,
            target_aud_cond_list=target_aud_cond_list,
            peri_event_time=model_output['peri_event_time']
        )

        all_cv_model_inactivation_behaviour_df.append(cv_model_inactivation_behaviour_df)

    all_cv_model_inactivation_behaviour_df = pd.concat(all_cv_model_inactivation_behaviour_df)
    # pdb.set_trace()

    # Extract only the go trials
    all_cv_model_inactivation_behaviour_df['chooseRight'] = all_cv_model_inactivation_behaviour_df['choice']
    go_all_cv_model_inactivation_behaviour_df = all_cv_model_inactivation_behaviour_df.loc[
        (all_cv_model_inactivation_behaviour_df['reactionTime'] >= 0)
    ]

    # subject_behave_df = go_all_cv_model_inactivation_behaviour_df

    vis_exp = 0.5879  # same as visual inactivation

    vis_exp_lower_bound = 0.55  # originally 0.6
    vis_exp_init_guess = 0.57
    vis_exp_upper_bound = 0.59  # orignally 3

    add_one_trial_to_stim_cond_with_one_choice = False  # orignally True
    if add_one_trial_to_stim_cond_with_one_choice:
        pdb.set_trace()
        stim_cond_w_one_choice = subject_behave_df.groupby(['visCond', 'audCond']).agg('mean')['goRight'] == 1
        stim_cond_w_one_choice = stim_cond_w_one_choice.loc[
            stim_cond_w_one_choice == True
            ].reset_index()

        for df_idx, sub_df in stim_cond_w_one_choice.iterrows():
            trial_df = pd.DataFrame.from_dict({
                'visCond': [sub_df['visCond']],
                'audCond': [sub_df['audCond']],
                'reactionTime': [0.2],
                'choice': [0],
                'chooseRight': [0],
                'goRight': [0],
                'visDiff': [sub_df['visCond']],
                'audDiff': [sub_df['audCond']]
            })
            go_all_cv_model_inactivation_behaviour_df = pd.concat(
                [go_all_cv_model_inactivation_behaviour_df, trial_df])

    """
    trial_df = pd.DataFrame.from_dict({
        'visCond': [-0.4, -0.1, 0.8],
        'audCond': [np.inf, np.inf, np.inf],
        'reactionTime': [0.2, 0.2, 0.2],
        'choice': [2, 2, 2],
        'chooseRight': [1, 1, 1],
        # 'goRight': [0, 0],
        # 'visDiff': [-0.4, -0.1],
        # 'audDiff': [np.inf, np.inf]
    })
    # pdb.set_trace()
    go_all_cv_model_inactivation_behaviour_df = pd.concat(
        [go_all_cv_model_inactivation_behaviour_df, trial_df])
    """
    small_norm_term = 0.01
    # small_norm_term = 0

    subject_behave_df = go_all_cv_model_inactivation_behaviour_df

    # pdb.set_trace()
    # sigma = None
    # subject_behave_df = control_go_model_behaviour_df

    # put less weight on the low visual contrast trials
    sigma = np.ones(14) * 0.1
    sigma[[8, 9, 10, 11]] = 0.25  # 0.25
    # sigma = 'auto'

    model_p_right, stim_cond_val, model_prediction_val, explained_var, model_popt = psychmodel.fit_and_predict_psych_model(
        subject_behave_df=subject_behave_df, small_norm_term=small_norm_term,
        vis_exp_lower_bound=vis_exp_lower_bound, vis_exp_init_guess=vis_exp_init_guess,
        vis_exp_upper_bound=vis_exp_upper_bound,
        sigma=sigma)

    model_vis_exponent = model_popt[3]

    # custom mouse dots
    mouse_aud_right_points = np.log10([2.05, 4.79, 8.05, 7.92, 17.7, 19.2, 10.6, 106, 46.5, 117, np.nan])
    mouse_aud_center_points = np.log10([0.558, 1.01, 1.26, 3.82, 4.38, 3.56, 5.98, 6.98, 13.4, 23.9, 20.6])
    mouse_aud_left_points = np.log10([np.nan, 0.2, 0.62, 0.64, 0.62, 0.86, 1.24, 1.42, 3.31, 4.35, 8.70])
    vis_cond_levels = np.array([0.04, 0.1, 0.2, 0.4, 0.8])
    vis_cond_vals = np.concatenate([-(np.flip(vis_cond_levels) ** model_vis_exponent), [0],
                                    vis_cond_levels ** model_vis_exponent])

    plot_model_points = True
    model_scatter_marker = ['o', 'o', 'o']
    log_base = '10'
    # small_norm_term = 0.01

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)
        fig, ax = vizbehave.plot_psychometric_model_fit(
            model_p_right, stim_cond=stim_cond_val,
            model_prediction=model_prediction_val,
            vis_exponent=model_vis_exponent,
            aud_center_to_off=True, fig=fig, ax=ax,
            open_circ_for_sim=False,
            open_circ_for_multimodal=False,
            removed_highest_coherent=False,
            include_scatter=plot_model_points,
            scatter_marker=model_scatter_marker,
            include_legend=False, log_base=log_base,
            small_norm_term=small_norm_term)

        ax.scatter(vis_cond_vals, mouse_aud_right_points, color='none', edgecolor='red')
        ax.scatter(vis_cond_vals, mouse_aud_left_points, color='none',
                   edgecolor='blue')
        ax.scatter(vis_cond_vals, mouse_aud_center_points, color='none', edgecolor='gray')

        xticks = np.concatenate(
            [-np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.3, 0.1, 0]) ** model_vis_exponent,
             np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) ** model_vis_exponent])

        ax.set_ylim([-2.5, 2.5])

        # ax.set_ylim([-4, 4])

        ax.grid()

        ax.set_xticks(xticks)
        ax.set_xticklabels([-0.8, None, None, None, None,
                            None, None, None, None, None, None, None, None, None, None, None, 0.8])

        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    main()