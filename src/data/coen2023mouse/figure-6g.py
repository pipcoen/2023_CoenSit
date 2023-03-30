"""
This plots figure 6g of the paper.
This is the psychometric fit of the behaviour output of the trained model after inactivating visual neurons


Internal notes:
This is from notebook 21.25-drift-model-change-decision-threshold
Note that it currently requires xarray == 0.16.2
TODO: there are some pickle files that should be converted to something more readable and independent of xarray version

"""

import numpy as np

import jax
# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')


import xarray as xr
import pandas as pd

import os
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import sciplotlib.style as splstyle

import src.data.analyse_spikes as anaspikes
import src.data.analyse_behaviour as anabehave
import src.data.process_ephys_data as pephys
import src.models.network_model as nmodel
import src.models.psychometric_model as psychmodel
import src.data.stat as stat
import scipy.stats as sstats

# Plotting
import src.visualization.vizmodel as vizmodel
import src.visualization.vizstat as vizstat
import src.visualization.vizbehaviour as vizbehave

import sklearn.model_selection as sklselection
import sklearn.linear_model as sklinear

from tqdm import tqdm


# packages for state space model analysis
import statsmodels.api as sm


# Jax
import jax.numpy as jnp
from jax import grad, jit, vmap

import src.models.jax_decision_model as jax_dmodel
import functools

from jax.experimental import optimizers

# saving
import pickle as pkl

# Distribution plot
from sklearn.neighbors import KernelDensity

import src.data.struct_to_dataframe as stdf


fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
fig_name = 'fig-6g.pdf'

def main():
    model_inactivation_output_path = '/Volumes/Ultra Touch/multispaceworld-rnn-models/drift-model-20/inactivation_model_behaviour/visLeft0p083_allTime_testSet_modelOutput.pkl'
    model_inactivation_output = pd.read_pickle(model_inactivation_output_path)

    # left_decision_threshold_val = -0.89
    # right_decision_threshold_val = 1.06

    # This is for 0.1 vis Left scaling I think...
    # Best inactivation fit to the parameters: np.array([0.5691,  -0.7523,  2.6661,  0.5879, -2.2095, 1.7806])
    left_decision_threshold_val = -1.34
    right_decision_threshold_val = 0.890

    # left_decision_threshold_val = -1.41
    # right_decision_threshold_val = 0.75

    # 0.1 * 5/6 = 0.083
    left_decision_threshold_val = -1.31
    right_decision_threshold_val = 0.92

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
            left_choice_val=0, right_choice_val=1, target_vis_cond_list=target_vis_cond_list,
            target_aud_cond_list=target_aud_cond_list,
            peri_event_time=model_output['peri_event_time']
        )

        all_cv_model_inactivation_behaviour_df.append(cv_model_inactivation_behaviour_df)

    all_cv_model_inactivation_behaviour_df = pd.concat(all_cv_model_inactivation_behaviour_df)

    # Extract only the go trials
    all_cv_model_inactivation_behaviour_df['chooseRight'] = all_cv_model_inactivation_behaviour_df['choice']
    go_all_cv_model_inactivation_behaviour_df = all_cv_model_inactivation_behaviour_df.loc[
        (all_cv_model_inactivation_behaviour_df['reactionTime'] >= 0)
    ]

    # Pre 2021-03-09
    small_norm_term = 0
    vis_exp_lower_bound = 1  # originally 1
    vis_exp_init_guess = 1.5
    vis_exp_upper_bound = 2

    # After 2021-03-09
    small_norm_term = 0
    vis_exp_lower_bound = 0.6  # originally 1
    vis_exp_init_guess = 0.8
    vis_exp_upper_bound = 1

    inact_model_p_right, inact_model_stim_cond_val, inact_model_model_prediction_val, inact_model_explained_var, inact_model_popt = \
        psychmodel.fit_and_predict_psych_model(
            subject_behave_df=go_all_cv_model_inactivation_behaviour_df, small_norm_term=small_norm_term,
            vis_exp_lower_bound=vis_exp_lower_bound,
            vis_exp_init_guess=vis_exp_init_guess,
            vis_exp_upper_bound=vis_exp_upper_bound)

    mouse_inactivation_fits = stdf.loadmat(
        '/Volumes/Partition 1/data/interim/multispaceworld-mice-fits/miceFitsInactivationNew_april2021.mat')
    mouse_inactivation_fits = mouse_inactivation_fits['miceFitsInactivationNew']

    mouse_inactivation_fits_vis_vals = np.array(mouse_inactivation_fits['visValues']).flatten()
    mouse_inactivation_fits_aud_vals = np.array(mouse_inactivation_fits['audValues']).flatten()
    mouse_inactivation_fits_pRight = np.array(mouse_inactivation_fits['fracRightTurnsLog']).flatten()

    # Mouse first
    # mouse_popt_inact = np.array([-0.76063857,  0.03611851,  3.35999308,  0.85019176, -1.2402565 ,
    #     1.86516393])

    # mouse_popt_inact = np.array([0.5691, 2.6661, 0.7523, 0.5879, 1.7806, 2.2095])

    # New values (bias, visScaleR, visScaleL, N, audScaleR, audScaleL)
    mouse_popt_inact = np.array([0.5691, 2.6661, 0.7523, 0.5879, 1.7806, 2.2095])

    # Changed to the ordering I use : beta, s_vl, s_vr, y, s_al, s_ar
    mouse_popt_inact = np.array([0.5691, -0.7523, 2.6661, 0.5879, -2.2095,
                                 1.7806])

    unscale_data_points = True

    if unscale_data_points:
        # Unscale the data points
        max_v_val = 0.8 ** mouse_popt_inact[3]
        min_v_val = -0.8 ** mouse_popt_inact[3]
        mouse_inactivation_fits_vis_vals = (mouse_inactivation_fits_vis_vals + 1) / 2 * (
                    max_v_val - min_v_val) + min_v_val

    # mouse_inactivation_aud_left_loc = np.where(mouse_inactivation_fits_aud_vals == -1)[0]
    # mouse_inactivation_aud_right_loc = np.where(mouse_inactivation_fits_aud_vals == 1)[0]

    # April 2021 new coding of aud
    mouse_inactivation_aud_left_loc = np.where(mouse_inactivation_fits_aud_vals == -60)[0]
    mouse_inactivation_aud_right_loc = np.where(mouse_inactivation_fits_aud_vals == 60)[0]

    mouse_inactivation_aud_left_vis_vals = mouse_inactivation_fits_vis_vals[mouse_inactivation_aud_left_loc]
    mouse_inactivation_aud_left_logPright = mouse_inactivation_fits_pRight[mouse_inactivation_aud_left_loc]

    mouse_inactivation_aud_right_vis_vals = mouse_inactivation_fits_vis_vals[mouse_inactivation_aud_right_loc]
    mouse_inactivation_aud_right_logPright = mouse_inactivation_fits_pRight[mouse_inactivation_aud_right_loc]

    mouse_inactivation_aud_center_loc = np.where(mouse_inactivation_fits_aud_vals == 0)[0]
    mouse_inactivation_aud_center_vis_vals = mouse_inactivation_fits_vis_vals[mouse_inactivation_aud_center_loc]
    mouse_inactivation_aud_center_logPright = mouse_inactivation_fits_pRight[mouse_inactivation_aud_center_loc]

    log_base = '10'
    add_grid_lines = True
    plot_mouse_fit_lines = False
    include_model_points = True
    model_marker = ['X', 'o', 's']
    plot_mouse_points_w_model_exponent = True

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

        # Plot mouse dots
        if log_base == '10':

            if plot_mouse_points_w_model_exponent:
                model_vis_exponent = inact_model_popt[3]
                model_aud_left_vis_vals = np.concatenate([
                    -np.array([0.8, 0.4, 0.2, 0.1, 0.05]) ** model_vis_exponent,
                    np.array([0, 0.05, 0.1, 0.2, 0.4, 0.8]) ** model_vis_exponent])
                model_aud_center_vis_vals = np.concatenate([
                    -np.array([0.8, 0.4, 0.2, 0.1, 0.05]) ** model_vis_exponent,
                    np.array([0, 0.05, 0.1, 0.2, 0.4, 0.8]) ** model_vis_exponent])
                model_aud_right_vis_vals = np.concatenate([
                    -np.array([0.8, 0.4, 0.2, 0.1, 0.05]) ** model_vis_exponent,
                    np.array([0, 0.05, 0.1, 0.2, 0.4, 0.8]) ** model_vis_exponent])

                ax.scatter(model_aud_left_vis_vals, mouse_inactivation_aud_left_logPright, color='blue')
                ax.scatter(model_aud_center_vis_vals, mouse_inactivation_aud_center_logPright, color='black')
                ax.scatter(model_aud_right_vis_vals, mouse_inactivation_aud_right_logPright, color='red')

            else:
                ax.scatter(mouse_inactivation_aud_left_vis_vals, mouse_inactivation_aud_left_logPright, color='blue')
                ax.scatter(mouse_inactivation_aud_center_vis_vals, mouse_inactivation_aud_center_logPright,
                           color='black')
                ax.scatter(mouse_inactivation_aud_right_vis_vals, mouse_inactivation_aud_right_logPright, color='red')

        else:

            ax.scatter(mouse_inactivation_aud_left_vis_vals,
                       np.log(np.power(10, mouse_inactivation_aud_left_logPright)), color='blue')
            ax.scatter(mouse_inactivation_aud_center_vis_vals,
                       np.log(np.power(10, mouse_inactivation_aud_center_logPright)), color='black')
            ax.scatter(mouse_inactivation_aud_right_vis_vals,
                       np.log(np.power(10, mouse_inactivation_aud_right_logPright)), color='red')

        all_17_mice_popt = [-0.1268, -2.5418, 2.7152, 0.6510, -1.4541, 1.7149]
        mouse_aud_r_vis_left_vals = -np.linspace(0.8, 0, 50) ** all_17_mice_popt[3]

        mouse_aud_c_inact_fitted_line, aud_c_xvals = psychmodel.get_fitted_line(popt=mouse_popt_inact,
                                                                                vis_cond_range=[-0.8, 0.8], aud_cond=0,
                                                                                num_space=100)

        mouse_aud_l_inact_fitted_line, aud_l_xvals = psychmodel.get_fitted_line(popt=mouse_popt_inact,
                                                                                vis_cond_range=[-0.4, 0.8], aud_cond=-1,
                                                                                num_space=100)
        mouse_aud_r_inact_fitted_line, aud_r_xvals = psychmodel.get_fitted_line(popt=mouse_popt_inact,
                                                                                vis_cond_range=[-0.8, 0.4], aud_cond=1,
                                                                                num_space=100)

        """
        fig, ax = vizbehave.plot_psychometric_model_fit(subject_p_right_inact, 
                                        stim_cond=mouse_stim_cond_val_inact, 
                                        model_prediction=mouse_model_prediction_val_inact,
                                        vis_exponent=mouse_popt_inact[3], include_scatter=False,
                                        linestyle='--', include_legend=False,
                                        aud_center_to_off=True, fig=fig, ax=ax)
        """
        if plot_mouse_fit_lines:

            if log_base == '10':
                mouse_aud_c_inact_fitted_line = np.log10(np.exp(mouse_aud_c_inact_fitted_line))
                mouse_aud_l_inact_fitted_line = np.log10(np.exp(mouse_aud_l_inact_fitted_line))
                mouse_aud_r_inact_fitted_line = np.log10(np.exp(mouse_aud_r_inact_fitted_line))

            ax.plot(aud_c_xvals, mouse_aud_c_inact_fitted_line, linestyle='--', color='black')
            ax.plot(aud_l_xvals, mouse_aud_l_inact_fitted_line, linestyle='--', color='blue')
            ax.plot(aud_r_xvals, mouse_aud_r_inact_fitted_line, linestyle='--', color='red')

        # TODO: figure out what is going on here...
        # model
        model_vis_exponent = inact_model_popt[3]
        # model_vis_exponent = mouse_popt_inact[3]

        # interpolate to greater contrast values (exponetiated)
        # inact_model_stim_cond_val[inact_model_stim_cond_val == 0.8] = 0.87705454
        # inact_model_stim_cond_val[inact_model_stim_cond_val == 0.87705454] = 0.8

        fig, ax = vizbehave.plot_psychometric_model_fit(inact_model_p_right,
                                                        stim_cond=inact_model_stim_cond_val,
                                                        model_prediction=inact_model_model_prediction_val,
                                                        vis_exponent=model_vis_exponent, scatter_marker=model_marker,
                                                        include_scatter=include_model_points,
                                                        linestyle='-', include_legend=False, log_base=log_base,
                                                        aud_center_to_off=True, fig=fig, ax=ax,
                                                        interpolate_center=False)

        custom_legend_objs = [mpl.lines.Line2D([0], [0], color='black', lw=2),
                              mpl.lines.Line2D([0], [0], color='black', linestyle='--', lw=2)
                              ]

        ax.legend(custom_legend_objs, ['Drift model', 'Mice'])

        if log_base == 'e':
            ax.text(0.6, 4, r'$A_R$', size=12, color='red')
            ax.text(0.6, -0.8, r'$A_L$', size=12, color='blue')
            ax.set_ylim([-4, 4])
        elif log_base == '10':
            ax.text(0.6, 2, r'$A_R$', size=12, color='red')
            ax.text(0.6, -0.8, r'$A_L$', size=12, color='blue')
            ax.set_ylim([-2.75, 2.75])

        if add_grid_lines:
            plt.grid(True)

        xticks = np.concatenate(
            [-np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.3, 0.1, 0]) ** model_vis_exponent,
             np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) ** model_vis_exponent])

        ax.set_xticks(xticks)
        ax.set_xticklabels([-0.8, None, None, None, None,
                            None, None, None, None, None, None, None, None, None, None, None, 0.8])

        fig.tight_layout()
        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()



