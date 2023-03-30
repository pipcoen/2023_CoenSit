"""
This plots figure 6e of the paper.
This is the psychometric fit of the behaviour output of the naive model.


Internal notes:
This is from notebook 21.25
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
fig_name = 'fig-6e.pdf'


def main():


    # NOTE: This requires xarray == 0.16.2
    alignment_ds_path = '/Volumes/Partition 1/data/interim/multispaceworld-rnn/new_window_permutation_test_passive_poisson_sampled_spike_count_to_rate_360_trials.nc'
    model_number = 20
    # model_result_folder = '/media/timsit/Partition 1/data/interim/multispaceworld-rnn/drift-model-%.f/'% model_number
    model_result_folder = '/Volumes/Ultra Touch/multispaceworld-rnn-models/drift-model-%.f' % model_number
    target_vis_cond_list = [-0.8, -0.4, -0.2, 0.2, 0.4, 0.8, 0.0, 0.0, 0.8, 0.8, -0.8, -0.8, -0.1, 0.1]
    target_aud_cond_list = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                            -60, 60, 60, -60, 60, -60, np.inf, np.inf]
    all_stim_cond_pred_matrix_dict, pre_preprocessed_alignment_ds_dev = jax_dmodel.load_model_outputs(
        model_result_folder, alignment_ds_path, target_random_seed=None,
        target_vis_cond_list=target_vis_cond_list,
        target_aud_cond_list=target_aud_cond_list, drift_param_N=1)

    no_aud_off_subset_active_behaviour_df = pd.read_pickle(
        '/Volumes/Partition 1/data/interim/multispaceworld-rnn/mouse_ephys_behaviour_compare_to_compare_w_drift_model_20.pkl')

    mouse_fit_behaviour_path = '/Volumes/Partition 1/data/interim/multispaceworld-mice-fits/miceFitsBehaviourNew_april2021.mat'
    # mouse_fit_behaviour = stdf.loadmat(mouse_fit_behaviour_path)['miceFitsBehaviour']
    mouse_fit_behaviour = stdf.loadmat(mouse_fit_behaviour_path)['miceFitsBehaviourNew']
    mouse_fit_vis_vals = np.array(mouse_fit_behaviour['visValues']).flatten()
    mouse_fit_aud_vals = np.array(mouse_fit_behaviour['audValues']).flatten()
    mouse_fit_logPright = np.array(mouse_fit_behaviour['fracRightTurnsLog']).flatten()

    all_17_mice_popt = [-0.1268, -2.5418, 2.7152, 0.6510, -1.4541, 1.7149]

    # mouse_fit_vis_vals = mouse_fit_vis_vals

    # Unscale the data points
    max_v_val = 0.8 ** all_17_mice_popt[3]
    min_v_val = -0.8 ** all_17_mice_popt[3]
    mouse_fit_vis_vals = (mouse_fit_vis_vals + 1) / 2 * (max_v_val - min_v_val) + min_v_val

    aud_left_loc = np.where(mouse_fit_aud_vals == -1)[0]
    aud_left_vis_vals = mouse_fit_vis_vals[aud_left_loc]
    aud_left_logPright = mouse_fit_logPright[aud_left_loc]

    aud_right_loc = np.where(mouse_fit_aud_vals == 1)[0]
    aud_right_vis_vals = mouse_fit_vis_vals[aud_right_loc]
    aud_right_logPright = mouse_fit_logPright[aud_right_loc]

    aud_center_loc = np.where(mouse_fit_aud_vals == 0)[0]
    aud_center_vis_vals = mouse_fit_vis_vals[aud_center_loc]
    aud_center_logPright = mouse_fit_logPright[aud_center_loc]

    log_base = '10'
    add_grid_lines = True
    plot_mouse_fit = False
    plot_model_points = True
    model_scatter_marker = ['X', 'o', 's']
    plot_mouse_points_w_model_exponent = True

    # Best fit parameters
    left_decision_threshold_val = -0.89
    right_decision_threshold_val = 1.06

    # Original parameters
    # left_decision_threshold_val = -1
    # right_decision_threshold_val = 1

    model_type = 'drift'
    model_behaviour_df = jax_dmodel.all_stim_cond_pred_matrix_dict_to_model_behaviour_df(
        all_stim_cond_pred_matrix_dict=all_stim_cond_pred_matrix_dict,
        alignment_ds=pre_preprocessed_alignment_ds_dev,
        left_decision_threshold_val=left_decision_threshold_val,
        right_decision_threshold_val=right_decision_threshold_val,
        model_type=model_type,
        left_choice_val=0, right_choice_val=1,
        target_vis_cond_list=target_vis_cond_list, target_aud_cond_list=target_aud_cond_list
    )

    # remove early and no response trials
    go_model_behaviour_df = model_behaviour_df.loc[
        model_behaviour_df['reactionTime'] >= 0
        ]

    vis_exp_lower_bound = 0.6
    vis_exp_init_guess = 0.6
    vis_exp_upper_bound = 3
    small_norm_term = 0

    subject_p_right, mouse_stim_cond_val, mouse_model_prediction_val, explained_var, mouse_popt = psychmodel.fit_and_predict_psych_model(
        subject_behave_df=no_aud_off_subset_active_behaviour_df, small_norm_term=0)
    go_model_behaviour_df['chooseRight'] = go_model_behaviour_df['choice']

    model_p_right, stim_cond_val, model_prediction_val, explained_var, model_popt = psychmodel.fit_and_predict_psych_model(
        subject_behave_df=go_model_behaviour_df, small_norm_term=small_norm_term,
        vis_exp_lower_bound=vis_exp_lower_bound, vis_exp_init_guess=vis_exp_init_guess,
        vis_exp_upper_bound=vis_exp_upper_bound)

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

        # Plot mouse dots
        if log_base == '10':

            if plot_mouse_points_w_model_exponent:
                model_vis_exponent = model_popt[3]
                model_aud_left_vis_vals = np.concatenate([
                    -np.array([0.8, 0.4, 0.2, 0.1]) ** model_vis_exponent,
                    np.array([0, 0.1, 0.2, 0.4, 0.8]) ** model_vis_exponent])
                model_aud_center_vis_vals = np.concatenate([
                    -np.array([0.8, 0.4, 0.2, 0.1]) ** model_vis_exponent,
                    np.array([0, 0.1, 0.2, 0.4, 0.8]) ** model_vis_exponent])
                model_aud_right_vis_vals = np.concatenate([
                    -np.array([0.8, 0.4, 0.2, 0.1]) ** model_vis_exponent,
                    np.array([0, 0.1, 0.2, 0.4, 0.8]) ** model_vis_exponent])

                ax.scatter(model_aud_left_vis_vals, aud_left_logPright, color='blue')
                ax.scatter(model_aud_center_vis_vals, aud_center_logPright, color='black')
                ax.scatter(model_aud_right_vis_vals, aud_right_logPright, color='red')
            else:
                ax.scatter(aud_left_vis_vals, aud_left_logPright, color='blue')
                ax.scatter(aud_center_vis_vals, aud_center_logPright, color='black')
                ax.scatter(aud_right_vis_vals, aud_right_logPright, color='red')

        else:

            ax.scatter(aud_left_vis_vals, np.log(np.power(10, aud_left_logPright)), color='blue')
            ax.scatter(aud_center_vis_vals, np.log(np.power(10, aud_center_logPright)), color='black')
            ax.scatter(aud_right_vis_vals, np.log(np.power(10, aud_right_logPright)), color='red')

        all_vis_cond_range = np.linspace(0, 0.8, 50) ** all_17_mice_popt[3]
        new_all_vis_cond_range = np.concatenate([-np.flip(all_vis_cond_range), all_vis_cond_range])

        model_vis_exponent = model_popt[3]
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
            include_legend=False, log_base=log_base)

        # Plot mouse fit
        if plot_mouse_fit:

            # parameters are: beta, s_vl, s_vr, y, s_al, s_ar
            all_17_mice_popt = [-0.1268, -2.5418, 2.7152, 0.6510, -1.4541, 1.7149]

            all_vis_cond_range = np.linspace(0, 0.8, 50) ** all_17_mice_popt[3]
            new_all_vis_cond_range = np.concatenate([-np.flip(all_vis_cond_range), all_vis_cond_range])

            mouse_aud_c_fitted_line, aud_c_xvals = psychmodel.get_fitted_line(popt=all_17_mice_popt,
                                                                              vis_cond_range=[-0.8, 0.8], aud_cond=0,
                                                                              num_space=100,
                                                                              raise_power=False)

            mouse_aud_l_fitted_line, aud_l_xvals = psychmodel.get_fitted_line(popt=all_17_mice_popt,
                                                                              vis_cond_range=[-0.4, 0.8], aud_cond=-1,
                                                                              num_space=100)
            mouse_aud_r_fitted_line, aud_r_xvals = psychmodel.get_fitted_line(popt=all_17_mice_popt,
                                                                              vis_cond_range=[-0.8, 0.4], aud_cond=1,
                                                                              num_space=100)

            if log_base == '10':
                mouse_aud_c_fitted_line = np.log10(np.exp(mouse_aud_c_fitted_line))
                mouse_aud_l_fitted_line = np.log10(np.exp(mouse_aud_l_fitted_line))
                mouse_aud_r_fitted_line = np.log10(np.exp(mouse_aud_r_fitted_line))

            if plot_mouse_points_w_model_exponent:
                # TODO: replace exponent with model exponent
                mouse_aud_r_vis_left_vals = -np.linspace(0.8, 0, 50) ** all_17_mice_popt[3]
                mouse_aud_r_vis_right_vals = np.linspace(0, 0.4, 50) ** all_17_mice_popt[3]
                mouse_aud_right_vis_vals = np.concatenate([mouse_aud_r_vis_left_vals, mouse_aud_r_vis_right_vals])

                mouse_aud_l_vis_left_vals = -np.linspace(0.4, 0, 50) ** all_17_mice_popt[3]
                mouse_aud_l_vis_right_vals = np.linspace(0.0, 0.8, 50) ** all_17_mice_popt[3]
                mouse_aud_left_vis_vals = np.concatenate([mouse_aud_l_vis_left_vals, mouse_aud_l_vis_right_vals])
            else:
                mouse_aud_r_vis_left_vals = -np.linspace(0.8, 0, 50) ** all_17_mice_popt[3]
                mouse_aud_r_vis_right_vals = np.linspace(0, 0.4, 50) ** all_17_mice_popt[3]
                mouse_aud_right_vis_vals = np.concatenate([mouse_aud_r_vis_left_vals, mouse_aud_r_vis_right_vals])

                mouse_aud_l_vis_left_vals = -np.linspace(0.4, 0, 50) ** all_17_mice_popt[3]
                mouse_aud_l_vis_right_vals = np.linspace(0.0, 0.8, 50) ** all_17_mice_popt[3]
                mouse_aud_left_vis_vals = np.concatenate([mouse_aud_l_vis_left_vals, mouse_aud_l_vis_right_vals])

            ax.plot(new_all_vis_cond_range, mouse_aud_c_fitted_line, color='black', linestyle='--')
            ax.plot(mouse_aud_right_vis_vals, mouse_aud_r_fitted_line, color='red', linestyle='--')
            ax.plot(mouse_aud_left_vis_vals, mouse_aud_l_fitted_line, color='blue', linestyle='--')

        custom_legend_objs = [mpl.lines.Line2D([0], [0], color='black', lw=2),
                              mpl.lines.Line2D([0], [0], color='black', linestyle='--', lw=2)
                              ]

        if log_base == '10':
            ax.text(0.6, 1.8, r'$A_R$', size=12, color='red')
            ax.text(0.6, -0.8, r'$A_L$', size=12, color='blue')
        else:
            ax.text(0.6, 4, r'$A_R$', size=12, color='red')
            ax.text(0.6, -0.8, r'$A_L$', size=12, color='blue')

        ax.legend(custom_legend_objs, ['Drift model', 'Mice'])

        if log_base == '10':
            ax.set_ylim([-2.5, 2.5])  # without include model marker
            # ax.set_ylim([-4, 4]) # including model maker
        else:
            ax.set_ylim([-4, 4])

        if add_grid_lines:
            plt.grid(True)

        fig.tight_layout()

        xticks = np.concatenate(
            [-np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.3, 0.1, 0]) ** model_vis_exponent,
             np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) ** model_vis_exponent])

        ax.set_xticks(xticks)
        ax.set_xticklabels([-0.8, None, None, None, None,
                            None, None, None, None, None, None, None, None, None, None, None, 0.8])

        fig.savefig(os.path.join(fig_folder, fig_name), bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()