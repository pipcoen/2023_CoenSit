import os
import numpy as np
import pickle as pkl
import pandas as pd
import src.models.kernel_regression as kernel_regression
import src.models.psth_regression as psth_regression
import src.visualization.vizregression as vizregression
import src.visualization.vizmodel as vizmodel
import src.data.process_ephys_data as pephys
import src.data.analyse_spikes as anaspikes

import sklearn.linear_model as sklinear
import src.data.alignment_mean_subtraction as ams

from tqdm import tqdm
import xarray as xr

import scipy.stats as sstats

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sciplotlib.polish as splpolish
import sciplotlib.style as splstyle
import pdb
import glob

from collections import defaultdict



def plot_active_kernel_lateralisation(df_dict, kernel_info_df, df_w_hemisphere_info,
                                      min_var_explained=0.02, min_kernel_abs_mean=None, fig_folder=None,
                                      fig_name=None):
    """

    Parameters
    ----------
    df_dict
    kernel_info_df
    min_var_explained
    fig_folder

    Returns
    -------

    """

    additive_model_df = df_dict['additiveModel']
    additive_model_remove_movLeft_df = df_dict['removeMovLeft']
    additive_model_remove_movRight_df = df_dict['removeMovRight']

    # Sort everything by exp and cell
    additive_model_df = additive_model_df.sort_values(by=['expRef', 'cellIdx'])
    additive_model_remove_movLeft_df = additive_model_remove_movLeft_df.sort_values(by=['expRef', 'cellIdx'])
    additive_model_remove_movRight_df = additive_model_remove_movRight_df.sort_values(by=['expRef', 'cellIdx'])

    movLeft_cpd = (additive_model_remove_movLeft_df['ss_residual'] - additive_model_df['ss_residual']) / (additive_model_df['ss_residual'])
    movRight_cpd = (additive_model_remove_movRight_df['ss_residual'] - additive_model_df['ss_residual']) / (additive_model_df['ss_residual'])

    additive_model_df['movLeft_cpd'] = movLeft_cpd.values
    additive_model_df['movRight_cpd'] = movRight_cpd.values


    additive_model_df['moveLeftKernelMean'] = kernel_info_df['movLeftKernelMean'].values
    additive_model_df['moveRightKernelMean'] = kernel_info_df['moveRightKernelMean'].values

    # Add hemisphere effect
    MOs_passive_neuron_df_w_hemisphere = df_w_hemisphere_info.loc[
        (df_w_hemisphere_info['cellLoc'] == 'MOs') &
        (df_w_hemisphere_info['subjectRef'] != 1)
        ]
    additive_model_df['hemisphere'] = MOs_passive_neuron_df_w_hemisphere['hemisphere'].values

    additive_model_df['leftVsRightKernelMean'] = (additive_model_df['moveRightKernelMean'] - additive_model_df['moveLeftKernelMean']) / \
                                                   (additive_model_df['moveRightKernelMean'] + additive_model_df['moveLeftKernelMean'])

    # additive_model_df['leftVsRightKernelMean'] = (additive_model_df['moveRightKernelMean'] - additive_model_df['moveLeftKernelMean'])
    movement_selective_neurons_df = additive_model_df.loc[
        ((additive_model_df['movLeft_cpd'] > 0) |
        (additive_model_df['movRight_cpd'] > 0)) &
        (additive_model_df['explained_variance'] >= min_var_explained)
    ]

    if min_kernel_abs_mean is not None:
        movement_selective_neurons_df = movement_selective_neurons_df.loc[
            (np.abs(movement_selective_neurons_df['moveRightKernelMean']) >= min_kernel_abs_mean) |
            (np.abs(movement_selective_neurons_df['moveLeftKernelMean']) >= min_kernel_abs_mean)
            ]


    print('Number of neurons %.f' % len(movement_selective_neurons_df))


    if fig_folder is not None:
        if fig_name is None:
            fig_name = 'SupX_choice_kernel_mean_left_right_hemisphere_sig_neurons.pdf'
        with plt.style.context(splstyle.get_style('nature-reviews')):
            """
            fig, ax = vizmodel.plot_hemisphere_and_kernel_weights(movement_selective_neurons_df.dropna(),
                                                                  metric_to_compare='leftVsRightKernelMean',
                                                                  jitter_level=0.03,
                                                                  min_var_explained=None,
                                                                  fig=None, ax=None)
            """




            fig, ax = vizmodel.plot_hemisphere_and_kernel_weights(movement_selective_neurons_df.dropna(),
                                                                  metric_to_compare='leftVsRightKernelMean',
                                                                  jitter_level=0.03,
                                                                  min_var_explained=None,
                                                                  fig=None, ax=None)

            ax.set_ylabel('Choice kernel right - left', size=12)
            # ax.set_ylabel(r'$\frac{\overline{k(R)} - \overline{k(L)}}{\overline{k(R)} + \overline{k(L)}}$')
            ax.set_ylim([-18, 18])
            fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


def main():

    fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
    fig_name = 'fig-s7e.pdf'

    process_params = dict(
        plot_active_kernel_lateralisation=dict(
            neuron_types=['all', 'sig_kernel'],
            model_paths={
                'additiveModel': '/Volumes/Partition 1/data/interim/14.2-psth-addition-to-get-active-psth/all_neurons_3_stim_param_plus_2_movement_param_movement_aligned_multiple_template_smooth_30_50_insert_movenent_0p2_all_vis_fit_active_only_v3b_stimOnRidge_wkernel_and_varExplained.pkl',
                'removeMovLeft': '/Volumes/Partition 1/data/interim/14.2-psth-addition-to-get-active-psth/all_neurons_3_stim_param_plus_2_movement_param_movement_aligned_multiple_template_smooth_30_50_insert_movenent_0p2_all_vis_fit_active_only_v3b_stimOnRidge_wkernel_and_varExplained_removeMoveLeft.pkl',
                'removeMovRight': '/Volumes/Partition 1/data/interim/14.2-psth-addition-to-get-active-psth/all_neurons_3_stim_param_plus_2_movement_param_movement_aligned_multiple_template_smooth_30_50_insert_movenent_0p2_all_vis_fit_active_only_v3b_stimOnRidge_wkernel_and_varExplained_removeMoveRight.pkl',
            },
            main_model_folder='/Volumes/Partition 1/data/interim/14.2-psth-addition-to-get-active-psth/all_neurons_3_stim_param_plus_2_movement_param_movement_aligned_multiple_template_smooth_30_50_insert_movenent_0p2_all_vis_fit_active_only_v3b_stimOnRidge_wkernel_and_varExplained/model_data',
            # main_model_folder='/Volumes/Partition 1/reports/figures/passive-kernel-regression/regression-psth-cv-include-unimodal-models-2-fold-cv-include-mse-w-kernels-2021-03-16/',
            kernel_mean_window=[-0.2, 0.4],
            min_var_explained=0.02,
            min_kernel_abs_mean=0.05,
            take_kernel_abs=False,
            # fig_folder='/media/timsit/Partition 1/reports/figures/supp-fig-for-pip/',
            fig_folder=fig_folder,
            fig_name=fig_name,
            neuron_df_w_hem_path='/Volumes/Partition 1/data/interim/passive-m2-new-parent/subset/neuron_df_with_hem.pkl',
            kernels_to_plot=['moveLeft', 'moveRight', 'audSign', 'visSign'],
        ),
    )

    param_dict = process_params['plot_active_kernel_lateralisation']

    model_paths = param_dict['model_paths']
    kernel_mean_window = param_dict['kernel_mean_window']
    min_var_explained = param_dict['min_var_explained']

    # First get model fit results
    model_df_dict = dict()
    for model_name, fpath in model_paths.items():
        model_df_dict[model_name] = pd.read_pickle(fpath)

    # Also get kernel results
    main_model_folder = param_dict['main_model_folder']
    exp_fpaths = glob.glob(os.path.join(main_model_folder, '*.pkl'))

    kernel_info = defaultdict(list)

    for exp_fpath in tqdm(exp_fpaths):
        active_regression_model_data = pd.read_pickle(exp_fpath)
        movement_kernel_ds = active_regression_model_data['mov_kernels']
        num_cell = len(movement_kernel_ds.Cell)

        if param_dict['take_kernel_abs']:
            move_left_kernel_mean_vals = np.abs(movement_kernel_ds.sel(Feature='moveLeft',
                                                                       Time=slice(kernel_mean_window[0],
                                                                                  kernel_mean_window[1]),
                                                                       )).mean('Time').values

            move_right_kernel_mean_vals = np.abs(movement_kernel_ds.sel(Feature='moveRight',
                                                                        Time=slice(kernel_mean_window[0],
                                                                                   kernel_mean_window[1]),
                                                                        )).mean('Time').values
        else:
            move_left_kernel_mean_vals = movement_kernel_ds.sel(Feature='moveLeft',
                                                                Time=slice(kernel_mean_window[0],
                                                                           kernel_mean_window[1]),
                                                                ).mean('Time').values
            move_right_kernel_mean_vals = movement_kernel_ds.sel(Feature='moveRight',
                                                                 Time=slice(kernel_mean_window[0],
                                                                            kernel_mean_window[1]),
                                                                 ).mean('Time').values
        exp = np.int(os.path.basename(exp_fpath)[-6:-4])
        kernel_info['Exp'].extend(np.repeat(exp, num_cell))
        kernel_info['Cell'].extend(movement_kernel_ds.Cell.values)
        kernel_info['movLeftKernelMean'].extend(move_left_kernel_mean_vals)
        kernel_info['moveRightKernelMean'].extend(move_right_kernel_mean_vals)

    kernel_info_df = pd.DataFrame.from_dict(kernel_info)
    kernel_info_df = kernel_info_df.sort_values(by=['Exp', 'Cell'])

    # Get hemisphere info
    passive_neuron_df_w_hemisphere = pd.read_pickle(
        param_dict['neuron_df_w_hem_path']
    )

    plot_active_kernel_lateralisation(df_dict=model_df_dict, kernel_info_df=kernel_info_df,
                                      df_w_hemisphere_info=passive_neuron_df_w_hemisphere,
                                      fig_folder=param_dict['fig_folder'], fig_name=param_dict['fig_name'],
                                      min_kernel_abs_mean=param_dict['min_kernel_abs_mean'],
                                      min_var_explained=min_var_explained)

if __name__ == '__main__':
    main()