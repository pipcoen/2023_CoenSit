"""
This scripts makes figure S7o in the paper.
"""


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



def plot_kernel_weight_and_depth_in_probe(weight_and_pos_df,
                                          metric_to_plot='audKernelMean', depth_metric='depth_in_probe',
                                          neuron_ave_metric='mean', probe_ave_metric='mean',
                                          probe_spread_metric='std',
                                          show_individual_probes=True,
                                          include_ave_line=False, num_depth_bins=20, custom_cap=None,
                                          include_stats=False, min_neuron=None,
                                          fig=None, ax=None):
    """
    Plot each probe as a line
    x axis : depth of neuron from surface (currently using dorsal-ventral axis value as a proxy)
    y axis : metric_to_plot, which is usually the absolute auditory kernesl weight over time,
             or the same for visual, movement, or movement direction
    Parameters
    -----------
    weight_and_pos_df : pandas dataframe
    metric_to_plot : str
    depth_metric : str
        metric used to define "depth"
        "depth_in_probe" is relative to the surface at bregma
        "depthFromSurface" is relative to the surface of the brain given the neuron's AP ML DV position
    min_neuron : int
        minimum number of neurons that the probe needs to have to be included in the plot and analysis
    fig : matplotlib figure object
    ax : matplotlib axes object
    """


    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)


    # Get only probes with a minimum of x neurons
    num_neuron_per_probe = weight_and_pos_df.groupby('penRef').agg('count')['cellPos']
    subset_penRef = num_neuron_per_probe[num_neuron_per_probe > min_neuron].index.values
    weight_and_pos_df = weight_and_pos_df.loc[
        weight_and_pos_df['penRef'].isin(subset_penRef)
    ]

    depth_in_probe_all = [row['cellPos'][1] for n_row, row in weight_and_pos_df.iterrows()]  # DV axis of cell
    weight_and_pos_df['depth_in_probe'] = depth_in_probe_all

    min_depth = np.min(weight_and_pos_df[depth_metric])
    max_depth = np.max(weight_and_pos_df[depth_metric])
    depth_in_probe_bins = np.linspace(min_depth, max_depth, num_depth_bins)
    # pdb.set_trace()
    bin_width = np.mean(np.diff(depth_in_probe_bins))

    if show_individual_probes:
        for probe_id in np.unique(weight_and_pos_df['penRef']):

            probe_df = weight_and_pos_df.loc[
                weight_and_pos_df['penRef'] == probe_id
            ]

            probe_df = probe_df.sort_values(by=depth_metric)

            y_vals = probe_df[metric_to_plot]

            if custom_cap is not None:
                y_vals[y_vals > custom_cap] = custom_cap

            if include_ave_line:
                ax.plot(probe_df[depth_metric], y_vals, lw=0.5, color='gray', alpha=0.2)
            else:
                ax.plot(probe_df[depth_metric], y_vals, lw=1)


    if include_ave_line:
        metric_per_depth_bin = weight_and_pos_df.groupby(
            ['penRef', pd.cut(weight_and_pos_df[depth_metric], depth_in_probe_bins)]).agg(neuron_ave_metric)[metric_to_plot].unstack()

        if probe_ave_metric == 'mean':
            mean_metric_per_depth_bin = np.nanmean(metric_per_depth_bin.values, axis=0)
        elif probe_ave_metric == 'median':
            mean_metric_per_depth_bin = np.nanmedian(metric_per_depth_bin.values, axis=0)

        if probe_spread_metric == 'std':
            std_metric_per_depth_bin = np.nanstd(metric_per_depth_bin.values, axis=0)
        elif probe_spread_metric == 'mad':
            std_metric_per_depth_bin = sstats.median_abs_deviation(metric_per_depth_bin.values, axis=0, nan_policy='omit')

        ax.plot(depth_in_probe_bins[0:-1] + bin_width/2, mean_metric_per_depth_bin, color='black', lw=1.5)
        ax.fill_between(depth_in_probe_bins[0:-1] + bin_width/2,
                        mean_metric_per_depth_bin - std_metric_per_depth_bin,
                        mean_metric_per_depth_bin + std_metric_per_depth_bin, color='gray', alpha=0.5, lw=0)

    if include_stats:
        probe_mean_depth_and_metric = weight_and_pos_df.groupby('penRef').agg(neuron_ave_metric)
        corr_rval, corr_pval = sstats.pearsonr(probe_mean_depth_and_metric[depth_metric],
                                                probe_mean_depth_and_metric[metric_to_plot])
        ax.set_title(r'$r = %.2f, p = %.2f$' % (corr_rval, corr_pval), size=11)

    ax.set_xlabel('Depth', size=11)
    ax.set_ylabel(metric_to_plot, size=11)

    return fig, ax



def plot_mean_metric_and_mean_position_per_probe(weight_and_pos_df, metric_to_plot='audKernelMean',
                                                 position_metric='AP', bregma_position=None, min_neuron=0,
                                                 include_fitted_line=False, put_stats_in_title=False,
                                                 fit_separate_lines=False, flip_ml=False,
                                                 fig=None, ax=None):
    """
    Parameters
    ----------
    weight_and_pos_df : pandas dataframe
    metric_to_plot : str
    position_metric : str
    include_fitted_line : bool
    fit_separate_lines : bool
    flip_ml : bool
        whether to flip on medial-lateral axis
    fig : matplotlib figure object
    ax : matplotlib axes object

    Returns
    -------
    fig : matplotlib figure object
    ax : matplotlib axes object
    """

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

    mean_position_per_probe = []
    mean_metric_per_probe = []

    for probe_id in np.unique(weight_and_pos_df['penRef']):

        probe_df = weight_and_pos_df.loc[
            weight_and_pos_df['penRef'] == probe_id
        ]

        if len(probe_df) < min_neuron:
            continue

        if position_metric == 'AP':
            probe_position = [row['cellPos'][2] for n_row, row in probe_df.iterrows()]
            xlabel = 'Anterior-Posterior'
        elif position_metric == 'ML':
            probe_position = [row['cellPos'][0] for n_row, row in probe_df.iterrows()]
            xlabel = 'Medial-Lateral'
        else:
            print('WARNING: no valid position_metric specified')
            return None, None

        probe_mean_position = np.mean(probe_position)
        probe_neuron_mean_metric = np.mean(probe_df[metric_to_plot])

        if bregma_position is not None:
            # see : https://github.com/pipcoen/expPipeline/search?q=bregma
            if position_metric == 'AP':
                probe_mean_position = bregma_position[0] - probe_mean_position
            elif position_metric == 'ML':
                probe_mean_position = bregma_position[2] - probe_mean_position

                if flip_ml:
                    if probe_mean_position < 0:
                        probe_mean_position = -probe_mean_position

        mean_position_per_probe.append(probe_mean_position)
        mean_metric_per_probe.append(probe_neuron_mean_metric)

        ax.scatter(probe_mean_position, probe_neuron_mean_metric, color='black')

    if flip_ml:
        fit_separate_lines = False

    if include_fitted_line:

        if fit_separate_lines:

            mean_position_per_probe = np.array(mean_position_per_probe)
            mean_metric_per_probe = np.array(mean_metric_per_probe)

            group_1_idx = np.where(mean_position_per_probe < 0)[0]  # left hemisphere
            group_2_idx = np.where(mean_position_per_probe > 0)[0]  # right hemisphere

            group_1_mean_position_per_probe = mean_position_per_probe[group_1_idx]
            group_2_mean_position_per_probe = mean_position_per_probe[group_2_idx]
            group_1_mean_metric_per_probe = mean_metric_per_probe[group_1_idx]
            group_2_mean_metric_per_probe = mean_metric_per_probe[group_2_idx]

            group_1_fitted_line_result = sstats.linregress(group_1_mean_position_per_probe,
                                                           group_1_mean_metric_per_probe)
            group_2_fitted_line_result = sstats.linregress(group_2_mean_position_per_probe,
                                                           group_2_mean_metric_per_probe)

            group_1_x_vals_to_interpolate = np.linspace(np.min(group_1_mean_position_per_probe),
                                                        np.max(group_1_mean_position_per_probe), 100)
            group_1_y_vals = group_1_fitted_line_result.intercept + group_1_x_vals_to_interpolate * group_1_fitted_line_result.slope


            group_2_x_vals_to_interpolate = np.linspace(np.min(group_2_mean_position_per_probe),
                                                        np.max(group_2_mean_position_per_probe), 100)
            group_2_y_vals = group_2_fitted_line_result.intercept + group_2_x_vals_to_interpolate * group_2_fitted_line_result.slope


            if put_stats_in_title:
                ax.set_title(r'L : r = %.2f, p = %.3f, R: r = %.2f, p = %.3f' %
                             (group_1_fitted_line_result.rvalue, group_1_fitted_line_result.pvalue,
                              group_2_fitted_line_result.rvalue, group_2_fitted_line_result.pvalue), size=6)
            else:
                ax.text(0.7, 0.85, r'L : r = %.2f, p = %.3f, R: r = %.2f, p = %.3f' %
                        (group_1_fitted_line_result.rvalue, group_1_fitted_line_result.pvalue,
                         group_2_fitted_line_result.rvalue, group_2_fitted_line_result.pvalue), size=6,
                        transform=ax.transAxes, color='gray')

            ax.plot(group_1_x_vals_to_interpolate, group_1_y_vals, color='gray', lw=1, alpha=0.8)
            ax.plot(group_2_x_vals_to_interpolate, group_2_y_vals, color='gray', lw=1, alpha=0.8)

        else:

            fitted_line_result = sstats.linregress(mean_position_per_probe, mean_metric_per_probe)

            x_vals_to_interpolate = np.linspace(np.min(mean_position_per_probe), np.max(mean_position_per_probe), 100)
            y_vals = fitted_line_result.intercept + x_vals_to_interpolate * fitted_line_result.slope

            if put_stats_in_title:
                ax.set_title(r'r = %.2f, p = %.3f' % (fitted_line_result.rvalue, fitted_line_result.pvalue), size=11)
            else:
                ax.text(0.7, 0.85, r'r = %.2f, p = %.3f' % (fitted_line_result.rvalue, fitted_line_result.pvalue), size=11,
                        transform=ax.transAxes, color='gray')
            ax.plot(x_vals_to_interpolate, y_vals, color='gray', lw=1, alpha=0.8)

    if bregma_position is not None:
        if flip_ml:
            if position_metric == 'ML':
                xlabel = 'lateral distance from bregma midline'
            else:
                xlabel = xlabel + ' (relative to bregma)'
        else:
            xlabel = xlabel + ' (relative to bregma)'

    ax.set_xlabel(xlabel, size=11)
    ax.set_ylabel(metric_to_plot, size=11)

    return fig, ax


def get_sig_kernel_neurons(all_models_df, passive_neuron_df_w_hemisphere,
                           min_var_explained=0.02, min_kernel_mean_amp=None, min_cpd=0, include_cell_pos=False,
                           return_aud_plus_vis_model=False):
    """
    Get kernels with significant variance explained compared to some baseline model
    cpd stands for : ???
    """
    aud_only_model = all_models_df.loc[
        all_models_df['model'] == 'audOnly'
        ].groupby(['Exp', 'Cell']).agg('mean')

    vis_only_model = all_models_df.loc[
        all_models_df['model'] == 'visOnly'
        ].groupby(['Exp', 'Cell']).agg('mean')

    aud_plus_vis_model = all_models_df.loc[
        all_models_df['model'] == 'addition'
        ].groupby(['Exp', 'Cell']).agg('mean')

    MOs_passive_neuron_df_w_hemisphere = passive_neuron_df_w_hemisphere.loc[
        (passive_neuron_df_w_hemisphere['cellLoc'] == 'MOs') &
        (passive_neuron_df_w_hemisphere['subjectRef'] != 1)
        ]

    aud_plus_vis_model['hemisphere'] = MOs_passive_neuron_df_w_hemisphere['hemisphere'].values
    aud_plus_vis_model['penRef'] = MOs_passive_neuron_df_w_hemisphere['penRef'].values

    if include_cell_pos:
        aud_plus_vis_model['cellPos'] = MOs_passive_neuron_df_w_hemisphere['cellPos'].values

    # To look at Vis CPD, we compare full model (aud + vis) with model with vis removed (aud only)
    vis_cpd = (aud_only_model['varExplained'] - aud_plus_vis_model['varExplained']) / (aud_only_model['varExplained'])
    aud_cpd = (vis_only_model['varExplained'] - aud_plus_vis_model['varExplained']) / (vis_only_model['varExplained'])

    aud_plus_vis_model['aud_cpd'] = aud_cpd
    aud_plus_vis_model['vis_cpd'] = vis_cpd

    cpd_sig_a_and_v_neurons_df = aud_plus_vis_model.loc[
        (aud_plus_vis_model['aud_cpd'] > min_cpd) &
        (aud_plus_vis_model['vis_cpd'] > min_cpd) &
        (aud_plus_vis_model['varExplained'] >= 0.02)
        ]

    cpd_sig_a_neurons_df = aud_plus_vis_model.loc[
        (aud_plus_vis_model['aud_cpd'] > min_cpd) &
        (aud_plus_vis_model['varExplained'] >= min_var_explained)
        ]

    cpd_sig_v_neurons_df = aud_plus_vis_model.loc[
        (aud_plus_vis_model['vis_cpd'] > min_cpd) &
        (aud_plus_vis_model['varExplained'] >= min_var_explained)
        ]

    if min_kernel_mean_amp is not None:
        cpd_sig_a_neurons_df = cpd_sig_a_neurons_df.loc[
            np.abs(cpd_sig_a_neurons_df['audKernelMean']) >= min_kernel_mean_amp
            ]

        cpd_sig_v_neurons_df = cpd_sig_v_neurons_df.loc[
            np.abs(cpd_sig_v_neurons_df['visKernelMean']) >= min_kernel_mean_amp
            ]

    if return_aud_plus_vis_model:
        return cpd_sig_v_neurons_df, cpd_sig_a_neurons_df, aud_plus_vis_model
    else:
        return cpd_sig_v_neurons_df, cpd_sig_a_neurons_df




def main():
    fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
    fig_name = 'fig-s7o.pdf'

    process_params = dict(
        plot_kernel_weights_and_depth_in_probe=dict(
            plot_active_dataset=True,
            neuron_types=['all', 'sig_kernel'],
            fit_separate_lines_for_ML=True,
            kernels_to_plot=['audSign'],
            cal_depth_rel_surface=True,  # calculate depth relative to surface of brain rather than bregma DV axis
            annotation_volume_path='/Users/timothysit/multisensory-integration/annotation_volume_10um_by_index.npy',
            # annotation_volume_path='/home/timothysit/Desktop/annotation_volume_10um_by_index.npy',
            # neuron_df_w_hem_path='/media/timothysit/Partition 1/data/interim/passive-m2-new-parent/subset/neuron_df_with_hem.pkl',
            # model_results_folder='/media/timothysit/Partition 1/reports/figures/passive-kernel-regression/regression-psth-cv-include-unimodal-models-2-fold-cv-include-mse-w-kernels-2021-03-16/',
            # active_model_results_folder='/media/timothysit/Partition 1/data/interim/14.2-psth-addition-to-get-active-psth/all_neurons_3_stim_param_plus_2_movement_param_movement_aligned_multiple_template_smooth_30_50_insert_movenent_0p2_all_vis_fit_active_only_v3b_stimOnRidge_wkernel_and_varExplained/model_data',
            neuron_df_w_hem_path='/Volumes/Partition 1/data/interim/passive-m2-new-parent/subset/neuron_df_with_hem.pkl',
            # model_results_folder='/Volumes/Partition 1/reports/figures/passive-kernel-regression/regression-psth-cv-include-unimodal-models-2-fold-cv-include-mse-w-kernels-2021-03-16/',
            model_results_folder='/Volumes/Partition 1/reports/figures/passive-kernel-regression/regression-psth-mean-divided-cv-include-unimodal-models-2-fold-cv-include-mse-w-kernels',
            # active_model_results_folder='/Volumes/Partition 1/data/interim/14.2-psth-addition-to-get-active-psth/all_neurons_3_stim_param_plus_2_movement_param_movement_aligned_multiple_template_smooth_30_50_insert_movenent_0p2_all_vis_fit_active_only_v3b_stimOnRidge_wkernel_and_varExplained/model_data',
            active_model_results_folder='/Volumes/Partition 1/data/interim/fit-active-only-model',
            filetype='not-pickle',
            behave_df_path='/Volumes/Partition 1/data/interim/passive-m2-new-parent/subset/ephys_behaviour_df.pkl',
            # fig_folder='/Volumes/Partition 1/reports/figures/supp-fig-for-pip/',
            # behave_df_path='/media/timothysit/Partition 1/data/interim/passive-m2-new-parent/subset/ephys_behaviour_df.pkl',
            # fig_folder='/media/timothysit/Partition 1/reports/figures/supp-fig-for-pip/',
            # fig_folder='/home/timothysit/Desktop/figs-for-pip/',
            # fig_folder='/Volumes/Partition 1/reports/figures/supp-fig-for-pip/normalized-activity-regression',
            fig_folder=fig_folder,
            min_var_explained=0.02, min_kernel_mean_amp=0.05,
            sig_neurons_to_plot=['aud'],  # ['aud'], ['vis'], or ['aud', 'vis']
            plot_metric='kernel_mean',  # 'sig_cpd', or 'kernel_mean'
            neuron_ave_metric='mean',  # 'mean' or 'median'
            probe_ave_metric='median',  # 'mean' or 'median'
            probe_spread_metric='mad',  # 'mad' or 'std'
            include_ave_line=True,
            fig_ext=['.png'],
            bregma_position=[540, 0, 570],  # ML / DV / AP
            min_neuron=30,
            flip_ml=True,  # whether to flip medial / lateral axis
        ),
    )

    process = 'plot_kernel_weights_and_depth_in_probe'
    param_dict = process_params[process]


    sig_neurons_to_plot = param_dict['sig_neurons_to_plot']
    fig_folder = param_dict['fig_folder']
    fig_ext = param_dict['fig_ext']
    plot_metric = param_dict['plot_metric']
    include_ave_line = process_params[process]['include_ave_line']
    bregma_position = process_params[process]['bregma_position']
    flip_ml = process_params[process]['flip_ml']
    annotation_volume_path = process_params[process]['annotation_volume_path']
    cal_depth_rel_surface = process_params[process]['cal_depth_rel_surface']
    show_individual_probes = False
    neuron_ave_metric = process_params[process]['neuron_ave_metric']
    probe_ave_metric = process_params[process]['probe_ave_metric']
    probe_spread_metric = process_params[process]['probe_spread_metric']

    print('Saving figures to %s' % fig_folder)

    # cellPos : (anterior-posterior, dorsal-ventral, medial-lateral)
    passive_neuron_df_w_hemisphere = pd.read_pickle(
        param_dict['neuron_df_w_hem_path']
    )

    passive_MOs_df = passive_neuron_df_w_hemisphere.loc[
        passive_neuron_df_w_hemisphere['cellLoc'] == 'MOs'
        ]

    cell_ml_dv_ap = np.stack(passive_MOs_df['cellPos']).astype(float)

    label_size = 11
    dot_size = 5

    # Load passive regression weights weights
    behave_df_path = param_dict['behave_df_path']

    if param_dict['plot_active_dataset']:
        all_models_df = psth_regression.load_active_combined_models_df(param_dict['active_model_results_folder'],
                                                                       behave_df_path, take_kernel_abs=True,
                                                                       kernel_mean_window=[0, 0.4],
                                                                       filetype=param_dict['filetype'])

        MOs_passive_df = passive_neuron_df_w_hemisphere.loc[
            (passive_neuron_df_w_hemisphere['cellLoc'] == 'MOs') &
            (passive_neuron_df_w_hemisphere['subjectRef'] != 1)
            ]

        MOs_passive_df = MOs_passive_df.sort_values(by='cellId')

        passive_exp_ref = np.unique(MOs_passive_df['expRef'])
        active_exp_ref = np.unique(all_models_df['Exp'])
        active_passive_shared_exp = np.intersect1d(passive_exp_ref, active_exp_ref)
        all_models_df = all_models_df.loc[
            all_models_df['Exp'].isin(active_passive_shared_exp)
        ]

        MOs_passive_df = MOs_passive_df.loc[
            MOs_passive_df['expRef'].isin(active_passive_shared_exp)
        ]

        all_models_df = all_models_df.sort_values(by='Exp')
        MOs_passive_df = MOs_passive_df.sort_values(by='expRef')

        # metrics_to_plot = ['audSignKernelMean', 'visSignKernelMean',
        #                    'movLeftKernelMean', 'movRightKernelMean']

        metrics_to_plot = ['movLeftKernelMean']

        all_models_df['cellPos'] = MOs_passive_df['cellPos'].values
        all_models_df['penRef'] = MOs_passive_df['penRef'].values

        weight_and_pos_df_list = [
            all_models_df
        ]

    else:
        all_models_df = psth_regression.load_combined_models_df(param_dict['model_results_folder'],
                                                                behave_df_path, filetype=param_dict['filetype'])
        # metrics_to_plot = ['absAudKernelMean', 'absVisKernelMean']
        metrics_to_plot = ['absAudKernelMean']

        all_models_df['absAudKernelMean'] = np.abs(all_models_df['audKernelMean'])
        all_models_df['absVisKernelMean'] = np.abs(all_models_df['visKernelMean'])

        cpd_sig_v_neurons_df, cpd_sig_a_neurons_df, aud_plus_vis_model_df = get_sig_kernel_neurons(all_models_df,
                                                                                                   passive_neuron_df_w_hemisphere,
                                                                                                   min_var_explained=
                                                                                                   param_dict[
                                                                                                       'min_var_explained'],
                                                                                                   min_kernel_mean_amp=
                                                                                                   param_dict[
                                                                                                       'min_kernel_mean_amp'],
                                                                                                   min_cpd=0,
                                                                                                   include_cell_pos=True,
                                                                                                   return_aud_plus_vis_model=True)

        # weight_and_pos_df_list = [aud_plus_vis_model_df, aud_plus_vis_model_df]
        weight_and_pos_df_list = [aud_plus_vis_model_df]

    positions_to_plot = ['AP', 'ML']

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

        for n_df, weight_and_pos_df in enumerate(weight_and_pos_df_list):

            if cal_depth_rel_surface:
                annotation_volume_data = np.load(annotation_volume_path)  # AP x DV x ML
                # TODO: convert DV position to depth from surface
                all_cell_ml = np.array([row['cellPos'][0] for n_row, row in weight_and_pos_df.iterrows()])
                all_cell_ap = np.array([row['cellPos'][2] for n_row, row in weight_and_pos_df.iterrows()])
                all_cell_dv = np.array([row['cellPos'][1] for n_row, row in weight_and_pos_df.iterrows()])

                all_cell_dv_rel_surface = np.zeros((len(all_cell_ml),)) + np.nan
                for cell_idx in np.arange(len(all_cell_ml)):
                    cell_ml = all_cell_ml[cell_idx]
                    cell_ap = all_cell_ap[cell_idx]
                    cell_dv = all_cell_dv[cell_idx]
                    annotation_cell_dv = annotation_volume_data[cell_ap, :, cell_ml]
                    brain_surface_idx = np.where(annotation_cell_dv != 1)[0][0]

                    all_cell_dv_rel_surface[cell_idx] = cell_dv - brain_surface_idx

                weight_and_pos_df['depthRelSurface'] = all_cell_dv_rel_surface

            metric_to_plot = metrics_to_plot[n_df]

            # dimensions (ML, DV, AP)
            if cal_depth_rel_surface:
                depth_metric = 'depthRelSurface'
            else:
                depth_metric = 'depth_in_probe'

            fig, ax = plot_kernel_weight_and_depth_in_probe(weight_and_pos_df,
                                                            metric_to_plot=metric_to_plot,
                                                            show_individual_probes=show_individual_probes,
                                                            include_ave_line=include_ave_line,
                                                            depth_metric=depth_metric,
                                                            custom_cap=None,
                                                            min_neuron=process_params[process]['min_neuron'],
                                                            neuron_ave_metric=neuron_ave_metric,
                                                            probe_ave_metric=probe_ave_metric,
                                                            probe_spread_metric=probe_spread_metric,
                                                            include_stats=True,
                                                            fig=None, ax=None)

            """
            # Also plot the mean AP position and mean ML position against the mean weights across neurons in each probe
            for n_position, position_metric in enumerate(positions_to_plot):

                if process_params[process]['fit_separate_lines_for_ML']:
                    if position_metric == 'ML':
                        fit_separate_lines = True
                    else:
                        fit_separate_lines = False
                else:
                    fit_separate_lines = False


                fig, axs[n_position] = plot_mean_metric_and_mean_position_per_probe(weight_and_pos_df,
                                                                       metric_to_plot=metric_to_plot,
                                                                       position_metric=position_metric,
                                                                       include_fitted_line=True,
                                                                       bregma_position=bregma_position,
                                                                       min_neuron=process_params[process]['min_neuron'],
                                                                       fit_separate_lines=fit_separate_lines,
                                                                       put_stats_in_title=False,
                                                                       flip_ml=flip_ml,
                                                                       fig=fig, ax=axs[n_position])
            """

        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    main()
