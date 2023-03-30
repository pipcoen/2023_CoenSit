import pandas as pd
import os
import glob


def load_choice_decoding_results(files_folder_path='/Users/timothysit/Dropbox/msi-key-data/active-m2-choice-init-decoding/decodeChoiceThreshDir/'):
    """
    Load choice decoding results used in the main figure for Coen 2022
    INPUT
    --------
    files_folder_path : (str)
        path to pickle file with choice decoding results
    """
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

    # Some experiments only have FRP, so those are renamed to be MOs (since here we group FRP and MOs neurons)
    all_exp_classification_results_df['brainRegion'] = all_exp_classification_results_df['brainRegion'].replace(
        {'FRP': 'MOs'})

    subset_brain_regions = ['MOs', 'ACA', 'PL', 'OLF', 'ORB', 'ILA']
    subset_all_exp_classification_results_df = all_exp_classification_results_df.loc[
        all_exp_classification_results_df['brainRegion'].isin(subset_brain_regions)
    ]

    return subset_all_exp_classification_results_df


def load_vis_decoding_results(decoding_result_path='/Volumes/Partition 1/data/interim/active-decode-stim/vis_lr_subset_30_neurons_no_balancing_0_to_300ms_5_subsamples_min_25_labels.pkl'):
    """
    Load visual decoding results used in the main figure for Coen 2022

    """
    all_classification_results_df = pd.read_pickle(decoding_result_path)
    brain_region_var_name = 'brainRegion'
    # acc_metric = 'accuracyRelBaseline'
    custom_subset_brain_regions = ['MOs', 'ACA', 'PL', 'OLF', 'ORB', 'ILA']
    all_classification_results_df = all_classification_results_df.loc[
        all_classification_results_df[brain_region_var_name].isin(custom_subset_brain_regions)
    ]

    # Calculate relative accuracy
    all_classification_results_df['accuracyRelBaseline'] = (all_classification_results_df['mean_classifier_score'] -
                                                            all_classification_results_df['mean_control_score']) / \
                                                           (1 - all_classification_results_df['mean_control_score'])
    all_classification_results_df = all_classification_results_df.reset_index()
    all_classification_results_df['exp'] = all_classification_results_df['exp'].astype(int)
    all_classification_results_df['Exp'] = all_classification_results_df['exp']

    all_classification_results_df['brainRegion'] = all_classification_results_df['brainRegion'].replace(
        {'FRP': 'MOs'})

    subset_brain_regions = ['MOs', 'ACA', 'PL', 'OLF', 'ORB', 'ILA']
    subset_all_exp_classification_results_df = all_classification_results_df.loc[
        all_classification_results_df['brainRegion'].isin(subset_brain_regions)]

    return subset_all_exp_classification_results_df


def load_aud_decoding_results(decoding_result_path = '/Volumes/Partition 1/data/interim/active-decode-stim/aud_lr_subset_30_neurons_no_balancing_0_to_300ms_5_subsamples_min_25_labels.pkl'):

    all_classification_results_df = pd.read_pickle(decoding_result_path)
    brain_region_var_name = 'brainRegion'
    acc_metric = 'accuracyRelBaseline'
    custom_subset_brain_regions = ['MOs', 'ACA', 'PL', 'OLF', 'ORB', 'ILA']
    all_classification_results_df = all_classification_results_df.loc[
        all_classification_results_df[brain_region_var_name].isin(custom_subset_brain_regions)
    ]

    # Calculate relative accuracy
    all_classification_results_df['accuracyRelBaseline'] = (all_classification_results_df['mean_classifier_score'] -
                                                            all_classification_results_df['mean_control_score']) / \
                                                           (1 - all_classification_results_df['mean_control_score'])
    all_classification_results_df = all_classification_results_df.reset_index()
    all_classification_results_df['exp'] = all_classification_results_df['exp'].astype(int)
    all_classification_results_df['Exp'] = all_classification_results_df['exp']

    subset_brain_regions = ['MOs', 'ACA', 'PL', 'OLF', 'ORB', 'ILA']
    subset_all_exp_classification_results_df = all_classification_results_df.loc[
        all_classification_results_df['brainRegion'].isin(subset_brain_regions)]

    exp_var_name = 'Exp'
    brain_region_var_name = 'brainRegion'
    cv_ave_method = 'mean'
    acc_metric = 'rel_score'

    exp_and_brain_region_grouped_df = subset_all_exp_classification_results_df.groupby(
        [exp_var_name, brain_region_var_name]).agg(
        cv_ave_method).reset_index()
    exp_and_brain_region_grouped_df['Exp'] = exp_and_brain_region_grouped_df['Exp'].astype(int)

    return exp_and_brain_region_grouped_df

def aggregate_decoding_cross_val_repeats(all_exp_classification_results_df,
                                         exp_var_name='Exp', acc_metric='classifier_score',
                                         brain_region_var_name='brainRegion',
                                         cv_ave_method='mean'):
    """
    Aggregates (takes the mean or median) of decoding accuracy across cross-validation sets

    """

    exp_and_brain_region_grouped_df = all_exp_classification_results_df.groupby([exp_var_name, brain_region_var_name]).agg(
        cv_ave_method).reset_index()

    all_exp_grouped_accuracy = exp_and_brain_region_grouped_df.groupby('brainRegion').agg(cv_ave_method)

    brain_region_order = all_exp_grouped_accuracy.sort_values(acc_metric, ascending=False).reset_index()['brainRegion'].values
    exp_and_brain_region_grouped_df['Exp'] = exp_and_brain_region_grouped_df['Exp'].astype(int)

    return exp_and_brain_region_grouped_df

def plot_decoding_vs_behaviour_performance():


    return fig, axs


def main():

    print('Running analyse decoding')

    processe_to_run = ['plot_decoding_vs_behaviour_performance']


    for process in processe_to_run:

        if process == 'plot_decoding_vs_behaviour_performance':

            print('TODO: move code from 3.20b')

            # Load beahviour data
            data_folder = '/Volumes/Partition 1/data/interim/active-m2-good-reliable-movement/subset'
            behave_df_file_name = 'ephys_behaviour_df.pkl'

            active_behave_df = pd.read_pickle(os.path.join(data_folder, behave_df_file_name))

            # This is for active-m2-good-w-two-movement
            """
            subset_active_behave_df = pephys.subset_behaviour_df(
            behaviour_df=active_behave_df, remove_invalid=True,
            remove_no_go=True, remove_no_stim=True, 
            min_reaction_time=0.1, reaction_time_variable_name='firstTimeToWheelMove',
            remove_rt_mismatch=True,
            remove_reverse_wheel_move_trials=True,
            time_range_to_cal_gradient=[-0.2, 0.2]
            )
            """

            # This is with newer folders
            subset_active_behave_df = pephys.subset_behaviour_df(
                behaviour_df=active_behave_df, remove_invalid=True,
                remove_no_go=True, remove_no_stim=True,
                min_reaction_time=0.1, reaction_time_variable_name='firstTimeToWheelMove',
                remove_rt_mismatch=False,
                remove_reverse_wheel_move_trials=False,
                time_range_to_cal_gradient=[-0.2, 0.2]
            )

            # Load decoding results
            # This is for reliable movement
            decoding_file_folder = '/Volumes/Partition 1/data/interim/active-m2-good-reliable-decode-left-right-shuffle-window-20'
            decoding_df_paths = glob.glob(os.path.join(decoding_file_folder, '*classification*.pkl'))
            decoding_df = pd.concat([pd.read_pickle(x) for x in decoding_df_paths])

            decoding_df = decoding_df.loc[decoding_df['brain_region'] == 'MOs']

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

            random_slope_and_intercept_md = Lmer(
                "relativeScore ~ correctResponse + (1 + correctResponse | subjectRef) ",
                data=decoding_and_performance_df)

            # random_slope_and_intercept_md = Lmer("relativeScore ~ correctResponse + (correctResponse | subjectRef) + (0 + correctResponse | subjectRef)",
            #            data=decoding_and_performance_df)
            print(random_slope_and_intercept_md.fit())

            random_slope_and_intercept_param_df = random_slope_and_intercept_md.fixef

            connect_dots = True
            plot_mean_lme_model_line = True
            interpolation_range = [0.6, 0.95]

            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, ax = vizmodel.plot_decoding_and_performance(decoding_and_performance_df,
                                                                 accuracy_metric='relativeScore',
                                                                 lme_model_fitted_param_df=random_slope_and_intercept_param_df,
                                                                 connect_dots=connect_dots,
                                                                 plot_mean_lme_model_line=plot_mean_lme_model_line,
                                                                 plot_lme_model_line_per_subject=True,
                                                                 subject_lme_line_alpha=0.5,
                                                                 interpolation_range=interpolation_range)
                ax.set_ylabel('Classifier score relative to baseline', size=10)
                ax.legend(title='Mouse', fontsize=10, bbox_to_anchor=(1.04, 1.04))
                ax.set_title('Random interval, random slope model', fontsize=10, weight='bold')

                # fig_folder = '/media/timsit/Partition 1/reports/figures/multispaceworld-decoding-and-behaviour-performance/'
                # fig_name = 'l2_svm_untuned_window_20_scatter-new-movement-times_random_interval_and_random_slope_model_fit'

                fig_folder = '/Volumes/Partition 1/reports/figures/multispaceworld-decoding-and-behaviour-performance/'
                fig_name = 'l2_svm_untuned_window_20_scatter-reliable-movement-times_random_interval_and_random_slope_model_fit'
                fig_ext = ['.pdf', '.png']

                for ext in fig_ext:
                    fig.savefig(os.path.join(fig_folder, fig_name + ext), bbox_inches='tight', dpi=300)


