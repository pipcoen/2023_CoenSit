import pdb

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sciplotlib.style as splstyle
import xarray as xr
import pandas as pd
import scipy.signal as ssignal
import scipy.stats as sstats
import sklearn as skl
import src.data.process_facecam as pface
import src.data.analyse_2p as ana2p
import src.data.analyse_videos as anavid
import src.data.analyse_filmworld as anafilmworld
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSCanonical
import sklearn.decomposition as skldecomposition

import glob
import os
import scipy.io as sio
from tqdm import tqdm
import json
import pickle as pkl


def neural_face_cca(neural_X, face_X, n_components=10, pre_processing_steps=[], cross_val_method=None):
    """
    Performs canonical correlation analysis on neural data and video data

    Arguments
    ---------------
    neural_X (numpy ndarray)
        array of shape (num_time_points, num_features)
        features are either neurons or neural principal components (PCs)
    face_X (numpy ndarray)
        array of shape (num_time_points, num_features)
        features are either raw motion energy per pixel or facemap SVDs
    cross_val_method (str or None)
        how to do cross validation
        None : run the entire set without train/test split
        'split-half' : splits the data into the first half and the second half
    pre_processing_steps (list)
        list of preprocessing steps
    """

    neural_face_cca_results = {}

    neural_face_cca = CCA(n_components=n_components)

    if cross_val_method is None:
        neural_cc, face_cc = neural_face_cca.transform(neural_X, face_X)
        neural_face_cca_results['nerual_cc'] = neural_cc
        neural_face_cca_results['face_cc'] = face_cc
        neural_face_cca_results['cca_model'] = neural_face_cca

    elif cross_val_method == 'split-half':
        n_time_points = np.shape(neural_X)[0]
        train_idx = np.arange(0, int(n_time_points/2))
        test_idx = np.arange(int(n_time_points/2), n_time_points)
        neural_X_train = neural_X[train_idx, :]
        neural_X_test = neural_X[test_idx, :]
        face_X_train = face_X[train_idx, :]
        face_X_test = face_X[test_idx, :]

        if 'neuralScale' in pre_processing_steps:
            scale_model = StandardScaler()
            scale_model.fit(neural_X_train)
            neural_X_train = scale_model.transform(neural_X_train)
            neural_X_test = scale_model.transform(neural_X_test)

        if 'neuralPCA' in pre_processing_steps:
            pca_model = PCA(n_components=100)
            pca_model.fit(neural_X_train)
            neural_X_train = pca_model.transform(neural_X_train)
            neural_X_test = pca_model.transform(neural_X_test)


        neural_face_cca.fit(neural_X_train, face_X_train)

        neural_cc_train, face_cc_train = neural_face_cca.transform(neural_X_train, face_X_train)
        neural_cc_test, face_cc_test = neural_face_cca.transform(neural_X_test, face_X_test)

        train_scores = neural_face_cca.score(neural_X_train, face_X_train)
        test_scores = neural_face_cca.score(neural_X_test, face_X_test)

        neural_face_cca_results['neural_cc'] = neural_cc_train
        neural_face_cca_results['face_cc'] = face_cc_train
        neural_face_cca_results['neural_cc_test'] = neural_cc_test
        neural_face_cca_results['face_cc_test'] = face_cc_test
        neural_face_cca_results['train_scores'] = train_scores
        neural_face_cca_results['test_scores'] = test_scores
        neural_face_cca_results['cca_model'] = neural_face_cca

    return neural_face_cca_results



def plot_neural_face_cc(neural_cc, face_cc, num_cc_to_plot=9, plot_corr=True, fig=None, axs=None):

    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
        fig.set_size_inches(7, 7)

    for n_cc in np.arange(num_cc_to_plot):

        axs.flatten()[n_cc].scatter(neural_cc[:, n_cc], face_cc[:, n_cc], s=10, color='black')
        if plot_corr:
            cc_corr = np.corrcoef(neural_cc[:, n_cc], face_cc[:, n_cc])
            axs.flatten()[n_cc].text(0.5, 0.8, '$r^2  = %.2f$' % (cc_corr[1, 0]), size=12,
                                 transform=axs.flatten()[n_cc].transAxes)
            axs.flatten()[n_cc].set_title('CC %.f' % (n_cc+1), size=10)


    return fig, axs


def plot_each_mouse_face(fig=None, axs=None):



    return fig, axs


def smooth_signal(signal, signal_time, smooth_window, smooth_method='mean'):
    """
    INPUTS
    ----------
    signal : (numpy ndarray)
    signal_time : (numpy ndarray)
    smooth_window : (float)
    smooth_method : (str)
        what method to do smoothing
    """

    if smooth_method == 'mean':
        smoothed_time_sec = np.arange(signal_time[0], signal_time[-1], smooth_window)
        smoothed_signal, _, _ = sstats.binned_statistic(
            x=signal_time, values=signal, bins=smoothed_time_sec,
            statistic='mean')

    return smoothed_signal


def extract_2p_activity_matrix(exp_folder, special_subfolder, target_planes,
                                custom_time_bins=None, start_time=None, end_time=None,
                                activity_name='Spikes', fps_per_plane=2.5,
                                resample_time_bin=1.2):

    """
    Extract spontaneous activity in an experiment where stimulus periods are
    interleaved by (>30 seconds) gray screen periods (spontaneous period)
    start_time : (float or None)
    end_time : (float or None)
    """

    all_planes_neural_spontaneous_activity = []
    cell_planes = []
    # cell_x_locs = []
    # cell_y_locs = []

    for plane in target_planes:
        plane_file = glob.glob(os.path.join(exp_folder, special_subfolder, '*plane%.f.nc' % plane))[0]
        plane_ds = xr.open_dataset(plane_file)

        plane_acquisition_times = plane_ds.Time.values

        if start_time is None:
            start_time = plane_acquisition_times[0]
        if end_time is None:
            end_time = plane_acquisition_times[-1]

        plane_neural_activity = plane_ds.isel(Plane=0)['Spikes'].transpose('Time', 'Cell').values
        # plane_resampled_activity = []

        if resample_time_bin is not None:
            neural_spontaneous_activity_resampled = []
            new_resampled_time = np.arange(start_time, end_time, resample_time_bin)

            if custom_time_bins is not None:
                new_resampled_time = custom_time_bins[0]

            for cell in np.arange(np.shape(plane_neural_activity)[1]):
                cell_resampled = np.interp(
                    x=new_resampled_time, xp=plane_acquisition_times, fp=plane_neural_activity[:, cell]
                )
                neural_spontaneous_activity_resampled.append(cell_resampled)

            plane_resampled_activity = np.stack(neural_spontaneous_activity_resampled, axis=1)
            # plane_resampled_activity.append(neural_spontaneous_activity_resampled)
        else:
            neural_spontaneous_activity.append(
                neural_spontaneous_activity_subset
            )

        all_planes_neural_spontaneous_activity.append(plane_resampled_activity)
        # cell_x_locs.extend(plane_ds['X_loc'].values)
        # cell_y_locs.extend(plane_ds['Y_loc'].values)
        cell_planes.extend(np.repeat(plane, len(plane_ds.Cell)))

    # Combine spontaneous activity across all planes
    time_bins_per_plane = [np.shape(x)[0] for x in all_planes_neural_spontaneous_activity]
    num_cell_per_plane = [np.shape(x)[1] for x in all_planes_neural_spontaneous_activity]
    all_plane_neural_spontaneous_activity_combined = np.zeros((np.min(time_bins_per_plane),
                                                               np.sum(num_cell_per_plane)))
    each_plane_spont_activity = [x[0:np.min(time_bins_per_plane), :] for x in all_planes_neural_spontaneous_activity]

    cell_counter = 0
    for plane_spontaneous_activity in all_planes_neural_spontaneous_activity:

        num_cell = np.shape(plane_spontaneous_activity)[1]
        all_plane_neural_spontaneous_activity_combined[:, cell_counter:cell_counter+num_cell] = \
        plane_spontaneous_activity[0:np.min(time_bins_per_plane), :]
        cell_counter += num_cell

    return all_plane_neural_spontaneous_activity_combined, cell_planes, each_plane_spont_activity




def extract_spont_activity_matrix(exp_folder, special_subfolder, target_planes,
                                block_df, custom_time_bins=None, custom_spontaneous_starts=None,
                                custom_spontaneous_ends=None,
                                activity_name='Spikes', fps_per_plane=2.5,
                                resample_time_bin=1.2):
    """
    Extract spontaneous activity in an experiment where stimulus periods are
    interleaved by (>30 seconds) gray screen periods (spontaneous period)
    """

    all_planes_neural_spontaneous_activity = []
    cell_planes = []
    # cell_x_locs = []
    # cell_y_locs = []

    for plane in target_planes:
        plane_file = glob.glob(os.path.join(exp_folder, special_subfolder, '*plane%.f.nc' % plane))[0]
        plane_ds = xr.open_dataset(plane_file)

        neural_spontaneous_activity, two_p_spontaneous_time = ana2p.extract_spontaneous_activity(
            plane_ds, block_df, custom_spontaneous_starts=custom_spontaneous_starts,
            custom_spontaneous_ends=custom_spontaneous_ends,
            fps_per_plane=fps_per_plane, resample_time_bin=resample_time_bin,
            activity_name=activity_name
        )

        all_planes_neural_spontaneous_activity.append(neural_spontaneous_activity)
        # cell_x_locs.extend(plane_ds['X_loc'].values)
        # cell_y_locs.extend(plane_ds['Y_loc'].values)
        cell_planes.extend(np.repeat(plane, len(plane_ds.Cell)))

    # Combine spontaneous activity across all planes
    time_bins_per_plane = [np.shape(x)[0] for x in all_planes_neural_spontaneous_activity]
    num_cell_per_plane = [np.shape(x)[1] for x in all_planes_neural_spontaneous_activity]
    all_plane_neural_spontaneous_activity_combined = np.zeros((np.min(time_bins_per_plane),
                                                               np.sum(num_cell_per_plane)))

    each_plane_spont_activity = [x[0:np.min(time_bins_per_plane), :] for x in all_planes_neural_spontaneous_activity]

    cell_counter = 0
    for plane_spontaneous_activity in all_planes_neural_spontaneous_activity:

        num_cell = np.shape(plane_spontaneous_activity)[1]
        all_plane_neural_spontaneous_activity_combined[:, cell_counter:cell_counter+num_cell] = \
        plane_spontaneous_activity[0:np.min(time_bins_per_plane), :]
        cell_counter += num_cell

    return all_plane_neural_spontaneous_activity_combined, cell_planes, each_plane_spont_activity



def load_stringer_matlab_data(spont_data_fpaths, subsampling_methods=['interp', 'mean-bin'], resample_time_bin=1.2,
                              sampling_rate=2.5):
    # load everything into memory (can save time later if memory can take it)
    all_mat_data = []
    all_exp_data = []

    for fpath in tqdm(spont_data_fpaths):
        mat_data = sio.loadmat(fpath)
        all_mat_data.append(mat_data)

        # TODO: deal with 10 plane acquisition and red cells

        filename = os.path.basename(fpath)
        subject = filename.split('_')[2]
        exp_date = filename.split('_')[-1].split('.')[0]
        mean_image = mat_data['beh'][0][0][0][0][0][2]
        num_neuron, num_time_points = np.shape(mat_data['Fsp'])
        recording_duration = num_time_points / sampling_rate
        time = np.linspace(0, recording_duration, num_time_points)
        motion_svd = mat_data['beh'][0][0][0][0][0][0]

        exp_info = dict()
        exp_info['subject'] = subject
        exp_info['exp_date'] = exp_date
        exp_info['mean_image'] = mean_image
        exp_info['svd_mask'] = mat_data['beh'][0][0][0][0][0][1]
        exp_info['spikes'] = mat_data['Fsp']
        exp_info['num_neuron'] = num_neuron
        exp_info['num_time_points'] = num_time_points
        exp_info['time'] = time
        exp_info['motion_svd'] = motion_svd
        exp_info['plane'] = np.array([x[0][-3][0][0] for x in mat_data['stat']])

        # TODO: what is the difference between med and stat X Y?
        exp_info['X_loc'] = np.array([x[0][6][0][0] for x in mat_data['stat']])
        exp_info['Y_loc'] = np.array([x[0][6][0][1] for x in mat_data['stat']])
        # all_y_max = [np.max(x['Y_loc']) for x in stringer_all_exp_data]


        # exp_info['Y_loc'] = mat_data['med'][:, 0]
        # exp_info['X_loc'] = mat_data['med'][:, 1]


        # Compute resampled motion_svd and spikes
        if 'interp' in subsampling_methods:
            new_resampled_time = np.arange(0, recording_duration, resample_time_bin)

            spikes_resampled = np.zeros((num_neuron, len(new_resampled_time)))
            for cell in np.arange(num_neuron):
                spikes_resampled[cell, :] = np.interp(
                    x=new_resampled_time, xp=time, fp=mat_data['Fsp'][cell, :]
                )

            exp_info['spikes_resampled'] = spikes_resampled

            num_mot_svd = np.shape(motion_svd)[1]
            motion_svd_resampled = np.zeros((len(new_resampled_time), num_mot_svd))
            for svd_idx in np.arange(num_mot_svd):
                # TODO: modify the resample method here as well
                motion_svd_resampled[:, svd_idx] = np.interp(
                    x=new_resampled_time, xp=time, fp=motion_svd[:, svd_idx]
                )

            exp_info['motion_svd_resampled'] = motion_svd_resampled

        if 'mean-bin' in subsampling_methods:
            Ff = mat_data['Fsp'].T
            num_bins = np.shape(Ff)[0]
            activity_time_sec = np.linspace(0, num_bins / sampling_rate, num_bins)
            Ff_resampled = ana2p.resample_activity(activity=Ff,
                                                   activity_time_sec=activity_time_sec,
                                                   resample_time_bin=resample_time_bin,
                                                   sp_start=None, sp_end=None,
                                                   method='binned-mean')
            exp_info['spikes_resampled_mean'] = Ff_resampled

        if 'stringer-bin-method' in subsampling_methods:
            # TODO: not quite ready yet
            NN, NT = np.shape(Ff_subset)  # number of neurons and number of time points

            num_planes_imaged = example_data['db']['nplanes']

            if num_planes_imaged == 10:
                tbin = 4  # imaging at 30 Hz in total, so 10 planes means 3 Hz per plane
                # so you are resampling to 1/3 * 4 = 1.3 seconds
            else:
                tbin = 3  # 12 plane imaging : 2.5 Hz per plane
                # so this is resampling to 1/2.5 * 3 = 1.2 seconds

            # Resampling by taking the mean across bins
            Ff_resampled = np.squeeze(
                np.mean(
                    np.reshape(Ff_subset[:, 0:int(np.floor(NT / tbin) * tbin)],
                               (NN, tbin, -1), order='F'),
                    axis=1)
            )  # cell x resmapled_time_bins

            exp_info['spikes_resampled_stringer'] = Ff_resampled

        # Do PCA on resampled data
        exp_info['spikes_resampled_pc'] = PCA(n_components=100).fit_transform(spikes_resampled.T)

        all_exp_data.append(exp_info)

    return all_exp_data


def load_sit_data(exps_to_analyse_df, activity_name='Spikes', special_subfolder='',
                  resample_time_bin=1.2,
                  use_alternative_roi=False, include_full_data=False, verbose=True):
    """
    Arguments
    ------------
    exps_to_analyse_df : (pandas dataframe)
        dataframe with information on the set of recordings to analyse
    activity_name : (str)
        what type of activity to use
            'Spikes' : deconvolved traces
            'Fluorescence' : raw fluorescence data
    special_subfolder : (str)
        whether there is a subfolder within the usual main_folder/subject/exp_date/exp_num structure to access
        leave empty by default
        'matlab-suite2p' : load suite2p data processed using matlab-suite2p (rather than python-suite2p)
    resample_time_bin : (float)
        time bin (in seconds) to resample data
    use_alternative_roi : (bool or str)
        if set to True: then use the second available ROI from facemap (usually a cropped FOV focusing on the face,
        and excludes areas with the floating ball / steering wheel)
    include_full_data : (bool)
        whether to include the full motion SVD and neural data (without subsetting the spntaneous period part out)
        will also create a vector to indicate which part is the spontaneous period and which part is not
    verbose : (bool)
        whether to print out some diagnostic properties (for sanity checks / debugging)
    Output
    -------------
    all_exp_data : (list of dict)
        list where each entry is a dictionary with data from a recording session

    """

    mean_images = {}
    motion_svd_masks = {}
    all_exp_data = []

    if special_subfolder == 'matlab-suite2p':
        target_planes = np.arange(2, 13)
    else:
        target_planes = np.arange(1, 12)

    # extract spontaneous period traces

    for num_exp, exp_info in tqdm(exps_to_analyse_df.iterrows()):

        exp_info = exp_info.to_dict()

        exp_folder = os.path.join(exp_info['main_folder'],
                                  exp_info['subject'],
                                  exp_info['exp_date'], str(exp_info['exp_num']))

        # Get face dataset
        face_ds_path = glob.glob(os.path.join(exp_folder, '*face*.nc'))

        if len(face_ds_path) == 0:
            print('Cannot find face_ds file for %s' % exp_folder)
            face_ds = None
            continue
        else:
            face_ds = xr.open_dataset(face_ds_path[0])

        # Get face_proc file
        facemap_output_path = glob.glob(os.path.join(exp_folder, '*proc*.npy'))

        if len(facemap_output_path) == 0:
            print('Cannot find face proc file for %s' % exp_folder)
            face_proc = None
        else:
            face_proc = np.load(facemap_output_path[0], allow_pickle=True).item()

            # TODO: can I get the mean image of a different ROI???
            mean_images[exp_folder] = face_proc['avgframe_reshape']
            motion_svd_masks[exp_folder] = face_proc['motMask_reshape']

        if exp_info['exp_type'] == 'nat_videos':

            # get block data
            block_csv_search_result = glob.glob(os.path.join(exp_folder, '*trial_info_processed.csv'))
            if len(block_csv_search_result) == 0:
                block_nc_path = glob.glob(os.path.join(exp_folder, '*blockTimeline.nc'))[0]
                block_ds = xr.open_dataset(block_nc_path)
                block_df = pd.DataFrame.from_dict({
                    'Trial': block_ds['Trial'].values,
                    'stimOnTime': block_ds['movieStartTimes'].values,
                })

                # Add video and audio name (this only applies to newer set of experiments, exclude TS010, TS008
                if exp_info['subject'] not in ['TS008', 'TS010']:
                    json_fpath = glob.glob(os.path.join(exp_folder, '*movie_params.json'))[0]

                    with open(json_fpath) as json_file:
                        param_info_dict = json.load(json_file)

                    block_df['video'] = [
                        param_info_dict[
                            trial_info['movie_fname'].split('_')[1].split('.')[0]
                        ]['video_id'] for _, trial_info in block_df.iterrows()
                    ]

                    block_df['audio'] = [
                        param_info_dict[
                            trial_info['movie_fname'].split('_')[1].split('.')[0]
                        ]['audio_id'] for _, trial_info in block_df.iterrows()
                    ]

                # TODO: need to add code here to get TS008 and TS010 audio and video from the block mat file

            else:
                block_df = pd.read_csv(block_csv_search_result[0])

            # Extract spontaneous period traces
            exp_info['block_df'] = block_df
            exp_info['stimOnTime'] = block_df['stimOnTime']
            large_diff_trials = np.where(np.diff(block_df['stimOnTime']) > 100)[0]
            spontaneous_starts = [block_df.loc[x]['stimOnTime'] + 10 for x in large_diff_trials]
            spontaneous_ends = [block_df.loc[x + 1]['stimOnTime'] - 1 for x in large_diff_trials]
        elif exp_info['exp_type'] == 'spontaneous':
            block_df = None
            spontaneous_starts = [0]
            spontaneous_ends = [face_ds.Time.values[-1]]

        face_time = face_ds.Time.values

        if use_alternative_roi == 'see-df':
            roi_idx = exp_info['roi_idx_to_use']
        else:
            if use_alternative_roi:
                if len(face_ds.ROI) > 1:
                    roi_idx = 1
                else:
                    roi_idx = 0
            else:
                roi_idx = 0

        face_SVD = face_ds.isel(ROI=roi_idx)['motion_svd'].values
        motion_energy = face_ds.isel(ROI=roi_idx)['motion_energy'].values
        is_spontaneous_vec = np.zeros((len(face_time), 1))

        # face_acq_times[exp_folder] = face_time
        face_spontaneous_activity = []
        spont_break_indices = []
        spont_durations = []
        spont_break_indices_count = 0
        for sp_start, sp_end in zip(spontaneous_starts, spontaneous_ends):
            subset_time = np.where(
                (face_time >= sp_start) &
                (face_time <= sp_end)
            )[0]

            is_spontaneous_vec[subset_time] = 1

            face_spontaneous_activity.append(
                face_SVD[subset_time, :]
            )

            spont_break_indices.append(
                spont_break_indices_count
            )
            spont_break_indices_count += len(subset_time)

            spont_durations.append(sp_end - sp_start)

        resampled_motion_svd, all_subset_time_sec, new_resampled_time = anavid.resample_motion_svd(face_SVD, face_time, spontaneous_starts,
                                                                               spontaneous_ends,
                                                                               resample_time_bin=resample_time_bin)

        if include_full_data:
            full_resampled_motion_svd, full_motion_time_sec, new_full_resampled_time = \
                anavid.resample_motion_svd(face_SVD, face_time,
                                          subset_time_starts=None,
                                          subset_time_ends=None,
                                          resample_time_bin=resample_time_bin)
        exp_info['full_motion_time_sec'] = full_motion_time_sec

        if verbose:
            print('Resampled motion svd matrix shape:')
            print(np.shape(resampled_motion_svd))

        # Also get spontaneous neural data
        if exp_info['exp_type'] == 'spontaneous':
            all_plane_neural_spontaneous_activity_combined, cell_planes, each_plane_spont_activity \
                = extract_spont_activity_matrix(
                exp_folder, special_subfolder, target_planes, block_df,
                custom_spontaneous_starts=spontaneous_starts,
                custom_spontaneous_ends=spontaneous_ends,
                custom_time_bins=None, activity_name=activity_name)
        else:

            all_plane_neural_spontaneous_activity_combined, cell_planes, each_plane_spont_activity \
                = extract_spont_activity_matrix(
                exp_folder, special_subfolder, target_planes, block_df,
                custom_time_bins=None, activity_name=activity_name)

        if include_full_data:
            # include full neural activity PC
            all_plane_full_activity_combined, _, _ = extract_2p_activity_matrix(
                exp_folder, special_subfolder, target_planes,
                custom_time_bins=new_full_resampled_time, activity_name=activity_name,
                resample_time_bin=resample_time_bin,
            )
            exp_info['new_full_resampled_time'] = new_full_resampled_time

        if verbose:
            print('Shape of all plane neural spontneous activity combined')
            print(np.shape(all_plane_neural_spontaneous_activity_combined))

        # Remove NaNs in cells (seem to happen with matlab suite2p, but not in python)
        cell_isnot_nan = np.sum(np.isnan(all_plane_neural_spontaneous_activity_combined), axis=0) == 0
        not_nan_index = np.where(cell_isnot_nan == 1)[0]
        all_plane_neural_spontaneous_activity_combined = all_plane_neural_spontaneous_activity_combined[
                                                         :, not_nan_index,
                                                         ]

        all_plane_neural_spontaneous_activity_combined_pc = PCA(n_components=100).fit_transform(
            all_plane_neural_spontaneous_activity_combined)

        for n_plane, plane in enumerate(target_planes):
            plane_activity = each_plane_spont_activity[n_plane]
            plane_cell_isnot_nan = np.sum(np.isnan(plane_activity), axis=0) == 0
            plane_cell_isnot_nan_idx = np.where(plane_cell_isnot_nan == 1)[0]
            plane_activity = plane_activity[:, plane_cell_isnot_nan_idx]
            num_cell = len(plane_cell_isnot_nan_idx)
            num_components_to_get = np.min([100, num_cell])

            exp_info['plane_%.f_svd' % plane] = PCA(n_components=num_components_to_get).fit_transform(
                plane_activity
            )

        # Get cell X loc and Y loc
        cell_x_locs = []
        cell_y_locs = []

        for plane in target_planes:

            plane_file = glob.glob(os.path.join(exp_folder, special_subfolder, '*plane%.f.nc' % plane))[0]
            plane_ds = xr.open_dataset(plane_file)
            if 'X_loc' in plane_ds.keys():
                cell_x_locs.extend(plane_ds['X_loc'].values)
                cell_y_locs.extend(plane_ds['Y_loc'].values)
            else:
                print('WARNING: missing cell loc data, setting those to None for now ')
                # TODO: resolve this

        # Spontaneous activity PCA
        raster_sorted, sort_idx, corr_w_pc = ana2p.cal_corr_with_PC(
            neural_activity=all_plane_neural_spontaneous_activity_combined,
            neural_PC=all_plane_neural_spontaneous_activity_combined_pc, pc_index=0)

        exp_info['spontaneous_pc'] = all_plane_neural_spontaneous_activity_combined_pc
        exp_info['spontaneous_raster_sorted'] = raster_sorted
        exp_info['spontaneous_sort_idx'] = sort_idx
        exp_info['corr_w_pc'] = corr_w_pc
        exp_info['mean_image'] = face_proc['avgframe_reshape']
        exp_info['X_loc'] = cell_x_locs
        exp_info['Y_loc'] = cell_y_locs
        exp_info['cell_planes'] = cell_planes
        exp_info[
            'all_plane_neural_spontaneous_activity_combined_pc'] = all_plane_neural_spontaneous_activity_combined_pc
        exp_info['all_plane_neural_spontaneous_activity_combined'] = all_plane_neural_spontaneous_activity_combined
        exp_info['spont_durations'] = spont_durations
        exp_info['face_svd_spontaneous'] = face_spontaneous_activity
        exp_info['resampled_motion_svd'] = resampled_motion_svd
        exp_info['facecam_fps'] = 30
        exp_info['spont_break_indices'] = spont_break_indices
        exp_info['face_acq_times'] = face_time
        exp_info['motion_energy'] = motion_energy
        exp_info['motion_svd'] = face_SVD
        exp_info['is_spontaneous_vec'] = is_spontaneous_vec
        exp_info['spontaneous_starts'] = spontaneous_starts
        exp_info['spontaneous_ends'] = spontaneous_ends

        if include_full_data:
            exp_info['full_resampled_motion_svd'] = full_resampled_motion_svd
            exp_info['all_plane_full_activity_combined'] = all_plane_full_activity_combined
            # exp_info['full_motion_svd'] = face_SVD

        all_exp_data.append(exp_info)

    return all_exp_data



def cal_subsample_PCA():


    return


def subsample_spont_time_and_calculate_mov_neural_corr(exp_data, mot_svd_idx=0, neural_pc_idx=0, num_segments=3,
                                                       time_segment_lengths=np.linspace(1, 45, 15) * 60,
                                                       num_random_samples_per_time=10, min_distance_sec=10,
                                                       num_sec_per_idx=1.2, neural_activity_name='spikes_resampled',
                                                       movement_activity_name='motion_svd_resampled',
                                                       neural_time_first_dim=False, movement_time_first_dim=True, verbose=True):

    """
    Arguments
    -------------
    exp_data : (dict)
        dictionary containing experiment data
    mot_svd_idx : (int)
        which SVD component to use to correlate with neural activity, uses 0 indexing, so 0 means the first
        SVD component
    neural_pc_idx : (int)
        which neural PC component to use to correlate with movement, uses 0 indexing, so 0 means the first PC component
    num_segments : (int)
        number of time segments or chunks to get
    time_segment_lengths : (list or 1D numpy ndarray)
        list of time segment length duration (in seconds) to use
        if None or np.inf is in the time_segment_lengths, then the entire recording is used (no subsampling)
    subsampling_method : (str)
        how you want to subsample the segments
        'maximum-spacing' : space the segments so they are maximally apart (deterministic, no need to run
        randomisations
        'random-w-min-spacing' : select random segments but with a minimum of x seconds apart
        x is controlled by min_distance_sec
    min_distance_sec : (int)
        if using 'random-w-min-spacing', the duration in seconds in which random segments needs to be apart
    num_sec_per_idx : (float)
        time width of each index of the activity matrix (in seconds)
    neural_activity_name : (str)
        name of neural activity to use
            'spikes_resampled' for Stringer's data
            'all_plane_neural_spontaneous_activity_combined' for Sit's data
    movement_activity_name : (str)
        name of movement activity to use
            'motion_svd_resampled' for Stringer's data
            'resampled_motion_svd' for Sit's data
    neural_time_first_dim : (bool)
        whether time is in the first dimension of neural data
    movement_time_first_dim : (bool)
        whether time is in the first dimension of movement data
    Output
    --------------
    r_results_da : (xarray dataarray)
        Dataarray with dimensions time_segment_length and randomisation
        Each entry is the Pearson's r value of the correlation between movement and neural component (1 dimensional signals)
    """

    if neural_time_first_dim:
        exp_data[neural_activity_name] = exp_data[neural_activity_name].T
    if not movement_time_first_dim:
        exp_data[movement_activity_name] = exp_data[movement_activity_name].T

    num_pc = mot_svd_idx + 1
    subsampling_method = 'random-w-min-spacing'

    num_recording_idx = np.shape(exp_data[neural_activity_name])[1]
    all_spikes = exp_data[neural_activity_name]

    # Check recording duration is long enough
    if verbose:
        print('Recording duration: %.1f seconds' % (num_recording_idx * num_sec_per_idx))

    r_2_results = np.zeros((len(time_segment_lengths), num_random_samples_per_time))

    for n_t, t_length in tqdm(enumerate(time_segment_lengths)):

        if (t_length == np.inf) or (t_length is None):
            # No subsampling
            subsampled_spikes = exp_data[neural_activity_name]
            spikes_subsampled_pc = PCA(n_components=num_pc).fit_transform(subsampled_spikes.T)
            spike_component = spikes_subsampled_pc[:, neural_pc_idx]
            motion_svd_component = exp_data[movement_activity_name][:, mot_svd_idx]
            r_2 = np.corrcoef(motion_svd_component, spike_component)[0, 1]
            r_2_results[n_t, :] = r_2
        else:
            for n_try, random_try in enumerate(np.arange(num_random_samples_per_time)):
                time_indices_segs = ana2p.subsample_segments_from_time_series(num_recording_idx,
                                                                              time_seg_length=t_length,
                                                                              num_sec_per_idx=num_sec_per_idx,
                                                                              method=subsampling_method,
                                                                              num_segments=num_segments,
                                                                              min_distance_sec=min_distance_sec)
                time_indices = np.concatenate(time_indices_segs)
                subsampled_spikes = exp_data[neural_activity_name][:, time_indices]
                spikes_subsampled_pc = PCA(n_components=num_pc).fit_transform(subsampled_spikes.T)
                spike_component = spikes_subsampled_pc[:, neural_pc_idx]
                motion_svd_component = exp_data[movement_activity_name][time_indices, mot_svd_idx]

                r_2 = np.corrcoef(motion_svd_component, spike_component)[0, 1]
                r_2_results[n_t, n_try] = r_2

    r_results_da = xr.DataArray(r_2_results, dims=['time_seg_length', 'randomisation'],
                    coords=[time_segment_lengths, np.arange(0, num_random_samples_per_time)])
    r_results_da.attrs['mot_svd_idx'] = mot_svd_idx
    r_results_da.attrs['num_segments'] = num_segments
    r_results_da.attrs['subsampling_method'] = subsampling_method
    r_results_da.attrs['recording_duration'] = num_recording_idx * num_sec_per_idx
    r_results_da.attrs['subject'] = exp_data['subject']
    r_results_da.attrs['exp_date'] = exp_data['exp_date']

    if 'exp_num' in exp_data.keys():
        r_results_da.attrs['exp_num'] = exp_data['exp_num']

    return r_results_da


def subtract_stim_activity(plane_ds, block_df, aligned_plane_ds=None, time_before_event=0,
                           time_after_event=4.5, activity_name='Spikes',
                           cal_zscore=False):


    # Get simulus-aligned activity
    if aligned_plane_ds is None:
        alignment_unit = 'frames'
        event_type = 'custom_times'
        custom_times = block_df['stimOnTime'].values
        remove_overstep_right_edge = False
        time_rounding_decimal_places = None
        miss_alignment_replacement_method = None
        include_cell_loc = True
        variables_to_keep = ['Fluorescence', 'Spikes']
        # Calculate z-scored activity as well
        if cal_zscore:
            variables_to_keep = ['Fluorescence', 'Spikes', 'zScoredSpikes', 'zScoredF']
            zscored_spikes = ana2p.normalise_activity(neural_xarray=plane_ds['Spikes'],
                                                  across='Time', method='zscore')
    
            plane_ds['zScoredSpikes'] = (['Plane', 'Cell', 'Time'], np.expand_dims(zscored_spikes, axis=0))
            zscored_F = ana2p.normalise_activity(neural_xarray=plane_ds['Fluorescence'],
                                             across='Time', method='zscore')
    
            plane_ds['zScoredF'] = (['Plane', 'Cell', 'Time'], np.expand_dims(zscored_F, axis=0))
            aligned_ds_save_path = os.path.join(exp_folder, 'aligned_plane_%.f.nc' % plane_number)
    
        aligned_plane_ds = ana2p.align_neural_xarray(neural_xarray=plane_ds,
                                                     block_data=None, timeline_xarray=None,
                                                     alignment_event_time_variable='stimOnTime',
                                                     alignment_groupby_variable=None, event_type=event_type,
                                                     custom_times=custom_times,
                                                     time_before_event=time_before_event,
                                                     time_after_event=time_after_event,
                                                     time_rounding_decimal_places=time_rounding_decimal_places,
                                                     variables_to_keep=variables_to_keep,
                                                     alignment_unit=alignment_unit,
                                                     output_type='xarray-concat',
                                                     reindex_trial=False,
                                                     miss_alignment_replacement_method=miss_alignment_replacement_method,
                                                     remove_overstep_right_edge=remove_overstep_right_edge)
    
        aligned_plane_ds['audio'] = ('Trial', block_df['audio'])
        aligned_plane_ds['video'] = ('Trial', block_df['video'])

    # Get video and audio component of response
    video_response_component, audio_response_component = ana2p.cal_video_and_audio_components(
        nat_video_aligned_plane_ds=aligned_plane_ds,
        activity_name=activity_name,
        subtract_mean_response=True)

    # Get matrix with audio and video component added
    num_cell = len(plane_ds.Cell)
    num_time_points = len(plane_ds.Time)
    audio_video_component_matrix = np.zeros((num_time_points, num_cell))

    for trial_idx, trial_df in block_df.iterrows():

        time_index_to_insert = np.argmin(np.abs(
            trial_df['stimOnTime'] - plane_ds.Time.values
        ))

        start_idx = time_index_to_insert
        end_idx = time_index_to_insert + len(aligned_plane_ds.FrameIndex)

        if trial_df['audio'] != -1:
            aud_component = np.squeeze(audio_response_component.where(
                audio_response_component['audio'] == trial_df['audio'], drop=True
            ).values)
        else:
            aud_component = 0

        if trial_df['video'] != -1:
            vid_component = np.squeeze(video_response_component.where(
                video_response_component['video'] == trial_df['video'], drop=True
            ).values)
        else:
            vid_component = 0

        aud_vid_component = aud_component + vid_component

        try:
            if type(aud_vid_component) is int:
                audio_video_component_matrix[start_idx:end_idx, :] = aud_vid_component
            else:
                audio_video_component_matrix[start_idx:end_idx, :] = aud_vid_component.T
        except:
            pdb.set_trace()

    # Subtract stimulus activity from original activity
    audio_video_component_da = xr.DataArray(audio_video_component_matrix, dims=['Time', 'Cell'])
    stim_subtracted_activity = plane_ds.isel(Plane=0)[activity_name] - audio_video_component_da

    return stim_subtracted_activity


def run_svca(activity, x_loc, y_loc, division_method='row', num_sec_per_idx=1,
             time_seg_length=60, npc=1000, return_full_info=False):

    activity_sum_per_cell = np.sum(activity, axis=1)
    activity_subset = activity[activity_sum_per_cell > 0, :]  # remove cells with no activity (deconvovled is always positive)
    # activity_mean_subtracted = Ff_subset - np.mean(activity_subset, axis=1).reshape(-1, 1)  # mean across time for each neuron
    activity_mean_subtracted = activity_subset - np.mean(activity_subset, axis=1).reshape(-1,
                                                                                    1)  # mean across time for each neuron
    x_loc_subset = x_loc[activity_sum_per_cell > 0]
    y_loc_subset = y_loc[activity_sum_per_cell > 0]
    num_neurons, num_time_points = np.shape(activity_mean_subtracted)

    ntrain, ntest = ana2p.get_train_test_neurons(
        x_loc_subset, y_loc_subset, image_x=None, image_y=None, num_dividers=16,
        division_method=division_method
    )

    num_recording_idx = num_time_points

    itrain, itest = ana2p.get_train_test_time_index(num_recording_idx, time_seg_length=time_seg_length,
                                                    num_sec_per_idx=num_sec_per_idx)

    sneur, varneur, u, v = ana2p.SVCA(Ff=activity_mean_subtracted,
                                      npc=npc, ntrain=ntrain, ntest=ntest, itrain=itrain, itest=itest)

    # Obtain the projection to shared variance components at training time points
    Ff_ntrain_ttrain = activity_mean_subtracted[ntrain, :][:, itrain]
    Ff_ntest_ttrain = activity_mean_subtracted[ntest, :][:, itrain]

    svc_ntrain_ttrain = np.matmul(u.T, Ff_ntrain_ttrain)
    svc_ntest_ttrain = np.matmul(v.T, Ff_ntest_ttrain)

    Ff_ntrain_ttest = activity_mean_subtracted[ntrain, :][:, itest]
    Ff_ntest_ttest = activity_mean_subtracted[ntest, :][:, itest]
    svc_ntrain_ttest = np.matmul(u.T, Ff_ntrain_ttest)
    svc_ntest_ttest = np.matmul(v.T, Ff_ntest_ttest)

    if return_full_info:
        return sneur, varneur, u, v, svc_ntrain_ttest, svc_ntest_ttest, svc_ntrain_ttrain, svc_ntest_ttrain, itrain, itest
    else:
        return sneur, varneur, u, v, svc_ntrain_ttest, svc_ntest_ttest


def run_svca_stringer_data(stringer_all_exp_data, save_folder, npc=1024,
                           activity_name='spikes_resampled',
                           num_sec_per_idx=1.2):


    for n_exp, exp_data in tqdm(enumerate(stringer_all_exp_data)):
        subject = exp_data['subject']
        exp_date = exp_data['exp_date']
        Ff = exp_data[activity_name]
        pdb.set_trace()
        Ff_sum_per_cell = np.sum(Ff, axis=1)
        Ff_subset = Ff[Ff_sum_per_cell > 0, :]  # remove cells with no activity (deconvovled is always positive)
        Ff_mean_subtracted = Ff_subset - np.mean(Ff_subset, axis=1).reshape(-1, 1)  # mean across time for each neuron

        x_loc_subset = exp_data['X_loc'][Ff_sum_per_cell > 0]
        y_loc_subset = exp_data['Y_loc'][Ff_sum_per_cell > 0]

        # num_time_points, num_neurons = np.shape(exp_data['spikes_resampled'].T)
        num_neurons, num_time_points = np.shape(Ff_mean_subtracted)

        ntrain, ntest = ana2p.get_train_test_neurons(
            x_loc_subset, y_loc_subset, image_x=None, image_y=None, num_dividers=16,
            division_method='row'
        )

        num_recording_idx = num_time_points

        # time_seg_length = time_seg_length * 1.2
        itrain, itest = ana2p.get_train_test_time_index(num_recording_idx, time_seg_length=60,
                                                        num_sec_per_idx=num_sec_per_idx)

        sneur, varneur, u, v = ana2p.SVCA(Ff=Ff_mean_subtracted,
                                          npc=npc, ntrain=ntrain, ntest=ntest, itrain=itrain, itest=itest)

        # Obtain the projection to shared variance components at training time points
        Ff_ntrain_ttrain = Ff_mean_subtracted[ntrain, :][:, itrain]
        Ff_ntest_ttrain = Ff_mean_subtracted[ntest, :][:, itrain]
        svc_ntrain_ttrain = np.matmul(u.T, Ff_ntrain_ttrain)
        svc_ntest_ttrain = np.matmul(v.T, Ff_ntest_ttrain)

        Ff_ntrain_ttest = Ff_mean_subtracted[ntrain, :][:, itest]
        Ff_ntest_ttest = Ff_mean_subtracted[ntest, :][:, itest]
        svc_ntrain_ttest = np.matmul(u.T, Ff_ntrain_ttest)
        svc_ntest_ttest = np.matmul(v.T, Ff_ntest_ttest)

        svca_results = {
            'subject': subject,
            'exp_date': exp_date,
            'sneur': sneur,
            'varneur': varneur,
            'u': u,
            'v': v,
            'svc_ntrain_ttrain': svc_ntrain_ttrain,
            'svc_ntest_ttrain': svc_ntest_ttrain,
            'svc_ntrain_ttest': svc_ntrain_ttest,
            'svc_ntest_ttest': svc_ntest_ttest,
        }

        save_name = '%s_%s_svca_results.pkl' % (subject, exp_date)
        save_path = os.path.join(save_folder, save_name)

        with open(save_path, 'wb') as handle:
            pkl.dump(svca_results, handle)


def plot_svca_train_vs_test_corr(svca_result_fpaths, fig_folder, fig_name):

    svc_indices = [0, 9, 99]

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, axs = plt.subplots(len(svca_result_fpaths), len(svc_indices),
                                sharex=True, sharey=True)
        fig.set_size_inches(8, 14)

        for n_fpath, fpath in enumerate(svca_result_fpaths):

            with open(fpath, 'rb') as handle:
                svca_results = pkl.load(handle)

            for n_svc, svc_idx in enumerate(svc_indices):

                cell_set_1_svc = svca_results['svc_ntrain_ttest'][svc_idx, :]
                cell_set_2_svc = svca_results['svc_ntest_ttest'][svc_idx, :]
                axs[n_fpath, n_svc].scatter(
                    cell_set_1_svc,
                    cell_set_2_svc, color='black', lw=0, s=9
                )

                r_2 = np.corrcoef(cell_set_1_svc, cell_set_2_svc)[0, 1] ** 2
                axs[n_fpath, n_svc].text(0.3, 0.8, '$r^2 = %.2f$' % (r_2), ha='center', va='center',
                                         transform=axs[n_fpath, n_svc].transAxes)

                if svc_idx == 0:
                    axs[n_fpath, n_svc].set_ylabel('%s \n %s' % (svca_results['subject'], svca_results['exp_date']),
                                                   size=9)
                if n_fpath == 0:
                    axs[n_fpath, n_svc].set_title('SVC %.f' % (svc_idx + 1), size=10)

        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300)


def plot_movement_aligned_to_stim(face_ds, block_df,
                                  num_repeats=4, time_before_event=0.4, time_after_event=5,
                                  variables_to_keep=['motion_energy', 'motion_svd',
                                                     'area', 'area_smooth',
                                                     'com', 'com_smooth'
                                                     ]
                                  ):

    stim_aligned_face_ds = pface.align_face_ds(face_ds,
                                               behave_df=block_df,
                                               alignment_event_time_variable='stimOnTime',
                                               variables_to_keep=variables_to_keep,
                                               time_before_event=time_before_event,
                                               time_after_event=time_after_event,
                                               reindex_trial=False, trial_var_name='trial',
                                               time_rounding_decimal_places=None)

    repeat = np.repeat(np.arange(0, num_repeats), len(stim_aligned_face_ds.Trial) / num_repeats)

    stim_aligned_face_ds['audio'] = ('Trial', block_df['audio'])
    stim_aligned_face_ds['video'] = ('Trial', block_df['video'])
    stim_aligned_face_ds['repeat'] = ('Trial', repeat)

    audio_by_video_time_motion_matrix_list = list()

    for repeat in np.arange(num_repeats):
        audio_by_video_time_motion_matrix = anafilmworld.get_audio_by_videotime_matrix(
            cell_aligned_ds=stim_aligned_face_ds.isel(ROI=0),
            activity_name='motion_energy',
            repeat_id=repeat, const_modality='audio',
            var_modality='video', time_dim_name='FrameIndex')
        audio_by_video_time_motion_matrix_list.append(audio_by_video_time_motion_matrix)

    video_by_audio_time_motion_matrix_list = list()

    for repeat in np.arange(num_repeats):
        video_by_audio_time_motion_matrix = anafilmworld.get_audio_by_videotime_matrix(
            cell_aligned_ds=stim_aligned_face_ds.isel(ROI=0),
            activity_name='motion_energy',
            repeat_id=repeat, const_modality='video',
            var_modality='audio', time_dim_name='FrameIndex')
        video_by_audio_time_motion_matrix_list.append(video_by_audio_time_motion_matrix)

    num_time_bins = len(stim_aligned_face_ds['FrameIndex'])
    repeat_to_plot = 0
    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        fig, ax = anafilmworld.plot_audio_by_vidoetime_matrix(
            video_by_audio_time_motion_matrix_list[repeat_to_plot], num_time_bins,
            fig=fig, ax=ax)
        ax.set_xlabel('Audio and time', size=12)
        ax.set_ylabel('Video', size=12)


def load_filmworld_motion_and_neural_resampled(exp_folder, activity_name='zScoredSpikes', resample_time_bin=1.2):
    """

    """
    # Load facemap data
    face_ds_path = glob.glob(os.path.join(exp_folder, '*face*.nc'))
    face_ds_path = [x for x in face_ds_path if 'aligned' not in x]

    if len(face_ds_path) != 1:
        print('Combined ds path not found, please debug')
        return None, None

    face_ds = xr.open_dataset(face_ds_path[0])

    combined_ds_path = glob.glob(os.path.join(exp_folder, '*combined*.nc'))
    combined_ds_path = [x for x in combined_ds_path if 'aligned' not in x]
    if len(combined_ds_path) != 1:
        print('Combined ds path not found, please debug')
        return None, None, None
    combined_ds = xr.open_dataset(combined_ds_path[0])  # not this is not aligned
    combined_ds_time_bins = combined_ds['interpolatedTime'].values
    plane_per_cell = combined_ds['Plane'].values

    # load the stimulus info
    trial_info_processed_df_path = glob.glob(os.path.join(exp_folder, '*trial_info_processed.csv'))[0]
    trial_info_processed_df = pd.read_csv(trial_info_processed_df_path)
    big_iti_times = np.where(np.diff(trial_info_processed_df['stimOnTime']) > 60)[0]
    spontaneous_activity_start_times = trial_info_processed_df.iloc[big_iti_times][
                                           'stimOnTime'].values + 10
    spontaneous_activity_end_times = trial_info_processed_df.iloc[big_iti_times + 1][
                                         'stimOnTime'].values - 10

    # resample face ds to match the time bins of combine_ds
    mot_svd = face_ds.isel(ROI=0)['motion_svd'].values
    face_time = face_ds['Time'].values
    new_resampled_time = combined_ds.interpolatedTime.values
    resampled_motion_svd, all_subset_time_sec, all_new_resampled_time = anavid.resample_motion_svd(
        mot_svd, face_time, subset_time_starts=None, subset_time_ends=None,
        new_resampled_time=new_resampled_time,
        resample_time_bin=resample_time_bin,
        default_start_time_at_zero=True,
        resampling_method='interp')

    subset_time_idx = []
    for start_time, end_time in zip(spontaneous_activity_start_times, spontaneous_activity_end_times):
        subset_time_idx.append(
            np.where((new_resampled_time > start_time) &
                     (new_resampled_time < end_time)
                     )[0]
        )
    subset_time_idx = np.concatenate(subset_time_idx)

    cell_activity_spont = combined_ds[activity_name].values[:, subset_time_idx].T
    resampled_motion_svd_spont = resampled_motion_svd[subset_time_idx, :]

    return cell_activity_spont, resampled_motion_svd_spont, plane_per_cell


def cal_mov_and_neural_cca_per_plane(cell_activity_spont, resampled_motion_svd_spont, plane_per_cell, time_seg_length=60,
                                     num_sec_per_idx=1.2, tot_planes_to_use=11, use_neural_pc=False,
                                     num_mot_svd_to_use=100, num_neural_PCs=10, total_num_CCA_components=10,
                                     return_projections=False):
    """

    """

    plane_cca_corr = np.zeros((tot_planes_to_use, total_num_CCA_components))

    num_recording_idx = np.shape(cell_activity_spont)[0]
    itrain, itest = ana2p.get_train_test_time_index(num_recording_idx, time_seg_length=time_seg_length,
                                                    num_sec_per_idx=num_sec_per_idx)

    # Store projections
    neural_X_c_train_per_plane = []
    face_X_c_train_per_plane = []
    neural_X_c_test_per_plane = []
    face_X_c_test_per_plane = []

    for n_plane, plane_number in enumerate(np.arange(0, tot_planes_to_use)):

        subset_cell_idx = np.where(plane_per_cell == plane_number)[0]
        plane_activity = cell_activity_spont[:, subset_cell_idx]

        if use_neural_pc:
            neural_pcs = skldecomposition.PCA(num_neural_PCs).fit_transform(plane_activity)
            neural_pcs = sstats.zscore(neural_pcs, axis=0)
            Y_train, Y_test = neural_pcs[itrain, :], neural_pcs[itest, :]
        else:
            Y_train, Y_test = plane_activity[itrain, :], plane_activity[itest, :]

        motion_svd_predictors = resampled_motion_svd_spont[:, 0:num_mot_svd_to_use]
        motion_svd_predictors = sstats.zscore(motion_svd_predictors, axis=0)
        X_train, X_test = motion_svd_predictors[itrain, :], motion_svd_predictors[itest, :]

        cca_model = CCA(n_components=total_num_CCA_components)
        cca_model.fit(Y_train, X_train)
        neural_X_c_train, face_X_c_train = cca_model.transform(Y_train, X_train)
        neural_X_c_test, face_X_c_test = cca_model.transform(Y_test, X_test)

        if return_projections:
            neural_X_c_train_per_plane.append(neural_X_c_train)
            face_X_c_train_per_plane.append(face_X_c_train)
            neural_X_c_test_per_plane.append(neural_X_c_test)
            face_X_c_test_per_plane.append(face_X_c_test)

        for component_idx in np.arange(0, total_num_CCA_components):
            plane_cca_corr[n_plane, component_idx] = np.corrcoef(neural_X_c_test[:, component_idx],
                                                                 face_X_c_test[:, component_idx])[0, 1]

    neural_X_c_train_per_plane = np.array(neural_X_c_train_per_plane)
    face_X_c_train_per_plane = np.array(face_X_c_train_per_plane)
    neural_X_c_test_per_plane = np.array(neural_X_c_test_per_plane)
    face_X_c_test_per_plane = np.array(face_X_c_test_per_plane)

    if return_projections:
        return plane_cca_corr, neural_X_c_train_per_plane, face_X_c_train_per_plane,  neural_X_c_test_per_plane,  face_X_c_test_per_plane
    else:
        return plane_cca_corr

def main():

    dataset_to_use = ['sit']

    main_folder = '/Volumes/timsitHD-2020-03:media:timsit:timsitHD-2020-03/cortexlab'

    # load my data (Sit's)
    # exps_to_analyse_path = '/media/timsit/timsitHD-2020-03/cortexlab/neural-vs-movement/exps_to_analyse.csv'
    exps_to_analyse_path = '/Volumes/timsitHD-2020-03:media:timsit:timsitHD-2020-03/cortexlab/neural-vs-movement/exps_to_analyse.csv'
    exps_to_analyse_df = pd.read_csv(exps_to_analyse_path)
    exps_to_analyse_df['main_folder'] = '/Volumes/timsitHD-2020-03:media:timsit:timsitHD-2020-03/cortexlab/subjects'

    # Drop TS mice for now because spontaneous period too short
    exps_to_analyse_df = exps_to_analyse_df.loc[
        ~exps_to_analyse_df['subject'].isin(['TS008', 'TS010'])
    ]

    exps_to_analyse_df = exps_to_analyse_df.loc[
        exps_to_analyse_df['exp_date'] == '2021-12-10'
    ]

    # sit_all_exp_data = anamovneural.load_sit_data(exps_to_analyse_df)
    sit_save_folder = '/Volumes/timsitHD-2020-03:media:timsit:timsitHD-2020-03/cortexlab/subjects/timSpont'

    # Load Stringer's data
    stringer_data_folder = '/Volumes/timsitHD-2020-03:media:timsit:timsitHD-2020-03/cortexlab/subjects/stringer2018spont'
    spont_data_fpaths = np.sort(glob.glob(os.path.join(stringer_data_folder, 'spont_*.mat')))
    stringer_save_folder = os.path.join(stringer_data_folder, 'analysisResults')

    supported_processes = ['cal_mov_neural_corr',
                          'subsample_spont_time_and_calculate_mov_neural_corr', 'load_stringer_matlab_data',
                         'get_stimulus_subtracted_activity', 'plot_svca_train_vs_test_corr',
                          'run_svca_stringer_data']

    processes_to_run = ['run_svca_stringer_data']

    process_params = {
                      'run_svca_stringer_data': dict(
                        plot_folder=''
                      ),
                      'subsample_spont_time_and_calculate_mov_neural_corr':
                      dict(mot_svd_idx=0, num_segments=3,
                           time_segment_lengths=np.concatenate([np.linspace(1, 14, 15) * 60,
                                                                np.array([np.inf])
                                                                ]),
                           num_random_samples_per_time=10,
                           num_sec_per_idx=1.2),
                      'load_stringer_matlab_data': dict(
                        save_folder=stringer_save_folder
                      ),
                      'plot_svca_train_vs_test_corr': dict(
                       #  svca_result_fpaths=glob.glob(
                       #  os.path.join(main_folder, 'neural-vs-movement/svca', '*.pkl')),
                        svca_result_fpaths=glob.glob(os.path.join(
                            '/Volumes/timsitHD-2020-03:media:timsit:timsitHD-2020-03/cortexlab/subjects/stringer2018spont/analysisResults', '*svca_results.pkl')),
                        fig_folder=os.path.join(main_folder, 'neural-vs-movement/svca/'),
                        fig_name='MP_mice_cell_train_vs_test_corr_in_test_set_2022-01-10_new_train_test.png',
                      ),
                      'run_svca_stringer_data': dict(
                          save_folder=stringer_save_folder, npc=1024,
                          activity_name='spikes_resampled',
                           num_sec_per_idx=1.2
                      ),
                      'plot_movement_aligned_to_stim': dict(
                          save_folder=None,
                      )
    }

    for process_name in processes_to_run:
        if process_name == 'cal_mov_neural_corr':
            print('Loading data')
            if 'sit' in dataset_to_use:
                all_exp_data = load_sit_data(exps_to_analyse_df)
                process_params['subsample_spont_time_and_calculate_mov_neural_corr']['neural_activity_name']\
                    = 'all_plane_neural_spontaneous_activity_combined'
                process_params['subsample_spont_time_and_calculate_mov_neural_corr']['movement_activity_name']\
                    = 'resampled_motion_svd'
                process_params['subsample_spont_time_and_calculate_mov_neural_corr']['neural_time_first_dim']\
                    = True
                save_folder = sit_save_folder
            elif 'stringer' in dataset_to_use:
                all_exp_data = load_stringer_matlab_data(spont_data_fpaths)
                save_folder = stringer_save_folder
            print('Plotting correlation of neural PC1 and motion PC1')

        if process_name == 'subsample_spont_time_and_calculate_mov_neural_corr':
            print('Loading data')
            if 'sit' in dataset_to_use:
                all_exp_data = load_sit_data(exps_to_analyse_df)
                process_params['subsample_spont_time_and_calculate_mov_neural_corr']['neural_activity_name']\
                    = 'all_plane_neural_spontaneous_activity_combined'
                process_params['subsample_spont_time_and_calculate_mov_neural_corr']['movement_activity_name']\
                    = 'resampled_motion_svd'
                process_params['subsample_spont_time_and_calculate_mov_neural_corr']['neural_time_first_dim']\
                    = True
                save_folder = sit_save_folder
            elif 'stringer' in dataset_to_use:
                all_exp_data = load_stringer_matlab_data(spont_data_fpaths)
                save_folder = stringer_save_folder
            print('Running pca with time subsamples')
            for n_exp, exp_data in enumerate(all_exp_data):
                r_results_da = subsample_spont_time_and_calculate_mov_neural_corr(exp_data,
                                                                   **process_params[process_name])
                file_name = 'spont_%s_%s_subsample_spont_time_results.nc' % (exp_data['subject'],
                                                                              exp_data['exp_date'])
                save_path = os.path.join(save_folder, file_name)
                r_results_da.to_netcdf(save_path)
                print('Processed %.f out of %.f files' % (n_exp+1, len(all_exp_data)))
        if process_name == 'get_stimulus_subtracted_activity':
            print('Getting stimulus subtracted activity')
            if 'sit' in dataset_to_use:
                for n_exp, exp_info in exps_to_analyse_df.iterrows():
                    exp_folder = os.path.join(exp_info['main_folder'], exp_info['subject'],
                                              exp_info['exp_date'], str(exp_info['exp_num']))

                    plane_ds_paths = glob.glob(os.path.join(
                        exp_folder, '*plane*.nc'
                    ))

                    plane_ds_paths = [x for x in plane_ds_paths if x not in ['aligned', 'stim_subtracted']]


                    # Get stimulus info
                    block_csv_search_result = glob.glob(os.path.join(exp_folder, '*trial_info_processed.csv'))
                    if len(block_csv_search_result) == 0:
                        block_nc_path = glob.glob(os.path.join(exp_folder, '*blockTimeline.nc'))[0]
                        block_ds = xr.open_dataset(block_nc_path)
                        block_df = pd.DataFrame.from_dict({
                            'Trial': block_ds['Trial'].values,
                            'stimOnTime': block_ds['movieStartTimes'].values,
                            'audio': block_ds['audio'].values,
                            'video': block_ds['video'].values
                        })

                    else:
                        block_df = pd.read_csv(block_csv_search_result[0])
                        # Add video and audio name
                        json_fpath = glob.glob(os.path.join(exp_folder, '*movie_params.json'))[0]

                        with open(json_fpath) as json_file:
                            param_info_dict = json.load(json_file)

                        block_df['video'] = [
                            param_info_dict[
                                trial_info['movie_fname'].split('_')[1].split('.')[0]
                            ]['video_id'] for _, trial_info in block_df.iterrows()
                        ]

                        block_df['audio'] = [
                            param_info_dict[
                                trial_info['movie_fname'].split('_')[1].split('.')[0]
                            ]['audio_id'] for _, trial_info in block_df.iterrows()
                        ]

                    # Do subtraction
                    print('Doing stimulus subtraction for %s %s exp %s' % (exp_info['subject'],
                                              exp_info['exp_date'], str(exp_info['exp_num'])))
                    for path in plane_ds_paths:
                        plane_ds = xr.open_dataset(path)
                        stim_subtracted_plane_ds = subtract_stim_activity(plane_ds, block_df=block_df,
                                               aligned_plane_ds=None, time_before_event=0,
                                               time_after_event=4.5, activity_name='Spikes',
                                               cal_zscore=False)
                        plane_number = plane_ds.Plane.values[0]
                        stim_sub_save_path = os.path.join(
                            exp_folder, 'stim_subtracted_plane%.f.nc' % (plane_number)
                        )
                        stim_subtracted_plane_ds.to_netcdf(stim_sub_save_path)

                    print('Processed %.f out of %.f files' % (n_exp, len(exps_to_analyse_df)))
        if process_name == 'load_stringer_matlab_data':
            print('Compiling stringer data')
            stringer_all_exp_data = load_stringer_matlab_data(spont_data_fpaths)
            save_path = os.path.join(process_params[process_name]['save_folder'],
                                   'stringer_all_exp_data.pkl')
            # TODO: this saves a large file and takes almost as long as the loading process... need to do this better
            with open(save_path, 'wb') as handle:
                pkl.dump(stringer_all_exp_data, handle)
            print('Saved data to: %s' % save_path)
        if process_name == 'run_svca_stringer_data':
            print('Loading data')
            stringer_all_exp_data = load_stringer_matlab_data(spont_data_fpaths)
            print('Running SVCA through stringer dataset')
            run_svca_stringer_data(stringer_all_exp_data, **process_params[process_name])

        if process_name == 'plot_svca_train_vs_test_corr':
            print('Plotting SVCA ntrain and ntest neuron correlation')
            plot_svca_train_vs_test_corr(**process_params[process_name])

if __name__ == '__main__':
    main()