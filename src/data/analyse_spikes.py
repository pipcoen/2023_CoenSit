import pdb

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import scipy.ndimage as spimage
import scipy.signal as spsignal
import scipy.stats as sstats
import xarray as xr
import pickle as pkl
from collections import defaultdict
import itertools
import os

import src.data.process_ephys_data as pr_ephys
import collections
from src.visualization import vizbehaviour

# pop analysis / similarity anlaysis
import scipy.spatial as sspatial
from sklearn.decomposition import PCA

# For alignment to stimulus first, peform mean subtraction, then align back to movement
import src.data.alignment_mean_subtraction as ams


def spike_df_to_xr(spike_df, neuron_df=None, behave_df=None, event_name_to_set_bin=None, bin_width=0.05,
                   include_cell_loc=False, verbose=False):
    """
    Converting dataframe containing spike times to a xarray dataset with binned spikes
    Parameters
    ------------
    spike_df : (pandas dataframe)
        dataframe containing spike information. Contains two columns
        (1) spikeTime : time of spike (in seconds)
        (2) cellId : id specific to the cell (across all experiments and all mice)
    neuron_df : (pandas dataframe)
        optional pandas dataframe with neuron information.
        used here to extract the number of neurons, useful there are neurons that has zero firing rate
        in the experiment (and therefore won't show up in spike_df) but we still want to keep
        track of it in any case (eg. subthreshold activity)
    bin_width: (float)
        bin width in seconds
    behave_df : (pandas dataframe)
        optional pandas dataframe with trial information
    event_name_to_set_bin : (str)
        optional alternative for selecting the range of binning times
        special keyword 'trialStartEnd' (which I think is the sensible for thing to do 99% of the time), which
        set the bin start to the start of the first trial and bin end to the end of the last trial
    :return:
    """

    first_bin_time = np.min(spike_df['spikeTime'].values)
    last_bin_time = np.max(spike_df['spikeTime'].values)

    if behave_df is not None and event_name_to_set_bin is not None:
        if event_name_to_set_bin == 'trialStartEnd':
            first_bin_time = np.min(behave_df['trialStart'])
            last_bin_time = np.max(behave_df['trialEnd'])
        else:
            first_bin_time = np.min(behave_df[event_name_to_set_bin].values)
            last_bin_time = np.max(behave_df[event_name_to_set_bin].values)

    assert last_bin_time > first_bin_time

    time_bin_width_sec = bin_width

    if verbose:
        print('First bin time' + str(first_bin_time))
        print('Last bin time' + str(last_bin_time))
        print('Time bin width' + str(time_bin_width_sec))

    time_bin_vec = np.arange(first_bin_time, last_bin_time, time_bin_width_sec)

    if neuron_df is not None:
        num_cells = len(np.unique(neuron_df['cellId']))
        cell_coords = np.sort(np.unique(neuron_df['cellId']))
    else:
        num_cells = len(np.unique(spike_df['cellId']))
        cell_coords = np.sort(np.unique(spike_df['cellId']))

    num_time_bins = len(time_bin_vec) - 1
    binned_spike_matrix = np.zeros((num_cells, num_time_bins))
    for cell_idx, cell_id in enumerate(np.sort(np.unique(spike_df['cellId']))):
        binned_spikes, bins = np.histogram(spike_df['spikeTime'].loc[
                                               spike_df['cellId'] == cell_id], time_bin_vec
                                           )
        binned_spike_matrix[cell_idx, :] = binned_spikes

    binned_spike_ds = xr.Dataset({'SpikeRate': (['Cell', 'Time'], binned_spike_matrix
                                                )},
                                 coords={'Time': ('Time', time_bin_vec[:-1]),
                                         'Cell': ('Cell', cell_coords)
                                         })

    if include_cell_loc:
        binned_spike_ds = binned_spike_ds.assign_coords({'CellLoc': ('Cell', neuron_df['cellLoc'])})

    return binned_spike_ds


# TODO: please subset ephys_spike_df by subject and reference before inputting to below


def aligned_dicts_to_xdataset(event_aligned_spike_dicts, aligned_event=None,
                              meta_fields=['cell_ref'], combine_exp=False,
                              meta_dict=None, make_meta_dict_dim=False,
                              fields_to_variables=None, include_trial=False,
                              include_cell_loc=False, include_cell_hemisphere=False):
    """
    Converts dictionary containing alingment data (obtained using align_and_bin_spikes function)
    to an xarray dataset object, which makes processing it later easier.

    Arguments
    ------------
    combine_exp         : (bool)
        if False, returns a list of xarrays, if True, combines all experiments into one dataframe.
    fields_to_variables : (list)
        list of extra dictionary keys to make into xarray varaible (other than aligned firing rate)
    For now, I assume all fields have dimension (trial), ie. they are features that occur on a trial by trial basis.
    meta_fields : (list of str)
        ???
    include_cell_hemisphere (bool)
        whether to include the hemisphere of the brain each cell is from
    --------------
    Returns
    --------------
    dataset_list  : (list)
        a list of xarray dataset object, each list represent a single experiment.
        each xarray dataset will have the following dimensions:
            Cell : cell number
            Time : with two coordinates (1) 'Time': time bins, 'PeriEventTime': time (seconds) relative to event onset.
            Trial : Trial number

    include_trial : (bool)
        whether to preserve the original trial number/index from behaviour df in the trial
        dimension. Default is False.
    include_cell_loc: : (bool)
        whether to include cell location as a coordinate of the cell dimension.
    Assumptions:
    ------------
    - Assume that certain meta-data are shared across all elements of event_aligned_spike_dict: bin_width, num_time_bin, time_before_align, time_after_align
    """

    # create list of dataset which we will later merge
    dataset_list = list()

    num_unique_mice_exp = len(event_aligned_spike_dicts['binned_spikes'])

    for exp_index in np.arange(num_unique_mice_exp):
        exp_binned_spikes = event_aligned_spike_dicts['binned_spikes'][exp_index]
        rate_matrix = exp_binned_spikes['rate_matrix']
        num_cell = np.shape(rate_matrix)[0]
        num_bin = np.shape(rate_matrix)[1]
        num_trials = np.shape(rate_matrix)[2]
        time_before_align = event_aligned_spike_dicts['binned_spikes'][exp_index]['time_before_align']
        time_after_align = event_aligned_spike_dicts['binned_spikes'][exp_index]['time_after_align']
        num_time_bin = event_aligned_spike_dicts['binned_spikes'][exp_index]['num_time_bin']
        bin_times = np.linspace(-time_before_align, time_after_align, num_time_bin)

        if include_trial:
            trial_coord = event_aligned_spike_dicts['binned_spikes'][exp_index]['trial_number']
            # print(trial_coord)
            # print('Shape of trial coordinates:', np.shape(trial_coord))
        else:
            trial_coord = np.arange(0, num_trials)

        if combine_exp is False:
            rate_xr = xr.DataArray(rate_matrix,
                                   dims=('Cell', 'Time', 'Trial'),
                                   coords={'Cell': np.arange(0, num_cell),
                                           'Time': np.arange(0, num_bin),
                                           'Trial': trial_coord,
                                           'PeriEventTime': (('Time'), bin_times)}
                                   )
        elif (make_meta_dict_dim is False) and (combine_exp is True):
            rate_xr = xr.DataArray(np.expand_dims(rate_matrix, axis=-1),
                                   dims=('Cell', 'Time', 'Trial', 'Exp'),
                                   coords={'Cell': np.arange(0, num_cell),
                                           'Time': np.arange(0, num_bin),
                                           'Trial': trial_coord,
                                           'Exp': [event_aligned_spike_dicts['exp_ref'][exp_index]],
                                           'PeriEventTime': (('Time'), bin_times)}
                                   )

        elif make_meta_dict_dim is True:
            num_dim_to_add = 2
            for n in np.arange(num_dim_to_add):
                rate_matrix = np.expand_dims(rate_matrix, axis=-1)
            rate_xr = xr.DataArray(rate_matrix,
                                   dims=('Cell', 'Time', 'Trial', 'Exp', 'TrialTypeRef'),
                                   coords={'Cell': np.arange(0, num_cell),
                                           'Time': np.arange(0, num_bin),
                                           'Trial': trial_coord,
                                           'Exp': [event_aligned_spike_dicts['exp_ref'][exp_index]],
                                           'TrialTypeRef': [meta_dict['trial_type_ref']],
                                           # 'Aud': [meta_dict['aud']],
                                           # 'Vis': [meta_dict['vis']],
                                           'PeriEventTime': (('Time'), bin_times)}
                                   )

        aligned_ds = rate_xr.to_dataset(name='firing_rate')

        # add new variabels to the dataset
        if fields_to_variables is not None:
            for field in fields_to_variables:
                aligned_ds[field] = xr.DataArray(np.expand_dims(event_aligned_spike_dicts[field][exp_index], axis=1),
                                                 dims=['Trial', 'Exp'], coords=
                                                 {'Trial': trial_coord,
                                                  'Exp': [event_aligned_spike_dicts['exp_ref'][exp_index]]})

        # include cell location as a coordinate for the cell dimension
        if include_cell_loc:
            cell_loc_coord = event_aligned_spike_dicts['binned_spikes'][exp_index]['cell_loc'].values
            aligned_ds = aligned_ds.assign_coords(CellLoc=('Cell', cell_loc_coord))

        if include_cell_hemisphere:
            cell_hem_coord = event_aligned_spike_dicts['binned_spikes'][exp_index]['cell_hem'].values
            aligned_ds = aligned_ds.assign_coords(CellHem=('Cell', cell_hem_coord))

        # extract some meta-data
        aligned_ds.attrs['bin_width'] = event_aligned_spike_dicts['binned_spikes'
        ][exp_index]['bin_width']  # seconds
        aligned_ds.attrs['num_time_bin'] = event_aligned_spike_dicts['binned_spikes'
        ][exp_index]['num_time_bin']

        if combine_exp is False:
            if 'cluAndPen' in meta_fields:
                aligned_ds.attrs['clu_and_pen'] = event_aligned_spike_dicts['binned_spikes'
                ][exp_index]['cluAndPen']
            if 'cell_ref' in meta_fields:
                aligned_ds.attrs['cell_ref'] = event_aligned_spike_dicts['binned_spikes'
                ][exp_index]['cell_ref']

                aligned_ds.attrs['exp_ref'] = event_aligned_spike_dicts['exp_ref'][exp_index]

            if aligned_event is not None:
                aligned_ds.attrs['aligned_event'] = aligned_event

        dataset_list.append(aligned_ds)

    if combine_exp is True:
        # WARNING: the output using this option is no longer a list, it is dataset
        dataset_list = xr.concat(dataset_list, dim='Exp')
        # if 'minimal', only variables with the specified dimension will be concatenated

    if meta_dict is not None:
        for key, val in meta_dict.items():
            dataset_list.attrs[key] = val

    # TODO: include event times as a variable along the time and trial dimension

    return dataset_list


def align_and_bin_spikes(ephys_behaviour_df, spike_df, neuron_df=None, event_name='movementTimes',
                         time_before_align=0.3,
                         time_after_align=0.3, num_time_bin=50, method='one_hist', include_spikeless_cells=False,
                         save_path=None, cell_index='cluAndPen', include_trial=False, include_cell_loc=False,
                         include_cell_hem=False):
    """
    Bin and align spikes via the following method:
    1. For each trial, we take the movement time
    2. Using this, we create time bins
    3. For each neuron (cluster) we bin the spike times
    4. This gives a a matrix of shape (num_cluster, num_bins, num_trials)
    --------
    Arguments
    ephys_behaviour_df : (pandas dataframe)
        dataframe which contains trial by trial information
    target_cell_id    : unique identifies of neurons, this only needs to be used if you want to include cell without spikes
    time_before_align : (float)
        time before event of interest to align (eg. movement, stimulus onset) (seconds)
    time_after_align  : (float)
        time after event of interest to align (seconds)
    num_time_bin      : (int)
        total number of time bins
    method            : (str)
        processing options; affects processing time and has different levels of flexibility.
    include_spike_less_cells : (bool)
        whether to include cells without spikes in the spike matrix (they will be zero vectors)
    event_name : (str)
        event to align the spikes to, has to be a column in the ephys_behaviour_df
    include_trial : (bool)
        whether to include the trial index/number obtained from the ephys_behaviour_df
        this can later be used by xarray so that trial number is preserved (so that xarray can be
        cross-referenced with behaviour_df info, such as the response made on each trial)
    include_cell_loc : (bool)
        whether to include brain area of each cell.
    include_cell_hem : (booL)
        whether to include brain hesmisphere of each cell.
    ------------
    Output
    spike_matrix      : (numpy ndarray)
        numpy array of shape: (num_cluster, num_bins, num_trials)
    --------
    TODO: The loop implementation seems inefficient to me... perhaps there is something to
    take advantage of the spike time format.
    """

    # get times of event of interest (eg. movement)
    event_times = ephys_behaviour_df[event_name]
    # NOTE: currently assumes there are no NA (ie. the event occured in all trials)
    if include_trial:
        trial_number = ephys_behaviour_df.index.values

    # Get the unique cells

    if cell_index == 'cluAndPen':
        if 'cluAndPen' not in spike_df.columns:
            print('Warning cluster and pen is not a column, doing this is very inefficient.')
            spike_df['cluAndPen'] = list(zip(spike_df['cluNum'], spike_df['penRef']))

        cell_refs = np.unique(spike_df['cluAndPen'])
    elif cell_index == 'cellId':
        if include_spikeless_cells:
            cell_refs = np.unique(neuron_df['cellId'])
        else:
            cell_refs = np.unique(spike_df['cellId'])
    else:
        print('Warning: no valid cell_index specified')

    # TODO: need to loop through the penetrations, because the array only handles cluster numbers

    # Pre-allocate 3D matrix (rank 3 tensor)
    binned_spike_matrix = np.zeros(shape=(len(cell_refs), num_time_bin,
                                          len(event_times)))

    total_time = time_before_align + time_after_align
    bin_width = total_time / (num_time_bin)

    if method == 'one_hist':

        start_time_vec = event_times - time_before_align
        end_time_vec = event_times + time_after_align
        time_bin = np.linspace(start_time_vec, end_time_vec, num_time_bin + 1).T  # +1 to ensure num_time_bin
        time_bin_vec = time_bin.flatten()
        # make sure it is monotically increasing
        assert min(np.diff(time_bin_vec)) > 0, print(min(np.diff(time_bin_vec)))

        for n_ref, ref in enumerate(cell_refs):
            # note the 1 indexing
            all_event_bin = np.histogram(spike_df['spikeTime'].loc[
                                             spike_df[cell_index] == ref], time_bin_vec)[0]

            # remove the bins between bin vectors
            all_e_bin = np.delete(all_event_bin, np.arange(num_time_bin, len(all_event_bin),
                                                           num_time_bin + 1))  # note the 0-indexing,

            # assert the shape make sense
            assert np.shape(all_e_bin)[0] == (num_time_bin) * len(event_times)

            # reshape to a raster matrix for this particular cell
            all_e_bin_matrix = np.reshape(all_e_bin, [len(event_times),
                                                      num_time_bin])
            # make sure row represents trial and column represent bins
            assert np.sum(all_e_bin_matrix[0, :] - all_e_bin[0:num_time_bin]) == 0
            binned_spike_matrix[n_ref, :, :] = all_e_bin_matrix.T

            # make sure first trial (event) spike bins match
            assert np.sum(all_e_bin_matrix[0, :] - binned_spike_matrix[n_ref, :, 0]) == 0

    elif method == 'spike_time_df':
        print('Aligning spike whilst preserving spike time, no binning is performed.')

    else:
        # pure for-loop implementation, handles most cases but rather inefficient
        # this implementation also still assumes the dataframe is from the same experiment (otherwise, it will align event times in one experiment to cells from a different experiment)
        for n_event, event_time in enumerate(tqdm(event_times, leave=True, position=0)):
            # get cell binned firing rate before movement_times
            start_time = event_time - time_before_align
            end_time = event_time + time_after_align
            time_bin = np.linspace(start_time, end_time, num_time_bin)
            # ? Is this necessary: making histogram faster vs. computing a new dataframe
            target_time_spike_df = spike_df.loc[spike_df['spikeTime'].between(start_time, end_time)
            ]
            # Consider using apply along axis to use the histogram
            for n_ref, ref in enumerate(cell_refs):
                # note the 1 indexing
                binned_spike_matrix[n_ref, :, n_event] = np.histogram(
                    target_time_spike_df['spikeTime'].loc[
                        target_time_spike_df['cluAndPen'] == ref], time_bin)[0]

    binned_spike_dict = dict()
    binned_spike_dict['spike_matrix'] = binned_spike_matrix
    binned_spike_dict['rate_matrix'] = binned_spike_matrix / bin_width  # assume all equal size!
    binned_spike_dict['time_before_align'] = time_before_align
    binned_spike_dict['time_after_align'] = time_after_align
    binned_spike_dict['num_time_bin'] = num_time_bin
    binned_spike_dict['aligned_bins'] = np.linspace(-time_before_align, time_after_align, num_time_bin + 1)
    binned_spike_dict['bin_width'] = bin_width
    binned_spike_dict['cell_ref'] = cell_refs
    binned_spike_dict['event_times'] = event_times

    if include_trial:
        binned_spike_dict['trial_number'] = trial_number

    if include_cell_loc:
        cell_loc = neuron_df.set_index('cellId').loc[cell_refs]['cellLoc']
        binned_spike_dict['cell_loc'] = cell_loc

    if include_cell_hem:
        cell_hem = neuron_df.set_index('cellId').loc[cell_refs]['hemisphere']
        binned_spike_dict['cell_hem'] = cell_hem

    # TODO: also include experiment and mouse number

    if method == 'one_hist':
        binned_spike_dict['time_bin_vec'] = time_bin_vec
        start_end_vec = np.stack([start_time_vec, end_time_vec]).T.flatten()
        binned_spike_dict['start_end_vec'] = start_end_vec

        # ensure start and end are monotonically increasing
        assert min(np.diff(start_end_vec)) > 0

    if save_path is not None:
        with open(save_path, 'wb') as handle:
            pkl.dump(binned_spike_dict, handle)
    else:
        return binned_spike_dict


def align_and_bin_spikes_exp(ephys_behaviour_df, spike_df, cell_df, event_name='movementTimes', time_before_align=0.3,
                             time_after_align=0.3, num_time_bin=50,
                             method='one_hist', cell_index='cluAndPen',
                             extra_fields=None, save_path=None, include_trial=False,
                             include_cell_loc=False):
    """
    Align and bin spikes across multiple experiments.
    This basically loops through the align_and_bin_spikes function.
    Arguments
    -------------
    ephys_behaviour_df  : (pandas dataframe)
        dataframe with trial-by-trial information
    spike_df            :(pandas dataframe)
        dataframe with spike information
    cell_df             : (pandas dataframe)
        dataframe with neuron information
    event_name          : (str)
        the event to align the spikes to
    extra_fields        : (list of strings)
        list of strings referring to the fields of behaviour df you want to include
                         to the dictionary, eg. reaction time so that you can later use it to
                         sort alignment raster by reaction time
    cell_index        : (str)
        name of column in spike_df and cell_df which uniquely identifies the cell of each spike
        old format datasets (before October 2019) uses cluAndPen
        new format daatasets (November 2019 - Present) uses cellId
    include_cell_loc : (bool)
        whether to include the brain area of each cell.
    Some subtleties / defects:
     - spike_df is already subsetted by brain region (I think)
     - but cell_df is not subsetted by brain region yet...

    Output
    -----------
    exp_binned_dict (dict)
        dictionary with two keys
        'binned_spikes' : list of length corresponding to the number of experiment session in
        the input ephys_behaviour_df. Each element of this list is a dictionary with the
        alignment information corresponding to that experiment. Those dictionary have the
        following keys:
            - 'spike_matrix' : numpy ndarray with dimensions (num_bins, num_neurons)
            - 'rate_matrix' : numpy ndarray with dimensions (num_bins, num_neurons)
            - '
        'exp_ref': a list where each element is the expRef number associated with the 'binned_spikes'
    """

    assert cell_index in ['cluAndPen', 'cellId'], print('Invalid cell identification column.')

    exp_binned_dict = defaultdict(list)
    exp_ref_list = np.unique(ephys_behaviour_df['expRef'])

    # TODO: parallelize this across experiments.

    for n_exp, exp_ref in enumerate(exp_ref_list):
        exp_behave_df = ephys_behaviour_df.loc[ephys_behaviour_df['expRef'] == exp_ref]
        pen_to_get = np.unique(cell_df['penRef'].loc[cell_df['expRef'] == exp_ref])

        if cell_index == 'cluAndPen':
            exp_spike_df = spike_df.loc[spike_df['penRef'].isin(pen_to_get)]
        elif cell_index == 'cellId':
            # Why is it needed to get the matching penetration, why isn't cellId enough???
            target_cell_id = cell_df['cellId'].loc[cell_df['penRef'].isin(pen_to_get)]

            # target_cell_id is not empty
            exp_spike_df = spike_df.loc[spike_df['cellId'].isin(target_cell_id)]

        # This is only needed if we want to include spikeless cells
        exp_neuron_df = cell_df.loc[cell_df['expRef'] == exp_ref]

        binned_spike_dict = align_and_bin_spikes(exp_behave_df, exp_spike_df, neuron_df=exp_neuron_df,
                                                 event_name=event_name,
                                                 time_before_align=time_before_align,
                                                 time_after_align=time_after_align,
                                                 num_time_bin=num_time_bin, method=method,
                                                 cell_index=cell_index, include_spikeless_cells=True,
                                                 save_path=None, include_trial=include_trial,
                                                 include_cell_loc=include_cell_loc)

        exp_binned_dict['binned_spikes'].append(binned_spike_dict)
        exp_binned_dict['exp_ref'].append(exp_ref)

        if extra_fields is not None:  # consider just check if type is list or array
            for field_entry in extra_fields:
                exp_binned_dict[field_entry].append(exp_behave_df[field_entry].values)

    # make searching through experiment easier:
    exp_binned_dict['exp_ref'] = np.array(exp_binned_dict['exp_ref'])

    return exp_binned_dict


def combine_exp_aligned_spikes(exp_binned_dict, cell_id_axis=0, event_time_axis=2):
    spike_matrix_list = list()
    for n_exp in range(len(exp_binned_dict['exp_ref'])):
        spike_matrix_list.append(exp_binned_dict['binned_spikes'][n_exp]['spike_matrix'])

    spike_matrix = np.concatenate(spike_matrix_list, axis=[cell_id_axis, event_time_axis])

    return spike_matrix


def exp_binned_dict_to_df(exp_binned_dict):
    """
    Extracts the cluster number, penetration and corresponding index of cells into a pandas dataframe so individual neuron's identity is easier to find.
    """

    clu_and_pen_list = list()
    within_exp_index_list = list()
    exp_list = list()

    for binned_dict, exp_ref in zip(exp_binned_dict['binned_spikes'], exp_binned_dict['exp_ref']):
        clu_and_pen_list.append(binned_dict['cluAndPen'])
        within_exp_index = np.arange(0, len(binned_dict['cluAndPen']))
        within_exp_index_list.append(within_exp_index)
        exp_list.append(np.repeat([exp_ref], len(binned_dict['cluAndPen'])))

    clu_and_pen_list = np.concatenate(clu_and_pen_list)
    within_exp_index_list = np.concatenate(within_exp_index_list)
    exp_list = np.concatenate(exp_list)

    exp_binned_df = pd.DataFrame(
        data={'clu_and_pen': clu_and_pen_list, 'within_exp_index': within_exp_index_list, 'exp': exp_list})

    return exp_binned_df


def align_spike_times(neuron_idx, binned_spike_df, ephys_spike_df, binned_spike_exp_dict):
    """
    Align spike times for a single neuron to a particular event over multiple trials.

    Arguments
    ----------
    neuron_idx  :
    spike_df    :

    """

    neuron_exp = binned_spike_df.iloc[neuron_idx]['exp']
    neuron_clu_and_pen = binned_spike_df.iloc[neuron_idx]['clu_and_pen']
    neuron_spike_times = ephys_spike_df.loc[
        ephys_spike_df['cluAndPen'] == neuron_clu_and_pen]['spikeTime']

    neuron_spike_times = neuron_spike_times.to_frame()

    exp_index = np.where(binned_spike_exp_dict['exp_ref'] == neuron_exp)[0][0]
    event_times = binned_spike_exp_dict['binned_spikes'][exp_index]['event_times']
    start_end_vec = binned_spike_exp_dict['binned_spikes'][exp_index]['start_end_vec']

    neuron_spike_times['binned'] = pd.cut(neuron_spike_times['spikeTime'], start_end_vec)

    # TODO: look into breaking up this long line.
    group_spike_times = neuron_spike_times.groupby('binned')['spikeTime'].apply(list).reset_index(name='spike_times')

    group_spike_times = group_spike_times.iloc[::2]  # drop row corresponding to end-start
    group_spike_times = group_spike_times.reset_index()
    assert len(group_spike_times == len(event_times))

    return group_spike_times, event_times


def extract_neuron_by_trial_spike_matrix(binned_spike_exp_dict, binned_spike_df, neuron_idx,
                                         units='spikes'):
    """
    Subset the spike_matrix to get the by-trial spike matrix of a particular neuron
    Arguments
    --------
    binned_spike_matrix :
    binned_spike_df     :
    neuron_idx          :

    Output
    ------
    neuron_by_trial_spike_matrix : numpy array of shape (num_trial, num_time_bin)
    """

    neuron_exp = binned_spike_df.iloc[neuron_idx]['exp']
    neuron_within_exp_index = binned_spike_df['within_exp_index'].iloc[neuron_idx]

    exp_index = np.where(binned_spike_exp_dict['exp_ref'] == neuron_exp)[0][0]
    binned_spike_matrix = binned_spike_exp_dict['binned_spikes'][exp_index]['spike_matrix']
    neuron_by_trial_spike_matrix = np.squeeze(binned_spike_matrix[neuron_within_exp_index, :, :]).T

    bin_width = binned_spike_exp_dict['binned_spikes'][exp_index]['bin_width']

    if units == 'rate':
        neuron_by_trial_spike_matrix = neuron_by_trial_spike_matrix / bin_width

    peri_stimulus_time = binned_spike_exp_dict['binned_spikes'][exp_index]['aligned_bins']

    return neuron_by_trial_spike_matrix, peri_stimulus_time


def summarise_binned_spikes(binned_spike_array, trial_axis=0, bin_axis=1):
    """
    Computes some summary statistic from a given spike array.
    By default, assumes the the array has shape (numTrial, numTimeBin)

    Arguments
    --------
    binned_spike_array : numpy array of shape (num_time_bin, num_trial)
    trial_axis         : the axis representing trials, eg. trial_axis = 0 means each row is a trial
    bin_axis           : the axis representing time bins
    """

    trial_spike_mean = np.mean(binned_spike_array, trial_axis)
    trial_spike_std = np.std(binned_spike_array, trial_axis)
    trial_spike_sem = trial_spike_std / np.sqrt(np.shape(binned_spike_array)[trial_axis])

    spike_summary = {'trial_mean': trial_spike_mean,
                     'trial_sem': trial_spike_sem}

    return spike_summary


# TODO: write function to go through each mice, and go through each condition of interest
# and do the align spike and save things

def align_multiple_penetrations(ephys_behaviour_df, spike_df, event_name='movementTimes', time_before_align=0.3,
                                time_after_align=0.3, num_time_bin=50,
                                save_path=None):
    multiple_pen_binned_spike = list()

    for penetration in np.unique(spike_df['penRef']):
        pen_subset_spike_df = None

    return None


def smooth_spikes(spike_train, method='full_gaussian', sigma=2, window_width=50, custom_window=None,
                  custom_smooth_axis=None):
    """
    Smooths spike trains.
    Parameters
    -------------
    spike_train   : option (1): numpy array of shape (num_time_bin, ) to smooth
                    option (2): numpy array of shape (num_time_bin, num_trial)
    method        : method to perform the smoothing
                   'half_gaussian': causal half-gaussian filter
                   'full_gaussian': standard gaussian filter
    sigma : (int)
    window_width : (int)
    custom_window : if not None, then convolve the spike train with the provided window
    TODO: need to polish handling of edge cases for convolution.
    """

    if custom_smooth_axis is not None:
        smooth_axis = custom_smooth_axis
    elif spike_train.ndim == 2:
        smooth_axis = 1
    elif spike_train.ndim == 1:
        smooth_axis = -1
    else:
        raise ValueError('spike_train dimension is ambigious as to how to perform smoothing')

    # scipy ndimage cnovolution methods used here does not work with ints (round's down)
    if type(spike_train[0]) != np.float64:
        spike_train = spike_train.astype(float)

    if custom_window is None:

        if method == 'full_gaussian':

            smoothed_spike_train = spimage.filters.gaussian_filter1d(spike_train, sigma=sigma,
                                                                     axis=smooth_axis)
            # note that there is a slight offset between the np.convolve and the scipy ndimage implementation, mainly due to handling edge cases I think.

        elif method == 'half_gaussian':
            gaussian_window = spsignal.windows.gaussian(M=window_width, std=sigma)

            # note that the mid-point is included (ie. the peak of the gaussian)
            half_gaussian_window = gaussian_window.copy()
            half_gaussian_window[:int((window_width - 1) / 2)] = 0

            # normalise so it sums to 1
            half_gaussian_window = half_gaussian_window / np.sum(half_gaussian_window)

            smoothed_spike_train = spimage.filters.convolve1d(spike_train, weights=half_gaussian_window,
                                                              axis=smooth_axis)
            # TODO: see https://scipy-cookbook.readthedocs.io/ section on convolutino comparisons

        else:
            print('No valid smoothing method specified')

    else:
        # TODO: convolve custom kernel with the spike train.
        smoothed_spike_train = spsignal.convolve(spike_train, custom_window, mode='same')

    return smoothed_spike_train


def multi_condition_find_sig_response(exp_combined_ds_list, exp_cell_idx_list=None, sig_threshold=0.01, save_path=None,
                                      time_coord_name='PeriStimTime', aud_cond_attr_name='aud',
                                      vis_cond_attr_name='vis', cell_dim_name='expCell'):
    # TODO: make this dask dataframe so it can have meta-data
    """
    Performs a significance test to test for difference in firing rate before and after stimulus.
    Does this across trials, and divided into specific trial conditions: eg. auditory left, visual right
    TODO: add option to do audio on / off and visual on / off only
    :param exp_combined_ds_list:
    :param exp_cell_idx_list:
    :param sig_threshold:
    :param save_path:
    :return:
    """

    sig_df_dict = defaultdict(list)

    if (exp_cell_idx_list is None) or ('all' in exp_cell_idx_list):
        num_exp_cell = len(exp_combined_ds_list[0][cell_dim_name])
        # we use the first condition as to get the number of cells
        # the number of cells should be equal across all conditions,
        # otherwise, something is wrong... (TODO: assert this to explain potential later errors)
        exp_cell_idx_list = np.arange(0, num_exp_cell)

    for exp_cell_idx in exp_cell_idx_list:

        for exp_combined_ds in exp_combined_ds_list:
            # neuron_xr = exp_combined_ds
            # TODO: find how to generalise this
            if cell_dim_name == 'expCell':
                neuron_xr = exp_combined_ds.isel(expCell=exp_cell_idx)
            elif cell_dim_name == 'Cell':
                neuron_xr = exp_combined_ds.isel(Cell=exp_cell_idx)

            neuron_xr_pre_stimulus = neuron_xr.isel(Time=(neuron_xr[time_coord_name] < 0))
            neuron_xr_post_stimulus = neuron_xr.isel(Time=(neuron_xr[time_coord_name] > 0))

            neuron_xr_pre_stimulus_mean = neuron_xr_pre_stimulus.mean('Time')
            neuron_xr_post_stimulus_mean = neuron_xr_post_stimulus.mean('Time')

            pre_stim_mean_vals = neuron_xr_pre_stimulus_mean.to_array().values[0]
            post_stim_mean_vals = neuron_xr_post_stimulus_mean.to_array().values[0]

            # some NaN because the trial number is the max of the trial numbers
            # just need to ignore them in the paired t-test
            pre_stim_mean_vals = pre_stim_mean_vals[~np.isnan(pre_stim_mean_vals)]
            post_stim_mean_vals = post_stim_mean_vals[~np.isnan(post_stim_mean_vals)]

            if np.sum(post_stim_mean_vals - pre_stim_mean_vals) == 0:
                # print('Warning: neuron firing rate is identical pre and post-stimulus')
                test_stat = np.nan
                p_val = np.nan
            else:
                test_stat, p_val = sstats.wilcoxon(pre_stim_mean_vals,
                                                   post_stim_mean_vals)

            sig_df_dict['test_stat'].append(test_stat)
            sig_df_dict['p_val'].append(p_val)
            sig_df_dict['cell_idx'].append(exp_cell_idx)

            sig_df_dict['sig_response'].append(p_val < sig_threshold)

            sig_df_dict['aud_cond'].append(exp_combined_ds.attrs[aud_cond_attr_name])
            sig_df_dict['vis_cond'].append(exp_combined_ds.attrs[vis_cond_attr_name])

            # get the mean (over trials) activity for the mean (over time bins) activity before
            # and after stimulus. Then take the mean of that as well to get proxy of general cell activity.
            sig_df_dict['mean_pre_stim_mean'].append(np.mean(pre_stim_mean_vals))
            sig_df_dict['mean_post_stim_mean'].append(np.mean(post_stim_mean_vals))
            sig_df_dict['mean_peri_stim_mean'].append(np.mean([
                np.mean(pre_stim_mean_vals), np.mean(post_stim_mean_vals)]
            ))

    sig_df = pd.DataFrame.from_dict(sig_df_dict)

    if save_path is not None:
        with open(save_path, 'wb') as handle:
            pkl.dump(sig_df, handle)
    else:
        return sig_df


def rank_sig_df(sig_df):
    """
    Calculate some metric for ranking (for sorted plot orders)
    :param sig_df: pandas dataframe generate using multi_condition_find_sig_response
    :return:
    """

    min_p_val_sig_df = sig_df.groupby('cell_idx').agg('min')['p_val']
    num_sig_response_sig_df = sig_df.groupby('cell_idx').agg('sum')['sig_response']
    mean_firing_rate_sig_df = sig_df.groupby('cell_idx').agg('mean')['mean_peri_stim_mean']

    ranked_sig_df = pd.DataFrame({'cell_idx': min_p_val_sig_df.index,
                                  'min_p_val': min_p_val_sig_df,
                                  'num_sig_conditions': num_sig_response_sig_df,
                                  'mean_firing_rate': mean_firing_rate_sig_df})

    ranked_sig_df['min_p_val_rank'] = ranked_sig_df['min_p_val'].rank(ascending=True)
    ranked_sig_df['num_sig_conditions_rank'] = ranked_sig_df['num_sig_conditions'].rank(ascending=False)
    ranked_sig_df['mean_firing_rate'] = ranked_sig_df['mean_firing_rate'].rank(ascending=False)

    # overall rank : currently just a equal weight sum (mean) of the three ranks.
    ranked_sig_df['overall_rank'] = ranked_sig_df[['min_p_val_rank',
                                                   'num_sig_conditions_rank',
                                                   'mean_firing_rate']].mean(axis=1)

    return ranked_sig_df


def find_sig_response_neuron(binned_spike_exp_dict, method='diff_pos',
                             trial_axis=2, neuron_axis=0, bin_axis=1):
    """
    Find neurons to significantly respond to 'event'

    Arguments
    ---------
    binned_spike_exp_dict : list of dictionary containing binned spike matrices and binning information
                            spike_matrix should have shape (num_neuron, num_bins, num_trials) by default
                            otherwise, please specify the axis of the array
    method                : method to find significantly responstive neurons
                            'diff_pos'   - naive method: sum(after event) - sum(before event)
                            'diff_neg'   - sum(before_event) - sum(after_event)
                            'signed_rank'- Wilcoxon signed rank test (two-tailed)

    Output

    neuron_sig : array of shape (numNeuron, 2)
                 first column   - neuron indices sorted from most significant to least (0-indexing)
                 second columnn - signigicance metric (eg. difference in firing rate, or statistical test result) (the latter is a TODO)

    TODO: generalise code so that it does not assume symmetric window (will come down to adding information to binned_spike_exp_dict
    """

    if (method == 'diff_pos') or (method == 'diff_neg'):
        trial_average_list = list()
        for n_exp in range(len(binned_spike_exp_dict['exp_ref'])):
            spike_matrix = binned_spike_exp_dict['binned_spikes'][n_exp]['spike_matrix']
            trial_average_spike_matrix = np.mean(spike_matrix, axis=trial_axis)
            trial_average_list.append(trial_average_spike_matrix)

        all_exp_trial_average = np.concatenate(trial_average_list, axis=neuron_axis)

        # assumed to be the same across experiments
        num_time_bin = binned_spike_exp_dict['binned_spikes'][n_exp]['num_time_bin']
        assert num_time_bin % 2 == 0
        event_bin = int(num_time_bin / 2)  # bin when the event occured (assume symmetric for now)

        fire_rate_difference = (np.sum(all_exp_trial_average[:, (event_bin + 1):], axis=bin_axis) - \
                                np.sum(all_exp_trial_average[:, 0:event_bin], axis=bin_axis))

        neuron_idx_sorted = fire_rate_difference.argsort()

        if method == 'diff_pos':
            neuron_idx_sorted = neuron_idx_sorted[::-1]
            sig_metric_sorted = fire_rate_difference[neuron_idx_sorted[::-1]]
        elif method == 'diff_neg':
            sig_metric_sorted = fire_rate_difference[neuron_idx_sorted]

        neuron_sig = np.stack([neuron_idx_sorted, sig_metric_sorted], axis=1)

    elif (method == 'signed_rank'):

        all_exp_test_result = list()
        all_exp_change_sign = list()

        for n_exp in range(len(binned_spike_exp_dict['exp_ref'])):
            spike_matrix = binned_spike_exp_dict['binned_spikes'][n_exp]['spike_matrix']
            # again, the below code assumes the bin_axis is 1 (really need to make subsetting by a dimension more intuitive
            # assumed to be the same across experiments
            num_time_bin = binned_spike_exp_dict['binned_spikes'][n_exp]['num_time_bin']
            assert num_time_bin % 2 == 0
            event_bin = int(num_time_bin / 2)  # bin when the event occured (assume symmetric for now)

            post_stim_rate = np.mean(spike_matrix[:, (event_bin + 1):, :], axis=bin_axis)
            pre_stim_rate = np.mean(spike_matrix[:, 0:event_bin, :], axis=bin_axis)
            # should return ndarray of shape (num_neuron, num_trial)

            # apply statistical test along the neuron dimension (to test each neuron)
            # np.apply_along_axis(sstats.wilcoxon, axis=neuron_axis) # TODO: need to check that it is working
            # properly.

            # For loop implemenation (still working on getting numpy apply along axis to work properly
            # stat_result_list = list()
            pre_post_axis = 2
            pre_post_stim_rate = np.stack([post_stim_rate, pre_stim_rate], pre_post_axis)

            for neuron in np.arange(np.shape(pre_post_stim_rate)[neuron_axis]):
                neuron_pre_stim_rate = pre_post_stim_rate[neuron, :, 0]
                neuron_post_stim_rate = pre_post_stim_rate[neuron, :, 1]
                try:
                    test_stat, p_val = sstats.wilcoxon(neuron_pre_stim_rate, neuron_post_stim_rate)
                except:
                    test_stat = np.nan
                    p_val = np.nan

                all_exp_test_result.append(p_val)
                # stat_result_list.append(p_val)
                # stat_result = np.array(stat_result_list)

                # look at whether the difference was positive or negative
                change_sign = np.sign(np.mean(neuron_post_stim_rate) - np.mean(neuron_pre_stim_rate))
                all_exp_change_sign.append(change_sign)

            # all_exp_test_result.append(stat_result_list)

        # sort neuron by p value (lowest first)
        all_exp_test_result = np.array(all_exp_test_result)
        all_exp_change_sign = np.array(all_exp_change_sign)
        neuron_idx_sorted = all_exp_test_result.argsort()
        sig_metric_sorted = all_exp_test_result[neuron_idx_sorted]
        change_sign_sorted = all_exp_change_sign[neuron_idx_sorted]

        neuron_sig = pd.DataFrame({'neuron_idx': neuron_idx_sorted,
                                   'p_val': sig_metric_sorted,
                                   'sign': change_sign_sorted})

    return neuron_sig


def aud_vis_subset_trial_df(subset_ephys_behave_df, aud_conditions=[-60, 60, np.inf],
                            visual_conditions=[[-0.8, -0.4, -0.2, -0.1], 0.0, [0.1, 0.2, 0.4, 0.8]],
                            check_trials=False):
    """

    Parameters
    ----------
    subset_ephys_behave_df
    aud_conditions
    visual_conditions
    check_trials (bool)
        whether to check that the trial numbers are equal.
        meaning no trials are loss / overlapped in the process of splitting the input behaviour df
        according to audio and visual conditions.

    Returns
    -------

    """
    # generate all condtions above

    stimuli_subset_df_list = list()

    # TODO: include aud and visual conditions
    aud_condition_list = list()
    vis_condition_list = list()

    tot_trials = 0

    for aud_c, visual_c in list(itertools.product(aud_conditions,
                                                  visual_conditions)):

        if (np.sum(aud_c) > 0) & (np.sum(aud_c) != np.inf):
            aud_condition_list.append('right')
        elif np.sum(aud_c) < 0:
            aud_condition_list.append('left')
        elif np.sum(aud_c) == np.inf:
            aud_condition_list.append('off')
        else:
            aud_condition_list.append('center')

        if np.sum(visual_c) > 0:
            vis_condition_list.append('right')
        elif np.sum(visual_c) < 0:
            vis_condition_list.append('left')
        else:
            vis_condition_list.append('off')

        if type(aud_c) is not list:
            aud_c = [aud_c]

        if type(visual_c) is not list:
            visual_c = [visual_c]

        subset_df = subset_ephys_behave_df.loc[
            (subset_ephys_behave_df['audDiff'].isin(aud_c)) &
            (subset_ephys_behave_df['visDiff'].isin(visual_c))
            ]

        # print(aud_c)
        # print(visual_c)
        # print(len(subset_df))
        tot_trials = tot_trials + len(subset_df)
        stimuli_subset_df_list.append(subset_df)

    if check_trials:
        assert tot_trials == len(subset_ephys_behave_df)

    return stimuli_subset_df_list, aud_condition_list, vis_condition_list


def multi_condition_alignment(query_cell_loc, query_exp_ref, query_subject_ref,
                              spike_df, ephys_behave_df, neuron_df,
                              align_event_name='stimOnTime',
                              extra_fields=None,
                              time_before_align=0.3, time_after_align=0.3, num_time_bin=50,
                              include_response_made=False, include_trial=False,
                              aud_conditions=[-60, 60, np.inf],
                              visual_conditions=[[-0.8, -0.4, -0.2, 0.1], 0.0, [0.1, 0.2, 0.4, 0.8]]):
    """
    Align spikes to multiple audio-visaul condition pairs, then bin them.
    Parameters
    ----------------------
    query_cell_loc: (list of str)
         list of cell locations to get
    query_exp_ref : (list of int)
        list of experiment numbers to compute alignment
    query_subject_ref: (list of int)
        list of subject numbers to compute alignment
    align_event_name (str)
        name of event contained in ephys_behave_df to align neural activity to.
        Examples:
        'stimOnTime' : onset of stimulus

    :param extra_fields (list of str)
        list of extra fields you want to include in alignment data (on trial by trial basis)
        this is also used to convert into data variables in the alignment xarray
    :param include_response_made (bool)
        whether to include the response made during each trial by the subject (only applies to active condition)
        the current coding scheme is: 0 = no-go, 1 = left, 2 = right.
    :param include_trial (bool)
        whether to preserve the original trial number/index from behaviour df in the trial
        dimension. Default is False.
    :param aud_conditions: (list)
        list of auditory conditions to align to. This function aligns all combination of auditory and visual conditions.
    :param vis_conditions: (list or list of list)
        list of visual conditions to align to
        if list contains a list, then those list are grouped into one condition
        eg. [[1, 2, 3], 4, [5, 6, 7]] means there are 3 group of conditions (1) [1, 2, 3], (2) [4], (3) [5, 6 7]
    ----------------------
    :return:

    exp_combined_ds_list : (list of xarray dataset)
        list of xarray dataset, each element within the list of a dataset corresponding to one of the
        audio and visual conditions
    subset_ephys_behave_df : (pandas dataframe)
        subtted behaviour dataframe based on the subject and experient number
    -----------------------
    """

    if type(query_subject_ref) != list:
        query_subject_ref = [query_subject_ref]
    if type(query_cell_loc) != list:
        query_cell_loc = [query_cell_loc]
    if type(query_exp_ref) != list:
        query_exp_ref = [query_exp_ref]

    subset_spike_df = pr_ephys.subset_spikes(spike_df,
                                             ephys_cell_df=neuron_df,
                                             query_cell_loc=query_cell_loc,
                                             query_subject_ref=query_subject_ref,
                                             query_exp_ref=query_exp_ref,
                                             method='isinCellId')

    subset_neuron_df = pr_ephys.subset_spikes(neuron_df,
                                              ephys_cell_df=neuron_df,
                                              query_cell_loc=query_cell_loc,
                                              query_subject_ref=query_subject_ref,
                                              query_exp_ref=query_exp_ref,
                                              method='isinCellId')

    subset_ephys_behave_df = pr_ephys.subset_behaviour_df(ephys_behave_df,
                                                          query_subject_ref=query_subject_ref,
                                                          query_exp_ref=query_exp_ref)

    stimuli_subset_df_list, aud_condition_list, vis_condition_list = aud_vis_subset_trial_df(
        subset_ephys_behave_df,
        aud_conditions=aud_conditions,
        visual_conditions=visual_conditions)

    exp_combined_ds_list = list()
    trial_type_ref = np.arange(len(stimuli_subset_df_list))

    for n_condition, condition_df in enumerate(stimuli_subset_df_list):

        if len(condition_df) == 0:
            print('Warning: condition_df is empty.')
            continue

        event_aligned_spike_dicts = align_and_bin_spikes_exp(
            condition_df, subset_spike_df, subset_neuron_df,  # neuron_df, 2019-11-26 to allow for cells without spikes
            event_name=align_event_name,
            time_before_align=time_before_align, time_after_align=time_after_align,
            num_time_bin=num_time_bin,
            save_path=None, method='one_hist', cell_index='cellId', extra_fields=extra_fields,
            include_trial=include_trial)

        meta_dict = {'aud': aud_condition_list[n_condition],
                     'vis': vis_condition_list[n_condition],
                     'trial_type_ref': trial_type_ref[n_condition]}

        exp_combined_ds = aligned_dicts_to_xdataset(
            event_aligned_spike_dicts=event_aligned_spike_dicts,
            aligned_event=align_event_name, combine_exp=True, meta_dict=meta_dict,
            make_meta_dict_dim=False, fields_to_variables=extra_fields,
            include_trial=include_trial)

        # stack experiment and cell dimension into a single dimension
        exp_combined_ds = exp_combined_ds.stack(expCell=('Exp', 'Cell'))

        exp_combined_ds_list.append(exp_combined_ds)

    return exp_combined_ds_list, subset_ephys_behave_df


def single_conditon_alignment_ds_to_multialignment(alignment_ds, aud_conditions=[-60, 60, np.inf],
                                                   visual_conditions=[[-0.8, -0.4, -0.2, 0.1], 0.0,
                                                                      [0.1, 0.2, 0.4, 0.8]]):
    """
    Splits a single alignment_ds (eg. aligned to stimulus onset) to a list of xarray subsetted
    based on stimulus connditions (eg. visLeftAudLeft, visLeftAudRight ...).
    Output type is the same as first output of multi_condition_alignment, but operates on
    already aligned data instead of doing the alingment directly from spikes.
    Parameters
    -----------
    alignment_ds : (xarray dataset)
        dataset of neural activity aligned to one event, eg. aligned to movement or stimulus onset
    :param aud_conditions : (list)
    :param visual_conditions:
    :return:
    """

    # convert conditions list to string, this is to make this compatible with
    # vizpikes.plot_grid_psth()
    aud_cond_dict = {-60: 'left',
                     60: 'right',
                     np.inf: 'off'}

    exp_combined_ds_list = list()

    for n_condition, (aud_cond, vis_cond) in enumerate(
            itertools.product(aud_conditions, visual_conditions)):

        subset_stim_cond_ds = alignment_ds.where(
            (alignment_ds['visDiff'].isin(vis_cond)) &
            (alignment_ds['audDiff'].isin(aud_cond)), drop=True
        )

        subset_stim_cond_ds.attrs['aud'] = aud_cond_dict[aud_cond]
        if np.sum(vis_cond) < 0:
            subset_stim_cond_ds.attrs['vis'] = 'left'
        elif np.sum(vis_cond) > 0:
            subset_stim_cond_ds.attrs['vis'] = 'right'
        else:
            subset_stim_cond_ds.attrs['vis'] = 'off'
        subset_stim_cond_ds.attrs['trial_type_ref'] = n_condition

        if len(subset_stim_cond_ds['Trial'].values) == 0:
            print('Condition do not have any trials, skipping')
        else:
            exp_combined_ds_list.append(subset_stim_cond_ds)

    return exp_combined_ds_list


def get_event_aligned_xarray(condition_df, spike_df, neuron_df, time_before_align=0.03, time_after_align=0.05):
    """
    Aligns spikes to a specific event, then turn the data into xarray form.
    Work in progress...

    Parameters
    -----------
    condition_df (pandas dataframe)
    spike_df (pandas dataframe)
        dataframe containing information associated with each spike.
    neuron_df (pandas dataframe)
        dataframe containing information associated with each neuron.

    Returns
    ----------
    exp_combined_ds (xarray dataset)
        xarray dataset with the aligned binned neural activity
    """

    event_aligned_spike_dicts = align_and_bin_spikes_exp(
        condition_df, subset_spike_df, subset_neuron_df,  # neuron_df, 2019-11-26 to allow for cells without spikes
        event_name=event_name,
        time_before_align=time_before_align, time_after_align=time_after_align,
        num_time_bin=num_time_bin,
        save_path=None, method='one_hist', cell_index='cellId', extra_fields=extra_fields)

    meta_dict = {'aud': aud_condition_list[n_condition],
                 'vis': vis_condition_list[n_condition],
                 'trial_type_ref': trial_type_ref[n_condition]}

    exp_combined_ds = aligned_dicts_to_xdataset(
        event_aligned_spike_dicts=event_aligned_spike_dicts,
        aligned_event=event_name, combine_exp=True, meta_dict=meta_dict,
        make_meta_dict_dim=False, fields_to_variables=extra_fields)

    # stack experiment and cell dimension into a single dimension
    exp_combined_ds = exp_combined_ds.stack(expCell=('Exp', 'Cell'))

    return exp_combined_ds


def batch_multi_condition_alignment(query_cell_loc_list, query_exp_ref_list,
                                    query_subject_ref_list, ephys_behave_df,
                                    neuron_df, spike_df, alignment_folder=None,
                                    pool_subject=False,
                                    pool_exp=True, pool_cell_loc=False,
                                    file_ext='.nc', include_sig_test=False,
                                    save_subset_behaviour=True,
                                    align_event_name='stimOnTime',
                                    extra_fields=None,
                                    time_before_align=0.3, time_after_align=0.3, num_time_bin=50,
                                    include_response_made=False, include_trial=False,
                                    aud_conditions=[-60, 60, np.inf],
                                    visual_conditions=[[-0.8, -0.4, -0.2, 0.1], 0.0, [0.1, 0.2, 0.4, 0.8]]
                                    ):
    """
    Aligns neuron firing rate to stimulus in all mice / brain area.
    :param query_cell_loc_list:
    :param query_exp_ref_list:
    :param query_subject_ref_list:
    :param pool_subject: whether to save separate files or just combine them on subject level.
    :param pool_exp:
    :param pool_cell_loc:
    :param save_subset_behaviour: (bool)
        whether to save ths subetted beahaviour df (for later plotting)
    :param include_response_made: (bool)
        whether to include the response made during each trial by the subject (only applies to active condition)
        the current coding scheme is: 0 = no-go, 1 = left, 2 = right.
    :param

    Returns
    --------

    """

    # TODO: think about whether it makes sense to include significance

    if not os.path.exists(alignment_folder):
        os.mkdir(alignment_folder)

    # TODO: save the list of ds to a specific folder by subject / brain area

    if 'all' in query_subject_ref_list and (not pool_subject):
        query_subject_ref_list = np.unique(ephys_behave_df['subjectRef'])

    for query_subject_ref in query_subject_ref_list:

        if not pool_subject:
            subject_folder = 'subject-' + str(query_subject_ref)
            subject_level_folder = os.path.join(alignment_folder, subject_folder)

            if not os.path.exists(subject_level_folder):
                os.mkdir(subject_level_folder)

        for query_cell_loc in query_cell_loc_list:

            if not pool_cell_loc:
                cell_loc_level_folder = os.path.join(alignment_folder, subject_folder,
                                                     query_cell_loc)
                if not os.path.exists(cell_loc_level_folder):
                    os.mkdir(cell_loc_level_folder)

            if pool_exp:
                exp_level_folder = os.path.join(alignment_folder, subject_folder,
                                                query_cell_loc, 'all-exp')

                if not os.path.exists(exp_level_folder):
                    os.mkdir(exp_level_folder)
            elif 'all' in query_exp_ref_list:
                query_exp_ref_list = np.unique(ephys_behave_df['expRef'])

            for query_exp_ref in query_exp_ref_list:

                # TODO: query exp ref currently does not really work...
                # since it assumes each subject has the same expRef;

                if not pool_exp:
                    exp_level_folder = os.path.join(alignment_folder, subject_folder,
                                                    query_cell_loc, 'exp-' + str(query_exp_ref))

                exp_combined_ds_list, subset_ephys_behave_df = multi_condition_alignment(query_cell_loc=query_cell_loc,
                                                                                         query_exp_ref=query_exp_ref,
                                                                                         query_subject_ref=query_subject_ref,
                                                                                         spike_df=spike_df,
                                                                                         ephys_behave_df=ephys_behave_df,
                                                                                         neuron_df=neuron_df,
                                                                                         align_event_name=align_event_name,
                                                                                         extra_fields=extra_fields,
                                                                                         time_before_align=time_before_align,
                                                                                         time_after_align=time_after_align,
                                                                                         num_time_bin=num_time_bin,
                                                                                         include_response_made=include_response_made,
                                                                                         include_trial=include_trial,
                                                                                         aud_conditions=aud_conditions,
                                                                                         visual_conditions=visual_conditions)
                if len(exp_combined_ds_list) == 0:
                    print('No matching exp: skipping')
                    continue
                else:
                    if not os.path.exists(exp_level_folder):
                        os.mkdir(exp_level_folder)

                # Save Significance Test of alignment
                if include_sig_test:
                    sig_test_file_name = 'multi-condition-sig.pkl'
                    multi_condition_find_sig_response(exp_combined_ds_list, exp_cell_idx_list=None,
                                                      sig_threshold=0.01,
                                                      save_path=os.path.join(exp_level_folder, sig_test_file_name))

                # Save Alignment Data
                file_name = 'multi-condition-alignment' + file_ext

                with open(os.path.join(exp_level_folder, file_name), 'wb') as handle:
                    pkl.dump(exp_combined_ds_list, handle)

                # Save subsetted trial data associated with alignment
                if save_subset_behaviour:
                    subset_behaviour_file_name = 'subset-behaviour-df.pkl'
                    with open(os.path.join(exp_level_folder, subset_behaviour_file_name), 'wb') as handle:
                        pkl.dump(subset_ephys_behave_df, handle)


def batch_all_condition_alignment(query_cell_loc_list, query_exp_ref_list,
                                  query_subject_ref_list, ephys_behave_df,
                                  neuron_df, spike_df, alignment_folder=None,
                                  time_before_align=0.3, time_after_align=0.3, num_time_bin=50,
                                  pool_subject=False, aligned_event='stimOnTime', cellIndex='cellId',
                                  pool_exp=False, pool_cell_loc=False, fields_to_variables=None,
                                  file_ext='.nc', save_in_top_folder=False, include_cell_loc=False,
                                  remove_no_go=True, remove_invalid=True, remove_no_stim=False,
                                  include_trial=False, remove_no_go_method='responseMade',
                                  reaction_time_variable_name='choiceInitTimeRelStim',
                                  custom_neuron_idx=None):
    """
    Parameters
    ----------
    query_cell_loc_list: (list of str)
        list of brain areas you want to align
        special keyword: 'all', which either aligns cell in all regions at once (default), or do each brain region
        individually (if pool_cell_loc set to False)
    query_exp_ref_list: (list of int/str)
        list of experiment you want to align
    query_subject_ref_list: (list of int/str)
        list of subjects you want to align
    ephys_behave_df: (pandas dataframe)
        dataframe containing trial by trial information
    neuron_df (pandas dataframe)
        dataframe containing information about each neuron
    spike_df: (pandas dataframe)
        dataframe containing information about each spike
    alignment_folder: (str)
        path to the folder to save this alignment data
    time_before_align: (float)
        time in seconds to align before the event of interest
    time_after_align: (float)
        time in seconds to align after the event of interest
    :param num_time_bin:
    :param pool_subject:
    :param aligned_event:
    :param cellIndex:
    :param pool_exp:
    :param pool_cell_loc:
    :param fields_to_variables:
    :param file_ext:
    :param save_in_top_folder:
    :param include_cell_loc: (bool)
        whether to include cell location.
    remove_no_go (bool)
        whether to remove no-go trials.
    remove_invalid (bool)
        wehther to remove invalid trials
    remove_no_stim (bool)
        whether to remove trials with no (audio or visual) stimulus
        this happens rarely, but usually means they are trials where the animal gets a reward
    include_trial (bool)
        whether to include the original trial index value from behaviour_df (unique to each subject and experiment)
        otherwise, the trial index in the alignment dataset will be re-indexed from 0 to the number of trials
    custom_neuron_idx : (list of int)
        list of cell indices to subset, normally used if you want to get alignment ds
        for a subsample of neurons at high time resolution
    :return:
    """

    if not os.path.exists(alignment_folder):
        os.mkdir(alignment_folder)

    # TODO: save the list of ds to a specific folder by subject / brain area
    if 'all' in query_subject_ref_list and (not pool_subject):
        query_subject_ref_list = np.unique(ephys_behave_df['subjectRef'])

    for query_subject_ref in query_subject_ref_list:

        if type(query_subject_ref) == str:
            query_subject_ref = int(query_subject_ref)

        if not pool_subject:
            subject_folder = 'subject-' + str(query_subject_ref)
            subject_level_folder = os.path.join(alignment_folder, subject_folder)

            if not os.path.exists(subject_level_folder):
                os.mkdir(subject_level_folder)

        if pool_exp:
            exp_level_folder = os.path.join(alignment_folder, subject_folder, 'all-exp')

            if not os.path.exists(exp_level_folder):
                os.mkdir(exp_level_folder)

        elif 'all' in query_exp_ref_list and (not pool_exp):
            # get all exp for this particular subject.
            subject_specific_ephys_behave_df = ephys_behave_df.loc[
                ephys_behave_df['subjectRef'] == query_subject_ref]
            query_exp_ref_list_inloop = np.unique(subject_specific_ephys_behave_df['expRef'])
            # query_exp_ref_list = np.unique(ephys_behave_df['expRef'])
        elif not pool_exp:
            # TODO: this still has to somehow be specific to the subject...
            query_exp_ref_list_inloop = query_exp_ref_list

        for query_exp_ref in query_exp_ref_list_inloop:

            if type(query_exp_ref) == str:
                query_exp_ref = int(query_exp_ref)

            if not pool_exp:
                exp_level_folder = os.path.join(alignment_folder, subject_folder, 'exp-' + str(query_exp_ref))
                if not os.path.exists(exp_level_folder):
                    os.mkdir(exp_level_folder)

            if 'all' in query_cell_loc_list and (not pool_cell_loc):
                neuron_df_specific_exp = neuron_df.loc[neuron_df['expRef'] == query_exp_ref]
                query_cell_loc_list_inloop = np.unique(neuron_df_specific_exp['cellLoc'])
                # we don't want to modify the original query list, otherwise it will persist for the other subjects.
            elif not pool_cell_loc:
                query_cell_loc_list_inloop = query_cell_loc_list

            for query_cell_loc in query_cell_loc_list_inloop:

                if not pool_cell_loc:
                    cell_loc_level_folder = os.path.join(exp_level_folder,
                                                         query_cell_loc)
                    if not os.path.exists(cell_loc_level_folder):
                        os.mkdir(cell_loc_level_folder)

                # get subset behaviour df and spike df
                spike_df_subsetted = pr_ephys.subset_spikes(spike_df,
                                                            neuron_df,
                                                            query_cell_loc=query_cell_loc,
                                                            query_subject_ref=query_subject_ref,
                                                            query_exp_ref=query_exp_ref,
                                                            method='isinCellId')

                behaviour_df_subsetted = pr_ephys.subset_behaviour_df(
                    behaviour_df=ephys_behave_df,
                    query_subject_ref=query_subject_ref,
                    query_exp_ref=query_exp_ref,
                    remove_invalid=remove_invalid, remove_no_go=remove_no_go,
                    remove_no_stim=remove_no_stim,
                    remove_no_go_method=remove_no_go_method,
                    reaction_time_variable_name=reaction_time_variable_name)

                neuron_df_subsetted = pr_ephys.subset_neuron_df(neuron_df=neuron_df,
                                                                query_subject_ref=query_subject_ref,
                                                                query_exp_ref=query_exp_ref,
                                                                query_cell_loc=query_cell_loc,
                                                                method='isinCellId')

                if custom_neuron_idx is not None:
                    if len(neuron_df_subsetted) > 0:
                        neuron_df_subsetted = neuron_df_subsetted.reset_index()
                        neuron_df_subsetted = neuron_df_subsetted.iloc[custom_neuron_idx]

                if len(behaviour_df_subsetted) == 0:
                    print('No trials remained after subsetting, skipping ...')
                    continue

                # this is a special keyword to perform alignment to both stimulus and movement
                if aligned_event in ['stimSubtractionThenMovement', 'stimAndMovement']:

                    if len(spike_df_subsetted) == 0:
                        print('Warning: no cells found in subject %s experiment %s with query cell location: '
                              % (str(query_subject_ref), str(query_exp_ref)) + query_cell_loc)
                        continue

                    try:
                        binned_spike_ds = spike_df_to_xr(spike_df_subsetted, neuron_df=neuron_df_subsetted,
                                                         bin_width=0.02)
                        aligned_ds = ams.align_stim_and_movement(binned_spike_ds, behave_df=behaviour_df_subsetted,
                                                                 time_before_alignment=time_before_align,
                                                                 time_after_alignment=time_after_align)
                    except:
                        print('Alignment using first and last spike problematic, '
                              'switching to alignment using trial start and end')
                        binned_spike_ds = spike_df_to_xr(spike_df_subsetted, neuron_df=neuron_df_subsetted,
                                                         bin_width=0.02, behave_df=behaviour_df_subsetted,
                                                         event_name_to_set_bin='trialStartEnd'
                                                         )
                        aligned_ds = ams.align_stim_and_movement(binned_spike_ds, behave_df=behaviour_df_subsetted,
                                                                 time_before_alignment=time_before_align,
                                                                 time_after_alignment=time_after_align)

                    if aligned_event == 'stimAndMovement':
                        aligned_xarray = aligned_ds
                    elif aligned_event == 'stimSubtractionThenMovement':
                        mean_subtracted_ds = ams.aligned_stim_mean_subtraction(aligned_ds,
                                                                               pre_stim_time=0.1, post_stim_time=0.1,
                                                                               subset_stim_cond=None)
                        aligned_xarray = ams.realign_ds_to_movement(stim_and_movement_aligned_ds=mean_subtracted_ds,
                                                                    pre_movement_time=time_before_align,
                                                                    post_movement_time=time_after_align)

                    for field in fields_to_variables:
                        aligned_xarray = aligned_xarray.assign(
                            {field: ('Trial', behaviour_df_subsetted[field].values)})
                    if include_cell_loc:
                        aligned_xarray = aligned_xarray.assign_coords(
                            {'CellLoc': ('Cell', neuron_df_subsetted['cellLoc'])}
                        )

                else:
                    stim_aligned_spike_dicts = align_and_bin_spikes_exp(
                        ephys_behaviour_df=behaviour_df_subsetted,
                        spike_df=spike_df_subsetted, cell_df=neuron_df_subsetted,
                        event_name=aligned_event,
                        cell_index=cellIndex,
                        time_before_align=time_before_align,
                        num_time_bin=num_time_bin,
                        time_after_align=time_after_align,
                        extra_fields=fields_to_variables,
                        include_cell_loc=include_cell_loc,
                        include_trial=include_trial)

                    # convert to xarray format
                    aligned_xarray = aligned_dicts_to_xdataset(
                        event_aligned_spike_dicts=stim_aligned_spike_dicts,
                        aligned_event=aligned_event,
                        meta_fields=['cell_ref'], combine_exp=True,
                        meta_dict=None, make_meta_dict_dim=False,
                        fields_to_variables=fields_to_variables, include_trial=include_trial,
                        include_cell_loc=include_cell_loc)

                # save the data to alignment folder
                alignment_file_name = 'subject_' + str(query_subject_ref) + '_' + 'exp_' + \
                                      str(query_exp_ref) + '_' + query_cell_loc + \
                                      '_aligned_to_' + aligned_event + file_ext

                if save_in_top_folder:
                    save_folder = alignment_folder
                else:
                    save_folder = cell_loc_level_folder

                if file_ext == '.pkl':
                    with open(os.path.join(save_folder, alignment_file_name), 'wb') as handle:
                        pkl.dump(aligned_xarray, handle)
                elif file_ext == '.nc':
                    # pdb.set_trace()
                    aligned_xarray.to_netcdf(os.path.join(save_folder, alignment_file_name))
                else:
                    print('Warning: no valid file extension specified, not saving data.')


def find_peak(activity_array, method='max'):
    """
    Find peak of neural response
    :param activity_array:
    :param method:
    :return:
    """
    if method == 'max':
        peak_loc = np.nanargmax(activity_array)

    elif method == 'scipy':
        peak_loc = spsignal.find_peak(activity_array)

    return peak_loc


def find_peri_stimulus_activity_peak(alignment_ds, target_cell_idx, peak_per='neuron',
                                     peak_finding_method='max',
                                     cell_dim_name='expCell', time_coord_name='PeriStimTime',
                                     activity_name='firing_rate', cell_index_selection='isel'):
    """
    Find peak of neural response given xarray structure.
    Basically a thin wrapper around the core peak finding function: find_peak
    Currently uses a naive max method. (scipy.signal.find_peak can be unstable)
    Parameters
    ----------
    :param alignment_ds:
    :param target_cell_idx: (numpy array or int)
        if argument is int, then this does peak finding for one cell
        if argument is a numpy 1-d array, then this performs peak finding for all specified cells
    :param peak_per:
    Output
    ----------
    peak_df
    TODO: this currently assumes a single peak... may be better to allow for two peaks
    """

    if peak_per == 'neuron':
        # target_cell_idx = target_cond_df['cell_idx']

        if cell_index_selection == 'isel':
            all_cell_alignment = alignment_ds.isel({cell_dim_name: target_cell_idx})
        elif cell_index_selection == 'sel':
            all_cell_alignment = alignment_ds.sel({cell_dim_name: target_cell_idx})

        mean_activity = all_cell_alignment.mean(dim='Trial')[activity_name]

        mean_activity = mean_activity.transpose(cell_dim_name, 'Time')
        mean_activity_array = np.squeeze(mean_activity.values)

        if len(mean_activity[cell_dim_name]) > 1:
            if peak_finding_method == 'max':
                peak_loc = np.nanargmax(mean_activity_array, axis=1)  # Along Time dimension
        else:
            # only has time dimension if one cell
            if peak_finding_method == 'max':
                peak_loc = np.nanargmax(mean_activity_array)

        peak_time = all_cell_alignment[time_coord_name].values[peak_loc]

        peak_df = pd.DataFrame({'cell_idx': target_cell_idx,
                                'peak_time': peak_time})
    elif peak_per == 'trial':
        # for each neuron, for each trial, for the time of peak

        print('Work on finding peak times per trial')

    return peak_df


def multi_condition_find_peak(sig_response_df, multicondition_alignment_ds, aud_condition_list, vis_condition_list):
    """
    Applies find_peri_stimulus_activity_peak across all auditory and visual conditions.
    :param sig_response_df:
    :param aud_condition_list:
    :param vis_condition_list:
    :return:
    """

    peak_df_list = list()

    for aud_c, visual_c in list(itertools.product(aud_condition_list,
                                                  vis_condition_list)):

        # subset sig response
        target_cond_df = sig_response_df.loc[
            (sig_response_df['aud_cond'] == aud_c) &
            (sig_response_df['vis_cond'] == visual_c)
            ]

        # get the corresponding alignment ds

        aud_condition_list = [x.attrs['aud'] for x in multicondition_alignment_ds]
        vis_condition_list = [x.attrs['vis'] for x in multicondition_alignment_ds]

        aud_cond_indices = np.where(np.array(aud_condition_list) == aud_c)[0]
        vis_cond_indices = np.where(np.array(vis_condition_list) == visual_c)[0]

        target_idx_list = np.intersect1d(aud_cond_indices, vis_cond_indices)

        if len(target_idx_list) > 0:
            target_idx = target_idx_list[0]
        else:
            print('Warning: audio-visual condition pair not found, skipping.')
            print(aud_c + visual_c)
            continue

        target_alignment_ds = multicondition_alignment_ds[target_idx]

        # get neuron indices
        target_cell_idx = target_cond_df['cell_idx']

        # find peaks

        peak_df = find_peri_stimulus_activity_peak(
            alignment_ds=target_alignment_ds, target_cell_idx=target_cell_idx,
            peak_per='neuron')

        peak_df['aud_cond'] = np.repeat(aud_c, len(peak_df))
        peak_df['vis_cond'] = np.repeat(visual_c, len(peak_df))

        peak_df_list.append(peak_df)

    all_peak_df = pd.concat(peak_df_list)

    return all_peak_df


def subset_df(subject_num=1, exp_num=1, target_cell_loc=['MOs'],
              behave_df=None, neuron_df=None, spike_df=None):
    """
    Subsets 3 dataframes used for ephys data analysis based on
    subject number, experiment number and target cell location.
    experiment number is actually already unique (subject nunber is redundant,
    but just as a double check)
    """

    if behave_df is not None:
        subset_behave_df = behave_df.loc[
            (behave_df['subjectRef'] == subject_num) &
            (behave_df['expRef'] == exp_num)
            ]
    else:
        subset_behave_df = None

    if neuron_df is not None:
        subset_neuron_df = neuron_df.loc[
            (neuron_df['expRef'] == exp_num) &
            (neuron_df['cellLoc'].isin(target_cell_loc))]
    else:
        subset_neuron_df = None

    if (neuron_df is not None) and (spike_df is not None):
        neuron_id = subset_neuron_df['cellId'].tolist()
        subset_spike_df = spike_df.loc[spike_df['cellId'].isin(neuron_id)]
    else:
        subset_spike_df = None

    return subset_behave_df, subset_neuron_df, subset_spike_df


# Population analysis code (perhaps will start a new module analyse_pop.py)


def get_target_condition_ds(multi_condition_ds, aud='left', vis='right', response=None,
                            response_dict={'left': 1, 'right': 2}):
    """
    Subsets alignment ds to get a particular stimulus condition.
    Parameters
    ----------
    multi_condition_ds: (list or xarray dataset)
    aud : (str, float or int)
        auditory condition to get
    :param vis:
    :param response:
    :param response_dict:
    :return:
    """
    found_target = False

    if type(multi_condition_ds) is list:

        for ds in multi_condition_ds:
            if (ds.attrs['aud'] == aud) and (ds.attrs['vis'] == vis):
                target_ds = ds
                found_target = True

        if found_target and (response is not None):
            target_ds = target_ds.where(target_ds['responseMade'] == response_dict[response], drop=True)

        if not found_target:
            print('Target dataset not found, returning None')
            target_ds = None
        # assert found_target, print('Target dataset not found')

    elif type(multi_condition_ds) is xr.core.dataset.Dataset:
        target_ds = multi_condition_ds
        if aud is not None:
            if aud == 'left':
                target_ds = target_ds.where(
                    target_ds['audDiff'] < 0, drop=True
                )
            elif aud == 'right':
                target_ds = target_ds.where(
                    target_ds['audDiff'] > 0, drop=True
                )
            elif (aud == 'center') or (aud == 0):
                target_ds = target_ds.where(
                    target_ds['audDiff'] == 0, drop=True
                )
            elif (aud == 'off'):
                target_ds = target_ds.where(
                    ~np.isfinite(target_ds['audDiff']), drop=True
                )
            elif type(aud) is float or type(aud) is int:
                target_ds = target_ds.where(
                    target_ds['audDiff'] == aud, drop=True
                )
        if vis is not None:
            if type(vis) is float:
                target_ds = target_ds.where(
                    target_ds['visDiff'] == vis, drop=True
                )
            elif vis == 'left':
                target_ds = target_ds.where(
                    target_ds['visDiff'] < 0, drop=True
                )
            elif vis == 'right':
                target_ds = target_ds.where(
                    target_ds['visDiff'] > 0, drop=True
                )
            elif vis == 'off':
                target_ds = target_ds.where(
                    target_ds['visDiff'] == 0, drop=True
                )
        if response is not None:
            target_ds = target_ds.where(
                target_ds['responseMade'] == response_dict[response], drop=True
            )
    else:
        Warning('Invalid datatype as input')

    return target_ds


def get_pop_vector_ds(input_ds, mean_across_trial=True,
                      peri_stim_time_start=0,
                      peri_stim_time_end=None,
                      mean_across_time=True,
                      take_pre_post_diff=False):
    if mean_across_trial:
        mean_across_trial_ds = input_ds.mean(dim='Trial')

    if peri_stim_time_end is None:
        peri_stim_time_end = max(mean_across_trial_ds.PeriStimTime)

    time_selected_ds = mean_across_trial_ds.where(
        (mean_across_trial_ds.PeriStimTime >=
         peri_stim_time_start) &
        (mean_across_trial_ds.PeriStimTime <= peri_stim_time_end)
    )

    if mean_across_time:
        pop_vector_ds = time_selected_ds.mean(dim='Time')

    if take_pre_post_diff:
        pre_stim_ds = mean_across_trial_ds.where(
            mean_across_trial_ds.PeriStimTime < 0)

        pre_stim_pop_vector_ds = pre_stim_ds.mean(dim='Time')

        pop_vector_ds = pop_vector_ds - pre_stim_pop_vector_ds

    return pop_vector_ds


def make_condition_pair_ds(alignment_data, aud_dir='left', vis_dir='left',
                           take_pre_post_diff=True, activity_name='firing_rate',
                           peri_stim_time_start=0, peri_stim_time_end=None):
    """
    For a given auditory and visual condition (eg. visual left and auditory right),
    returns the sum of the two conditions with auditory and visual only, and the
    condition with auditory and visual together.
    :param alignment_data: (list of xarray datasets)
    :param aud_dir: (str)
        auditory condition to get
    :param vis_dir: (str)
        vis condition to get
    :param take_pre_post_diff: (bool)
        whether to subtract the post-stimulus activity mean with the pre-stimulus mean
    :param activity_name: (str)
        variable within the xarray dataset to get as a proxy of neural activity
    :param peri_stim_time_start: (float)
        time relative to the stimulus to start slicing  (eg. to take the mean of)
    :param peri_stim_time_end: (float or None)
        time relative to the stimulus to end slicing
    :return:
    """

    vis_dir_only_ds = get_target_condition_ds(alignment_data,
                                              aud='off', vis=vis_dir)
    aud_dir_only_ds = get_target_condition_ds(alignment_data,
                                              aud=aud_dir, vis='off')
    vis_dir_aud_dir_ds = get_target_condition_ds(alignment_data,
                                                 aud=aud_dir, vis=vis_dir)

    vis_dir_only_trial_time_mean = get_pop_vector_ds(
        input_ds=vis_dir_only_ds,
        mean_across_trial=True,
        peri_stim_time_start=peri_stim_time_start,
        peri_stim_time_end=peri_stim_time_end,
        mean_across_time=True,
        take_pre_post_diff=take_pre_post_diff)

    aud_dir_only_trial_time_mean = get_pop_vector_ds(
        input_ds=aud_dir_only_ds,
        mean_across_trial=True,
        peri_stim_time_start=peri_stim_time_start,
        peri_stim_time_end=peri_stim_time_end,
        mean_across_time=True,
        take_pre_post_diff=take_pre_post_diff)

    vis_dir_aud_dir_ds_trial_time_mean = get_pop_vector_ds(
        input_ds=vis_dir_aud_dir_ds,
        mean_across_trial=True,
        peri_stim_time_start=peri_stim_time_start,
        peri_stim_time_end=peri_stim_time_end,
        mean_across_time=True,
        take_pre_post_diff=take_pre_post_diff)

    two_single_condition_rate = vis_dir_only_trial_time_mean[activity_name] + \
                                aud_dir_only_trial_time_mean[activity_name]

    one_combined_condition_rate = vis_dir_aud_dir_ds_trial_time_mean[activity_name]

    return two_single_condition_rate, one_combined_condition_rate


def compute_similarity(vec_1, vec_2, method='dot'):
    """
    Computes the similarity of two vectors.
    """

    if method in ['cosine', 'scipy-cosine']:
        if np.count_nonzero(vec_1) == 0:
            print('Warning: vector 1 is a zero vector, some similarity methods will return NaNs')
        if np.count_nonzero(vec_2) == 0:
            print('Warning: vector 2 is a zero vector, some similarity methods will return NaNs')

    supported_method_list = ['dot', 'cosine', 'scipy-cosine', 'mean-subtracted-cosine', 'pearson']
    assert method in supported_method_list, print('Unsupported similarity measure method.')

    if method == 'dot':
        similarity_score = np.dot(vec_1, vec_2)
    elif method == 'cosine':
        similarity_score = 1 - (np.dot(vec_1, vec_2) / (np.sqrt(np.dot(vec_1, vec_1)) * np.sqrt(np.dot(vec_2, vec_2))))
    elif method == 'scipy-cosine':
        similarity_score = sspatial.distance.cosine(vec_1, vec_2)
    elif (method == 'mean-subtracted-cosine') or (method == 'pearson'):
        # this is basically Pearson correlation I think
        # https://www.researchgate.net/post/Can_someone_differentiate_between_Cosine_Adjusted_cosine_and_Pearson_correlation_similarity_measuring_techniques
        # https://brenocon.com/blog/2012/03/cosine-similarity-pearson-correlation-and-ols-coefficients/
        similarity_score = sspatial.distance.cosine(vec_1 - np.mean(vec_1), vec_2 - np.mean(vec_2))

    else:
        similarity_score = None

    return similarity_score


def compute_similarity_trajectory(query_vec, target_vec, similarity_metric='dot',
                                  activity='firing_rate', method='bin-by-bin',
                                  vec_preprocessing_method=None):
    """
    Computes the moment by moment (ie. over time) simliarity of some population vector with another population vector.
    Parameters
    -----------
    query_vec : (xarray array)
        xarray object containing 'Time' dimension and some variable indicating the activity level
        of the neuron.
    target_vec: (xarray array)
        xarray object, if you wanto to use the 'bin-by-bin' method, then this also needs the 'Time' dimension
    similarity_metric : (str)
    """

    assert method in ['bin-by-bin', 'target-mean'], print('No valid method specified.')

    similarity_to_target_list = list()
    time_vec = target_vec.Time

    for time in time_vec:

        if method == 'bin-by-bin':
            # target varies at each time...
            similarity_to_target = compute_similarity(target_vec.sel(Time=time)[activity],
                                                      query_vec.sel(Time=time)[activity],
                                                      method=similarity_metric)
        elif method == 'target-mean':
            # TODO: check if Time dimension exists in query_vec, if so, mean across it
            similarity_to_target = compute_similarity(target_vec[activity],
                                                      query_vec.sel(Time=time)[activity])

        similarity_to_target_list.append(similarity_to_target)

    sim_to_target = np.array(similarity_to_target_list)

    return sim_to_target


def compute_multi_condition_vector_sim(alignment_data, response_left_ds_trial_mean,
                                       response_right_ds_trial_mean,
                                       aud_conditions=['left', 'off', 'center', 'right'],
                                       vis_conditions=['left', 'off', 'right'],
                                       response_conditions=['left', 'right'],
                                       vec_preprocessing_method=None,
                                       n_components=20,
                                       ):
    nested_dict = lambda: collections.defaultdict(nested_dict)  # creates an artbirarily nested dict
    aud_vis_res_dict = nested_dict()

    for aud_cond, vis_cond, res_cond in itertools.product(aud_conditions, vis_conditions, response_conditions):
        # print(aud_cond + vis_cond + res_cond)
        target_ds = get_target_condition_ds(ds_list=alignment_data,
                                            aud=aud_cond, vis=vis_cond, response=res_cond,
                                            response_dict={'left': 1, 'right': 2})

        if (target_ds is not None):
            if len(target_ds.Trial) > 0:
                trial_mean_ds = target_ds.mean(dim='Trial')
                # print(trial_mean_ds)
                # print(response_left_ds_trial_mean)

                if vec_preprocessing_method is not None:
                    trial_mean_ds_project_to_left, response_left_ds_trial_mean_reduced, dim_reduce_object \
                        = dim_reduce_xarray(
                        reduction_target=response_left_ds_trial_mean,
                        projection_target=trial_mean_ds, method=vec_preprocessing_method,
                        n_components=n_components,
                        output_format='xarray')

                    trial_mean_ds_project_to_right, response_right_ds_trial_mean_reduced, dim_reduce_object \
                        = dim_reduce_xarray(
                        reduction_target=response_right_ds_trial_mean,
                        projection_target=trial_mean_ds, method=vec_preprocessing_method,
                        n_components=n_components,
                        output_format='xarray')

                    sim_to_rL = compute_similarity_trajectory(query_vec=trial_mean_ds_project_to_left,
                                                              target_vec=response_left_ds_trial_mean_reduced,
                                                              similarity_metric='dot',
                                                              activity='firing_rate', method='bin-by-bin')
                    sim_to_rR = compute_similarity_trajectory(query_vec=trial_mean_ds_project_to_right,
                                                              target_vec=response_right_ds_trial_mean_reduced,
                                                              similarity_metric='dot',
                                                              activity='firing_rate', method='bin-by-bin')

                else:
                    sim_to_rL = compute_similarity_trajectory(query_vec=trial_mean_ds,
                                                              target_vec=response_left_ds_trial_mean,
                                                              similarity_metric='dot',
                                                              activity='firing_rate', method='bin-by-bin')
                    sim_to_rR = compute_similarity_trajectory(query_vec=trial_mean_ds,
                                                              target_vec=response_right_ds_trial_mean,
                                                              similarity_metric='dot',
                                                              activity='firing_rate', method='bin-by-bin')

                aud_vis_res_dict[aud_cond][vis_cond][res_cond]['sim_to_rL'] = sim_to_rL
                aud_vis_res_dict[aud_cond][vis_cond][res_cond]['sim_to_rR'] = sim_to_rR

        aud_vis_res_dict[aud_cond][vis_cond][res_cond]['trial_ds'] = target_ds

    return aud_vis_res_dict


def dim_reduce_xarray(reduction_target, projection_target=None, method='PCA',
                      variable_name='firing_rate', n_components=20, output_format='xarray'):
    if type(reduction_target) == xr.core.dataset.Dataset:
        reduction_target_time_coord = reduction_target.Time
        reduction_target = reduction_target[variable_name].values

    if type(projection_target) == xr.core.dataset.Dataset:
        projection_target_time_coord = projection_target.Time
        projection_target = projection_target[variable_name].values

        assert np.array_equal(reduction_target_time_coord.values,
                              projection_target_time_coord.values)

    assert method in ['PCA', 'PCA-whiten']

    if method == 'PCA':
        dim_reduce_object = PCA(n_components=n_components)
    elif method == 'PCA-whiten':
        dim_reduce_object = PCA(n_components=n_components, whiten=True)

    fitted_dim_reduced_model = dim_reduce_object.fit(reduction_target)
    reduced_target = fitted_dim_reduced_model.transform(reduction_target)
    reduced_projection_target = fitted_dim_reduced_model.transform(projection_target)

    if output_format == 'xarray':
        # wrap the reduced array back to xarray format
        reduced_target = xr.Dataset({'firing_rate': (['Time', 'PC'], reduced_target)},
                                    coords={'Time': reduction_target_time_coord,
                                            'PC': np.arange(1, n_components + 1)})

        reduced_projection_target = xr.Dataset({'firing_rate': (['Time', 'PC'],
                                                                reduced_projection_target)},
                                               coords={'Time': projection_target_time_coord,
                                                       'PC': np.arange(1, n_components + 1)})

    return reduced_target, reduced_projection_target, dim_reduce_object


def get_two_cond_ds(alignment_ds, cond_1='audOn', cond_2='audOff', unimodal_trials_only=False):
    """

    :param alignment_ds:
    :param cond_1:
    :param cond_2:
    :param unimodal_trials_only:
    :return:
    """

    if (cond_1 == 'beforeEvent') & (cond_2 == 'afterEvent'):
        cond_1_ds = alignment_ds.where(
            alignment_ds['PeriEventTime'] < 0, drop=True
        ).mean(dim='Time')

        cond_2_ds = alignment_ds.where(
            alignment_ds['PeriEventTime'] >= 0, drop=True
        ).mean(dim='Time')

    if cond_1 == 'audLeft':
        if unimodal_trials_only:
            cond_1_ds = alignment_ds.where(
                (alignment_ds['audDiff'] < 0) &
                (alignment_ds['visDiff'] == 0), drop=True)
        else:
            cond_1_ds = alignment_ds.where(
                (alignment_ds['audDiff'] < 0) &
                (np.isfinite(alignment_ds['audDiff'])), drop=True)
    elif cond_1 == 'visLeft':
        if unimodal_trials_only:
            cond_1_ds = alignment_ds.where(
                (alignment_ds['visDiff'] < 0) &
                (~np.isfinite(alignment_ds['audDiff'])), drop=True)
        else:
            cond_1_ds = alignment_ds.where(
                (alignment_ds['visDiff'] < 0) &
                (np.isfinite(alignment_ds['visDiff'])), drop=True)
    elif cond_1 == 'audOn':
        if unimodal_trials_only:
            cond_1_ds = alignment_ds.where(
                np.isfinite(alignment_ds['audDiff']) &
                (alignment_ds['visDiff'] == 0), drop=True)
        else:
            cond_1_ds = alignment_ds.where(
                np.isfinite(alignment_ds['audDiff']), drop=True)
    elif cond_1 == 'visOn':
        if unimodal_trials_only:
            cond_1_ds = alignment_ds.where(
                (~np.isfinite(alignment_ds['audDiff'])) &  # audio off conditions
                (alignment_ds['visDiff'] != 0),
                drop=True
            )
        else:
            cond_1_ds = alignment_ds.where(
                (alignment_ds['visDiff'] != 0),
                drop=True
            )
    elif cond_1 == 'respondLeft':
        cond_1_ds = alignment_ds.where(
            (alignment_ds['responseMade'] == 1), drop=True
        )
    elif (type(cond_1) is list) & (type(cond_1[0]) is float):
        # this is to perform a specific time subsetting
        cond_1_ds = alignment_ds.where(
            (alignment_ds['PeriEventTime'] >= cond_1[0]) &
            (alignment_ds['PeriEventTime'] < cond_1[0]), drop=True
        )
    elif cond_1 == 'beforeEvent':
        cond_1_ds = alignment_ds.where(
            alignment_ds['PeriEventTime'] < 0, drop=True
        )

    if cond_2 == 'audRight':
        if unimodal_trials_only:
            cond_2_ds = alignment_ds.where(
                (alignment_ds['audDiff'] > 0) &
                (alignment_ds['visDiff'] == 0), drop=True)
        else:
            cond_2_ds = alignment_ds.where(
                (alignment_ds['audDiff'] > 0) &
                (np.isfinite(alignment_ds['audDiff'])), drop=True)
    elif cond_2 == 'visRight':
        if unimodal_trials_only:
            cond_2_ds = alignment_ds.where(
                (alignment_ds['visDiff'] > 0) &
                (~np.isfinite(alignment_ds['audDiff'])), drop=True)
        else:
            cond_2_ds = alignment_ds.where(
                (alignment_ds['visDiff'] > 0) &
                (np.isfinite(alignment_ds['visDiff'])), drop=True)
    elif cond_2 == 'audOff':
        if unimodal_trials_only:
            cond_2_ds = alignment_ds.where(
                ~np.isfinite(alignment_ds['audDiff']) &
                (alignment_ds['visDiff'] == 0), drop=True)
        else:
            cond_2_ds = alignment_ds.where(
                ~np.isfinite(alignment_ds['audDiff']), drop=True)

    elif cond_2 == 'visOff':
        if unimodal_trials_only:
            """
            cond_2_ds = alignment_ds.where(
                (~np.isfinite(alignment_ds['audDiff'])) &  # audio off conditions
                (~alignment_ds['visDiff'] != 0),
                drop=True
            """

            cond_2_ds = alignment_ds.where(
                (alignment_ds['visDiff'] == 0),
                drop=True
            )
        else:
            cond_2_ds = alignment_ds.where(
                (alignment_ds['visDiff'] == 0),
                drop=True
            )
    elif cond_2 == 'respondRight':
        cond_2_ds = alignment_ds.where(
            (alignment_ds['responseMade'] == 2), drop=True
        )
    elif (type(cond_2) is list) & (type(cond_2[0]) is float):
        # this is to perform a specific time subsetting
        cond_2_ds = alignment_ds.where(
            (alignment_ds['PeriEventTime'] >= cond_2[0]) &
            (alignment_ds['PeriEventTime'] <= cond_2[1]), drop=True
        )

    return cond_1_ds, cond_2_ds


############################################################
# Code related to decoding analysis specifically for ephys #
############################################################

def get_choice_prob_subsetted_alignment_ds(alignment_ds, behave_df,
                                           choice_prob_lower_bound=0.4, choice_prob_upper_bound=0.6,
                                           mode='include', verbose=False, combine_conditions=True,
                                           choice_subsampling=False):
    """
    Gets alignment ds that has a certain range of left/right choice probability.
    This is mainly to find cases where the stimulus will have a poor prediction of choice,
    and see if the neural activity can do any better.
    :param alignment_ds:
    :param behave_df:
    :param choice_prob_lower_bound:
    :param choice_prob_upper_bound:
    :return:
    """

    # TODO: behave_df actually not required, can do the same thing with alignemnt_ds
    # but may require converting the data variable audDiff to coordinates...
    p_choice_per_stimuli_cond = behave_df.groupby(['visDiff', 'audDiff']).agg(np.mean)['goRight'].reset_index()

    # Count the number of trials in each stimulus condition
    # This is just to double check that the subsetting loop below is not missing any trials
    num_trials_per_stimuli_cond = behave_df.groupby(['visDiff', 'audDiff']).agg('count')['goRight'].reset_index()

    if (choice_prob_lower_bound is None) and (choice_prob_upper_bound is None):
        # no subsetting performed: we are taking all stimulus conditions
        if verbose:
            print('No subsetting based on choice probability performed.')
        target_range_choice_trial_count_df = num_trials_per_stimuli_cond
    else:
        if mode == 'include':
            target_range_choice_trial_count_df = num_trials_per_stimuli_cond.loc[
                (p_choice_per_stimuli_cond['goRight'] > choice_prob_lower_bound) &
                (p_choice_per_stimuli_cond['goRight'] < choice_prob_upper_bound)
                ]
        elif mode == 'exclude':
            target_range_choice_trial_count_df = num_trials_per_stimuli_cond.loc[
                ~(
                        (p_choice_per_stimuli_cond['goRight'] > choice_prob_lower_bound) &
                        (p_choice_per_stimuli_cond['goRight'] < choice_prob_upper_bound)
                )
            ]
        else:
            print('Warning: no valid subsetting mode specified.')

    if len(target_range_choice_trial_count_df) == 0:
        print('Warning: no stimulus conditions matches desired choice probability range.'
              'Returning None')
        return None, None

    subset_alignment_ds_list = list()
    for _, stim_cond_df in target_range_choice_trial_count_df.iterrows():
        subset_alignment_ds = alignment_ds.where(
            (alignment_ds['audDiff'] == stim_cond_df['audDiff']) &
            (alignment_ds['visDiff'] == stim_cond_df['visDiff']), drop=True
        )

        if choice_subsampling:
            # subsample left and right choice trials so that the number of trials are balanced
            # before performing the mean subtraction
            stim_left_choice = subset_alignment_ds.where(
                subset_alignment_ds['responseMade'] == 1, drop=True
            )

            stim_right_choice = subset_alignment_ds.where(
                subset_alignment_ds['responseMade'] == 2, drop=True
            )
            num_left_choice = len(stim_left_choice['Trial'])
            num_right_choice = len(stim_right_choice['Trial'])

            # if verbose:
            #    print('Number of left choices %s' % str(num_left_choice))
            #    print('NUmber of right choices %s' % str(num_right_choice))
            # if np.min([num_left_choice, num_right_choice]) == 0:
            #     print('Warning: no example of both choice types, skipping')
            #  continue

            if num_left_choice > num_right_choice:
                # there are more left choices, and so we need to subsample left choice
                random_trial_idx = np.random.choice(stim_left_choice['Trial'], size=num_right_choice,
                                                    replace=False)
                subset_left_choice_ds = stim_left_choice.sel(Trial=random_trial_idx)
                choice_combined_ds = xr.concat([subset_left_choice_ds,
                                                stim_right_choice], dim='Trial')
            elif num_left_choice < num_right_choice:
                random_trial_idx = np.random.choice(stim_right_choice['Trial'], size=num_left_choice,
                                                    replace=False)
                subset_right_choice_ds = stim_right_choice.sel(Trial=random_trial_idx)
                choice_combined_ds = xr.concat([subset_right_choice_ds,
                                                stim_left_choice], dim='Trial')
            elif num_left_choice == num_right_choice:
                choice_combined_ds = xr.concat([stim_left_choice, stim_right_choice],
                                               dim='Trial')

            stim_mean_activity = choice_combined_ds['firing_rate'].mean(dim='Trial')
            subset_alignment_ds = subset_alignment_ds.assign(
                mean_subtracted_activity=subset_alignment_ds['firing_rate'] - stim_mean_activity)

        else:
            # Calculate mean-subtracted activity (across all choice)
            stim_mean_activity = subset_alignment_ds['firing_rate'].mean(dim='Trial')
            subset_alignment_ds = subset_alignment_ds.assign(
                mean_subtracted_activity=subset_alignment_ds['firing_rate'] - stim_mean_activity)

        subset_alignment_ds_list.append(subset_alignment_ds)

    if combine_conditions:
        condition_combined_ds = xr.concat(subset_alignment_ds_list, dim='Trial')

        assert len(condition_combined_ds['Trial'].values) == np.sum(target_range_choice_trial_count_df['goRight'])
    else:
        condition_combined_ds = subset_alignment_ds_list

    return condition_combined_ds, p_choice_per_stimuli_cond


def subset_ds(template_ds, target_ds, constant_dim='Cell',
              constant_val=0, subset_dimension='Trial',
              subset_variable='responseMade'):
    """
    Subsets dataset based on some variable.
    Current use case is for subsetting trial numbers based on response (left/right) to ensure that the
    response distribution is the same for both datasets.
    Currently assumes that target_ds always contain more labels compared with template_ds,
    so we are subsetting by number of trials instead of proportion of trials.

    Arguments
    -----------
    template_ds: (xarray dataset)
        the dataset to use to generate label distribution or counts to subset
    target_ds: (xarray dataset)
        the dataset to subset
    constant_dim: (string or list of string)
        dimension(s) to subset first; these are redundant dimension such that we can just
        take a particular coordinate along this dimension and the values will be the same
    constant_val: (int, float or string, or list)
        coordinate value of the redundant dimension to take
    subset_dimension: (string)
        dimension to get subset indices
    subset_variable: (string)
        name of variable to use for subsettting.
    Returns
    -----------
    target_ds_subsampled (xarray dataset)
    """

    subset_variable_vals = template_ds.sel({constant_dim:
                                                constant_val})[subset_variable].values
    target_variable_vals = target_ds.sel({constant_dim:
                                              constant_val})[subset_variable].values

    subset_variable_val_vector, subset_variable_count_vector = np.unique(
        subset_variable_vals, return_counts=True)

    all_trial_idx_subsampled = list()

    for subset_variable_val, subset_variable_counts in zip(subset_variable_val_vector, subset_variable_count_vector):
        trial_idx_w_variable_val = np.where(
            target_variable_vals == subset_variable_val)[0]

        trial_idx_sampled = np.random.choice(
            trial_idx_w_variable_val, size=int(subset_variable_counts),
            replace=False)
        all_trial_idx_subsampled.append(trial_idx_sampled)

    target_ds_subsampled = target_ds.isel({subset_dimension:
                                               np.concatenate(all_trial_idx_subsampled)})

    if len(all_trial_idx_subsampled) == 0:
        print('No trials match condition, returning none')
        return None

    assert len(template_ds[subset_variable]) == len(target_ds_subsampled[subset_variable])
    target_subset_variable_val, target_subset_variable_counts = np.unique(target_ds_subsampled.sel({constant_dim:
                                                                                                        constant_val})[
                                                                              subset_variable].values,
                                                                          return_counts=True)
    assert np.all(np.isclose(subset_variable_count_vector, target_subset_variable_counts))

    return target_ds_subsampled


def cal_two_cond_mean_activity_difference(alignment_ds, cond_variable='responseMade',
                                          cond_variable_1=1, cond_variable_2=2, time_range=[-0.1, 0],
                                          time_variable='PeriEventTime', activity_variable='firing_rate',
                                          per='cell', normalise=False, small_coefficient=0):
    if time_range is not None:
        alignment_ds = alignment_ds.where(
            (alignment_ds[time_variable] >= time_range[0]) &
            (alignment_ds[time_variable] < time_range[1]), drop=True
        )

    # TODO: this is more generalisable with groupby

    cond_1_ds = alignment_ds.where(
        alignment_ds[cond_variable] == cond_variable_1, drop=True
    ).mean(['Time', 'Trial'])[activity_variable]

    cond_2_ds = alignment_ds.where(
        alignment_ds[cond_variable] == cond_variable_2, drop=True
    ).mean(['Time', 'Trial'])[activity_variable]

    cond_difference = cond_1_ds - cond_2_ds

    if normalise:
        cond_difference = cond_difference.values / (cond_1_ds.values + cond_2_ds.values + small_coefficient)

    return cond_difference


def circular_shift_raster(activity_ds, activity_name='firing_rate',
                          per_neuron=False):
    """

    :param activity_ds:
    :param activity_name:
    per_neuron : (bool)
        whether to shift each neuron separately (may take a long time)
    :return:
    """
    num_time_points = len(activity_ds.Time.values)

    if per_neuron:
        for cell in activity_ds['Cell'].values:
            cell_ds = activity_ds.sel(Cell=cell)

            for trial in activity_ds['Trial'].values:
                cell_trial_ds = cell_ds.sel(Trial=trial)
                shift_amount = np.random.randint(low=0, high=num_time_points)
                cell_trial_ds = cell_trial_ds.assign({activity_name: ('Time', np.roll(
                    cell_trial_ds[activity_name].values, shift_amount))})
    else:
        trial_ds_list = list()
        for trial in activity_ds['Trial'].values:
            trial_ds = activity_ds.sel(Trial=trial)
            shift_amount = np.random.randint(low=0, high=num_time_points)

            trial_ds = trial_ds.roll({'Time': shift_amount},
                                     roll_coords=False)

            # print(cell_trial_ds[activity_name])

            # trial_ds = trial_ds.assign({activity_name: (['Cell, Time'], np.roll(
            #         cell_trial_ds[activity_name].values, shift_amount))})
            trial_ds_list.append(trial_ds)

        shifted_raster_ds = xr.concat(trial_ds_list, dim='Trial')

    return shifted_raster_ds


def left_right_metric_to_contrast_ipsi(neuron_df, modality_metric_df, metric_column_names):
    """
    Convert metric relating involving left and right stimuli or movement to position relative to the recorded brain
    hemisphere (contralateral or ipsilateral to the recorded hemisphere)

    Parameters
    -----------
    :param neuron_df:
    :param modality_metric_df:
    metric_column_names: (list of str)
    :return:
    """

    hemisphere_df = neuron_df[['subjectRef', 'expRef',
                               'cellLoc', 'cluNum', 'hemisphere']]

    exp_included = np.unique(modality_metric_df['expRef'])

    hemisphere_subset_df = hemisphere_df.loc[
        hemisphere_df['expRef'].isin(exp_included)
    ]

    ### Loop through brain regino in each experiment to assign L/R index
    cell_loc_modality_df_list = list()
    for exp in np.unique(hemisphere_subset_df['expRef']):

        exp_hemisphere_subset_df = hemisphere_subset_df.loc[
            (hemisphere_subset_df['expRef'] == exp)]

        exp_modality_test_df = modality_metric_df.loc[
            modality_metric_df['expRef'] == exp
            ]

        for brain_region in np.unique(exp_hemisphere_subset_df['cellLoc']):
            cell_loc_hemisphere_df = exp_hemisphere_subset_df.loc[
                exp_hemisphere_subset_df['cellLoc'] == brain_region
                ]

            cell_loc_modality_df = exp_modality_test_df.loc[
                exp_modality_test_df['cellLoc'] == brain_region
                ]

            cell_loc_modality_df['hemisphere'] = cell_loc_hemisphere_df['hemisphere'].values

            cell_loc_modality_df_list.append(cell_loc_modality_df)

    contra_ipsi_df = pd.concat(cell_loc_modality_df_list)

    for metric_name in metric_column_names:
        # Right hemisphere values needs their sign flipped
        contra_ipsi_df['IpsiContra' + metric_name] = modality_metric_df[metric_name]
        contra_ipsi_df.loc[(contra_ipsi_df['hemisphere'] == 'R'), 'IpsiContra' + metric_name] = contra_ipsi_df.loc[
                                                                                                    contra_ipsi_df[
                                                                                                        'hemisphere'] == 'R'][
                                                                                                    metric_name] * -1
        # left hemisphere values stay the same

    return contra_ipsi_df


def cal_single_neuron_ave_psth(subject, experiment, cellIdx, alignment_folder,
                               cond_1='audLeft', cond_2='audRight', unimodal_trials_only=True,
                               run_
                               =False,
                               smooth_sigma=10, smooth_window_width=10):
    """
    Utility function to average and smooth spike trains across trial to compare between two trial conditions.
    This is mainly used for the plotting function plot_neuron_alignment_condition_trace()

    Parameters
    ----------
    subjectRef : (int)
        subject reference number
    experimentRef : (int)
        experiment reference number
    cellIdx : (int)
        cell index, currently expect the index to be across all brain regions
    alignment_folder : (str)
        path to folder that contain list of alignment files (aligned to stimulus or response)
    cond_1 : (str)
        first trial condition to obtain
        eg. audLeft : auditory stimuli is on the left side of the mouse
        eg. respondLeft : mouse made left choice
    cond_2 : (str)
        second trial condition to obtain
    unimodal_trials_only : (bool)
        whether to only get trials that have a single stimuli
        if True, then audLeft will mean audio left + no visual stimuli
        if False, then audLeft will mean audio left + any visual stimuli
    run_smooth_spikes : (bool)
        whether to smooth the spike trains
    smooth_sigma : (int)
        standard deviation of the half-Gaussian used to smooth the spikes
    smooth_window_width : (int)
        number of time bins to apply each half-Gaussian window
    Output
    ----------
    cond_1_cell_ds : (xarray dataset)
        dataset for the first condition
    """

    print('Reminder: the cell index here is assumed to be across all brain regions')
    alignment_ds = pr_ephys.load_subject_exp_alignment_ds(alignment_folder=alignment_folder,
                                                          subject_num=subject, exp_num=experiment,
                                                          target_brain_region=None,
                                                          aligned_event='stimOnTime',
                                                          alignment_file_ext='.nc')

    cell_ds = alignment_ds.isel(Cell=cellIdx)

    cond_1_cell_ds, cond_2_cell_ds = get_two_cond_ds(
        cell_ds, cond_1=cond_1, cond_2=cond_2, unimodal_trials_only=unimodal_trials_only)

    if run_smooth_spikes:
        if len(cond_1_cell_ds['Trial'].values) >= 1:
            if len(cond_1_cell_ds['Trial'].values) == 1:
                cond_1_smoothed_firing_rate = smooth_spikes(cond_1_cell_ds['firing_rate'],
                                                            method='half_gaussian', sigma=smooth_sigma,
                                                            window_width=smooth_window_width)
                cond_1_cell_ds['firing_rate'] = (['Time', 'Trial'], cond_1_smoothed_firing_rate)
            else:
                cond_1_cell_ds['firing_rate'] = cond_1_cell_ds['firing_rate'].groupby('Trial').apply(
                    smooth_spikes,
                    method='half_gaussian', sigma=smooth_sigma,
                    window_width=smooth_window_width)
        if len(cond_2_cell_ds['Trial'].values) >= 1:
            if len(cond_2_cell_ds['Trial'].values) == 1:
                cond_2_smoothed_firing_rate = smooth_spikes(cond_2_cell_ds['firing_rate'],
                                                            method='half_gaussian',
                                                            sigma=smooth_sigma,
                                                            window_width=smooth_window_width)
                cond_2_cell_ds['firing_rate'] = (['Time', 'Trial'], cond_2_smoothed_firing_rate)
            else:
                cond_2_cell_ds['firing_rate'] = cond_2_cell_ds['firing_rate'].groupby('Trial').apply(
                    smooth_spikes,
                    method='half_gaussian', sigma=smooth_sigma,
                    window_width=smooth_window_width)

    return cond_1_cell_ds, cond_2_cell_ds


def cal_multi_neuron_ave_psth(cell_df, alignment_folder, z_score_spikes=True,
                              use_pre_stim_as_baseline=True,
                              switch_cond_based_on_preference=False,
                              cond_1='audLeft', cond_2='audRight',
                              unimodal_trials_only=True, run_smooth_spikes=False,
                              smooth_sigma=10, smooth_window_width=10):
    """
    Calls the function 'cal_single_neuron_ave_psth' in a loop through each cell

    Parameters
    -----------
    cell_df : (pandas dataframe)
        pandas dataframe with at least 3 columns
        'subjectRef'
        'expRef'
        'cellIdx'
        this is usually a statistical test dataframe where each row is a cell
        that test significant for a particular condition pair
    z_score_spikes : (bool)
        whether to z-score each average spike train trace for each neuron
        this is so that the firing rate can be compared between neurons relative to their baselines
    use_pre_stim_as_baseline : (bool)
        whether to use the pre-stimulus activity as the baseline to calculate the mean and standard deviation
        then apply this to calculate the z-score of the entire trace
        only relevant of z_score_spikes is set to True.
    switch_cond_based_on_preference : (bool)
        whether to re-label the condition of each neuron from being based on the stimulus or the response direction
        to the preferred stimulus or response
        eg. audLeft and audRight will be re-coded to audPreferred and audNotPreferred
        by comparing the mean activity. The condition with the higher mean activity will
        be the preferred auditory stimulus.
    alignment_folder : (str)
        path to folder that contain list of alignment files (aligned to stimulus or response)
    cond_1 : (str)
        first trial condition to obtain
        eg. audLeft : auditory stimuli is on the left side of the mouse
        eg. respondLeft : mouse made left choice
    cond_2 : (str)
        second trial condition to obtain
    unimodal_trials_only : (bool)
        whether to only get trials that have a single stimuli
        if True, then audLeft will mean audio left + no visual stimuli
        if False, then audLeft will mean audio left + any visual stimuli
    run_smooth_spikes : (bool)
        whether to smooth the spike trains
    smooth_sigma : (int)
        standard deviation of the half-Gaussian used to smooth the spikes
    smooth_window_width : (int)
        number of time bins to apply each half-Gaussian window
    Output
    ----------
    """
    cond_1_list = list()
    cond_2_list = list()

    for df_idx, cell_df in cell_df.iterrows():
        subject = cell_df['subjectRef']
        experiment = cell_df['expRef']
        cellIdx = cell_df['cellIdx']

        cond_1_cell_ds, cond_2_cell_ds = cal_single_neuron_ave_psth(subject, experiment, cellIdx,
                                                                    alignment_folder=alignment_folder,
                                                                    cond_1=cond_1, cond_2=cond_2,
                                                                    unimodal_trials_only=unimodal_trials_only,
                                                                    run_smooth_spikes=run_smooth_spikes,
                                                                    smooth_sigma=smooth_sigma,
                                                                    smooth_window_width=smooth_window_width)

        cond_1_firing_rate_mean = cond_1_cell_ds['firing_rate'].mean(dim='Trial').values
        cond_2_firing_rate_mean = cond_2_cell_ds['firing_rate'].mean(dim='Trial').values

        # TODO: need to combine cond_1 and cond_2, or do the z_score way earlier
        if z_score_spikes:
            # Option 1: the entire trace
            if not use_pre_stim_as_baseline:
                cond_1_firing_rate_mean = sstats.zscore(cond_1_firing_rate_mean)
                cond_2_firing_rate_mean = sstats.zscore(cond_2_firing_rate_mean)
            else:
                # Option 2: use the pre-stimulus activity as baseline
                cond_1_pre_stim_mean = cond_1_cell_ds['firing_rate'].where(
                    cond_1_cell_ds['PeriEventTime'] < 0, drop=True
                ).mean(dim='Trial')
                cond_2_pre_stim_mean = cond_2_cell_ds['firing_rate'].where(
                    cond_2_cell_ds['PeriEventTime'] < 0, drop=True
                ).mean(dim='Trial')

                cond_1_firing_rate_mean = sstats.zmap(scores=cond_1_firing_rate_mean,
                                                      compare=cond_1_pre_stim_mean)
                cond_2_firing_rate_mean = sstats.zmap(scores=cond_2_firing_rate_mean,
                                                      compare=cond_2_pre_stim_mean)

        if switch_cond_based_on_preference:
            if np.mean(cond_1_firing_rate_mean) < np.mean(cond_2_firing_rate_mean):
                # switch cond_1 and cond_2
                # cond_1 is the preferred direction, cond_2 is the non-preferred direction
                temp = cond_1_firing_rate_mean
                cond_1_firing_rate_mean = cond_2_firing_rate_mean
                cond_2_firing_rate_mean = temp

        cond_1_list.append(cond_1_firing_rate_mean)
        cond_2_list.append(cond_2_firing_rate_mean)

    cond_1_combined = np.stack(cond_1_list)
    cond_2_combined = np.stack(cond_2_list)

    # TODO: tidy this up and make cond_1_combined a dataarray
    peri_event_time = cond_1_cell_ds['PeriEventTime'].values

    return cond_1_combined, cond_2_combined, peri_event_time


def get_aud_vis_projection_X(alignment_ds_time_mean, activity_name='firing_rate'):
    """
    Get feature matrix to do audiovisual dimensionality reduction.
    Parameters
    ----------
    alignment_ds_time_mean

    Returns
    -------

    """
    unimodal_alignment_ds = alignment_ds_time_mean.where(
        ((alignment_ds_time_mean['audDiff'] == np.inf) | (alignment_ds_time_mean['visDiff'] == 0)),
        drop=True)
    unimodal_alignment_ds = unimodal_alignment_ds.where(
        ((np.abs(unimodal_alignment_ds['visDiff']) > 0.7) | (alignment_ds_time_mean['visDiff'] == 0)),
        drop=True)
    arv0_index = np.where((unimodal_alignment_ds['audDiff'] == 60) & (unimodal_alignment_ds['visDiff'] == 0))
    alv0_index = np.where((unimodal_alignment_ds['audDiff'] == -60) & (unimodal_alignment_ds['visDiff'] == 0))
    a0vr_index = np.where((unimodal_alignment_ds['audDiff'] == np.inf) & (unimodal_alignment_ds['visDiff'] >= 0.7))
    a0vl_index = np.where((unimodal_alignment_ds['audDiff'] == np.inf) & (unimodal_alignment_ds['visDiff'] <= -0.7))
    arvr_alignment_ds = alignment_ds_time_mean.where(
        ((alignment_ds_time_mean['visDiff'] > 0.7) & (alignment_ds_time_mean['audDiff'] == 60)),
        drop=True)
    arvl_alignment_ds = alignment_ds_time_mean.where(
        ((alignment_ds_time_mean['visDiff'] < -0.7) & (alignment_ds_time_mean['audDiff'] == 60)),
        drop=True)
    alvr_alignment_ds = alignment_ds_time_mean.where(
        ((alignment_ds_time_mean['visDiff'] > 0.7) & (alignment_ds_time_mean['audDiff'] == -60)),
        drop=True)
    alvl_alignment_ds = alignment_ds_time_mean.where(
        ((alignment_ds_time_mean['visDiff'] < -0.7) & (alignment_ds_time_mean['audDiff'] == -60)),
        drop=True)
    X_arvr = arvr_alignment_ds[activity_name].values
    X_arvl = arvl_alignment_ds[activity_name].values
    X_alvl = alvl_alignment_ds[activity_name].values
    X_alvr = alvr_alignment_ds[activity_name].values
    y_aud = unimodal_alignment_ds['audDiff'].values
    y_vis = unimodal_alignment_ds['visDiff'].values
    y_aud[y_aud == -60] = -1
    y_aud[y_aud == 60] = 1
    y_aud[y_aud == np.inf] = 0
    y_vis = np.sign(y_vis)
    X = unimodal_alignment_ds['firing_rate']
    return (
        arv0_index, alv0_index, a0vr_index, a0vl_index, X, X_arvr, X_arvl, X_alvl, X_alvr, y_aud, y_vis)


def load_active_and_passive_ds(subject, experiment, brain_region, exp_cell_idx, active_alignment_folder,
                               passive_alignment_folder, smooth_sigma=30, smooth_window_width=50, min_rt=0.1,
                               max_rt=0.35, rt_variable_name='choiceInitTimeRelStim'):
    """

    Parameters
    ----------
    subject
    experiment
    brain_region
    exp_cell_idx
    active_alignment_folder
    passive_alignment_folder

    Returns
    -------

    """
    active_alignment_ds = pr_ephys.load_subject_exp_alignment_ds(alignment_folder=active_alignment_folder,
                                                                 subject_num=subject,
                                                                 exp_num=experiment,
                                                                 target_brain_region=brain_region,
                                                                 aligned_event='stimOnTime',
                                                                 alignment_file_ext='.nc')
    passive_alignment_ds = pr_ephys.load_subject_exp_alignment_ds(alignment_folder=passive_alignment_folder,
                                                                  subject_num=subject,
                                                                  exp_num=experiment,
                                                                  target_brain_region=brain_region,
                                                                  aligned_event='stimOnTime',
                                                                  alignment_file_ext='.nc')
    if min_rt is not None:
        active_alignment_ds = active_alignment_ds.where((active_alignment_ds[rt_variable_name] >= min_rt),
                                                        drop=True)
    if max_rt is not None:
        active_alignment_ds = active_alignment_ds.where((active_alignment_ds[rt_variable_name] <= max_rt),
                                                        drop=True)

    active_align_to_stim_ds = active_alignment_ds.isel(Cell=exp_cell_idx)
    passive_align_to_stim_ds = passive_alignment_ds.isel(Cell=exp_cell_idx)
    cell_active_stim_aligned_ds = active_align_to_stim_ds.stack(trialTime=['Trial', 'Time'])
    cell_active_stim_aligned_ds['smoothed_fr'] = ('trialTime',
                                                  smooth_spikes((cell_active_stim_aligned_ds['firing_rate']),
                                                                method='half_gaussian',
                                                                sigma=smooth_sigma,
                                                                window_width=smooth_window_width,
                                                                custom_window=None))
    cell_active_stim_aligned_ds = cell_active_stim_aligned_ds.unstack()
    cell_passive_stim_aligned_ds = passive_align_to_stim_ds.stack(trialTime=['Trial', 'Time'])
    cell_passive_stim_aligned_ds['smoothed_fr'] = ('trialTime',
                                                   smooth_spikes((cell_passive_stim_aligned_ds['firing_rate']),
                                                                 method='half_gaussian',
                                                                 sigma=smooth_sigma,
                                                                 window_width=smooth_window_width,
                                                                 custom_window=None))
    passive_cell_aligned_ds = cell_passive_stim_aligned_ds.unstack()
    return cell_active_stim_aligned_ds, passive_cell_aligned_ds
