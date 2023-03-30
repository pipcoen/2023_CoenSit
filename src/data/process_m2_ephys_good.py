from src.data.process_ephys_data import ephys_mat_to_pkl
import os
from src.utils import get_project_root
import numpy as np
import pandas as pd
import pdb

# Local data
# root = get_project_root()

# External Data
root = '/media/timsit/Partition 1/'

# Local Data
# ACTIVE_EPHYS_MAT_PATH = 'data/raw/active-m2-ephys-good-new-atlas.mat'
# ACTIVE_SAVE_PATH = 'data/interim/active-m2-good-w-nogo-and-invalid/'

# PASSIVE_EPHYS_MAT_PATH = 'data/raw/passive-m2-ephys-good-new-atlas.mat'
# PASSIVE_SAVE_PATH = 'data/interim/passive-m2-w-nogo-and-invalid/'

# 2020-03-20: Updated wheel time files

# ACTIVE_EPHYS_MAT_PATH = 'data/raw/active-m2-ephys-good-two-movement-times.mat'
# PASSIVE_EPHYS_MAT_PATH = 'data/raw/passive-m2-ephys-good-two-movement-times.mat'

# 2020-04-24: Updated wheel time files with reliable / unreliable movements

# ACTIVE_EPHYS_MAT_PATH = 'data/raw/active-m2-ephys-reliable-movement-times.mat'
# PASSIVE_EPHYS_MAT_PATH = 'data/raw/passive-m2-ephys-reliable-movement-times.mat'

# ACTIVE_SAVE_PATH = 'data/interim/active-m2-good-reliable-movement'
# PASSIVE_SAVE_PATH = 'data/interim/passive-m2-good-reliable-movement'

# 2020-09-19: Updated grouping of brain regions ("parent" field in cell location

# ACTIVE_EPHYS_MAT_PATH = 'data/raw/active-m2-ephys-new-parent.mat'
# PASSIVE_EPHYS_MAT_PATH = 'data/raw/passive-m2-ephys-new-parent.mat'

# ACTIVE_SAVE_PATH = 'data/interim/active-m2-new-parent'
# PASSIVE_SAVE_PATH = 'data/interim/passive-m2-new-parent'

# 2020-11-11: Updated movement initiation time definition (passive unchanged)

# ACTIVE_EPHYS_MAT_PATH = 'data/raw/active-m2-ephys-choice-init-mod.mat'
# PASSIVE_EPHYS_MAT_PATH = 'data/raw/passive-m2-ephys-new-parent.mat'

# ACTIVE_SAVE_PATH = 'data/interim/active-m2-choice-init/'
# PASSIVE_SAVE_PATH = 'data/interim/passive-m2-choice-init'

# 2021-01-27 : Updated trials
ACTIVE_EPHYS_MAT_PATH = 'data/raw/active-m2-ephys-choice-init-v2.mat'
PASSIVE_EPHYS_MAT_PATH = 'data/raw/passive-m2-ephys-new-parent.mat'

# ACTIVE_SAVE_PATH = 'data/interim/active-m2-choice-init-v2/'
# PASSIVE_SAVE_PATH = 'data/interim/passive-m2-choice-init'

# 2021-03-02 : Include experiment date
ACTIVE_SAVE_PATH = 'data/interim/active-m2-w-date'
PASSIVE_SAVE_PATH = 'data/interim/passive-m2-w-date'
include_exp_dates = True

data_date = '2021-01-27'

process_active_condition = True
process_passive_condition = True

def main():

    # Active condition
    if process_active_condition:
        ephys_mat_to_pkl(root=root, ephys_mat_path=ACTIVE_EPHYS_MAT_PATH,
                         save_folder=os.path.join(root, ACTIVE_SAVE_PATH),
                         active_condition=True,
                         include_no_go=True,
                         include_invalid=True,
                         spike_format='post-oct-2019-unique-cell-id',
                         dataframe_type='pandas',
                         extraction_list=['behaviour'],
                         data_date=data_date, include_exp_dates=include_exp_dates)

    # Passive Condition
    if process_passive_condition:
        ephys_mat_to_pkl(root=root, ephys_mat_path=PASSIVE_EPHYS_MAT_PATH,
                         save_folder=os.path.join(root, PASSIVE_SAVE_PATH),
                         active_condition=False,
                         include_no_go=False,
                         include_invalid=False,
                         spike_format='post-oct-2019-unique-cell-id',
                         dataframe_type='pandas', include_exp_dates=include_exp_dates)

    # Subset the data by removing mismatch data between active and passive condition
    print('Subsetting data')
    remove_passive_active_mismatch(active_data_folder_path=os.path.join(root, ACTIVE_SAVE_PATH),
                                   passive_data_folder_path=os.path.join(root, PASSIVE_SAVE_PATH))
    print('All done')


def remove_passive_active_mismatch(active_data_folder_path, passive_data_folder_path,
                                   ephys_behave_file='ephys_behaviour_df.pkl',
                                   ephys_cell_file='neuron_df.pkl', ephys_spike_file='spike_df.pkl'):
    """
    Remove mismatch data between passive and active dataset.
    Currently hard-coded. 
    TODO: make this not hard-coded; detect discrepancies automatically.

    Currently the hard coded version achieves this:
    (1) removing the one extra experiment in the passive condition
    (where there is no corresponding active condition recording)
    (2) remove the one extra penetration in the passive condition
    """

    active_behave_df = pd.read_pickle(os.path.join(root, active_data_folder_path, ephys_behave_file))
    active_neuron_df = pd.read_pickle(os.path.join(root, active_data_folder_path, ephys_cell_file))
    active_spike_df = pd.read_pickle(os.path.join(root, active_data_folder_path, ephys_spike_file))

    passive_behave_df = pd.read_pickle(os.path.join(root, passive_data_folder_path, ephys_behave_file))
    passive_neuron_df = pd.read_pickle(os.path.join(root, passive_data_folder_path, ephys_cell_file))
    passive_spike_df = pd.read_pickle(os.path.join(root, passive_data_folder_path, ephys_spike_file))

    # (1) remove the one extra experiment in the passive condition, and shift the references
    PASSIVE_EXP_TO_REMOVE = 17

    # will need to reindex the exp refs so that they match ...
    passive_behave_df = passive_behave_df.loc[
        passive_behave_df['expRef'] != PASSIVE_EXP_TO_REMOVE]

    passive_neuron_df = passive_neuron_df.loc[
        passive_neuron_df['expRef'] != PASSIVE_EXP_TO_REMOVE]

    # move the other exps one down : neuron df
    passive_neuron_df['expRef'].loc[
        passive_neuron_df['expRef'] > PASSIVE_EXP_TO_REMOVE] -= 1

    passive_behave_df['expRef'].loc[
        passive_behave_df['expRef'] > PASSIVE_EXP_TO_REMOVE] -= 1

    # remove penetration corresponding to the extra experiment
    PASSIVE_PEN_TO_SHIFT = 27

    passive_neuron_df['penRef'].loc[
        passive_neuron_df['penRef'] > PASSIVE_PEN_TO_SHIFT] -= 1


    # (2) remove the extra penetration in one experiment, and shift references down

    # number of neuron in each pen

    num_neuron_per_pen_active = active_neuron_df.groupby('penRef').agg('count')['cellId']
    num_neuron_per_pen_passive = passive_neuron_df.groupby('penRef').agg('count')['cellId']

    num_per_neuron = pd.DataFrame({'active': num_neuron_per_pen_active,
                                   'passive': num_neuron_per_pen_passive})

    pen_to_remove_idx = np.where(num_per_neuron['active'] - num_per_neuron['passive'] != 0)[0][0]
    pen_to_remove = num_per_neuron.index[pen_to_remove_idx]

    # a priori know that this is from the active condition
    active_neuron_df = active_neuron_df.loc[
        active_neuron_df['penRef'] != pen_to_remove
        ]

    active_neuron_df['penRef'].loc[
        active_neuron_df['penRef'] > pen_to_remove] -= 1


    # Re-count the neurons of each penRef and assert that they are the same

    num_neuron_per_pen_active = active_neuron_df.groupby('penRef').agg('count')['cellId']
    num_neuron_per_pen_passive = passive_neuron_df.groupby('penRef').agg('count')['cellId']

    num_per_neuron = pd.DataFrame({'active': num_neuron_per_pen_active,
                                   'passive': num_neuron_per_pen_passive})


    # remove neurons removed in the above process from spike_df
    active_spike_df = active_spike_df.loc[
        active_spike_df['cellId'].isin(np.unique(active_neuron_df['cellId']))
    ]

    passive_spike_df = passive_spike_df.loc[
        passive_spike_df['cellId'].isin(np.unique(passive_neuron_df['cellId']))
    ]

    # map old cell id to new cell id, and apply that to spike_df
    active_cell_id_conversion_dict = dict(zip(active_neuron_df['cellId'],
                                              np.arange(len(active_neuron_df))
                                              ))

    passive_cell_id_conversion_dict = dict(zip(passive_neuron_df['cellId'],
                                               np.arange(len(passive_neuron_df))
                                               ))

    active_spike_df['cellId'] = active_spike_df['cellId'].map(
        active_cell_id_conversion_dict)

    passive_spike_df['cellId'] = passive_spike_df['cellId'].map(
        passive_cell_id_conversion_dict)

    # Re-index to create matching cellId and pd.Index
    active_neuron_df = active_neuron_df.reset_index(drop=True)
    passive_neuron_df = passive_neuron_df.reset_index(drop=True)

    active_behave_df = active_behave_df.reset_index(drop=True)
    passive_behave_df = passive_behave_df.reset_index(drop=True)

    active_spike_df = active_spike_df.reset_index(drop=True)
    passive_spike_df = passive_spike_df.reset_index(drop=True)

    # resest unique (pen, clu) cell Ids

    active_neuron_df['cellId'] = np.arange(len(active_neuron_df))
    passive_neuron_df['cellId'] = np.arange(len(passive_neuron_df))


    # assert neuron counts per penetration are equal
    assert np.sum(num_per_neuron['active'].values - num_per_neuron['passive'].values) == 0


    # assert experiment references are equal
    assert np.sum(np.unique(active_behave_df['expRef']) -
                  np.unique(passive_behave_df['expRef'])) == 0


    # assert that all numerical fields of neuron df in both conditions
    field_to_test_list = ['subjectRef', 'expRef', 'penRef', 'cluNum', 'cellId']
    for field_to_test in field_to_test_list:
        assert np.sum(active_neuron_df[field_to_test] - \
                      passive_neuron_df[field_to_test]) == 0


    # save the new data

    ACTIVE_SUBSET_FOLDER_PATH = os.path.join(root, active_data_folder_path, 'subset')
    PASSIVE_SUBSET_FOLDER_PATH = os.path.join(root, passive_data_folder_path, 'subset')

    if not os.path.exists(ACTIVE_SUBSET_FOLDER_PATH):
        os.mkdir(ACTIVE_SUBSET_FOLDER_PATH)

    if not os.path.exists(PASSIVE_SUBSET_FOLDER_PATH):
        os.mkdir(PASSIVE_SUBSET_FOLDER_PATH)

    active_behave_df.to_pickle(os.path.join(root, ACTIVE_SUBSET_FOLDER_PATH, ephys_behave_file))
    active_neuron_df.to_pickle(os.path.join(root, ACTIVE_SUBSET_FOLDER_PATH, ephys_cell_file))
    active_spike_df.to_pickle(os.path.join(root, ACTIVE_SUBSET_FOLDER_PATH, ephys_spike_file))

    passive_behave_df.to_pickle(os.path.join(root, PASSIVE_SUBSET_FOLDER_PATH, ephys_behave_file))
    passive_neuron_df.to_pickle(os.path.join(root, PASSIVE_SUBSET_FOLDER_PATH, ephys_cell_file))
    passive_spike_df.to_pickle(os.path.join(root, PASSIVE_SUBSET_FOLDER_PATH, ephys_spike_file))


if __name__ == '__main__':
    main()