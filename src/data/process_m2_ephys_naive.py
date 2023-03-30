from src.data.process_ephys_data import ephys_mat_to_pkl
import os

# About
# This is a copy of process_m2_ephys_good.py but to process the data
# from recordings in naive mice. The main changes are
# (1) processing active condition is removed
# (2) there is no need to remove active / passive mismatch
# Created: 2021-09-17

# External Data
root = '/media/timsit/Partition 1/'
#PASSIVE_EPHYS_MAT_PATH = 'data/raw/naive-m2-ephys-passive.mat'
#PASSIVE_SAVE_PATH = 'data/interim/passive-m2-naive'

PASSIVE_EPHYS_MAT_PATH = 'data/raw/naive-m2-ephys-sorted-passive.mat'
PASSIVE_SAVE_PATH = 'data/interim/passive-m2-sorted-naive'

include_exp_dates = True
# data_date = '2021-09-17'
data_date = '2021-11-23'
process_passive_condition = True
include_depth = True  # TODO: currently on by default inside code
custom_cell_location = 'putativeMOs'
extraction_list = ['behaviour', 'spike', 'neuron'] # ['behaviour', 'spike', 'neuron']

def main():

    # Passive Condition
    if process_passive_condition:
        print('Processing passive data from naive mice')
        ephys_mat_to_pkl(root=root, ephys_mat_path=PASSIVE_EPHYS_MAT_PATH,
                         save_folder=os.path.join(root, PASSIVE_SAVE_PATH),
                         active_condition=False,
                         include_no_go=False,
                         include_invalid=False,
                         spike_format='post-oct-2019-unique-cell-id',
                         extraction_list=extraction_list,
                         dataframe_type='pandas', include_exp_dates=include_exp_dates,
                         custom_cell_location=custom_cell_location)
    print('All done')

if __name__ == '__main__':
    main()
