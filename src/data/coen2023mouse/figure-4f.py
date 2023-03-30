"""
This scripts generates figure 4f of the paper
This is the full model error (AV + interaction term) vs. additive model error, during passive conditions
"""


import pandas as pd
import matplotlib.pyplot as plt
import sciplotlib.style as splstyle
import numpy as np
import os

data_folder = '/Volumes/Partition 1/data/interim'
fig_folder = '/Volumes/Macintosh HD/Users/timothysit/coen2023mouse'
fig_name = 'fig-4f.pdf'

def main():
    # df = pd.read_csv('/Volumes/Partition 1/data/interim/temp/passive_3_plus_2_versus_4_plus_2_mse_error.csv')
    # df = pd.read_csv('/Volumes/Partition 1/data/interim/temp/Ridge1/passive_3_plus_2_versus_4_plus_2_mse_error.csv')
    df = pd.read_csv(
        os.path.join(data_folder, 'temp/passive_model_no_reg/passive_3_plus_2_versus_4_plus_2_mse_error_0ms_to_400ms_error_window.csv'))

    dot_size = 5
    min_val = 0
    max_val = 130

    high_light_neuron_df = df.loc[
        (df['Exp'] == 17) &
        (df['Cell'] == 2)
        ]

    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)
        ax.scatter(df['interaction'], df['addition'], color='black', s=dot_size)
        unity_line = np.linspace(min_val, max_val, 100)
        ax.plot(unity_line, unity_line, linestyle='--', color='gray', alpha=0.5)

        ax.scatter(high_light_neuron_df['interaction'],
                   high_light_neuron_df['addition'], color='red', s=dot_size)

        ax.set_xlabel('Interaction model error', size=11)
        ax.set_ylabel('Addition model error', size=11)

        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])


        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()

