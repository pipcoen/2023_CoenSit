import matplotlib.pyplot as plt
import os
from src.utils import get_project_root
root = get_project_root()
# plt.style.use(os.path.join(root, 'src/visualization/ts.mplstyle'))
FIG_DPI = 300

# from src.visualization.useSansMaths import use_sans_maths
# use_sans_maths()



def plot_toeplitz(fig, ax, toeplitz_matrix, global_time=None,
                  peri_event_time=None):

    if (fig is None) or (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)

    ax.matshow(toeplitz_matrix, aspect='auto')

    if peri_event_time is not None:
        ax.set_xticks(range(len(peri_event_time)))
        ax.set_xticklabels(peri_event_time)
        ax.set_xlabel('Peri-event time (a.u.)')

    if global_time is not None:

        ax.set_yticks(range(len(global_time)))
        ax.set_yticklabels(global_time)
        ax.set_ylabel('Time')

    return fig, ax




