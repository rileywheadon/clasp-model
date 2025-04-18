# This file contains exploratory data analysis and data preprocessing for trichoblast 
# and atrichoblast data. The raw data can be found in the 'data/' folder
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules import *

# Sensible default parameters for matplotlib
mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['figure.dpi'] = 360
mpl.rcParams['font.size'] = 18

# Mutant prefixes and labels
PREFIXES = ["WT-", "BC-", "C1-"]
LABELS = ["Wild Type", "BRIN-CLASP", "CLASP-1"]
COLORS = OKABE_ITO[:3]


##########################################################################
# PLOT #1: Visualization of unprocessed cell number vs. cell length data #
##########################################################################

# Plot raw mutant data for the specified genotype
def plot_raw_mutant_data(cell_type, mutant, label, color):

    # Load raw data from the specified mutant and compute the mean and SE
    raw = pd.read_csv(f"data/{cell_type}-areas.csv")
    columns = [c for c in raw.columns if c.startswith(mutant)]
    means = raw[columns].mean(axis = 1).to_numpy()
    sems = raw[columns].sem(axis = 1).to_numpy()

    # Remove noisy data from higher in the root
    if mutant == "BC-":
        means = means[:41]
        sems = sems[:41]
    if mutant == "C1-":
        means = means[:30]
        sems = sems[:30]

    # Plot the mean and standard error
    ps = np.linspace(1, means.size, num = means.size)
    plt.plot(ps, means, alpha = 1, zorder = 3, color = color, label = label)
    plt.errorbar(ps, means, yerr = sems, color = color, fmt = "o", lw = 1)

def plot_unprocessed_data(cell_type = "trichoblast"):

    # Generate plot of unprocessed cell number vs. cell length data
    ax = plt.subplot(111)
    ax.set_xlabel(r"Cell Number")
    ax.set_ylabel(r"Cell Area ($\mu$m)")

    # Disable top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot the raw mutant data for each genotype
    for p, l, c in zip(PREFIXES, LABELS, COLORS):
        plot_raw_mutant_data(cell_type, p, l, c)
        
    # Add a legend and save the resulting figure
    ax.legend()
    plt.savefig(f"{PATH}/{cell_type}-data-unprocessed.pdf", format ='pdf')
    print(f"Generated figure '{PATH}/{cell_type}-data-unprocessed.pdf'")
    ax.clear()



################################################################
# PLOT #2: Visualization of processed position vs. length data #
################################################################

# Plot data for a single mutant
def plot_processed_mutant_data(cell_type, mutant, label, color):
    data, fit, se = get_mutant_data(cell_type, mutant, 10, 12)
    style = {'s': 15, 'alpha': 0.4, 'color': color, 'edgecolor': 'none', 'label': label}
    plt.scatter(data[:, 0], data[:, 1], **style)


def plot_processed_data(cell_type = "trichoblast"):

    # Initialize the plot
    ax = plt.subplot(111)
    ax.set_xlim((0, 1000))
    ax.set_xlabel(r"Distance from Quiescent Center ($\mu$m)")
    ax.set_ylim((0, 150))
    ax.set_yticks(np.arange(0, 180, 30))
    ax.set_ylabel(r"Cell Length ($\mu$m)")

    # Disable top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot the transformed mutant data for each genotype
    for p, l, c in zip(PREFIXES, LABELS, COLORS):
        plot_processed_mutant_data(cell_type, p, l, c)
        
    # Add a legend and save the figure
    ax.legend()
    plt.savefig(f"{PATH}/{cell_type}-data-processed.pdf", format ='pdf')
    print(f"Generated figure '{PATH}/{cell_type}-data-processed.pdf'")
    ax.clear()


#############################################################
# PLOT #3: Visualization of binned position vs. length data #
#############################################################

# Plot binned data (see modules.py for an explanation of nbins, max_position)
def plot_binned_data(cell_type="trichoblast", nbins=20, max_position=500):

    # Initialize the plot
    ax = plt.subplot(111)
    ax.set_xlabel(r"Distance from Quiescent Center ($\mu$m)")
    ax.set_ylabel(r"Cell Length ($\mu$m)")
    ax.set_ylim((0, 60))
    ax.set_yticks(np.arange(0, 70, 10))
    ax.set_xlim((0, 500))

    # Disable top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot the binned mutant data with 20 bins
    for prefix, label, color in zip(PREFIXES, LABELS, COLORS):

        # Filter data to only include positions between 0um and 500um
        data, fit, se = get_mutant_data(cell_type, prefix, 10, 12)
        data = data[data[:, 0] < 500]

        # Generate the binned data, label midpoints, and steps
        means, errors = get_binned_data(data, nbins=nbins, max_position=max_position)
        mid, step = max_position / (2 * nbins), max_position / nbins
        ps = np.linspace(mid, (step * nbins) + mid, nbins)

        # Plot the means and standard errors
        plt.scatter(ps, means, alpha=1, zorder=3, color=color, label=label)
        plt.errorbar(ps, means, yerr=errors, color=color, fmt ="o", lw=2, alpha=0.6)
        
    # Configure the display and draw the plot
    ax.legend()
    plt.savefig(f"{PATH}/{cell_type}-data-binned.pdf", format ='pdf')
    print(f"Generated figure '{PATH}/{cell_type}-data-binned.pdf'")
    ax.clear()
