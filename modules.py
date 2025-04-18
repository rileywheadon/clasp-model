# This file contains various data processing functions used in the other scripts
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import curve_fit

# We use the Okabe & Ito colour palette (https://siegal.bio.nyu.edu/color-palette/)
# for all visualizations in this project to ensure our plots are accessible
OKABE_ITO = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7"
]

# This variable sets the folder in which plots are saved ('/img' by default)
PATH = "img"


# Compute the root-mean-squared error for a model
# - predicted (np.ndarray): The model predictions
# - observed (np.ndarray): The original data
@njit
def RMSE(predicted, observed):
    residuals = predicted - observed
    rss = np.sum(np.square(residuals))
    return np.sqrt(rss / np.size(observed))
    

# Compute the AICc for a given model
# - n (int): the number data points
# - k (int): the number of independent parameters
# - rmse (float): the root-mean-squared error of the model
def AICc(n, k, rmse):
    return (2 * k) + (n * rmse**2) + (2 * k ** 2 + 2 * k)/(n - k - 1)

   
# Load and transform BES1 data 
def get_bes1_data():

    # Load data and sort by position
    data = pd.read_csv(f"data/bes1.csv").to_numpy()
    data = data[data[:, 0].argsort()]

    # Bound data between 0um and 1000um
    data = data[data[:, 0] > 0]
    data = data[data[:, 0] < 1000]

    # Normalize data to a maximum of 1
    data[:, 1] = data[:, 1] / np.max(data[:, 1])
    return data


# Load and transform cell length data from mutants
# - cell_type (str): either "trichoblast" or "atrichoblast"
# - mutant (str): one of "WT" (wild type), "BC" (brinCLASPpro), or "C1" (clasp-1)
# - diameter (float): presumed cell diameter in microns (um)
# - impute_threshold (int): impute cell length data up to this position
def get_mutant_data(cell_type, mutant, diameter=10, impute_threshold=12):

    # Load the data into a Pandas dataframe and set "Cell Position" as the index
    raw = pd.read_csv(f"data/{cell_type}-areas.csv").set_index("Cell Position")

    # Compute cell lengths based on the presumed diameter
    columns = [c for c in raw.columns if c.startswith(mutant)]
    df_lengths = raw.filter(columns).div(diameter)

    # Create a dataframe containing the mean cell length for each cell number
    means = df_lengths.mean(axis = 1)
    df_means = pd.DataFrame({c : means for c in columns})

    # Compute cell position as the cumulative sum of the lengths
    df_positions = df_lengths.fillna(df_means[:impute_threshold]).cumsum(axis = 0)

    # Transform data into (position, length) tuples
    position_stack = df_positions.stack(-1, future_stack=True).to_numpy()
    length_stack = df_lengths.stack(-1, future_stack=True).to_numpy()
    data = np.stack((position_stack, length_stack), axis = 1)

    # Filter out tuples with NaN values and cells with positions above 1000um
    data = data[~np.isnan(data).any(axis = 1)]
    data = data[data[:, 0] < 1000]

    # Convert position to mm to prevent overflow, then fit an exponential curve 
    f = lambda x, A, B, C : A + (B * x) + (C * x ** 2)
    popt, pcov = curve_fit(f, (data[:, 0] / 1000), data[:, 1])
    fit = lambda x : f(x, *popt)

    # Compute the standard error proportional to length
    predicted = fit(data[:, 0] / 1000)
    observed = data[:, 1]
    std = np.std((predicted - observed) / observed) 
    se = std / np.sqrt(observed.size)

    # Return the transformed data, fitted curve, and vector of standard errors
    return data, fit, se


# Compute the binned averages of a dataset
# - data (np.ndarray): an array of (position, length) tuples
# - nbins (int): the number of bins
# - max_position (int): the maximum cell position (in microns) to use
@njit
def get_binned_data(data, nbins=20, max_position=500):

    # Unpack the data set
    positions, lengths = data[:, 0], data[:, 1]   
    indices = np.digitize(positions, np.linspace(0, max_position, nbins + 1))
    means, errors = np.empty(nbins), np.empty(nbins)

    # Iterate through the list of bins
    for i in range(nbins):
        group = lengths[indices == i + 1]

        # Return an array of zeros if any group sempty
        if group.size == 0: 
            return np.zeros(nbins), np.zeros(nbins)
            
        # Compute summary statistics on this bin
        means[i] = np.mean(group)
        errors[i] = np.std(group) / np.sqrt(group.size)

    return means, errors

