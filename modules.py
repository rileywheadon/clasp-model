import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import curve_fit


# Find the residual sum of squares and then the RMSE
@njit
def RMSE(predicted, observed):
    residuals = predicted - observed
    rss = np.sum(np.square(residuals))
    return np.sqrt(rss / np.size(observed))
    

# Compute the AIC using the residual sum of squares
def AIC(k, predicted, observed):
    residuals = predicted - observed
    rss = np.sum(np.square(residuals))
    n = observed.size
    return (2 * k) + n * np.log(rss / n)


# Curve fitting helper that also returns the RMSE and SE
def fit_curve(f, data):
    popt, pcov = curve_fit(f, data[:, 0], data[:, 1])
    fit = lambda x : f(x, *popt)
    rmse = RMSE(fit(data[:, 0]), data[:, 1])
    print(popt)
    return fit, rmse
    

# Take the moving average of the BL precursor data
# Returns a vector of size vP
def moving_average(vP, data, n): 
    
    # If n = 0 do not take a moving average, just interpolate
    if n == 0:
        interpolates = np.interp(vP, data[:, 0], data[:, 1])
        return np.stack((vP, interpolates), axis = 1)

    # Otherwise take a moving average
    totals = np.zeros(vP.size)
    counts = np.zeros(vP.size)
    scale = (vP.size - 1) / 1000

    for row in data:
        left = max(0, int(row[0] - scale * n))
        right = min(vP.size, int(row[0] + scale * n))
        totals[left:right] = totals[left:right] + row[1]
        counts[left:right] = counts[left:right] + 1

    averages = np.divide(totals, counts)
    return np.stack((vP, averages), axis = 1)


# Get the BR signalling (a weighted average of CPD and ROT3)
# vP must be a vector of evenly spaced values between 0 and 1000
# Prescribe bias = 0.5, and moving average period n = 0
# Normalize to a maximum value of of 1
def get_bl(vP, bias = 0.5, n = 0):

    # Load the CPD and ROT3 data from their data files
    CPD = pd.read_csv('data/cpd.csv', header = None).to_numpy()
    ROT3 = pd.read_csv('data/rot3.csv', header = None).to_numpy()

    # Sort the CPD and ROT3 arrays by position
    CPD = CPD[CPD[:, 0].argsort()]
    ROT3 = ROT3[ROT3[:, 0].argsort()]

    # Take a moving average of the CPD and ROT3 data if necessary
    CPD = moving_average(vP, CPD, n)
    ROT3 = moving_average(vP, ROT3, n)
    
    # Compute the BL function and normalize it
    BL = (CPD[:, 1] * bias) + ROT3[:, 1] * (1 - bias)
    return np.stack((vP, BL / np.max(BL)), axis = 1)


# Get the position vs. time function
def get_position():

    # Load the three lines of cell data
    LINES = []
    for i in [1, 2, 3]:
        data = pd.read_csv(f'data/line{i}.csv', header = None).to_numpy()

        # Sort the data by position
        data = data[data[:, 0].argsort()]

        # Shift time to 0 and position to 150um
        data[:, 0] = data[:, 0] - data[0, 0]
        data[:, 1] = data[:, 1] - data[0, 1] + 150
        LINES.append(data)

    # Construct the position data and sort it
    POS = np.concatenate(LINES)
    POS = POS[POS[:, 0].argsort()]

    # Fit an exponential to the data
    f = lambda x, A, B, C : A + B * np.exp(C * x)
    g = lambda x, A, B, C : (1 / C) * np.log((x - A) / B)
    popt, pcov = curve_fit(f, POS[:, 0], POS[:, 1])

    # Get the time to position and position to time functions
    time_to_position = lambda x : f(x, *popt)
    position_to_time = lambda x : g(x, *popt)

    # Return the data and the functions
    return POS, time_to_position, position_to_time


# Data parsing function for BES1 data
def get_bes1_data():

    # Load and sort data
    data = pd.read_csv(f"data/bes1.csv").to_numpy()
    data = data[data[:, 0].argsort()]

    # Bound data between 0um and 1000um
    data = data[data[:, 0] > 0]
    data = data[data[:, 0] < 1000]

    # Normalize data to a maximum of 1
    data[:, 1] = data[:, 1] / np.max(data[:, 1])
    return data

# Data parsing function for mutant data
def get_mutant_data(cell_type, prefix, diameter, threshold):

    # Load in the raw data
    raw = (pd
        .read_csv(f"data/{cell_type}-areas.csv")
        .set_index("Cell Position")
    )

    # Assume a fixed cell diameter given in the parameters (measured in um)
    # Since images give a cross section of the cell, divide by the diameter
    columns = [c for c in raw.columns if c.startswith(prefix)]
    df_lengths = (raw
        .filter(columns)
        .div(diameter)
    )

    # Create a dataframe containing the mean cell length for each cell number
    means = df_lengths.mean(axis = 1)
    df_means = pd.DataFrame({c : means for c in columns})

    # Compute cell position as the cumulative sum of the lengths
    # Fill missing data points with the cell number mean up to the threshold cell
    df_positions = (df_lengths
        .fillna(df_means[:threshold])
        .cumsum(axis = 0)
    )

    # Transform data into (position, length) tuples
    position_stack = df_positions.stack(-1, dropna = False).to_numpy()
    length_stack = df_lengths.stack(-1, dropna = False).to_numpy()
    data = np.stack((position_stack, length_stack), axis = 1)

    # Filter out tuples with NaN values and values with positions above 1000um
    data = data[~np.isnan(data).any(axis = 1)]
    data = data[data[:, 0] < 1000]

    # Compute an exponential curve of best fit
    # Convert positions to mm to prevent overflow
    f = lambda x, A, B, C : A + (B * x) + (C * x ** 2)
    popt, pcov = curve_fit(f, (data[:, 0] / 1000), data[:, 1])
    fit = lambda x : f(x, *popt)

    # Compute the standard error proportional to length
    predicted = fit(data[:, 0] / 1000)
    observed = data[:, 1]
    std = np.std((predicted - observed) / observed) 
    se = std / np.sqrt(observed.size)
    return data, fit, se
