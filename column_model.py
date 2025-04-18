# This file contains the cell column model
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import itertools
from numba import njit, prange
from sympy import symbols, sqrt, solve, lambdify
from scipy.optimize import minimize, direct
from modules import *


# Define additional configuration information for the simulation
STEP = 0.002
MAX_CELLS = 5000
MAX_TIME = 15
MAX_SIZE = 150
MAX_POSITION = 500
NBINS = 20
VT = np.arange(0, MAX_TIME + STEP, STEP)

# Sensible defaults for matplotlib
plt.rcParams['figure.dpi'] = 360
plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = (12, 8)


# Creates the CLASP and bound BRI1 receptor functions for the model given parameters
def setup(params, mutant):

    # Define parameters and variables
    a0, a1, b0, b1, kd = params
    C, RT, RB, B, P = symbols('C RT RB B P')
            
    # Override parameter values based on mutant
    if mutant == "BRIN-CLASP":
        b1 = 0
    if mutant == "CLASP-1":
        b0 = 0
        b1 = 0

    # NOTE: This system uses a linear BL function
    system = [
        b0 - b1 * RB - C,
        62 * (0.65 + 0.35 * C) - RT,
        (1/2) * ((B + RT + kd) - sqrt((B + RT + kd)**2 - 4 * RT * B)) - RB,
        a0 + ((1 - a0) * P / 1000) - B
    ]

    # Compute the steady states, then return only the CLASP and RB functions
    steady_states = solve(system, [C, RT, RB, B], dict = True)[-1]
    return [njit(lambdify(P, steady_states[C])), njit(lambdify(P, steady_states[RB]))]


# Simulate one time step of the cell column model
@njit
def simulate_step(params, fRB, fC, L, P, D, i):

    # Unpack the parameters, create lambda functions
    n = 10
    m, g0, g1, d0, d1, d2 = params
    dL = lambda l, p: ((g0 + g1 * fRB(p)) * l) * STEP
    hill = lambda l : (l ** n) / ((d0 ** n) + (l ** n))
    dD = lambda l, p: (1 + d1 * fC(p) - d2 * fC(p) ** 2) * (1 - hill(l)) * STEP
    
    # Unpack the data from the previous and current row
    L0, P0, D0 = L[i-1, :], P[i-1, :], D[i-1, :]
    L1, D1 = L[i, :], D[i, :]

    # Iterate through the (i-1)-th row and update
    j, k = 0, 0
    while j < MAX_CELLS and k < MAX_CELLS and L0[j] > 0:

        # Handle the division case
        if D0[j] >= 1 and L0[j] > m:
            D1[k], D1[k+1] = 0, 0
            L1[k], L1[k+1] = L0[j]/2, L0[j]/2
            k += 1

        # Handle the growth case
        elif L0[j] < MAX_SIZE:
            D1[k] = D0[j] + dD(L0[j], P0[j])
            L1[k] = L0[j] + dL(L0[j], P0[j])

        # Handle the differentiation case
        else:
            D1[k] = D0[j]
            L1[k] = L0[j]
        
        k += 1
        j += 1

    # Update the position vector and return
    return L1, np.cumsum(L1), D1


# Process data from a simulation, compute error of model means from experimental means
@njit
def analyze_simulation(data):

    # Unpack the data from the simulation, filter the data to remove points before t=5
    (L, P, D) = data
    L = L[int(5 / STEP):]
    P = P[int(5 / STEP):]

    # Then sample one point from every 100 time steps
    size = L.shape[0]
    L = L[np.arange(size) % 100 == 0, :]
    P = P[np.arange(size) % 100 == 0, :]

    # Flatten the L and P arrays
    L = L.flatten()
    P = P.flatten()

    # Only include nonzero lengths, then filter positions above 500um
    tups = np.stack((P, L), axis = 1)
    tups = tups[(tups[:, 1] > 4) & (tups[:, 1] < MAX_SIZE) & (tups[:, 0] < 500)]

    # Return the binned data
    return get_binned_data(tups)


# Run a single simulation of a root
@njit
def simulate_root(params, fRB, fC, observed):

    # Create arrays for cell lengths, positions, and division statuses
    L = np.zeros((VT.size, MAX_CELLS)) 
    P = np.zeros((VT.size, MAX_CELLS)) 
    D = np.zeros((VT.size, MAX_CELLS)) 

    # Set the initial lengths and positions
    L[0, :10] = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    P[0, :] = np.cumsum(L[0, :])

    # Run the simulation loop
    for i in np.arange(1, VT.size):
        L[i, :], P[i, :], D[i, :] = simulate_step(params, fRB, fC, L, P, D, i)

    # Simplify the predictions by running analyze_simulation
    predicted, se = analyze_simulation((L, P, D))

    # Penalize failed simulations by setting the error to 1000
    if (predicted == np.zeros(NBINS)).all():
        return (L, P, D), predicted, 1000

    # Compute the rmse and return
    rmse = np.sqrt((1 / observed.size) * np.sum((predicted - observed) ** 2))
    return (L, P, D), predicted, rmse


# Run a simulation of all three mutants given a set of parameters
@njit
def simulate_mutants(params, fWTRB, fWTC, fBCRB, fBCC, fC1RB, fC1C, datasets):

    # Unpack the datasets array
    wt_data, bc_data, c1_data = datasets

    # Simulate each of the mutants individually and compute their error from observations
    wt_raw, wt_model, wt_rmse = simulate_root(params, fWTC, fWTRB, wt_data)
    bc_raw, bc_model, bc_rmse = simulate_root(params, fBCC, fBCRB, bc_data)
    c1_raw, c1_model, c1_rmse = simulate_root(params, fC1C, fC1RB, c1_data)

    # Compute the rmse and package up the model results
    rmse = wt_rmse + bc_rmse + c1_rmse
    raws = (wt_raw, bc_raw, c1_raw)
    models = (wt_model, bc_model, c1_model)
    return raws, models, rmse


# Fit the model given intracellular_params and the cell_type
def fit_model(intracellular_params, cell_type="trichoblast", modified=False):

    # Unpack the setup tuples to get the intracellular functions
    fWTRB, fWTC = setup(intracellular_params, "Wild Type")
    fBCRB, fBCC = setup(intracellular_params, "BRIN-CLASP")
    fC1RB, fC1C = setup(intracellular_params, "CLASP-1")

    # Get the binned experimental data
    DATASETS, ERRORS = [], []
    for prefix in ["WT-", "BC-", "C1-"]:
        data, fit, se = get_mutant_data(cell_type, prefix)
        means, errors = get_binned_data(data)
        DATASETS.append(means)
        ERRORS.append(errors)

    # Use the d2 parameter if modified=True
    d2_max = 3 if modified else 0.000000001

    # Set parameter bounds
    bounds = [
        (5, 25),     # m
        (0, 1),      # g0
        (0, 10),     # g1
        (15, 25),    # d0
        (0, 3),      # d1
        (0, d2_max)  # d2
    ]

    # Define the cost function
    cost = lambda params : simulate_mutants(
        params,
        fWTRB,
        fWTC,
        fBCRB,
        fBCC,
        fC1RB,
        fC1C,
        DATASETS
    )[-1]

    # Find the parameters of best fit
    fit = direct(func = cost, bounds = bounds)

    # Run a simulation with the optimal parameters
    raws, models, rmse = simulate_mutants(
        fit.x,
        fWTRB,
        fWTC,
        fBCRB,
        fBCC,
        fC1RB,
        fC1C,
        DATASETS
    )

    m, g0, g1, d0, d1, d2 = fit.x

    # Log the simulation
    print("Success: ", fit.success, fit.message)
    print(f"Params: {m:.3e}, {g0:.3e}, {g1:.3e}, {d0:.3e}, {d1:.3e}, {d2:.3e}")
    print("Error: ", rmse)

    # Generate the visualizations
    file_prefix = f"{cell_type}-column-{'modified' if modified else 'original'}"
    plot_model_fit(models, DATASETS, ERRORS, file_prefix)
    plot_column_profile(raws, file_prefix)
    plot_division_histogram(raws, fit.x, file_prefix)


######################################################
# PLOT #1: Fitted cell column model compared to data #
######################################################

# Helper function to plot the model results for a mutant
def plot_mutant_fit(model, data, se, label, color):
    mid, step = MAX_POSITION / (2 * NBINS), MAX_POSITION / NBINS
    ps = np.linspace(mid, (step * NBINS) + mid, NBINS)

    # Plot the model results
    styles = {'alpha': 1, 'zorder': 4, 'markersize': 10, 'marker': '*', 'color': color}
    plt.plot(ps, model, label = f"{label} Model", **styles)

    # Plot the experimental results
    plt.scatter(ps, data, alpha = 1, zorder = 3, color = color, label = label)

# Plot the fit of the models to the data
def plot_model_fit(models, datasets, errors, file_prefix):
    mpl.rcParams['figure.figsize'] = (10, 8)

    # Create a scatterplot
    ax = plt.subplot(111)
    ax.set_xlabel(r"Position ($\mu$m)")
    ax.set_ylabel(r"Length ($\mu$m)")
    ax.set_xlim((0, 500))
    ax.set_ylim((0, 60))

    # Disable the axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot the fit of each mutant
    labels = ["Wild Type", "BRIN-CLASP", "CLASP-1"]
    colors = OKABE_ITO[:3]
    
    for m, d, e, l, c in zip(models, datasets, errors, labels, colors):
        plot_mutant_fit(m, d, e, l, c)

    # Add a legend and save the figure
    plt.legend(prop = {'size': 18})
    plt.savefig(f"{PATH}/{file_prefix}-fit.pdf", format = 'pdf')
    print(f"Generated figure '{PATH}/{file_prefix}-fit.pdf'")
    ax.clear()



#################################
# PLOT #2: Cell column profiles #
#################################

# Find the last division location
def find_last_division(data):
    L, P, D = data
    divisions = P[np.where(D > 1)]
    return np.max(divisions), np.max(P)

# Filter the data
def filter_raw_data(data):

    # Only incldue times between 5 and 7
    L, P, D = data
    T = np.tile(np.arange(5, 7, STEP), (MAX_CELLS, 1))
    P = P[int(5 / STEP):int(7 / STEP)]

    # Remove cells above 500um
    tups = np.stack((T.T, P), axis = 2).reshape(-1, 2)
    tups = tups[tups[:, 1] < 500]
    return tups

# Plot the division zone
def plot_column_profile(raws, file_prefix):
    
    WT, BC, C1 = raws
    
    # Get the division zone sizes and root sizes
    wt_div, wt_size = find_last_division(WT)
    bc_div, bc_size = find_last_division(BC)
    c1_div, c1_size = find_last_division(C1)
    
    # Filter the data so it doesn't take a year to generate the plot
    WT = filter_raw_data(WT)
    BC = filter_raw_data(BC)
    C1 = filter_raw_data(C1)

    mpl.rcParams['figure.figsize'] = (10, 15)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, layout="constrained")
    ax1.set_xlim((5, 7))
    ax1.set_xticks([5, 6, 7])

    # Plot the division zone for the wild type
    ax1.set_title(rf"Wild Type ({wt_size:.0f}$\mu$m)")
    ax1.scatter(WT[:, 0], WT[:, 1], color=OKABE_ITO[0], s=1, edgecolor="none")
    ax1.set_ylim((0, 450))
    wt_line = ax1.hlines(wt_div, 0, MAX_TIME, colors='k', linestyles="dashed", lw=4)

    # Plot the division zone for the BRIN-CLASP mutant
    ax2.set_title(rf"BRIN-CLASP ({bc_size:.0f}$\mu$m)")
    ax2.scatter(BC[:, 0], BC[:, 1], color=OKABE_ITO[1], s=1, edgecolor="none")
    ax2.set_ylim((0, 450))
    ax2.hlines(bc_div, 0, MAX_TIME, colors=OKABE_ITO[0], linestyles="dashed", lw=4)

    # Plot the division zone for the CLASP-1 mutant
    ax3.set_title(rf"CLASP-1 ({c1_size:.0f}$\mu$m)")
    ax3.scatter(C1[:, 0], C1[:, 1], color=OKABE_ITO[2], s=1, edgecolor="none")     
    ax3.set_ylim((0, 450))
    ax3.hlines(c1_div, 0, MAX_TIME, colors=OKABE_ITO[0], linestyles="dashed", lw=4)

    # Add a legend and save the plot
    fig.legend([wt_line], ["Last Division"], loc='outside upper left')
    fig.supxlabel(r"Time")
    fig.supylabel(r"Position ($\mu$m)")
    fig.savefig(f"{PATH}/{file_prefix}-profile.pdf", format = 'pdf')
    print(f"Generated figure '{PATH}/{file_prefix}-profile.pdf'")
    fig.clear()



############################################
# PLOT #3: Histogram of division locations #
############################################

# Get all of the division points
def get_divisions(data, params):
    L, P, D = data
    return P[np.where((D > 1) & (L > params[0]))]


# Plot the division locations
def plot_division_histogram(raws, params, file_prefix):
    WT, BC, C1 = raws
    mpl.rcParams['figure.figsize'] = (10, 12)
    
    # Get the divisions for each mutant
    wt_divs = get_divisions(WT, params)
    bc_divs = get_divisions(BC, params)
    c1_divs = get_divisions(C1, params)

    # Generate the histograms
    fig, axs = plt.subplots(3, 1, sharex = True, layout = "constrained")
    fig.supxlabel(r"Position ($\mu$m)")
    fig.supylabel("Number of Divisions")
    
    # Plot the histogram for each mutant
    labels = ["Wild Type", "brinCLASPpro", "clasp-1"]
    divisions = [wt_divs, bc_divs, c1_divs]

    for ax, color, div, label in zip(axs, OKABE_ITO[:3], divisions, labels):
        ax.set_ylim((0, 60))
        ax.hist(div, bins=20, range=(0,400), color=color, alpha=0.5, edgecolor="k")
        ax.set_title(f"{label} Model")

    # Save the figure
    fig.savefig(f"{PATH}/{file_prefix}-histogram.pdf", format = 'pdf')
    print(f"Generated figure '{PATH}/{file_prefix}-histogram.pdf'")
    fig.clear()

    # Print summary statistics about division locations
    for name, divs in zip(labels, divisions):
        print(f"\n{name} Divisions ({file_prefix})")
        print(f" - Mean: {np.mean(divs):.2f}um")
        print(f" - Median: {np.median(divs):.2f}um")
        print(f" - Max: {np.max(divs):.2f}um")
        print(f" - Count: {np.size(divs):.0f}")


