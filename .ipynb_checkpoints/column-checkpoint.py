import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import itertools
from numba import njit, prange
from sympy import symbols, sqrt, solve, lambdify
from scipy.optimize import minimize, direct
from modules import *

# Define the hyperparameter sets for each mutant
WT_PARAMS = (0.03, 1.416, 1.036, 8.949)
BC_PARAMS = (0.03, 1.416, 0.000, 8.949)
C1_PARAMS = (0.03, 0.000, 0.000, 8.949)

# Define additional configuration information for the simulation
STEP = 0.002
MAX_CELLS = 2000
MAX_TIME = 20
MAX_SIZE = 150
VT = np.arange(0, MAX_TIME + STEP, STEP) 

# Creates the CLASP and bound receptor functions for the model
def setup(params):

    # Define parameters and variables
    a, b0, b1, Kd = params
    C, RT, RB, B, P = symbols('C RT RB B P')
            
    # Solve the system at quasi-steady state
    system = [
        b0 - b1 * RB - C,
        62 * (0.65 + 0.35 * C) - RT,
        (1/2) * ((B + RT + Kd) - sqrt((B + RT + Kd)**2 - 4 * RT * B)) - RB,
        a + ((1 - a) * P**2)/(500**2 + P**2) - B
    ]

    # Compute the steady states, then return only the CLASP and RB functions
    steady_states = solve(system, [C, RT, RB, B], dict = True)[-1]
    return [njit(lambdify(P, steady_states[C])), njit(lambdify(P, steady_states[RB]))]

@njit
def simulate_step(params, fRB, fC, L, P, D, i):

    # Unpack the parameters, create lambda functions
    n = 10
    m, g0, g1, g2, d = params
    dL = lambda l, p: ((g0 + g1 * fRB(p)) * l - g2 * fC(p)) * STEP
    dD = lambda l: (1 - ((l ** n) / ((d ** n) + (l ** n)))) * STEP
    
    # Unpack the data from the previous and current row
    L0, P0, D0 = L[i-1, :], P[i-1, :], D[i-1, :]
    L1, D1 = L[i, :], D[i, :]

    # Iterate through the (i-1)-th row and update
    j, k = 0, 0
    while L0[j] > 0:

        # Handle the division case
        if D0[j] >= 1 and L0[j] > m:
            D1[k], D1[k+1] = 0, 0
            L1[k], L1[k+1] = L0[j]/2, L0[j]/2
            k += 1

        # Handle the growth case
        elif L0[j] < MAX_SIZE:
            D1[k] = D0[j] + dD(L0[j])
            L1[k] = L0[j] + dL(L0[j], P0[j])

        # Handle the differentiation case
        else:
            D1[k] = D0[j] + dD(L0[j])
            L1[k] = L0[j]
        
        k += 1
        j += 1

    # Update the position vector and return
    return L1, np.cumsum(L1), D1

# Process data from a simulation, compute error of model means from experimental means
@njit
def analyze_simulation(data):

    # Filter the data to remove points before t = 5 
    L, P, D = data
    L = L[int(5 / STEP):]
    P = P[int(5 / STEP):]

    # Then sample one point from every 200 time steps
    size = L.shape[0]
    L = L[np.arange(size) % 200 == 0, :]
    P = P[np.arange(size) % 200 == 0, :]

    # Flatten the L and P arrays
    L = L.flatten()
    P = P.flatten()

    # Create a new data array containing only nonzero lengths
    data = np.stack((P, L), axis = 1)
    data = data[(data[:, 1] > 4) & (data[:, 1] < MAX_SIZE)]
    return data

@njit
def simulate_root(params, fRB, fC, vT, max_cells, data):

    # Create arrays for cell lengths, positions, and division statuses
    L = np.zeros((vT.size, max_cells)) 
    P = np.zeros((vT.size, max_cells)) 
    D = np.zeros((vT.size, max_cells)) 

    # Set the initial lengths and positions
    L[0, :10] = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    P[0, :] = np.cumsum(L[0, :])

    # Run the simulation loop
    for i in np.arange(1, vT.size):
        L[i, :], P[i, :], D[i, :] = simulate_step(params, fRB, fC, L, P, D, i)

    # Simplify the predictions by running analyze_simulation, then check for failed simulations
    predicted = analyze_simulation((L, P, D))
    if predicted.size == 0:
        return predicted, 1000
    
    # Compute the rmse and return
    observed = np.interp(predicted[:, 0], data[:, 0], data[:, 1])
    rmse = np.sqrt((1 / observed.size) * np.sum((predicted[:, 1] - observed) ** 2))
    return predicted, rmse

# Run a simulation of all three mutants given a set of parameters
@njit
def simulate_mutants(params, fWTRB, fWTC, fBCRB, fBCC, fC1RB, fC1C, datasets):

    # Unpack the datasets array
    wt_data, bc_data, c1_data = datasets

    # Simulate each of the mutants individually and compute their error from observations
    wt_model, wt_rmse = simulate_root(params, fWTC, fWTRB, VT, MAX_CELLS, wt_data)
    bc_model, bc_rmse = simulate_root(params, fBCC, fBCRB, VT, MAX_CELLS, bc_data)
    c1_model, c1_rmse = simulate_root(params, fC1C, fC1RB, VT, MAX_CELLS, c1_data)

    # Compute the rmse and package up the model results
    rmse = wt_rmse + bc_rmse + c1_rmse
    models = [wt_model, bc_model, c1_model]
    print(params)
    print(rmse)
    return models, rmse

def fit_model():

    # Set parameter bounds
    bounds = [(7, 14), (0.4, 0.8), (2, 7), (1, 4), (15, 25)]
    cost = lambda params : simulate_mutants(params, fWTRB, fWTC, fBCRB, fBCC, fC1RB, fC1C, DATASETS)[-1]

    # Find the parameters of best fit
    fit = direct(
        func = cost, 
        bounds = bounds,
        maxiter = 10000
    )

    # Run a simulation with the optimal parameters
    models, rmse = simulate_mutants(fit.x, fWTRB, fWTC, fBCRB, fBCC, fC1RB, fC1C, DATASETS)
    m, g0, g1, g2, d = fit.x

    # Log the simulation
    print("Success: ", fit.success, fit.message)
    print(f"Params: {m:.3e}, {g0:.3e}, {g1:.3e}, {g2:.3e}, {d:.3e}")
    print("Error: ", rmse)
        
    return models, rmse, fit.x

# Unpack the setup tuples to get the intracellular functions
fWTRB, fWTC = setup(WT_PARAMS)
fBCRB, fBCC = setup(BC_PARAMS)
fC1RB, fC1C = setup(C1_PARAMS)

# Get the fit for the experimental data
vP = np.linspace(0, 1000, 1001)
wt_fit, wt_se = get_mutant_data("trichoblast", "WT-", 10, 12)[1:]
bc_fit, bc_se = get_mutant_data("trichoblast", "BC-", 10, 12)[1:]
c1_fit, c1_se = get_mutant_data("trichoblast", "C1-", 10, 12)[1:]

wt_data = np.stack((vP, wt_fit(vP / 1000)), axis = 1)
bc_data = np.stack((vP, bc_fit(vP / 1000)), axis = 1)
c1_data = np.stack((vP, c1_fit(vP / 1000)), axis = 1)

DATASETS = [wt_data, bc_data, c1_data]
ERRORS = [wt_se, bc_se, c1_se]

# Run the simulation
result, rmse, optimal = fit_model()

# Plot the results
plt.rcParams['figure.dpi'] = 360
plt.rcParams['font.size'] = 18
plt.rcParams['figure.figsize'] = (10, 10)

def plot_position_length(result):

    vP = np.linspace(0, 1000, 1001)

    # Create a scatterplot
    ax = plt.subplot(111)
    ax.set_xlabel(r"Position ($\mu$m)")
    ax.set_ylabel(r"Length ($\mu$m)")
    ax.set_title("Cell Column Model vs. Experimental Observations")
    ax.set_xlim((0, 1000))
    ax.set_ylim((0, 150))
    ax.set_yticks(np.arange(0, 180, 30))

    # Helper function to plot the model results for a mutant
    def plot_mutant(model, data, se, label, color):

        # Fit a quadratic curve to the model
        f = lambda x, A, B, C : A + (B * x) + (C * x ** 2)
        popt, pcov = curve_fit(f, (model[:, 0] / 1000), model[:, 1])
        fit = lambda x : f(x, *popt)

        # Then plot the model curve as well as the points themselves
        ax.plot(vP, fit(vP / 1000), "--", label = f"{label} Model", color = color, lw = 2)
        ax.scatter(model[:, 0], model[:, 1], alpha = 0.3, color = color, s = 10, edgecolor = "none")

        # Plot the experimental observations
        ci_low, ci_high = (1 - 1.96 * se), (1 + 1.96 * se)
        ax.plot(data[:, 0], data[:, 1], label = label, color = color, lw = 2)
        ax.fill_between(data[:, 0], data[:, 1] * ci_low, data[:, 1] * ci_high, color = color, alpha = 0.4)

    labels = ["Wild Type", "BRIN-CLASP", "CLASP-1"]
    colors = OKABE_ITO[:3]

    for m, d, e, l, c in zip(result, DATASETS, ERRORS, labels, colors):
        plot_mutant(m, d, e, l, c)

    plt.legend()
    plt.savefig(f"column-position-length.png")

plot_position_length(result)