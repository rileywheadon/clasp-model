import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import itertools
from numba import njit, prange
from sympy import symbols, sqrt, solve, lambdify
from scipy.optimize import minimize, direct
from modules import *

OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
PATH = "../paper"

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
        a + ((1 - a) * P / 1000) - B
    ]

    # Compute the steady states, then return only the CLASP and RB functions
    steady_states = solve(system, [C, RT, RB, B], dict = True)[-1]
    return [njit(lambdify(P, steady_states[C])), njit(lambdify(P, steady_states[RB]))]

# Define the hyperparameter sets for each mutant
# Currently using LINEAR model
WT_PARAMS = (0.001, 1.389, 0.895, 8.901)
BC_PARAMS = (0.001, 1.389, 0.000, 8.901)
C1_PARAMS  =(0.001, 0.000, 0.000, 8.901)

# Define additional configuration information for the simulation
STEP = 0.002
MAX_CELLS = 5000
MAX_TIME = 15
MAX_SIZE = 150
MAX_POSITION = 500
NBINS = 20
VT = np.arange(0, MAX_TIME + STEP, STEP) 

# Compute the binned averages of a dataset
@njit
def bin_data(data):
    positions, lengths = data[:, 0], data[:, 1]   
    indices = np.digitize(positions, np.linspace(0, MAX_POSITION, NBINS + 1))
    means, errors = np.empty(NBINS), np.empty(NBINS)
    for i in range(NBINS):
        
        group = lengths[indices == i + 1]

        # Send an error code if a bin is empty
        if group.size == 0: 
            return np.zeros(NBINS), np.zeros(NBINS)
            
        means[i] = np.mean(group)
        errors[i] = np.std(group) / np.sqrt(group.size)

    return means, errors

# Process the experimental data by getting the binned averages
def process_experimental_data(prefix):
    data, fit, se = get_mutant_data("trichoblast", prefix, 10, 12)
    return bin_data(data)

@njit
def simulate_step(params, fRB, fC, L, P, D, i):

    # Unpack the parameters, create lambda functions
    n = 10
    m, g0, g1, d1, s0, s1 = params
    dL = lambda l, p: ((g0 + g1 * fRB(p)) * l) * STEP
    dD = lambda l, p: (s0 + s1 * fC(p) - fC(p) ** 2) * (1 - ((l ** n) / ((d1 ** n) + (l ** n)))) * STEP
    
    # Unpack the data from the previous and current row
    L0, P0, D0 = L[i-1, :], P[i-1, :], D[i-1, :]
    L1, D1 = L[i, :], D[i, :]

    # Iterate through the (i-1)-th row and update
    j, k = 0, 0
    while L0[j] > 0:

        # Handle the division case, but only if D1 is not full
        if D0[j] >= 1 and L0[j] > m and k + 1 < MAX_CELLS:
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

    # Unpack the data from the simulation, filter the data to remove points before t = 5
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

    # Create a new data array containing only nonzero lengths, then filter positions above 500um
    tups = np.stack((P, L), axis = 1)
    tups = tups[(tups[:, 1] > 4) & (tups[:, 1] < MAX_SIZE) & (tups[:, 0] < 500)]

    # Return the binned data
    return bin_data(tups)

@njit
def simulate_root(params, fRB, fC, vT, max_cells, observed):

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
    predicted, se = analyze_simulation((L, P, D))
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
    wt_raw, wt_model, wt_rmse = simulate_root(params, fWTC, fWTRB, VT, MAX_CELLS, wt_data)
    bc_raw, bc_model, bc_rmse = simulate_root(params, fBCC, fBCRB, VT, MAX_CELLS, bc_data)
    c1_raw, c1_model, c1_rmse = simulate_root(params, fC1C, fC1RB, VT, MAX_CELLS, c1_data)

    # Compute the rmse and package up the model results
    rmse = wt_rmse + bc_rmse + c1_rmse
    raws = (wt_raw, bc_raw, c1_raw)
    models = (wt_model, bc_model, c1_model)

    print(params)
    print(rmse)
    return raws, models, rmse

# Unpack the setup tuples to get the intracellular functions
fWTRB, fWTC = setup(WT_PARAMS)
fBCRB, fBCC = setup(BC_PARAMS)
fC1RB, fC1C = setup(C1_PARAMS)

# Get the fit for the experimental data
vP = np.linspace(0, 1000, 1001)
wt_data, wt_se = process_experimental_data("WT-")
bc_data, bc_se = process_experimental_data("BC-")
c1_data, c1_se = process_experimental_data("C1-")
DATASETS = [wt_data, bc_data, c1_data]
ERRORS = [wt_se, bc_se, c1_se]


def fit_model():

    # Set parameter bounds
    bounds = [
        (5, 15),   # m
        (0, 1),    # g0
        (0, 10),   # g1
        (15, 25),  # d1
        (0, 1),    # s0
        (0, 3)     # s1
    ]
    
    cost = lambda params : simulate_mutants(params, fWTRB, fWTC, fBCRB, fBCC, fC1RB, fC1C, DATASETS)[-1]

    # Find the parameters of best fit
    fit = direct(
        func = cost, 
        bounds = bounds
    )

    # Run a simulation with the optimal parameters
    # OPTIMAL = [10.08230453, 0.5, 5.37037037, 24.44444444, 0.87037037, 1.48765432]
    raws, models, rmse = simulate_mutants(fit.x, fWTRB, fWTC, fBCRB, fBCC, fC1RB, fC1C, DATASETS)
    m, g0, g1, d1, s0, s1 = fit.x

    # Log the simulation
    print("Success: ", fit.success, fit.message)
    print(f"Params: {m:.3e}, {g0:.3e}, {g1:.3e}, {d1:.3e}, {s0:.3e}, {s1:.3e}")
    print("Error: ", rmse)
        
    return raws, models, rmse, fit.x

raws, models, rmse, optimal = fit_model()