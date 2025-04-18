# This file contains the intracellular signalling model
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import itertools
from numba import njit, prange
from sympy import symbols, sqrt, solve, lambdify
from scipy.optimize import minimize, direct
from modules import *

# Sensible defaults for matplotlib
mpl.rcParams['figure.dpi'] = 360
mpl.rcParams['font.size'] = 18

# VP is a vector of positions from 0um to 1000um (the domain of the model)
VP = np.linspace(0, 1000, 1001)

# BES1 is an array of shape (N, 2) which contains the BES1 data from Vukasinovic et al.
BES1 = get_bes1_data()

# Define candidate BL functions
BL_LINEAR = njit(lambda a0, a1: a0 + ((1 - a0) * VP / 1000))
BL_HILL = njit(lambda a0, a1: a0 + ((1 - a0) * VP ** 2 / (a1 ** 2 + VP ** 2)))
BL_QUADRATIC = njit(lambda a0, a1: a0 + ((1 - a0) * VP ** 2 / (1000 ** 2)))


# Compute the bound receptors given the B, RT, and KD using the mass-balance equations
@njit
def boundReceptors(b, rt, kd):
    a = b + rt + kd
    return (a - np.sqrt(a ** 2 - (4 * rt * b))) / 2

@njit
def simulate(params, mutant, fBL):

    # Unpack the parameters
    a0, a1, b0, b1, kd = params

    # Override parameter values if simulating a mutant
    if mutant == "BRIN-CLASP":
        b1 = 0

    if mutant == "CLASP-1":
        b0 = 0
        b1 = 0

    # Initialize the BL, CLASP, RT, and RB vectors
    vB = fBL(a0, a1)
    vC = np.zeros(vB.size)
    vRT = np.zeros(vB.size)
    vRB = np.zeros(vB.size)
    
    # Set the initial conditions
    vC[0] = 1
    vRT[0] = 62 * (0.65 + 0.35 * vC[0])
    vRB[0] = boundReceptors(vB[0], vRT[0], kd)

    # Run the model by iterating through vB
    for i in range(1, vB.size):
        vC[i] = b0 - b1 * vRB[i-1]
        vRT[i] = 62 * (0.65 + 0.35 * vC[i])
        vRB[i] = boundReceptors(vB[i], vRT[i], kd)

    # Compute the RMSE of the simulation based on BRI1 concentration and BES1
    # NOTE: The RMSE is only relevant when fitting the wild type model
    predicted = np.interp(BES1[:, 0], VP, vRB)
    bes1_rmse = RMSE(predicted, BES1[:, 1])
    bri1_rmse = np.abs(np.mean(vRT[1:] - 62) / 62)
    rmse = bes1_rmse + bri1_rmse
    return (vC, vRT, vRB), rmse


# Fit the intracellular signalling model to experimental data
def fit_model(fBL, name, log = True):

    # Place bounds on the parameters
    bounds = [
        (0, 1),    # a0
        (0, 1000), # a1
        (0, 5),    # b0
        (0, 5),    # b1
        (7.5, 55)  # kd
    ]

    # Define the cost function
    cost = lambda params : simulate(params, "Wild Type", fBL)[1]

    # Find the parameters that yield the best fit
    fit = direct(
        func = cost, 
        bounds = bounds,
        eps = 0.01,
        maxfun = 50000,
        maxiter = 10000,
    )

    # Run a simulation with the optimal parameters
    (vC, vRT, vRB), rmse = simulate(fit.x, "Wild Type", fBL)
    a0, a1, b0, b1, kd = fit.x

    # Get the number of parameters (k) and the size of the BES1 data (n)
    k = 6 if name == "Hill" else 5
    n = BES1[:, 0].size

    # Log the results of model fitting if log = True
    if log: 
        print(f"\n{name} BL Model:")
        print(f" - Success: {fit.success}, {fit.message}")
        print(f" - Params: {[round(float(x), 3) for x in fit.x]}")
        print(f" - Error: {rmse:.3e}")
        print(f" - AICc: {AICc(n, k, rmse):.3e}")
        print(f" - Mean Bound BRI1 (RT): {np.mean(vRT):.3e}")

    # Return the fitted model
    return (vC, vRT, vRB), rmse, fit.x


###########################################################
# PLOT #1: Fitted BL models compared to experimental data #
###########################################################


def simulate_bl_functions():

    # Simulate the model with each candidate BL function
    model_linear = fit_model(BL_LINEAR, "Linear")
    model_hill = fit_model(BL_HILL, "Hill")
    model_quadratic = fit_model(BL_QUADRATIC, "Quadratic")

    # Initialize the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim((0, 1000))
    ax.set_xlabel(r"Position ($\mu\text{m}$)")
    ax.set_ylim((0, 1.05))
    ax.set_ylabel(r"Bound BRI1 Receptors ($\text{nmol L}^{-1}$)")

    # Plot the raw BES1 data
    styles = {'color':'k', 's':50, 'alpha':0.5, 'edgecolor':'none'}
    ax.scatter(BES1[:, 0], BES1[:, 1], label="Experimental Data", **styles)

    # Remove the spines for clearer visuals 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot all three of the models
    models = [model_linear, model_hill, model_quadratic]
    labels = ["Linear", "Hill", "Quadratic"]
    colors = OKABE_ITO[5:] 
    
    for m, l, c in zip(models, labels, colors):
        (vC, vRT, vRB), fit, rmse = m
        plt.plot(VP, vRB, lw=3, color=c, label=l)

    # Add a legend and save the plot 
    ax.legend()
    plt.savefig(f"{PATH}/intracellular-bl-functions.pdf", format ='pdf')
    print(f"\nGenerated figure '{PATH}/intracellular-bl-functions.pdf'")
    ax.clear()

    # Return the simulated models and BL functions
    return models


########################################################################
# PLOT #2: Optimal intracellular signalling model simulated on mutants #
########################################################################


# NOTE: Requires one of the fitted models (and its name) from simulate_bl_functions
def simulate_intracellular_signalling(model, name):

    # Set the correct BL function
    bl_func = None
    if name == "Linear": bl_func = BL_LINEAR
    if name == "Hill": bl_func = BL_HILL
    if name == "Quadratic": bl_func = BL_QUADRATIC

    # Simulate the intracellular signalling model on wild type, brinCLASP, and clasp-1  
    *WT, params = model
    BC = simulate(params, "BRIN-CLASP", bl_func)
    C1 = simulate(params, "CLASP-1", bl_func)

    # Initialize the figure
    fig, axs = plt.subplots(2, 2, sharex=True, tight_layout=True, figsize=(10,10)) 
    ((ax1, ax2), (ax3, ax4)) = axs

    # Set axis titles and
    fig.supxlabel(r"Position ($\mu\text{m}$)")
    ax1.set_title(r"$R_B$ ($\text{nmol L}^{-1}$)")
    ax2.set_title(r"CLASP (a.u.)")
    ax3.set_title(r"$R_T$ ($\text{nmol L}^{-1}$)")
    ax4.set_title(r"BL ($\text{nmol L}^{-1}$)")

    # Set y-limits
    ax1.set_ylim((0, 1))
    ax2.set_ylim((0, 2))
    ax3.set_ylim((0, 80))  
    ax4.set_ylim((0, 1))

    # Set x-limits
    ax1.set_xlim((0, 1000))
    ax2.set_xlim((0, 1000))
    ax3.set_xlim((0, 1000))
    ax4.set_xlim((0, 1000))

    # Plot the intracellular signalling model for all three mutants
    datasets = [WT, BC, C1]
    labels = ["Wild Type", "BRIN-CLASP", "CLASP-1"]
    colors = OKABE_ITO[:3]

    for data, label, color in zip(datasets, labels, colors):
        (vC, vRT, vRB), rmse = data
        ax1.plot(VP[1:], vRB[1:], lw=3, color=color, label=label)
        ax2.plot(VP[1:], vC[1:], lw=3, color=color)
        ax3.plot(VP[1:], vRT[1:], lw=3, color=color)

    # Plot the BL function on the bototm-right axis
    ax4.plot(VP, bl_func(params[0], params[1]), lw=3, color=OKABE_ITO[0])

    # Add a legend and save the figure
    fig.legend(bbox_to_anchor=(-0.02, 0.14, 0.967, 0.15), loc="lower right")
    plt.savefig(f"{PATH}/intracellular-mutants.pdf", format="pdf")
    print(f"Generated figure '{PATH}/intracellular-mutants.pdf'")
    fig.clear()

