# This file contains the intracellular signalling model
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, sqrt, solve, lambdify
from scipy.optimize import minimize, direct
from modules import *
from numba import njit

# Sensible defaults for matplotlib
mpl.rcParams['figure.dpi'] = 360
mpl.rcParams['font.size'] = 18

# Load BES1 data and split into position and data components
data = get_bes1_data()
Z = data[:, 0]
BES1 = data[:, 1]

# Set intracellular constants KD, B1, A0, A1
KD = 10
B1 = 1.5
A0 = 62 * 0.65
A1 = 62 * 0.35

# Define BL functions
@njit
def bl_linear(z, params):
    c0, c1, n2, n3, n4 = params
    return c0 + c1 * z

@njit
def bl_hill1(z, params):
    c0, c1, c2, n3, n4  = params 
    return c0 + c1 * z / (1 + c2 * z)

@njit
def bl_hill2(z, params):
    c0, c1, c2, n3, n4  = params 
    return c0 + (c1 * z**2 / (1 + c2**2 * z**2))

@njit
def bl_full(z, params):
    c0, c1, c2, c3, c4 = params 
    term1 = c1 * z / (1 + c2 * z)
    term2 = c3 * z**2 / (1 + c4**2 * z**2)
    return c0 + term1 + term2


# Define RB function
@njit
def bound_receptors(z, bl_function, b0, b1, params):

    # Get the Bz curve
    Bz = bl_function(z, params)

    # Compute phi and psi (intermediate parameters)
    phi = (A0 + (A1 * b0)) / (1 + (A1 * b1))
    psi = KD / (1 + (A1 * b1))

    # Compute the discriminant and RB using mass balance
    discriminant = np.sqrt((phi + psi + Bz)**2 - (4 * phi * Bz))
    return ((phi + psi + Bz) - discriminant) / 2


# Fit a BL model to the data
#  - initial_params is a list of initial parameters to the model
#  - bl_function is the BL function being fitted
def fit_model(initial_params, bl_function, data = BES1, b1 = B1):

    # Ensure that all parameters are non-negative
    bounds = [(0, None)] * 6

    # Define the error function
    def RSS(params):

        # Compute error from BES1 data 
        b0, *bl_params = params
        predicted_bes1 = bound_receptors(Z, bl_function, b0, b1, bl_params)
        error_bes1 = np.sum((predicted_bes1 - data) ** 2)

        # Compute CLASP and RT 
        predicted_clasp = b0 - (b1 * predicted_bes1)
        predicted_rt = A0 + (A1 * predicted_clasp)

        # Compute error from mean CLASP of 1
        error_clasp = len(data) * ((np.mean(predicted_clasp) - 1) ** 2)

        # Return the sum of the errors
        return error_bes1 + error_clasp

    fit = minimize(RSS, initial_params, bounds=bounds, method='L-BFGS-B')
    return fit.x, fit.fun, fit.success


# Compute the AIC with a small sample size correction
#  - rss is the residual sum of squares
#  - n is the number of data points
#  - k is the number of parameters
def AICc(rss, n, k):
    aic = n * np.log(rss / n) + 2 * k
    correction = (2 * k * (k + 1)) / (n - k - 1)
    return aic + correction


# Compute confidence intervals for the parameters using a parametric bootstrap
#  - initial_params is a list of initial parameters, passed to fit_model
#  - bl_function is the BL function being fitted
#  - n_bootstrap is the number of bootstrap simulations (default: 1000)
#  - alpha is the significance level for the confidence interval (default: 0.05)
def bootstrap_CI(initial_params, bl_function, n_bootstrap = 10, alpha = 0.05, b1 = B1):

    # Fit to original data
    best_params, rss, success = fit_model(initial_params, bl_function)

    # Get the best paramers and the residuals
    b0, *best_bl_params = best_params
    residuals = BES1 - bound_receptors(Z, bl_function, b0, b1, best_bl_params)
    bootstrap_estimates = np.zeros((n_bootstrap, len(initial_params)))

    for i in range(n_bootstrap):

        # Resample residuals with replacement
        resampled_residuals = np.random.choice(
            residuals,
            size = len(residuals),
            replace = True
        )

        y_bootstrap = bound_receptors(
            Z,
            bl_function,
            b0,
            b1,
            best_bl_params
        ) + resampled_residuals

        try:

            p_bootstrap, _, success = fit_model(
                best_params,
                bl_function,
                data = y_bootstrap
            )

            if success:
                bootstrap_estimates[i, :] = p_bootstrap

        except:
            print("An error occurred in the bootstrap.")
            continue

    lower = np.percentile(bootstrap_estimates, 100 * alpha / 2, axis=0)
    upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2), axis=0)

    return best_params, rss, (lower, upper), bootstrap_estimates



# Generate RB, CLASP, and RT functions from a fittted model
#  - bl_function is the nam eof a B(z) form
#  - params is a tuple of parameters
#  - mutant is a string ("Wild Type", "BRIN-CLASP", or "CLASP-1")
def get_functions(bl_function, params, mutant, b1 = B1):

    # Unpack the parameters and compute the bound receptors
    b0 = params[0]
    bl_params = params[1:]

    # Override parameter values if simulating a mutant
    if mutant == "BRIN-CLASP":
        b1 = 0
    elif mutant == "CLASP-1":
        b0 = 0
        b1 = 0

    # Compute RB, CLASP and RT 
    fRB = njit(lambda z : bound_receptors(z / 1000, bl_function, b0, b1, bl_params))
    fC = njit(lambda z : b0 - (b1 * fRB(z)))
    fRT = njit(lambda z : A0 + (A1 * fC(z)))

    return [fRB, fC, fRT]


# Run the intracellular model
def intracellular_model():

    # Define the models
    bl_models = {
        'Linear': {
            'function': bl_linear,
            'initial_params': [1, 1, 1, 1, 1, 1],
            'param_names': ['b0', 'c0', 'c1'],
            'k': 3
        },
        'Hill (1)': {
            'function': bl_hill1,
            'initial_params': [1, 1, 1, 1, 1, 1],
            'param_names': ['b0', 'c0', 'c1', 'c2'],
            'k': 4
        },
        'Hill (2)': {
            'function': bl_hill2,
            'initial_params': [1, 1, 1, 1, 1, 1],
            'param_names': ['b0', 'c0', 'c1', 'c2'],
            'k': 4
        },
        'Full': {
            'function': bl_full,
            'initial_params': [1, 1, 1, 1, 1, 1],
            'param_names': ['b0', 'c0', 'c1', 'c2', 'c3'],
            'k': 6
        },
    }

   # Loop over the models
    for name, model in bl_models.items():

        # Run bootstrap_CI to get the best fitting parameters, RSS, lower and upper
        best_fit, rss, (lower, upper), samples = bootstrap_CI(
            model['initial_params'],
            model['function']
        )

        print(f"\n\nModel: {name}")
        print(f"RSS:   {rss:.4f}")
        print(f"AICc:  {AICc(rss, len(BES1), model['k']):.2f}\n")
        
        # Print parameter estimates with confidence intervals
        print(f"{'Param':<6} {'Estimate':>10} {'95% CI':>24}")
        for pname, estim, lo, hi in zip(model['param_names'], best_fit, lower, upper):
            bounds = f"({lo:.4f}, {hi:.4f})"
            print(f"{pname:<6} {estim:>10.4f} {bounds:>24}")

        # Add the model to bl_models list
        bl_models[name]['best_params'] = best_fit

        # Get RB, CLASP, and RB functions for all mutants and add them to bl_models
        for mutant in ['Wild Type', 'BRIN-CLASP', 'CLASP-1']:
            bl_models[name][mutant] = get_functions(model['function'], best_fit, mutant)

    return bl_models

# Uncomment for testing.
# intracellular_model()


###########################################################
# PLOT #1: Fitted BL models compared to experimental data #
###########################################################


# - bl_models is the list of models returned by intracellular_model()
def plot_bl_functions(bl_models):

    # Get the RB functions for each distribution
    rb_linear = bl_models['Linear']['Wild Type'][0]
    rb_hill1  = bl_models['Hill (1)']['Wild Type'][0]
    rb_hill2  = bl_models['Hill (2)']['Wild Type'][0]
    rb_full   = bl_models['Full']['Wild Type'][0]

    # Initialize the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim((0, 500))
    ax.set_xlabel(r"Position ($\mu\text{m}$)")
    ax.set_ylim((0, 1.05))
    ax.set_ylabel(r"Bound BRI1 Receptors ($\text{nmol L}^{-1}$)")

    # Plot the raw BES1 data
    styles = {'color':'k', 's':50, 'alpha':0.5, 'edgecolor':'none'}
    ax.scatter(Z * 1000, BES1, label="Experimental Data", **styles)

    # Remove the spines for clearer visuals 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot all four of the models
    rb_funcs = [rb_linear, rb_hill1, rb_hill2, rb_full]
    labels = ["Linear", "Hill (1)", "Hill (2)",  "Full"]
    colors = [OKABE_ITO[i] for i in [0, 5, 6, 7]]
    
    for f, l, c in zip(rb_funcs, labels, colors):
        vP = np.linspace(0, 500, 501)
        plt.plot(vP, f(vP), lw = 3, color = c, label = l)

    # Add a legend and save the plot 
    ax.legend()
    plt.savefig(f"{PATH}/intracellular-bl-functions.pdf", format ='pdf')
    print(f"\nGenerated figure '{PATH}/intracellular-bl-functions.pdf'")
    ax.clear()


########################################################################
# PLOT #2: Optimal intracellular signalling model simulated on mutants #
########################################################################


# - bl_models is the list of models returned by intracellular_model()
# - name is the name of one model ("Linear", "Hill (1)", "Hill (2)", "Full")
def plot_intracellular_signalling(bl_models, name):

    # Set the correct BL model
    model = bl_models[name]

    # Initialize the figure
    fig, axs = plt.subplots(2, 2, sharex = True, tight_layout = True, figsize = (10,10)) 
    ((ax1, ax2), (ax3, ax4)) = axs

    # Set axis titles and
    fig.supxlabel(r"Position ($\mu\text{m}$)")
    ax1.set_title(r"$R_B$ ($\text{nmol L}^{-1}$)")
    ax2.set_title(r"CLASP (a.u.)")
    ax3.set_title(r"$R_T$ ($\text{nmol L}^{-1}$)")
    ax4.set_title(r"BL ($\text{nmol L}^{-1}$)")

    # Set y-limits
    ax1.set_ylim((0, 1))
    ax2.set_ylim((-0.1, 2))
    ax3.set_ylim((0, 80))  
    ax4.set_ylim((0, 1.2))

    # Set x-limits
    ax1.set_xlim((0, 500))
    ax2.set_xlim((0, 500))
    ax3.set_xlim((0, 500))
    ax4.set_xlim((0, 500))

    # Plot the intracellular signalling model for all three mutants
    functions = [model['Wild Type'], model['BRIN-CLASP'], model['CLASP-1']]
    labels = ['Wild Type', 'BRIN-CLASP', 'CLASP-1']
    colors = OKABE_ITO[:3]

    for funcs, label, color in zip(functions, labels, colors):
        (fRB, fC, fRT) = funcs 
        vP = np.linspace(0, 500, 501)
        ax1.plot(vP, fRB(vP), lw = 3, color = color, label = label)
        ax2.plot(vP,  fC(vP), lw = 3, color = color)
        ax3.plot(vP, fRT(vP), lw = 3, color = color)

    # Plot the BL function on the bototm-right axis
    b0, *bl_params = model['best_params']
    ax4.plot(vP * 1000, model['function'](vP, bl_params), lw = 3, color = OKABE_ITO[0])

    # Add a legend and save the figure
    fig.legend(bbox_to_anchor=(-0.02, 0.14, 0.967, 0.15), loc="lower right")
    plt.savefig(f"{PATH}/intracellular-mutants.pdf", format="pdf")
    print(f"Generated figure '{PATH}/intracellular-mutants.pdf'")
    fig.clear()

