import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import itertools
from numba import njit
from sympy import symbols, sqrt, solve, lambdify
from scipy.optimize import minimize, direct
from modules import *

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

PATH = "img"

class intracellularModel:

    def __init__(self, omega, btype):
        self.setup(btype)
        self.load(omega)

    def setup(self, btype):

        # Define parameters and variables
        a0, a1, b0, b1, kd = symbols('a0 a1 b0 b1 kd')
        C, RT, RB, B, P = symbols('C RT RB B P')

        # Adjust the BL equation depending on self.bl
        bfunc = 0
        match btype:
            case "Linear":
                bfunc = a0 + ((1 - a0) * P / 1000)
            case "Quadratic":
                bfunc = a0 + ((1 - a0) * P**2 / (1000**2))
            case "Hill":
                bfunc = a0 + ((1 - a0) * P**2 / (a1**2 + P**2))
            case "_":
                print(f"Error: bfunc {self.bl} is undefined.")
                
        # Solve the system at quasi-steady state
        system = [
            b0 - b1 * RB - C,
            62 * (0.65 + 0.35 * C) - RT,
            (1/2) * ((B + RT + kd) - sqrt((B + RT + kd)**2 - 4 * RT * B)) - RB,
            bfunc - B
        ]

        # Compute the quasi-steady states, then assign functions to member variables
        steady_states = solve(system, [C, RT, RB, B], dict = True)[0]
        params = (a0, a1, b0, b1, kd, P)

        self.c = njit(lambdify(params, steady_states[C]))
        self.rt = njit(lambdify(params, steady_states[RT]))
        self.rb = njit(lambdify(params, steady_states[RB]))
        self.b = njit(lambdify(params, steady_states[RB]))

    # Get BES1 signalling data and rescale based on omega
    def load(self, omega):
        self.BES1 = get_bes1_data()
        self.BES1[:, 1] *= omega

    # Cost function for model fitting using the RMSE error metric
    def cost(self, params):
        RMSE = lambda p, o : np.sqrt(np.square(p - o).mean())
        bes1_p = self.rb(*params, self.BES1[:, 0])
        bri1_p = self.rt(*params, np.linspace(0, 1000, 1001))
        bes1_o = self.BES1[:, 1]
        return RMSE(bes1_p, bes1_o) + RMSE(bri1_p.mean(), 62)

    # Fits the model to experimental data. Requires setup() and load() 
    def fit(self):
        bounds = [(0, 1), (0, 1000), (0, 5), (0, 5), (7.5, 55)]

        # Find the parameters of best fit
        fit = direct(
            func = self.cost, 
            bounds = bounds,
            eps = 0.01,
            maxfun = 100000,
            maxiter = 100000,
        )

        print(f" - Success: {fit.success}, {fit.message}")
        print(f" - Params: {[round(x, 3) for x in fit.x]}")
        print(f" - Error: {fit.fun}")
        print(f"{self.cost((0.001, 0, 1.389, 0.895, 8.901))}")

model = intracellularModel(1, "Linear")
model.fit()
