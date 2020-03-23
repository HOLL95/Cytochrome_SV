import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import copy
from single_e_class_unified import single_electron
from harmonics_plotter import harmonics
param_list={
        "E_0":0.2, #Midpoint potnetial (V)
        'E_start': -0.1, #Sinusoidal input minimum (or starting) potential (V)
        'E_reverse': 0.5, #Sinusoidal input maximum potential (V)
        'omega':10,   #frequency Hz
        "original_omega":10, #Nondimensionalising value for frequency (Hz)
        'd_E': 300e-3,   #ac voltage amplitude - V
        'area': 0.07, #electrode surface area cm^2
        'Ru': 100.0,  #     uncompensated resistance ohms
        'Cdl': 1e-5, #capacitance parameters
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-10,   # surface coverage per unit area
        "original_gamma":1e-10,        # Nondimensionalising cvalue for surface coverage
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.5, #(Symmetry factor)
        'phase' : 3*(math.pi/2),#Phase of the input potential
        "cap_phase":3*(math.pi/2),
        'sampling_freq' : (1.0/400),
    }
likelihood_options=["timeseries", "fourier"]
simulation_options={
        "no_transient":False,
        "experimental_fitting":False,
        "method": "sinusoidal",
        "likelihood":"timeseries",
        "label": "MCMC",
        "optim_list":["E_0", "k_0", "Ru", "Cdl", "alpha"]
    }
other_values={
        "filter_val": 0.5,
        "harmonic_range":list(range(3,9,1)),
        "num_peaks":30,
    }
param_bounds={
        'E_0':[0.2, 0.3],
        'Ru': [0, 1e3],
        'Cdl': [0,1e-4],
        'k_0': [50, 1e3],
        "alpha":[0.4, 0.6],
    }
SV_test=single_electron(file_name=None, dim_parameter_dictionary=param_list, simulation_options=simulation_options, other_values=other_values, param_bounds=param_bounds)
