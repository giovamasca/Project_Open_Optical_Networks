from scipy.constants import c as speed_light
from scipy.constants import Planck as h_Plank
import numpy as np
from scipy.special import erfcinv # this is the function of interest for bit rate evaluation

from Project_Open_Optical_Networks.Core.parameters import *

##################### Constant Values #######################
phase_velocity = 2 / 3 * speed_light # velocity of the line is defined by light speed and a ratio
# n = 1.5, so c/n ~ (2/3)*c
#############################################################

def n_amplifier_evaluation(length):
    n_amplifier = int(np.floor(length/span_length)) # span length from parameters
    return n_amplifier
def latency_evaluation(length):
    latency = length / phase_velocity # obtained delay in a line
    return latency
def noise_generation(signal_power, length): # generates noise from length and power and a very low constant
    noise_power = noise_power_spectral_density * signal_power * length
    return noise_power

def dB_to_linear_conversion_power(dB_quantity):
    return np.power(10, dB_quantity/10)
def linear_to_dB_conversion_power(linear_quantity):
    return 10*np.log10(linear_quantity)

def bit_rate_evaluation(GSNR_lin, strategy):
    Rb = 0
    first_formula = 2 * Rs_symbol_rate / Bn_noise_band * (np.power(erfcinv(2 * BER_target), 2))
    second_formula = 14 / 3 * Rs_symbol_rate / Bn_noise_band * (np.power(erfcinv(3 / 2 * BER_target), 2))
    third_formula = 10 * Rs_symbol_rate / Bn_noise_band * (np.power(erfcinv(8 / 3 * BER_target), 2))
    if strategy == 'fixed_rate':
        if GSNR_lin >= first_formula:
            Rb = 100e9 # Gbps
    elif strategy == 'flex_rate':
        if GSNR_lin >= first_formula and GSNR_lin < second_formula:
            Rb = 100e9
        elif GSNR_lin >= second_formula and GSNR_lin < third_formula:
            Rb = 200e9
        elif GSNR_lin >= third_formula:
            Rb = 400e9
    elif strategy == 'shannon':
        Rb = 2 * Rs_symbol_rate * np.log2(1 + GSNR_lin * Rs_symbol_rate / Bn_noise_band)
    else:
        print('ERROR in strategy definition')
        exit(5)
    return Rb

def capacity_and_avarage_bit_rate(connections):
    capacity = np.nansum( [connections[i].bit_rate for i in range(0, len(connections))] )
    avarage_bit_rate = capacity / len(connections)
    return [capacity, avarage_bit_rate]

def alpha_from_dB_to_linear_value(alpha_in_dB):
    return alpha_in_dB/(20*np.log10(np.exp(1)))

