from scipy.constants import c as speed_light
import numpy as np
from scipy.special import erfcinv # this is the function of interest for bit rate evaluation

from Project_Open_Optical_Networks.Core.parameters import *

##################### Constant Values #######################
phase_velocity = 2 / 3 * speed_light # velocity of the line is defined by light speed and a ratio
# n = 1.5, so c/n ~ (2/3)*c
#############################################################

def n_amplifier_evaluation(length):
    n_amplifier = int(np.floor(length/span_length))
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
    first_formula = 2 * Rs / Bn * (np.power(erfcinv(2 * BER_target), 2))
    second_formula = 14 / 3 * Rs / Bn * (np.power(erfcinv(3 / 2 * BER_target), 2))
    third_formula = 10 * Rs / Bn * (np.power(erfcinv(8 / 3 * BER_target), 2))
    if strategy == 'fixed_rate':
        if GSNR_lin >= first_formula:
            Rb = 100
    elif strategy == 'flex_rate':
        if GSNR_lin >= first_formula and GSNR_lin < second_formula:
            Rb = 100
        elif GSNR_lin >= second_formula and GSNR_lin < third_formula:
            Rb = 200
        elif GSNR_lin >= third_formula:
            Rb = 400
    elif strategy == 'shannon':
        Rb = 2 * Rs * np.log2(1 + GSNR_lin * Rs / Bn)
    else:
        print('ERROR in strategy definition')
        exit(5)
    return Rb

