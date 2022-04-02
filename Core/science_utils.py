from scipy.constants import c as speed_light
from scipy.constants import Planck as h_Plank
import numpy as np
from scipy.special import erfcinv # this is the function of interest for bit rate evaluation

from Project_Open_Optical_Networks.Core.parameters import *

##################### Constant Values #######################
phase_velocity = 2 / 3 * speed_light # velocity of the line is defined by light speed and a ratio
# n = 1.5, so c/n ~ (2/3)*c
P_base = h_Plank * frequency_C_band * Bn_noise_band
#############################################################

def latency_evaluation(length):
    latency = length / phase_velocity # obtained delay in a line
    return latency
def eta_NLI_evaluation(alpha_dB=None, beta=None, gamma_NL=None, Rs=None, DeltaF=None, N_channels=None, L_eff=None):
    alpha_linear = alpha_from_dB_to_linear_value(alpha_dB)
    PHI_d = ((np.power(np.pi,2))/2)*(beta*(np.power(Rs,2)))/alpha_linear
    P_NL = 1/(gamma_NL*L_eff)
    parentesys = (1/Rs)*(np.log(PHI_d)/PHI_d) + (2/DeltaF)*(np.log(N_channels)/PHI_d)
    eta_NLI = (8/27)*np.pi*(1/(np.power(P_NL,2))) * parentesys
    return eta_NLI

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
            Rb = 100 # Gbps
    elif strategy == 'flex_rate':
        if GSNR_lin >= first_formula and GSNR_lin < second_formula:
            Rb = 100
        elif GSNR_lin >= second_formula and GSNR_lin < third_formula:
            Rb = 200
        elif GSNR_lin >= third_formula:
            Rb = 400
    elif strategy == 'shannon':
        Rb = 2 * Rs_symbol_rate * np.log2(1 + GSNR_lin * Rs_symbol_rate / Bn_noise_band) * 1e-9 # Gbps
    else:
        print('ERROR in strategy definition')
        exit(5)
    return Rb # returned in Gbps

def capacity_and_average_bit_rate(connections_list):
    capacity = np.nansum( [connections_list[i].bit_rate for i in range(0, len(connections_list))] )
    avarage_bit_rate = capacity / len(connections_list)
    return [capacity, avarage_bit_rate]
def SNR_metrics(connection_list):
    ### averaging on linear quantities
    SNR_list = [dB_to_linear_conversion_power( connection_list[i].snr ) for i in range(0, len(connection_list))]
    total_SNR_linear = np.nansum( SNR_list )
    SNR_per_link_in_dB = linear_to_dB_conversion_power( total_SNR_linear / len(connection_list) )

    SNR_max = max( linear_to_dB_conversion_power( SNR_list ) )
    SNR_min = min( linear_to_dB_conversion_power( SNR_list ) )
    return [SNR_per_link_in_dB, SNR_max, SNR_min]
def capacity_metrics(connections_list):
    [capacity, average_bit_rate] = capacity_and_average_bit_rate(connections_list=connections_list)
    bitrate_list = [connections_list[i].bit_rate for i in range(0, len(connections_list))]
    bitrate_max = max(bitrate_list)
    bitrate_min = min(bitrate_list)
    return [capacity, average_bit_rate, bitrate_max, bitrate_min]
def alpha_from_dB_to_linear_value(alpha_in_dB):
    return alpha_in_dB/(20*np.log10(np.exp(1)))

