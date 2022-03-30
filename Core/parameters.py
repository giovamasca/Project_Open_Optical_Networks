import os
import time

########### GLOBAL VARIABLES ###########
#### SET FIND BEST CONDITION at the beginning
set_latency_or_snr = 'SNR' # 'SNR' or 'latency' for find best
### CHANNELS
number_of_active_channels = 10 # if number of channels changes
# maximum_number_of_channels = 96 # this is the most diffused value for DeltaF of 50 GHz
### NOISE, BER PARAMETERS
noise_power_spectral_density = 1e-9
BER_target = 1e-3
Bn_noise_band = 12.5e9 # Ghz noise bandwith
### TX PARAMETERSgit
Rs_symbol_rate = 32e9 # GHz symbol rate
span_length = 80e3 # m, equal 80 km
frequency_C_band = 193.414e12 # is the central frequency at C-band for optical communications, equivalent to 193.414 THz
channel_spacing = 50e9 # channel spacing df from two consecutive tones
### SIMULATION PARAMETERS
Number_simulations = 100
###### LINEAR and NON-LINEARITY constants ##########
alpha_in_dB_ct = 0.2e-3 # dB/m
beta_abs_for_CD_ct = 2.13e-26 # 1/(m*Hz^2)
gamma_non_linearity_ct = 1.27e-3 # 1/(W*m)
###### amplified line parameters #############
G_gain_ct = 16 # dB
NF_noise_figure_ct = 3 # dB
########################################
############## STATE ###################
FREE = 1
OCCUPIED = 0
########################################

# from datetime import datetime
# now = datetime.now()
# date_string = now.strftime('%Y_%m_%d_%Hhr_%Mmin_%Ssec')
########################## path definition with path lib #####################################
from pathlib import Path
root = Path(__file__).parent.parent

### Transceiver json files
file_nodes_full_fixed_rate = root / 'Resources' / 'Lab_Networks' / 'nodes_full_fixed_rate.json'
file_nodes_full_flex_rate = root / 'Resources' / 'Lab_Networks' / 'nodes_full_flex_rate.json'
file_nodes_full_shannon = root / 'Resources' / 'Lab_Networks' / 'nodes_full_shannon.json'
### Traditional with and without switching matrix files
file_nodes_full = root / 'Resources' / 'Lab_Networks' / 'nodes_full.json'
file_nodes_not_full = root / 'Resources' / 'Lab_Networks' / 'nodes_not_full.json'

### Results saved
trial_path = 'best_' + set_latency_or_snr + '_of_' + time.strftime('%Y_%m_%d__%H_%M_%S')
if not os.path.isdir('../Results/Lab9/' + trial_path):  # if Results doesn't exists, it creates it
    os.makedirs('../Results/Lab9/' + trial_path)

file_console = root / 'Results' / 'Lab9' / trial_path / 'console.txt'
## LAB 8
lab8_fig1 = root / 'Results' / 'Lab8' / 'lab8_fig1_node_couples_full_and_not_full.png'
lab8_fig2 = root / 'Results' / 'Lab8' / 'lab8_fig2_node_couples_transceiver.png'
lab8_fig3 = root / 'Results' / 'Lab8' / 'lab8_fig3_SNR_full_and_not_full.png'
lab8_fig4 = root / 'Results' / 'Lab8' / 'lab8_fig4_SNR_transceiver.png'
lab8_fig5 = root / 'Results' / 'Lab8' / 'lab8_fig5_bit_rates.png'
## LAB 9
# point 5
lab9_fig1 = root / 'Results' / 'Lab9' / trial_path / ('lab9_fig1_point5_IO_best_' + set_latency_or_snr)
lab9_fig2 = root / 'Results' / 'Lab9' / trial_path / ('lab9_fig2_point5_SNR_best_' + set_latency_or_snr)
lab9_fig3 = root / 'Results' / 'Lab9' / trial_path / ('lab9_fig3_point5_Rb_best_' + set_latency_or_snr)
# point 7
# SNR
lab9_fig4 = root / 'Results' / 'Lab9' / trial_path / ('lab9_fig4_point7_SNR_fixed_best_' + set_latency_or_snr)
lab9_fig5 = root / 'Results' / 'Lab9' / trial_path / ('lab9_fig5_point7_SNR_flex_best_' + set_latency_or_snr)
lab9_fig6 = root / 'Results' / 'Lab9' / trial_path / ('lab9_fig6_point7_SNR_shannon_best_' + set_latency_or_snr)
# BITRATE
lab9_fig7 = root / 'Results' / 'Lab9' / trial_path / ('lab9_fig7_point7_Rb_fixed_best_' + set_latency_or_snr)
lab9_fig8 = root / 'Results' / 'Lab9' / trial_path / ('lab9_fig8_point7_Rb_flex_best_' + set_latency_or_snr)
lab9_fig9 = root / 'Results' / 'Lab9' / trial_path / ('lab9_fig9_point7_Rb_shannon_best_' + set_latency_or_snr)
# LAT
lab9_fig10 = root / 'Results' / 'Lab9' / trial_path / ('lab9_fig10_point7_latency_fixed_best_' + set_latency_or_snr)
lab9_fig11 = root / 'Results' / 'Lab9' / trial_path / ('lab9_fig11_point7_latency_flex_best_' + set_latency_or_snr)
lab9_fig12 = root / 'Results' / 'Lab9' / trial_path / ('lab9_fig12_point7_latency_shannon_best_' + set_latency_or_snr)
##############################################################################################
