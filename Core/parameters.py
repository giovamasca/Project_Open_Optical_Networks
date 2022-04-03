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
watchdog_limit = 200
########################## path definition with path lib #####################################
from pathlib import Path
root = Path(__file__).parent.parent

################## EXAM FILES
### Transceiver json files
file_nodes_full_fixed_rate = root / 'Resources' / 'Exam_Networks' / 'full_fixed_rate_network_269725.json'
file_nodes_full_flex_rate = root / 'Resources' / 'Exam_Networks' / 'full_flex_rate_network_269725.json'
file_nodes_full_shannon = root / 'Resources' / 'Exam_Networks' / 'full_shannon_network_269725.json'
### Traditional with and without switching matrix files
file_nodes_full = root / 'Resources' / 'Exam_Networks' / 'full_network_269725.json'
file_nodes_not_full = root / 'Resources' / 'Exam_Networks' / 'not_full_network_269725.json'

# ################## LABORATORY FILES
# ### Transceiver json files
# file_nodes_full_fixed_rate = root / 'Resources' / 'Lab_Networks' / 'nodes_full_fixed_rate.json'
# file_nodes_full_flex_rate = root / 'Resources' / 'Lab_Networks' / 'nodes_full_flex_rate.json'
# file_nodes_full_shannon = root / 'Resources' / 'Lab_Networks' / 'nodes_full_shannon.json'
# ### Traditional with and without switching matrix files
# file_nodes_full = root / 'Resources' / 'Lab_Networks' / 'nodes_full.json'
# file_nodes_not_full = root / 'Resources' / 'Lab_Networks' / 'nodes_not_full.json'

### Results saved
## LAB 8
lab8_fig1 = root / 'Results' / 'Lab8' / 'lab8_fig1_node_couples_full_and_not_full.png'
lab8_fig2 = root / 'Results' / 'Lab8' / 'lab8_fig2_node_couples_transceiver.png'
lab8_fig3 = root / 'Results' / 'Lab8' / 'lab8_fig3_SNR_full_and_not_full.png'
lab8_fig4 = root / 'Results' / 'Lab8' / 'lab8_fig4_SNR_transceiver.png'
lab8_fig5 = root / 'Results' / 'Lab8' / 'lab8_fig5_bit_rates.png'