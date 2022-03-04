########### GLOBAL VARIABLES ###########
number_channels = 10 # if number of channels changes
noise_power_spectral_density = 1e-9
BER_target = 1e-3
Rs = 32e9 # GHz symbol rate
Bn = 12.5e9 # Ghz noise bandwith
span_length = 80e3 # m, equal 80 km
frequency = 193.414e12 # is the central frequency at C-band for optical communications, equivalent to 193.414 THz
Number_simulations = 100
########################################
############## STATE ###################
FREE = 1
OCCUPIED = 0
########################################

########################## path definition with path lib #####################################
from pathlib import Path
root = Path(__file__).parent.parent

file_nodes_full_fixed_rate = root / 'Resources' / 'nodes_full_fixed_rate.json'
file_nodes_full_flex_rate = root / 'Resources' / 'nodes_full_flex_rate.json'
file_nodes_full_shannon = root / 'Resources' / 'nodes_full_shannon.json'

file_nodes_full = root / 'Resources' / 'nodes_full.json'
file_nodes_not_full = root / 'Resources' / 'nodes_not_full.json'
##############################################################################################
