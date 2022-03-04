########### GLOBAL VARIABLES ###########
number_channels = 10 # if number of channels changes
noise_power_spectral_density = 1e-9
BER_target = 1e-3
Rs = 32 # GHz symbol rate
Bn = 12.5 # Ghz noise bandwith
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
