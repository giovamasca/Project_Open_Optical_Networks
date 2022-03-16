########### GLOBAL VARIABLES ###########
number_channels = 10 # if number of channels changes
noise_power_spectral_density = 1e-9
BER_target = 1e-3
Rs_symbol_rate = 32e9 # GHz symbol rate
Bn_noise_band = 12.5e9 # Ghz noise bandwith
span_length = 80e3 # m, equal 80 km
frequency = 193.414e12 # is the central frequency at C-band for optical communications, equivalent to 193.414 THz
Number_simulations = 100
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
file_nodes_full_fixed_rate = root / 'Resources' / 'nodes_full_fixed_rate.json'
file_nodes_full_flex_rate = root / 'Resources' / 'nodes_full_flex_rate.json'
file_nodes_full_shannon = root / 'Resources' / 'nodes_full_shannon.json'
### Traditional with and without switching matrix files
file_nodes_full = root / 'Resources' / 'nodes_full.json'
file_nodes_not_full = root / 'Resources' / 'nodes_not_full.json'

### Results saved
lab8_fig1 = root / 'Results' / 'lab8_fig1_node_couples_full_and_not_full.png'
lab8_fig2 = root / 'Results' / 'lab8_fig2_node_couples_transceiver.png'
lab8_fig3 = root / 'Results' / 'lab8_fig3_SNR_full_and_not_full.png'
lab8_fig4 = root / 'Results' / 'lab8_fig4_SNR_transceiver.png'
lab8_fig5 = root / 'Results' / 'lab8_fig5_bit_rates.png'
##############################################################################################
