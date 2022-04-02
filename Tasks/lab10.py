############## Monte Carlo Analysis for OON ############à

# from Project_Open_Optical_Networks.Core.parameters import *
from Project_Open_Optical_Networks.Core.utils import *
from Project_Open_Optical_Networks.Core.science_utils import *

file_print = open(file_console, 'w')
file_print.close()
############ NETWORKs GENERATION
# these 3 networks has defined transceiver instance
network_fixed_rate = network_generation_from_file(file_nodes_full_fixed_rate)
network_flex_rate = network_generation_from_file(file_nodes_full_flex_rate)
network_shannon = network_generation_from_file(file_nodes_full_shannon)

# LABELS
fixed = 'fixed_rate'
flex = 'flex_rate'
shannon = 'shannon'

print_and_save(text='Lab 10 - Monte Carlo Simulation with Single Traffic Matrix', file=file_console)
############## Fixed M - Single Traffic Matrix Scenario ###########
M = 15 # fixed value of M

lab10_point1(network=network_fixed_rate, M = M, set_latency_or_snr=set_latency_or_snr, N_iterations=Number_simulations, label='Fixed Rate', file_console=file_console)
lab10_point1(network=network_flex_rate, M = M, set_latency_or_snr=set_latency_or_snr, N_iterations=Number_simulations, label='Flex Rate', file_console=file_console)
lab10_point1(network=network_shannon, M = M, set_latency_or_snr=set_latency_or_snr, N_iterations=90, label='Shannon', file_console=file_console)



