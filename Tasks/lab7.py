import numpy as np
# import pandas as pd
# import random
# import matplotlib.pyplot as plt

from Project_Open_Optical_Networks.Core.elements import Network #, Connection
from Project_Open_Optical_Networks.Core.parameters import *
from Project_Open_Optical_Networks.Core.utils import *

############ NETWORKs GENERATION
# these 3 networks has defined transceiver instance
network_fixed_rate = Network(file_nodes_full_fixed_rate)
network_flex_rate = Network(file_nodes_full_flex_rate)
network_shannon = Network(file_nodes_full_shannon)

# let's generate the full nodes network:
# it means that switching matricies are full, that means that giving a node a lightpath can travel to and from any line connected to the node
network_full = Network(file_nodes_full)

# then let's generate the network with no full switching matricies, that means that some node have not full switching matrix
network_not_full = Network(file_nodes_not_full)

print('Display for each Node switching matrix when it is OFF')
for node in network_not_full.nodes:
    network_not_full.node_switching_analyse(node, disp='OFF')

########################################################################################################################
########################################################################################################################
######## Evaluation of 100 random connections
# network_full.restore_free_state_lines() ## restore network as at the beginning

connections = {'full': [], 'not_full': [], 'fixed_rate': [], 'flex_rate': [], 'shannon': []}
connections['full'] = random_generation_for_network(network=network_full, Numb_sim=Number_simulations, network_label='full')
connections['not_full'] = random_generation_for_network(network=network_not_full, Numb_sim=Number_simulations, network_label='not_full')

connections['fixed_rate'] = random_generation_for_network(network=network_fixed_rate, Numb_sim=Number_simulations, network_label='fixed_rate')
connections['flex_rate'] = random_generation_for_network(network=network_flex_rate, Numb_sim=Number_simulations, network_label='flex_rate')
connections['shannon'] = random_generation_for_network(network=network_shannon, Numb_sim=Number_simulations, network_label='shannon')

################### FIGURE 1
plot_histogram(figure_num=1, list_data=[ [ connection_full.input + connection_full.output for connection_full in connections['full']],
               [connection_not_full.input + connection_not_full.output for connection_not_full in connections['not_full']] ],
               nbins=np.arange(31)-0.5, edge_color='k', color=['g', 'b'], label=['Switching Matrix Full','Switching Matrix Not Full'],
               title = 'Histogram of simulated nodes',
               ylabel='number of results', xlabel='path', savefig_path='../Results/lab7_fig1')

#################### FIGURE 2
plot_histogram(figure_num = 2, list_data=[ [connection_fixed_rate.input + connection_fixed_rate.output for connection_fixed_rate in connections['fixed_rate']],
            [connection_flex_rate.input + connection_flex_rate.output for connection_flex_rate in connections['flex_rate']],
            [connection_shannon.input + connection_shannon.output for connection_shannon in connections['shannon']] ],
               nbins=np.arange(31)-0.5, edge_color='k', color=['y', 'm', 'r'], label=['Fixed rate', 'Flex Rate', 'Shannon'],
               title = 'Histogram of simulated nodes',
               ylabel='number of results', xlabel='path', savefig_path='../Results/lab7_fig2')

############### FIGURE 3
plot_histogram(figure_num = 3, list_data=[ [connection_full_hist.snr for connection_full_hist in connections['full']],
            [connection_not_full_hist.snr for connection_not_full_hist in connections['not_full']] ],
               nbins=20, edge_color='k', color=['g', 'b'], label=['Switching Matrix Full ','Switching Matrix Not Full'],
               title = 'Histogram of obtained SNRs for Switching Matrix networks',
               ylabel='number of results', xlabel='SNR [dB]', savefig_path='../Results/lab7_fig3')

############## FIGURE 4
plot_histogram(figure_num = 4, list_data=[ [connection_fixed_rate_hist.snr for connection_fixed_rate_hist in connections['fixed_rate']],
            [connection_flex_rate_hist.snr for connection_flex_rate_hist in connections['flex_rate']],
            [connection_shannon_hist.snr for connection_shannon_hist in connections['shannon']] ],
               nbins=20, edge_color='k', color=['y', 'm', 'r'], label=['Fixed Rate', 'Flex Rate', 'Shannon'],
               title = 'Histogram of obtained SNRs for transceiver networks',
               ylabel='number of results', xlabel='SNR [dB]', savefig_path='../Results/lab7_fig4')

##################################################################################################################
# BIT RATE
##################################################################################################################
from Project_Open_Optical_Networks.Core.science_utils import capacity_and_avarage_bit_rate as cap_ave

[capacity_fixed_rate, avarage_fixed_rate] = cap_ave(connections['fixed_rate'])
fixed_rate_label = 'Fixed Rate with avarage Rb=' + str(avarage_fixed_rate*1e-9) + ' Gbps and C=' + str(capacity_fixed_rate*1e-9) + ' Gbps'

[capacity_flex_rate, avarage_flex_rate] = cap_ave(connections['flex_rate'])
flex_rate_label = 'Flex Rate with avarage Rb=' + str(avarage_flex_rate*1e-9) + ' Gbps and C=' + str(capacity_flex_rate*1e-9) + ' Gbps'

[capacity_shannon, avarage_shannon] = cap_ave(connections['shannon'])
shannon_label = 'Shannon with avarage Rb=' + str(avarage_shannon*1e-9) + ' Gbps and C=' + str(capacity_shannon*1e-9) + ' Gbps'

################## FIGURE 5
plot_histogram(figure_num = 5, list_data=[ [connection_fixed_rate.bit_rate for connection_fixed_rate in connections['fixed_rate']],
            [connection_flex_rate.bit_rate for connection_flex_rate in connections['flex_rate']],
            [connection_shannon.bit_rate for connection_shannon in connections['shannon']] ],
               nbins=10, edge_color='k', color=['y', 'm', 'r'], label=[fixed_rate_label, flex_rate_label, shannon_label],
               title = 'Histogram of Bit Rates', bbox_to_anchor=(0.5, -0.35), loc='lower center', ylabel='', bottom=0.25,
               xlabel='Bit rate [Gbps]', savefig_path='../Results/lab7_fig5_bit_rates.png')

plt.show()