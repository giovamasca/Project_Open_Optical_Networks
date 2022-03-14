import numpy as np
# import pandas as pd
# import random
import matplotlib.pyplot as plt

from Project_Open_Optical_Networks.Core.elements import Network, Connection
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

plt.figure(1)
plt.hist( [ [ connection_full.input + connection_full.output for connection_full in connections['full']],
            [connection_not_full.input + connection_not_full.output for connection_not_full in connections['not_full']] ] , bins=20,
          edgecolor='k', color=['g','b'], label=['Switching Matrix Full','Switching Matrix Not Full'] )
plt.title('Histogram of simulated nodes')
plt.legend()
plt.ylabel('number of results')
plt.xlabel('path')
figure = plt.gcf() # get current figure
figure.set_size_inches(8, 6)
plt.savefig('../Results/lab7_fig1')

plt.figure(2)
plt.hist( [ [connection_fixed_rate.input + connection_fixed_rate.output for connection_fixed_rate in connections['fixed_rate']],
            [connection_flex_rate.input + connection_flex_rate.output for connection_flex_rate in connections['flex_rate']],
            [connection_shannon.input + connection_shannon.output for connection_shannon in connections['shannon']] ] , bins=20,
          edgecolor='k', color=['y', 'm', 'r'], label=['Fixed rate', 'Flex Rate', 'Shannon'] )
plt.title('Histogram of simulated nodes')
plt.legend()
plt.ylabel('number of results')
plt.xlabel('path')
figure = plt.gcf() # get current figure
figure.set_size_inches(8, 6)
plt.savefig('../Results/lab7_fig2')

plt.figure(3)
plt.hist( [ [connection_full_hist.snr for connection_full_hist in connections['full']],
            [connection_not_full_hist.snr for connection_not_full_hist in connections['not_full']] ],
          edgecolor='k', color=['g','b'], label=['Switching Matrix Full ','Switching Matrix Not Full'], bins=20 )
plt.title('Histogram of obtained SNRs for Switching Matrix networks')
plt.legend(loc='upper left')
plt.ylabel('number of results')
plt.xlabel('SNR [dB]')
figure = plt.gcf() # get current figure
figure.set_size_inches(8, 6)
plt.savefig('../Results/lab7_fig3')

plt.figure(4)
plt.hist( [ [connection_fixed_rate_hist.snr for connection_fixed_rate_hist in connections['fixed_rate']],
            [connection_flex_rate_hist.snr for connection_flex_rate_hist in connections['flex_rate']],
            [connection_shannon_hist.snr for connection_shannon_hist in connections['shannon']] ],
          edgecolor='k', color=['y', 'm', 'r'], label=['Fixed Rate', 'Flex Rate', 'Shannon'], bins=20 )
plt.title('Histogram of obtained SNRs for transceiver networks')
plt.legend()
plt.ylabel('number of results')
plt.xlabel('SNR [dB]')
figure = plt.gcf() # get current figure
figure.set_size_inches(8, 6)
plt.savefig('../Results/lab7_fig4')

##################################################################################################################
# BIT RATE
##################################################################################################################
capacity_fixed_rate = np.nansum([connections['fixed_rate'][i].bit_rate for i in range(0, len(connections['fixed_rate']))])
avarage_fixed_rate = capacity_fixed_rate/len(connections['fixed_rate'])
fixed_rate_label = 'Fixed Rate with avarage Rb=' + str(avarage_fixed_rate) + ' and C=' + str(capacity_fixed_rate)

capacity_flex_rate = np.nansum([connections['flex_rate'][i].bit_rate for i in range(0, len(connections['flex_rate']))])
avarage_flex_rate = capacity_flex_rate/len(connections['fixed_rate'])
flex_rate_label = 'Flex Rate with avarage Rb=' + str(avarage_flex_rate) + ' and C=' + str(capacity_flex_rate)

capacity_shannon = np.nansum([connections['shannon'][i].bit_rate for i in range(0, len(connections['shannon']))])
avarage_shannon = capacity_shannon/len(connections['shannon'])
shannon_label = 'Shannon with avarage Rb=' + str(avarage_shannon) + ' and C=' + str(capacity_shannon)

fig = plt.figure(5)
fig.subplots_adjust(bottom=0.25)
plt.hist( [ [connection_fixed_rate.bit_rate for connection_fixed_rate in connections['fixed_rate']],
            [connection_flex_rate.bit_rate for connection_flex_rate in connections['flex_rate']],
            [connection_shannon.bit_rate for connection_shannon in connections['shannon']] ] , bins=10,
          edgecolor='k', color=['y', 'm', 'r'], label=[fixed_rate_label, flex_rate_label, shannon_label] )
plt.title('Histogram of Bit Rates')
plt.legend(title='Histogram of Bit Rates',bbox_to_anchor=(0.5, -0.4), loc='lower center')
plt.xlabel('Bit rate [Gbps]')
figure = plt.gcf() # get current figure
figure.set_size_inches(8, 6)
plt.savefig('../Results/lab7_fig5_bit_rates.png')

plt.show()