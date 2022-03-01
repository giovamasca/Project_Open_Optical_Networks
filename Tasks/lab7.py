import numpy as np
# import pandas as pd
import random
import matplotlib.pyplot as plt

from Project_Open_Optical_Networks.Core.elements import Network, Connection
from Project_Open_Optical_Networks.Core.parameters import *

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
def random_generation_for_network(network, Numb_sim, network_label=None): # network and sumber of simulations
    nodes_gener = list(network.nodes.keys())  # extracts all nodes

    avarage_bit_rate = 0
    connections_generated = []  # defined a list of connections
    for i in range(0, Numb_sim): # do a number of simulations equal to Numb_sim
        n1 = random.randint(0, len(nodes_gener) - 1)  # any position is ok
        n2 = random.randint(0, len(nodes_gener) - 1)
        while n2 == n1:
            n2 = random.randint(0, len(nodes_gener) - 1)  # repeat the random evaluation until there is a couple of nodes, not same values

        connection_generated = Connection(nodes_gener[n1], nodes_gener[n2], 1)  # creates connection
        connection_generated = network.stream(connection_generated, 'snr', use_state=True)  # stream it with state on and snr set
        # if connection_generated.latency==np.NaN:
        #     continue # avoid this connection
        # with np.NaN the histograms avoid the corresponding values
        connections_generated.append(connection_generated) # appends connection
        avarage_bit_rate += connection_generated.bit_rate
        # print('Evaluation in progress: ', np.round_(i/Numb_sim*100), ' %', end='\r')
    print('Evaluated ', Numb_sim, ' simulations for network ', network_label)
    return connections_generated

connections = {'full': [], 'not_full': [], 'fixed_rate': [], 'flex_rate': [], 'shannon': []}
connections['full'] = random_generation_for_network(network=network_full, Numb_sim=100, network_label='full')
connections['not_full'] = random_generation_for_network(network=network_not_full, Numb_sim=100, network_label='not_full')

connections['fixed_rate'] = random_generation_for_network(network=network_fixed_rate, Numb_sim=100, network_label='fixed_rate')
connections['flex_rate'] = random_generation_for_network(network=network_flex_rate, Numb_sim=100, network_label='flex_rate')
connections['shannon'] = random_generation_for_network(network=network_shannon, Numb_sim=100, network_label='shannon')

# plt.figure(1)
# plt.hist( [ [ connection_full.input + connection_full.output for connection_full in connections['full']],
#             [connection_not_full.input + connection_not_full.output for connection_not_full in connections['not_full']] ] , bins=20,
#           edgecolor='k', color=['g','b'], label=['Switching Matrix Full','Switching Matrix Not Full'] )
# plt.title('Histogram of simulated nodes')
# plt.legend()
# plt.ylabel('number of results')
# plt.xlabel('path')

plt.figure(2)
plt.hist( [ [connection_fixed_rate.input + connection_fixed_rate.output for connection_fixed_rate in connections['fixed_rate']],
            [connection_flex_rate.input + connection_flex_rate.output for connection_flex_rate in connections['flex_rate']],
            [connection_shannon.input + connection_shannon.output for connection_shannon in connections['shannon']] ] , bins=20,
          edgecolor='k', color=['y', 'm', 'r'], label=['Fixed rate', 'Flex Rate', 'Shannon'] )
plt.title('Histogram of simulated nodes')
plt.legend()
plt.ylabel('number of results')
plt.xlabel('path')

plt.figure(3)
plt.hist( [ [connection_full_hist.snr for connection_full_hist in connections['full']],
            [connection_not_full_hist.snr for connection_not_full_hist in connections['not_full']] ],
          edgecolor='k', color=['g','b'], label=['Switching Matrix Full ','Switching Matrix Not Full'], bins=20 )
plt.title('Histogram of obtained SNRs for Switching Matrix networks')
plt.legend()
plt.ylabel('number of results')
plt.xlabel('SNR [dB]')

plt.figure(4)
plt.hist( [ [connection_fixed_rate_hist.snr for connection_fixed_rate_hist in connections['fixed_rate']],
            [connection_flex_rate_hist.snr for connection_flex_rate_hist in connections['flex_rate']],
            [connection_shannon_hist.snr for connection_shannon_hist in connections['shannon']] ],
          edgecolor='k', color=['y', 'm', 'r'], label=['Fixed Rate', 'Flex Rate', 'Shannon'], bins=20 )
plt.title('Histogram of obtained SNRs for transceiver networks')
plt.legend()
plt.ylabel('number of results')
plt.xlabel('SNR [dB]')

##################################################################################################################
# FIXED RATE
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

plt.figure(5)
plt.hist( [ [connection_fixed_rate.bit_rate for connection_fixed_rate in connections['fixed_rate']],
            [connection_flex_rate.bit_rate for connection_flex_rate in connections['flex_rate']],
            [connection_shannon.bit_rate for connection_shannon in connections['shannon']] ] , bins=10,
          edgecolor='k', color=['y', 'm', 'r'], label=[fixed_rate_label, flex_rate_label, shannon_label] )
plt.title('Histogram of Bit Rates')
plt.legend()
plt.ylabel('number of results')
plt.xlabel('Bit rate [Gbps]')

plt.show()