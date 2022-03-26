from Project_Open_Optical_Networks.Core.parameters import *
from Project_Open_Optical_Networks.Core.utils import *

############ NETWORKs GENERATION
# these 3 networks has defined transceiver instance
network_fixed_rate = network_generation_from_file(file_nodes_full_fixed_rate)
network_flex_rate = network_generation_from_file(file_nodes_full_flex_rate)
network_shannon = network_generation_from_file(file_nodes_full_shannon)

# LABELS
fixed = 'fixed_rate'
flex = 'flex_rate'
shannon = 'shannon'

### Connections list
connections = {fixed:[], flex:[], shannon:[]}
connections[fixed] = random_generation_for_network(network=network_fixed_rate, Numb_sim=Number_simulations, network_label=fixed)
connections[flex] = random_generation_for_network(network=network_flex_rate, Numb_sim=Number_simulations, network_label=flex)
connections[shannon] = random_generation_for_network(network=network_shannon, Numb_sim=Number_simulations, network_label=shannon)

############# FIGURE 1
plot_histogram(figure_num=1, list_data=[connection_list_data_extractor(connection_list=connections[fixed], type_data='I/O'),
                                        connection_list_data_extractor(connection_list=connections[flex], type_data='I/O'),
                                        connection_list_data_extractor(connection_list=connections[shannon], type_data='I/O')],
               nbins=np.arange(31)-0.5, edge_color='k', color=['b', 'g', 'y'], label=[fixed, flex, shannon],
               title='Histogram of pairs of random extracted nodes for lightpath',
               ylabel='number of results', xlabel='pair of nodes', savefig_path=None )
############# FIGURE 2
plot_histogram(figure_num=2, list_data=[connection_list_data_extractor(connection_list=connections[fixed], type_data='SNR'),
                                        connection_list_data_extractor(connection_list=connections[flex], type_data='SNR'),
                                        connection_list_data_extractor(connection_list=connections[shannon], type_data='SNR')],
               nbins=20, edge_color='k', color=['g', 'm', 'y'], label=[fixed, flex, shannon],
               title='Histogram of SNR for each type of tranceiver',
               ylabel='number of results', xlabel='SNR [dB]', savefig_path=None)

########## BIT RATE and CAPACITY
from Project_Open_Optical_Networks.Core.science_utils import capacity_and_avarage_bit_rate as capacity_and_avarage
[capacity_fixed_rate, avarage_bit_rate_fixed_rate] = capacity_and_avarage(connections_list=connections[fixed])
[capacity_flex_rate, avarage_bit_rate_flex_rate] = capacity_and_avarage(connections_list=connections[flex])
[capacity_shannon, avarage_bit_rate_shannon] = capacity_and_avarage(connections_list=connections[shannon])
## labels
fixed_rate_label = 'Fixed Rate with avarage Rb=' + str(np.round(avarage_bit_rate_fixed_rate*1e-9, 3)) + ' Gbps and C=' + str(np.round(capacity_fixed_rate*1e-12, 3)) + ' Tbps'
flex_rate_label = 'Flex Rate with avarage Rb=' + str(np.round(avarage_bit_rate_flex_rate*1e-9, 3)) + ' Gbps and C=' + str(np.round(capacity_flex_rate*1e-12, 3)) + ' Tbps'
shannon_label = 'Shannon with avarage Rb=' + str(np.round(avarage_bit_rate_shannon*1e-9, 3)) + ' Gbps and C=' + str(np.round(capacity_shannon*1e-12, 3)) + ' Tbps'

################## FIGURE 3
plot_histogram(figure_num = 3, list_data=[ connection_list_data_extractor(connections[fixed], 'Rb'),
                                           connection_list_data_extractor(connections[flex], 'Rb'),
                                           connection_list_data_extractor(connections[shannon], 'Rb') ],
               nbins=40, edge_color='k', color=['y', 'm', 'r'], label=[fixed_rate_label, flex_rate_label, shannon_label],
               title = 'Histogram of Bit Rates per transceiver', bbox_to_anchor=(0.5, -0.35), loc='lower center', ylabel='number of results',
               bottom=0.25, xlabel='Bit rate [Gbps]', savefig_path = None)
print()
print('Blocking events for fixed_rate network: ', number_blocking_events_evaluation(connections[fixed]))
print('Blocking events for flex_rate network: ', number_blocking_events_evaluation(connections[flex]))
print('Blocking events for shannon network: ', number_blocking_events_evaluation(connections[shannon]))
plt.show() # has to be put at the end