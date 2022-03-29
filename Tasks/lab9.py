from Project_Open_Optical_Networks.Core.parameters import *
from Project_Open_Optical_Networks.Core.utils import *

from Project_Open_Optical_Networks.Core.science_utils import capacity_and_avarage_bit_rate as capacity_and_avarage

#### SET FIND BEST CONDITION at the beginning
set_latency_or_snr = 'snr' # 'snr' or 'latency'

############ NETWORKs GENERATION
# these 3 networks has defined transceiver instance
network_fixed_rate = network_generation_from_file(file_nodes_full_fixed_rate)
network_flex_rate = network_generation_from_file(file_nodes_full_flex_rate)
network_shannon = network_generation_from_file(file_nodes_full_shannon)

# LABELS
fixed = 'fixed_rate'
flex = 'flex_rate'
shannon = 'shannon'

print('\t\t\tPoint 6 - Lab 9\n')
### Connections list
connections = {fixed:[], flex:[], shannon:[]}
connections[fixed] = random_generation_for_network(network=network_fixed_rate, Numb_sim=Number_simulations, network_label=fixed, set_lat_or_snr=set_latency_or_snr)
connections[flex] = random_generation_for_network(network=network_flex_rate, Numb_sim=Number_simulations, network_label=flex, set_lat_or_snr=set_latency_or_snr)
connections[shannon] = random_generation_for_network(network=network_shannon, Numb_sim=Number_simulations, network_label=shannon, set_lat_or_snr=set_latency_or_snr)

############# FIGURE 1
plot_histogram(figure_num=1, list_data=[connection_list_data_extractor(connection_list=connections[fixed], type_data='I/O'),
                                        connection_list_data_extractor(connection_list=connections[flex], type_data='I/O'),
                                        connection_list_data_extractor(connection_list=connections[shannon], type_data='I/O')],
               nbins=np.arange(31)-0.5, edge_color='k', color=['b', 'g', 'y'], label=[fixed, flex, shannon],
               title='Point 6 Lab 9 - Histogram of pairs of random extracted nodes for lightpath',
               ylabel='number of results', xlabel='pair of nodes', savefig_path=None )
############# FIGURE 2
plot_histogram(figure_num=2, list_data=[connection_list_data_extractor(connection_list=connections[fixed], type_data='SNR'),
                                        connection_list_data_extractor(connection_list=connections[flex], type_data='SNR'),
                                        connection_list_data_extractor(connection_list=connections[shannon], type_data='SNR')],
               nbins=20, edge_color='k', color=['g', 'm', 'y'], label=[fixed, flex, shannon],
               title='Point 6 Lab 9 - Histogram of SNR for each type of tranceiver',
               ylabel='number of results', xlabel='SNR [dB]', savefig_path=None)

########## BIT RATE and CAPACITY
[capacity_fixed_rate, average_bit_rate_fixed_rate] = capacity_and_avarage(connections_list=connections[fixed])
[capacity_flex_rate, average_bit_rate_flex_rate] = capacity_and_avarage(connections_list=connections[flex])
[capacity_shannon, average_bit_rate_shannon] = capacity_and_avarage(connections_list=connections[shannon])
## labels
fixed_rate_label = 'Fixed Rate with average Rb=' + str(np.round(average_bit_rate_fixed_rate, 3)) + ' Gbps and C=' + str(np.round(capacity_fixed_rate*1e-3, 3)) + ' Tbps'
flex_rate_label = 'Flex Rate with average Rb=' + str(np.round(average_bit_rate_flex_rate, 3)) + ' Gbps and C=' + str(np.round(capacity_flex_rate*1e-3, 3)) + ' Tbps'
shannon_label = 'Shannon with average Rb=' + str(np.round(average_bit_rate_shannon)) + ' Gbps and C=' + str(np.round(capacity_shannon*1e-3, 3)) + ' Tbps'

################## FIGURE 3
plot_histogram(figure_num = 3, list_data=[ connection_list_data_extractor(connections[fixed], 'Rb'),
                                           connection_list_data_extractor(connections[flex], 'Rb'),
                                           connection_list_data_extractor(connections[shannon], 'Rb') ],
               nbins=40, edge_color='k', color=['y', 'm', 'r'], label=[fixed_rate_label, flex_rate_label, shannon_label],
               title = 'Point 6 Lab 9 - Histogram of Bit Rates per transceiver', bbox_to_anchor=(0.5, -0.35), loc='lower center',
               ylabel='number of results', bottom=0.25, xlabel='Bit rate [Gbps]', savefig_path = None)
print()
print('Blocking events for fixed_rate network: ', number_blocking_events_evaluation(connections[fixed]))
print('Blocking events for flex_rate network: ', number_blocking_events_evaluation(connections[flex]))
print('Blocking events for shannon network: ', number_blocking_events_evaluation(connections[shannon]))
# plt.draw() # has to be put at the end

#################### Point 7
print('\n\t\t\tPoint 7 Lab 9\n')
connections_fixed_per_M = []
connections_flex_per_M = []
connections_shannon_per_M = []
M_list = []
for M in range(1, 47, 9):
    M_list.append(M)

    print('\t\tFor M = ', str(M), ':')
    connections_fixed_rate = random_generation_with_traffic_matrix(network=network_fixed_rate, M_traffic=M, set_lat_or_snr=set_latency_or_snr)
    print('Fixed Rate')
    print('\tTotal connections for fixed_rate network: ', len(connections_fixed_rate))
    print('\tBlocking events for fixed_rate network: ', number_blocking_events_evaluation(connections_fixed_rate))
    connections_fixed_per_M.append(connections_fixed_rate)
    ############## CAPACITY and BITRATE
    [capacity_fixed_rate, average_bit_rate_fixed_rate] = capacity_and_avarage(connections_list=connections_fixed_rate)
    ## labels
    fixed_rate_label = 'Fixed Rate with average Rb=' + str(np.round(average_bit_rate_fixed_rate, 3)) + ' Gbps and C=' +\
                       str(np.round(capacity_fixed_rate * 1e-3, 3)) + ' Tbps'
    print('\t'+fixed_rate_label)

    connections_flex_rate = random_generation_with_traffic_matrix(network=network_flex_rate, M_traffic=M, set_lat_or_snr=set_latency_or_snr)
    print('Flex Rate')
    print('\tTotal connections for flex_rate network: ', len(connections_flex_rate))
    print('\tBlocking events for flex_rate network: ', number_blocking_events_evaluation(connections_flex_rate))
    connections_flex_per_M.append(connections_flex_rate)
    ############## CAPACITY and BITRATE
    [capacity_flex_rate, average_bit_rate_flex_rate] = capacity_and_avarage(connections_list=connections_flex_rate)
    ## labels
    flex_rate_label = 'Flex Rate with average Rb=' + str(np.round(average_bit_rate_flex_rate, 3)) + ' Gbps and C=' + \
                       str(np.round(capacity_flex_rate * 1e-3, 3)) + ' Tbps'
    print('\t' + flex_rate_label)

    connections_shannon = random_generation_with_traffic_matrix(network=network_shannon, M_traffic=M, set_lat_or_snr=set_latency_or_snr)
    print('Shannon Rate')
    print('\tTotal connections for shannon network: ', len(connections_shannon))
    print('\tBlocking events for shannon network: ', number_blocking_events_evaluation(connections_shannon))
    connections_shannon_per_M.append(connections_shannon)
    ############## CAPACITY and BITRATE
    [capacity_shannon, average_bit_rate_shannon] = capacity_and_avarage(connections_list=connections_shannon)
    ## labels
    shannon_label = 'Fixed Rate with average Rb=' + str(np.round(average_bit_rate_shannon, 3)) + ' Gbps and C=' + \
                       str(np.round(capacity_shannon * 1e-3, 3)) + ' Tbps'
    print('\t' + fixed_rate_label)
    print()

#################### SNR
plot_histogram(figure_num=(4), list_data=[connection_list_data_extractor(connection_list=connections_fixed_per_M[i], type_data='SNR') for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='SNR [dB]', ylabel='number of results', title=('Point 7 Lab 9 - Fixed Rate SNR - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=None)
plot_histogram(figure_num=(5), list_data=[connection_list_data_extractor(connection_list=connections_flex_per_M[i], type_data='SNR') for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='SNR [dB]', ylabel='number of results', title=('Point 7 Lab 9 - Flex Rate SNR - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=None)
plot_histogram(figure_num=(6), list_data=[connection_list_data_extractor(connection_list=connections_shannon_per_M[i], type_data='SNR') for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='SNR [dB]', ylabel='number of results', title=('Point 7 Lab 9 - Shannon Rate SNR - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=None)
##################### BITRATE
plot_histogram(figure_num=(7), list_data=[connection_list_data_extractor(connection_list=connections_fixed_per_M[i], type_data='Rb') for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='Rb [Gbps]', ylabel='number of results', title=('Point 7 Lab 9 - Fixed Rate bitrate - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=None)
plot_histogram(figure_num=(8), list_data=[connection_list_data_extractor(connection_list=connections_flex_per_M[i], type_data='Rb') for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='Rb [Gbps]', ylabel='number of results', title=('Point 7 Lab 9 - Flex Rate bitrate - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=None)
plot_histogram(figure_num=(9), list_data=[connection_list_data_extractor(connection_list=connections_shannon_per_M[i], type_data='Rb') for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='Rb [Gbps]', ylabel='number of results', title=('Point 7 Lab 9 - Shannon Rate bitrate - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=None)
##################### LAT
plot_histogram(figure_num=(10), list_data=[np.array(connection_list_data_extractor(connection_list=connections_fixed_per_M[i], type_data='LAT'))*1e3 for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='Latency [ms]', ylabel='number of results', title=('Point 7 Lab 9 - Fixed Rate latency - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=None)
plot_histogram(figure_num=(11), list_data=[np.array(connection_list_data_extractor(connection_list=connections_flex_per_M[i], type_data='LAT'))*1e3 for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='Latency [ms]', ylabel='number of results', title=('Point 7 Lab 9 - Flex Rate latency - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=None)
plot_histogram(figure_num=(12), list_data=[np.array(connection_list_data_extractor(connection_list=connections_shannon_per_M[i], type_data='LAT'))*1e3 for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='Latency [ms]', ylabel='number of results', title=('Point 7 Lab 9 - Shannon Rate latency - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=None)
# plt.ioff()
# plt.pause(1)
plt.show()