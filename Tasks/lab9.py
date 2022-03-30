from Project_Open_Optical_Networks.Core.parameters import *
from Project_Open_Optical_Networks.Core.utils import *

from Project_Open_Optical_Networks.Core.science_utils import capacity_and_avarage_bit_rate as capacity_and_avarage

############ NETWORKs GENERATION
# these 3 networks has defined transceiver instance
network_fixed_rate = network_generation_from_file(file_nodes_full_fixed_rate)
network_flex_rate = network_generation_from_file(file_nodes_full_flex_rate)
network_shannon = network_generation_from_file(file_nodes_full_shannon)

# LABELS
fixed = 'fixed_rate'
flex = 'flex_rate'
shannon = 'shannon'

file_print = open(file_console, 'w')
print('\t\t\tPoint 5 - Lab 9\n')
print('\t\t\tPoint 5 - Lab 9\n', file=file_print)
### Connections list
connections = {fixed:[], flex:[], shannon:[]}
connections[fixed] = random_generation_for_network(network=network_fixed_rate, Numb_sim=Number_simulations, network_label=fixed, set_lat_or_snr=set_latency_or_snr)
connections[flex] = random_generation_for_network(network=network_flex_rate, Numb_sim=Number_simulations, network_label=flex, set_lat_or_snr=set_latency_or_snr)
connections[shannon] = random_generation_for_network(network=network_shannon, Numb_sim=Number_simulations, network_label=shannon, set_lat_or_snr=set_latency_or_snr)
print()
############# FIGURE 1
plot_histogram(figure_num=1, list_data=[connection_list_data_extractor(connection_list=connections[fixed], type_data='I/O'),
                                        connection_list_data_extractor(connection_list=connections[flex], type_data='I/O'),
                                        connection_list_data_extractor(connection_list=connections[shannon], type_data='I/O')],
               nbins=np.arange(31)-0.5, edge_color='k', color=['b', 'g', 'y'], label=[fixed, flex, shannon],
               title='Point 6 Lab 9 - Histogram of pairs of random extracted nodes for lightpath',
               ylabel='number of results', xlabel='pair of nodes', savefig_path=lab9_fig1 )
############# FIGURE 2
plot_histogram(figure_num=2, list_data=[connection_list_data_extractor(connection_list=connections[fixed], type_data='SNR'),
                                        connection_list_data_extractor(connection_list=connections[flex], type_data='SNR'),
                                        connection_list_data_extractor(connection_list=connections[shannon], type_data='SNR')],
               nbins=20, edge_color='k', color=['g', 'm', 'y'], label=[fixed, flex, shannon],
               title='Point 6 Lab 9 - Histogram of SNR for each type of tranceiver',
               ylabel='number of results', xlabel='SNR [dB]', savefig_path=lab9_fig2)

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
               nbins=8, edge_color='k', color=['y', 'm', 'r'], label=[fixed_rate_label, flex_rate_label, shannon_label],
               title = 'Point 6 Lab 9 - Histogram of Bit Rates per transceiver', bbox_to_anchor=(0.5, -0.35), loc='lower center',
               ylabel='number of results', bottom=0.25, xlabel='Bit rate [Gbps]', savefig_path = lab9_fig3)

print('Blocking events for fixed_rate network: ', number_blocking_events_evaluation(connections[fixed]))
print('Blocking events for flex_rate network: ', number_blocking_events_evaluation(connections[flex]))
print('Blocking events for shannon network: ', number_blocking_events_evaluation(connections[shannon]))
print('Blocking events for fixed_rate network: ', number_blocking_events_evaluation(connections[fixed]), file=file_print)
print('Blocking events for flex_rate network: ', number_blocking_events_evaluation(connections[flex]), file=file_print)
print('Blocking events for shannon network: ', number_blocking_events_evaluation(connections[shannon]), file=file_print)
# plt.draw() # has to be put at the end

#################### Point 7 ##############################à
print('\n\t\t\tPoint 7 Lab 9\n')
print('\n\t\t\tPoint 7 Lab 9\n', file=file_print)
connections_fixed_per_M = []
number_connections_fixed_rate_per_M = []
number_blocking_events_fixed_rate_per_M = []
capacities_fixed_rate_per_M = []
average_bitrate_fixed_rate_per_M = []

connections_flex_per_M = []
number_connections_flex_rate_per_M = []
number_blocking_events_flex_rate_per_M = []
capacities_flex_rate_per_M = []
average_bitrate_flex_rate_per_M = []

connections_shannon_per_M = []
number_connections_shannon_per_M = []
number_blocking_events_shannon_per_M = []
capacities_shannon_per_M = []
average_bitrate_shannon_per_M = []

M_list = []
for M in [1, 5, 15, 25, 35, 45, 55]: # [1, 10, 25, 45, 60] # range(1, 47, 9)
    M_list.append(M)

    ################### FIXED RATE #####################
    print('\t\tFor M = ', str(M), ':')
    print('\t\tFor M = ', str(M), ':', file=file_print)
    connections_fixed_rate = random_generation_with_traffic_matrix(network=network_fixed_rate, M_traffic=M, set_lat_or_snr=set_latency_or_snr)
    print('Fixed Rate')
    print('Fixed Rate', file=file_print)
    number_connections_fixed_rate = len(connections_fixed_rate)
    number_blocking_events_fixed_rate = number_blocking_events_evaluation(connections_fixed_rate)
    print('\tTotal connections for fixed_rate network: ', number_connections_fixed_rate)
    print('\tBlocking events for fixed_rate network: ', number_blocking_events_fixed_rate)
    print('\tTotal connections for fixed_rate network: ', number_connections_fixed_rate, file=file_print)
    print('\tBlocking events for fixed_rate network: ', number_blocking_events_fixed_rate, file=file_print)
    connections_fixed_per_M.append(connections_fixed_rate)
    number_connections_fixed_rate_per_M.append(number_connections_fixed_rate)
    number_blocking_events_fixed_rate_per_M.append(number_blocking_events_fixed_rate)
    ############## CAPACITY and BITRATE
    [capacity_fixed_rate, average_bit_rate_fixed_rate] = capacity_and_avarage(connections_list=connections_fixed_rate)
    capacities_fixed_rate_per_M.append(capacity_fixed_rate)
    average_bitrate_fixed_rate_per_M.append(average_bit_rate_fixed_rate)
    ## labels
    fixed_rate_label = 'Fixed Rate with average Rb=' + str(np.round(average_bit_rate_fixed_rate, 3)) + ' Gbps and C=' +\
                       str(np.round(capacity_fixed_rate * 1e-3, 3)) + ' Tbps'
    print('\t'+fixed_rate_label)
    print('\t' + fixed_rate_label, file=file_print)

    ################### FLEX RATE #######################à
    connections_flex_rate = random_generation_with_traffic_matrix(network=network_flex_rate, M_traffic=M, set_lat_or_snr=set_latency_or_snr)
    print('Flex Rate')
    print('Flex Rate', file=file_print)
    number_connections_flex_rate = len(connections_flex_rate)
    number_blocking_events_flex_rate = number_blocking_events_evaluation(connections_flex_rate)
    print('\tTotal connections for flex_rate network: ', number_connections_flex_rate)
    print('\tBlocking events for flex_rate network: ', number_blocking_events_flex_rate)
    print('\tTotal connections for flex_rate network: ', number_connections_flex_rate, file=file_print)
    print('\tBlocking events for flex_rate network: ', number_blocking_events_flex_rate, file=file_print)
    connections_flex_per_M.append(connections_flex_rate)
    number_connections_flex_rate_per_M.append(number_connections_flex_rate)
    number_blocking_events_flex_rate_per_M.append(number_blocking_events_flex_rate)
    ############## CAPACITY and BITRATE
    [capacity_flex_rate, average_bit_rate_flex_rate] = capacity_and_avarage(connections_list=connections_flex_rate)
    capacities_flex_rate_per_M.append(capacity_flex_rate)
    average_bitrate_flex_rate_per_M.append(average_bit_rate_flex_rate)
    ## labels
    flex_rate_label = 'Flex Rate with average Rb=' + str(np.round(average_bit_rate_flex_rate, 3)) + ' Gbps and C=' + \
                       str(np.round(capacity_flex_rate * 1e-3, 3)) + ' Tbps'
    print('\t' + flex_rate_label)
    print('\t' + flex_rate_label, file=file_print)

    ################### SHANNON RATE ########################
    connections_shannon = random_generation_with_traffic_matrix(network=network_shannon, M_traffic=M, set_lat_or_snr=set_latency_or_snr)
    print('Shannon Rate')
    print('Shannon Rate', file=file_print)
    number_connections_shannon = len(connections_shannon)
    number_blocking_events_shannon = number_blocking_events_evaluation(connections_shannon)
    print('\tTotal connections for shannon network: ', number_connections_shannon)
    print('\tBlocking events for shannon network: ', number_blocking_events_shannon)
    print('\tTotal connections for shannon network: ', number_connections_shannon, file=file_print)
    print('\tBlocking events for shannon network: ', number_blocking_events_shannon, file=file_print)
    connections_shannon_per_M.append(connections_shannon)
    number_connections_shannon_per_M.append(number_connections_shannon)
    number_blocking_events_shannon_per_M.append(number_blocking_events_shannon)
    ############## CAPACITY and BITRATE
    [capacity_shannon, average_bit_rate_shannon] = capacity_and_avarage(connections_list=connections_shannon)
    capacities_shannon_per_M.append(capacity_shannon)
    average_bitrate_shannon_per_M.append(average_bit_rate_shannon)
    ## labels
    shannon_label = 'Shannon Rate with average Rb=' + str(np.round(average_bit_rate_shannon, 3)) + ' Gbps and C=' + \
                       str(np.round(capacity_shannon * 1e-3, 3)) + ' Tbps'
    print('\t' + shannon_label)
    print()
    print('\t' + shannon_label, file=file_print)
    print(file=file_print)

#################### SNR
plot_histogram(figure_num=(4), list_data=[connection_list_data_extractor(connection_list=connections_fixed_per_M[i], type_data='SNR') for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='SNR [dB]', ylabel='number of results', title=('Point 7 Lab 9 - Fixed Rate SNR - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=lab9_fig4)
plot_histogram(figure_num=(5), list_data=[connection_list_data_extractor(connection_list=connections_flex_per_M[i], type_data='SNR') for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='SNR [dB]', ylabel='number of results', title=('Point 7 Lab 9 - Flex Rate SNR - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=lab9_fig5)
plot_histogram(figure_num=(6), list_data=[connection_list_data_extractor(connection_list=connections_shannon_per_M[i], type_data='SNR') for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='SNR [dB]', ylabel='number of results', title=('Point 7 Lab 9 - Shannon Rate SNR - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=lab9_fig6)
##################### BITRATE
plot_histogram(figure_num=(7), list_data=[connection_list_data_extractor(connection_list=connections_fixed_per_M[i], type_data='Rb') for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='Rb [Gbps]', ylabel='number of results', title=('Point 7 Lab 9 - Fixed Rate bitrate - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=lab9_fig7)
plot_histogram(figure_num=(8), list_data=[connection_list_data_extractor(connection_list=connections_flex_per_M[i], type_data='Rb') for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='Rb [Gbps]', ylabel='number of results', title=('Point 7 Lab 9 - Flex Rate bitrate - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=lab9_fig8)
plot_histogram(figure_num=(9), list_data=[connection_list_data_extractor(connection_list=connections_shannon_per_M[i], type_data='Rb') for i in range(0, len(M_list))],
               nbins=None, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='Rb [Gbps]', ylabel='number of results', title=('Point 7 Lab 9 - Shannon Rate bitrate - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=lab9_fig9)
##################### LAT
plot_histogram(figure_num=(10), list_data=[np.array(connection_list_data_extractor(connection_list=connections_fixed_per_M[i], type_data='LAT'))*1e3 for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='Latency [ms]', ylabel='number of results', title=('Point 7 Lab 9 - Fixed Rate latency - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=lab9_fig10)
plot_histogram(figure_num=(11), list_data=[np.array(connection_list_data_extractor(connection_list=connections_flex_per_M[i], type_data='LAT'))*1e3 for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='Latency [ms]', ylabel='number of results', title=('Point 7 Lab 9 - Flex Rate latency - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=lab9_fig11)
plot_histogram(figure_num=(12), list_data=[np.array(connection_list_data_extractor(connection_list=connections_shannon_per_M[i], type_data='LAT'))*1e3 for i in range(0, len(M_list))],
               nbins=8, edge_color='k', color=None, label=['M='+str(M_list[i]) for i in range(0, len(M_list))],
               xlabel='Latency [ms]', ylabel='number of results', title=('Point 7 Lab 9 - Shannon Rate latency - with best '+set_latency_or_snr), alpha=0.75,
               savefig_path=lab9_fig12)

################ Number Connections
plot_bar(figure_num=(13), list_data=[[number_connections_fixed_rate_per_M[i] for i in range(0, len(M_list))],
                                           [number_connections_flex_rate_per_M[i] for i in range(0, len(M_list))],
                                           [number_connections_shannon_per_M[i] for i in range(0, len(M_list))]],
         x_ticks = [M for M in M_list], bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center',
         edge_color='k', color=None, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'],
         xlabel='M value for traffic matrix', ylabel='number of connections',
         title=('Point 7 Lab 9 - Number of connections - with best '+set_latency_or_snr), alpha=0.75,
         savefig_path=None)
################ Number Blocking Events
plot_bar(figure_num=(14), list_data=[[number_blocking_events_fixed_rate_per_M[i] for i in range(0, len(M_list))],
                                           [number_blocking_events_flex_rate_per_M[i] for i in range(0, len(M_list))],
                                           [number_blocking_events_shannon_per_M[i] for i in range(0, len(M_list))]],
         x_ticks = [M for M in M_list], bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center',
         edge_color='k', color=None, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'],
         xlabel='M value for traffic matrix', ylabel='number of blocking events',
         title=('Point 7 Lab 9 - Number of blocking events - with best '+set_latency_or_snr), alpha=0.75,
         savefig_path=None)
################ Capacity
plot_bar(figure_num=(15), list_data=[[capacities_fixed_rate_per_M[i] for i in range(0, len(M_list))],
                                           [capacities_flex_rate_per_M[i] for i in range(0, len(M_list))],
                                           [capacities_shannon_per_M[i] for i in range(0, len(M_list))]],
         x_ticks = [M for M in M_list], bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center',
         edge_color='k', color=None, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'],
         xlabel='M value for traffic matrix', ylabel='Capacity [Gbps]',
         title=('Point 7 Lab 9 - Capacity - with best '+set_latency_or_snr), alpha=0.75,
         savefig_path=None)
################ Average Bit Rate
plot_bar(figure_num=(16), list_data=[[average_bitrate_fixed_rate_per_M[i] for i in range(0, len(M_list))],
                                           [average_bitrate_flex_rate_per_M[i] for i in range(0, len(M_list))],
                                           [average_bitrate_shannon_per_M[i] for i in range(0, len(M_list))]],
         x_ticks = [M for M in M_list], bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center',
         edge_color='k', color=None, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'],
         xlabel='M value for traffic matrix', ylabel='Average Bit Rate [Gbps]',
         title=('Point 7 Lab 9 - Average Bit Rate - with best '+set_latency_or_snr), alpha=0.75,
         savefig_path=None)
# plt.pause(1)

file_print.close()
plt.show()