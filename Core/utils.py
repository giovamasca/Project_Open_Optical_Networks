import random
# import matplotlib
# matplotlib.use('TkAgg')
# import time
import matplotlib.pyplot as plt
# import os

import numpy as np

from Project_Open_Optical_Networks.Core.elements import Connection, Network

def network_generation_from_file(network_file):
    network = Network(network_file)
    return network
def random_generation_for_network(network, Numb_sim, network_label=None, set_lat_or_snr=None): # network and sumber of simulations
    set_lat_or_snr = set_lat_or_snr if set_lat_or_snr else 'SNR' # default
    nodes_gener = list(network.nodes.keys())  # extracts all nodes

    # avarage_bit_rate = 0
    connections_generated = []  # defined a list of connections
    for i in range(0, Numb_sim): # do a number of simulations equal to Numb_sim
        n1 = random.randint(0, len(nodes_gener) - 1)  # any position is ok
        n2 = random.randint(0, len(nodes_gener) - 1)
        while n2 == n1:
            n2 = random.randint(0, len(nodes_gener) - 1)  # repeat the random evaluation until there is a couple of nodes, not same values

        connection_generated = Connection(input_node=nodes_gener[n1], output_node=nodes_gener[n2], signal_power=1e-3)  # creates connection
        connection_generated = network.stream(connection=connection_generated, set_latency_or_snr=set_lat_or_snr, use_state=True)  # stream it with state on and snr set
        # if connection_generated.latency==np.NaN:
        #     continue # avoid this connection
        # with np.NaN the histograms avoid the corresponding values
        connections_generated.append(connection_generated) # appends connection
        # avarage_bit_rate += connection_generated.bit_rate
    print('Evaluated ', Numb_sim, ' simulations for network ', network_label)
    return connections_generated
def random_generation_with_traffic_matrix_with_while(network, M_traffic=None, set_lat_or_snr=None):
    set_lat_or_snr = set_lat_or_snr if set_lat_or_snr else 'SNR'
    network.reset(M_traffic_matrix=M_traffic)
    connections = []
    # i=0
    while not network.traffic_matrix_saturated(): # generates a list until network is saturated
        connection_generated = network.connection_with_traffic_matrix(set_latency_or_snr=set_lat_or_snr, use_state=True)
        # input_node = connection_generated.input
        # output_node = connection_generated.output
        connections.append(connection_generated)
        # i += 1
        # print(i)
    return connections
def single_traffic_matrix_scenario(network, M_traffic_static, set_lat_or_snr, N_iterations):
    network.reset(M_traffic_matrix=M_traffic_static)
    connections = []
    for i in range(0, N_iterations):
        connection_generated = network.connection_with_traffic_matrix(set_latency_or_snr=set_lat_or_snr, use_state=True)
        connections.append(connection_generated)
    return connections
def number_blocking_events_evaluation(connection_list):
    blocking_events = sum(connection.channel is None for connection in connection_list)
    # blocking_events = sum(connection.latency is np.NaN for connection in connection_list)
    return blocking_events
def connection_list_data_extractor(connection_list, type_data):
    ############# EXTRACTS LIST OF DATA FROM LIST OF CONNECTIONS ###############
    ### Setting properties:
    ### 'I/O' for input and output node extraction
    ### 'SNR' for SNR extraction
    ### 'LAT' for latency extraction
    ### 'Rb' for bit rate extraction
    ### if needed other implementations possible
    list_data = ''
    if type_data == 'I/O':
        list_data = [connection.input + connection.output for connection in connection_list]
    elif type_data == 'SNR':
        list_data = [connection.snr for connection in connection_list]
    elif type_data == 'LAT':
        list_data = [connection.latency for connection in connection_list]
    elif type_data == 'Rb':
        list_data = [connection.bit_rate for connection in connection_list]
    else:
        print('Error in type_data for connection list extractor!')
        exit(5)
    return list_data
def print_and_save(text, file=None):
    print(text)
    if file:
        file_print = open(file, 'a')
        print(text, file=file_print)
        file_print.close()
def plot_histogram(figure_num, list_data, nbins, edge_color, color, label, title='', ylabel = '', xlabel = '', savefig_path = None, bbox_to_anchor = None, loc = None, bottom = None, NaN_display=False, alpha=None):
    if NaN_display:
        list_data = list(np.nan_to_num(list_data)) # replace NaN with 0

    fig = plt.figure(figure_num)
    fig.subplots_adjust(bottom=bottom)
    plt.hist( list_data, bins=nbins, edgecolor=edge_color, color = color, label = label, alpha=alpha)
    plt.title(title)
    plt.legend(bbox_to_anchor = bbox_to_anchor, loc = loc)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)

    # savefig_path = None ##### AVOIDED SAVE AS DEBUG
    if savefig_path:
        # if not os.path.isdir('../Results/Lab9'): # if Results doesn't exists, it creates it
        #     os.makedirs('../Results/Lab9')
        plt.savefig(savefig_path)

    # fig.canvas.draw()
    # plt.pause(0.25)
    return
def plot_bar(figure_num=None, list_data=None, x_ticks=None, edge_color='k', color=None, label='', title='', ylabel = '', xlabel = '', savefig_path = None, bbox_to_anchor = None, loc = None, bottom = None, NaN_display=False, alpha=None):
    if NaN_display:
        list_data = list(np.nan_to_num(list_data)) # replace NaN with 0

    x = np.arange(len(x_ticks))

    fig = plt.figure(figure_num)
    # fig, ax = plt.subplots()
    ax = plt.gca()

    fig.subplots_adjust(bottom=bottom)
    for index in range(0, len(list_data)):
        x_i = np.arange(len(x_ticks))
        width=0.25

        x_i = x + width*( index + 0.5 - len(list_data) / 2 )

        # if index + 0.5 < len(list_data) / 2 :
        #     x_i = x - width*(len(list_data) / 2 - index + 0.5 )
        # elif index + 0.5 > len(list_data) / 2 :
        #     x_i = x + width*index
        plt.bar(x = x_i , width=width, height=list_data[index], edgecolor=edge_color, color=color[index] if color else None, alpha=alpha)
    # plt.bar( x=x-0.25, width=0.25, height=list_data[0], edgecolor=edge_color, color = color, alpha=alpha)
    # plt.bar(x=x+0.0, width=0.25, height=list_data[1], edgecolor=edge_color, color=color, alpha=alpha)
    # plt.bar(x=x+0.25, width=0.25, height=list_data[2], edgecolor=edge_color, color=color, alpha=alpha)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    ax.set_xticks(x, x_ticks)
    ax.legend(labels=label, bbox_to_anchor = bbox_to_anchor, loc = loc)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)

    # savefig_path = None ##### AVOIDED SAVE AS DEBUG
    if savefig_path: # if None avoid save
        # if not os.path.isdir('../Results/Lab9'): # if Results doesn't exists, it creates it
        #     os.makedirs('../Results/Lab9')
        plt.savefig(savefig_path)

    # fig.canvas.draw()
    # plt.pause(0.25)
    return
###########################################      LAB 10     ############################################################
def lab10_point1(network, M, set_latency_or_snr, N_iterations, label, file_console=None):
    from Project_Open_Optical_Networks.Core.science_utils import SNR_metrics, capacity_metrics

    print_and_save(text='M=' + str(M) + ':', file=file_console)

    connections = single_traffic_matrix_scenario(network=network, M_traffic_static=M, set_lat_or_snr=set_latency_or_snr, N_iterations=N_iterations)

    print_and_save('Fixed Rate', file=file_console)

    number_connections = len(connections)
    number_blocking_events = number_blocking_events_evaluation(connections)

    print_and_save('\tTotal connections for ' + label + ' network: ' + str(number_connections), file=file_console)
    print_and_save('\tBlocking events for ' + label + ' network: ' + str(number_blocking_events), file=file_console)

    ############## CAPACITY and BITRATE
    [SNR_ave_per_link, SNR_max, SNR_min] = SNR_metrics(connection_list=connections)
    [capacity, average_bitrate, bitrate_max, bitrate_min] = capacity_metrics(connections_list=connections)

    ## labels
    SNR_fixed_rate_label = '\t' + label + ' with average SNR = ' + str(np.round(SNR_ave_per_link, 3)) + ' dB, maximum SNR = ' +\
                            str(np.round(SNR_max, 3)) + ' dB, minimum SNR = ' + str(np.round(SNR_min, 3)) + ' dB'
    capacity_fixed_rate_label = '\t' + label + ' with average Rb = ' + str(np.round(average_bitrate, 3)) + ' Gbps and C = ' +\
                           str(np.round(capacity * 1e-3, 3)) + ' Tbps, ' +\
                           'maximum bit rate = ' + str(np.round(bitrate_max, 3)) + ' Gbps and minimum bit rate = ' +\
                           str(np.round(bitrate_min, 3)) + ' Gbps.'

    print_and_save(text=capacity_fixed_rate_label, file=file_console)
    print_and_save(text=SNR_fixed_rate_label, file=file_console)

    # plot_bar()
########################################################################################################################