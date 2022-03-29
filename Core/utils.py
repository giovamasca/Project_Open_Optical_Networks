import random
# import matplotlib
# matplotlib.use('TkAgg')
import time
import matplotlib.pyplot as plt
import os

import numpy as np

from Project_Open_Optical_Networks.Core.elements import Connection, Network

def network_generation_from_file(network_file):
    network = Network(network_file)
    return network
def random_generation_for_network(network, Numb_sim, network_label=None): # network and sumber of simulations
    nodes_gener = list(network.nodes.keys())  # extracts all nodes

    # avarage_bit_rate = 0
    connections_generated = []  # defined a list of connections
    for i in range(0, Numb_sim): # do a number of simulations equal to Numb_sim
        n1 = random.randint(0, len(nodes_gener) - 1)  # any position is ok
        n2 = random.randint(0, len(nodes_gener) - 1)
        while n2 == n1:
            n2 = random.randint(0, len(nodes_gener) - 1)  # repeat the random evaluation until there is a couple of nodes, not same values

        connection_generated = Connection(input_node=nodes_gener[n1], output_node=nodes_gener[n2], signal_power=1e-3)  # creates connection
        connection_generated = network.stream(connection=connection_generated, set_latency_or_snr='snr', use_state=True)  # stream it with state on and snr set
        # if connection_generated.latency==np.NaN:
        #     continue # avoid this connection
        # with np.NaN the histograms avoid the corresponding values
        connections_generated.append(connection_generated) # appends connection
        # avarage_bit_rate += connection_generated.bit_rate
    print('Evaluated ', Numb_sim, ' simulations for network ', network_label)
    return connections_generated
def random_generation_with_traffic_matrix(network, M_traffic=None):
    network.reset(M_traffic_matrix=M_traffic)
    connections = []
    # i=0
    while not network.traffic_matrix_saturated(): # generates a list until network is saturated
        connection_generated = network.connection_with_traffic_matrix(set_latency_or_snr='snr', use_state=True)
        # input_node = connection_generated.input
        # output_node = connection_generated.output
        connections.append(connection_generated)
        # i += 1
        # print(i)
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
    return list_data

def plot_histogram(figure_num, list_data, nbins, edge_color, color, label, title='', ylabel = '', xlabel = '', savefig_path = None, bbox_to_anchor = None, loc = None, bottom = None, NaN_display=False):
    if NaN_display:
        list_data = list(np.nan_to_num(list_data)) # replace NaN with 0

    fig = plt.figure(figure_num)
    fig.subplots_adjust(bottom=bottom)
    plt.hist( list_data, bins=nbins, edgecolor=edge_color, color = color, label = label)
    plt.title(title)
    plt.legend(bbox_to_anchor = bbox_to_anchor, loc = loc)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)

    # savefig_path = None ##### AVOIDED SAVE AS DEBUG
    if savefig_path:
        if not os.path.isdir('../Results'): # if Results doesn't exists, it creates it
            os.makedirs('../Results')
        plt.savefig(savefig_path)
    # plt.ion()
    fig.canvas.draw()
    return