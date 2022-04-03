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
def random_generation_for_network(network, Numb_sim, network_label=None, set_lat_or_snr=None): # network and sumber of simulations - lab 9
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
def random_generation_with_traffic_matrix_with_while(network, M_traffic=None, set_lat_or_snr=None): # lab 9 Monte Carlo run
    from Project_Open_Optical_Networks.Core.parameters import watchdog_limit

    set_lat_or_snr = set_lat_or_snr if set_lat_or_snr else 'SNR'
    network.reset(M_traffic_matrix=M_traffic)
    connections = []
    watchdog = 0
    while not network.traffic_matrix_saturated(): # generates a list until network is saturated
        connection_generated = network.connection_with_traffic_matrix(set_latency_or_snr=set_lat_or_snr, use_state=True)
        # input_node = connection_generated.input
        # output_node = connection_generated.output
        watchdog += 1
        if connection_generated and watchdog<=watchdog_limit:
            connections.append(connection_generated)
        else:
            print('\nWhile stopped by WATCHDOG\nNo more connections available, even if network is not saturated')
            return connections
        # i += 1
        # print(i)
    return connections
def single_traffic_matrix_scenario(network, M_traffic_static, set_lat_or_snr, N_iterations, file_console=None):
    network.reset(M_traffic_matrix=M_traffic_static)
    connections = []
    for i in range(0, N_iterations):
        connection_generated = network.connection_with_traffic_matrix(set_latency_or_snr=set_lat_or_snr, use_state=True)
        if connection_generated:
            connections.append(connection_generated)
        else:
            print_and_save('While stopped by WATCHDOG.', file=file_console)
            if network.traffic_matrix_saturated():
                print_and_save('Following connections have saturated traffic matrix.', file=file_console)
            else:
                print_and_save('Following connections have unsaturated traffic matrix, but all states occupied.', file=file_console)
            return connections
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
def print_and_save(text, file=None): # used to print and save in file
    print(text)
    if file:
        file_print = open(file, 'a')
        print(text, file=file_print)
        file_print.close()
def plot_histogram(figure_num=None, list_data=None, nbins=None, edge_color='k', color=None, label=None, title='', ylabel = '', xlabel = '', savefig_path = None, bbox_to_anchor = None, loc = None, bottom = None, NaN_display=False, alpha=None):
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
def plot_bar(figure_num=None, list_data=None, x_ticks=None, edge_color='k', color=None, label=None, title='', ylabel = '', xlabel = '', savefig_path = None, bbox_to_anchor = None, loc = None, bottom = None, NaN_display=False, alpha=None):
    if NaN_display:
        list_data = list(np.nan_to_num(list_data)) # replace NaN with 0

    x = np.arange(len(x_ticks)) if x_ticks else 1

    fig = plt.figure(figure_num)
    # fig, ax = plt.subplots()
    ax = plt.gca()

    fig.subplots_adjust(bottom=bottom)
    for index in range(0, len(list_data)):
        # x_i = np.arange(len(x_ticks))
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

    if x_ticks:
        ax.set_xticks(x, x_ticks)
    else:
        plt.tick_params( axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
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
def lab10_point1_results(network, M, set_latency_or_snr, N_iterations, label, file_console=None):
    from Project_Open_Optical_Networks.Core.science_utils import SNR_metrics, latency_metrics, capacity_metrics

    results = {} # contains all results as a dictionary

    # print_and_save(text='M=' + str(M) + ':', file=file_console)

    connections = single_traffic_matrix_scenario(network=network, M_traffic_static=M, set_lat_or_snr=set_latency_or_snr, N_iterations=N_iterations, file_console=file_console)
    results['connections'] = connections

    print_and_save(label + ' Rate with N interactions = ' + str(N_iterations), file=file_console)

    number_connections = len(connections)
    number_blocking_events = number_blocking_events_evaluation(connections)
    results['number_connections'] = number_connections
    results['number_blocking_events'] = number_blocking_events


    print_and_save('\tTotal connections for ' + label + ' network: ' + str(number_connections), file=file_console)
    print_and_save('\tBlocking events for ' + label + ' network: ' + str(number_blocking_events), file=file_console)

    ############## CAPACITY and BITRATE
    [SNR_ave_per_link, SNR_max, SNR_min] = SNR_metrics(connection_list=connections)
    [latency_average, latency_max, latency_min] = latency_metrics(connection_list=connections)
    [capacity, average_bitrate, bitrate_max, bitrate_min] = capacity_metrics(connections_list=connections)
    results['SNR_ave_per_link'] = SNR_ave_per_link
    results['SNR_max'] = SNR_max
    results['SNR_min'] = SNR_min
    results['latency_average'] = latency_average
    results['latency_max'] = latency_max
    results['latency_min'] = latency_min
    results['capacity'] = capacity
    results['average_bitrate'] = average_bitrate
    results['bitrate_max'] = bitrate_max
    results['bitrate_min'] = bitrate_min

    ## labels
    SNR_fixed_rate_label = '\t' + label + ' with average SNR = ' + str(np.round(SNR_ave_per_link, 3)) + ' dB, maximum SNR = ' +\
                            str(np.round(SNR_max, 3)) + ' dB, minimum SNR = ' + str(np.round(SNR_min, 3)) + ' dB'
    capacity_fixed_rate_label = '\t' + label + ' with average Rb = ' + str(np.round(average_bitrate, 3)) + ' Gbps and C = ' +\
                           str(np.round(capacity * 1e-3, 3)) + ' Tbps, ' +\
                           'maximum bit rate = ' + str(np.round(bitrate_max, 3)) + ' Gbps and minimum bit rate = ' +\
                           str(np.round(bitrate_min, 3)) + ' Gbps.'

    print_and_save(text=capacity_fixed_rate_label, file=file_console)
    print_and_save(text=SNR_fixed_rate_label, file=file_console)

    return results
def lab10_point1_graphs(initial_fig, images_folder, results, set_latency_or_snr, M, N_iterations):
    ########################################            FIGURE    SNR         #################################################
    fig_num = initial_fig
    preface_title = 'Lab 10 Point 1 - '
    savefig_path = images_folder / (
                'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_SNR_' + \
                '_with_M_' + str(M) + '_and_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'SNRs with M = ' + str(M) + ', N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_histogram(figure_num=fig_num, list_data=[
        connection_list_data_extractor(connection_list=results[label]['connections'], type_data='SNR') for label in
        results],
                   xlabel='SNR [dB]', ylabel='Number of results', nbins=15, alpha=0.75,
                   label=[label.replace('_', ' ') for label in results], title=title, savefig_path=savefig_path)
    #######################################################################################################################
    ########################################            FIGURE   LAT         #################################################
    fig_num = fig_num + 1
    # preface_title = 'Lab 10 Point 1 - '
    savefig_path = images_folder / (
                'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_latency_' + \
                '_with_M_' + str(M) + '_and_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Latencies with M = ' + str(M) + ', N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_histogram(figure_num=fig_num, list_data=[
        np.array(connection_list_data_extractor(connection_list=results[label]['connections'], type_data='LAT')) * 1e3
        for label in results],
                   xlabel='delay [ms]', ylabel='Number of results', nbins=15, alpha=0.75,
                   label=[label.replace('_', ' ') for label in results], title=title, savefig_path=savefig_path)
    #######################################################################################################################
    ########################################            FIGURE    BITRATES        #################################################
    fig_num = fig_num + 1
    # preface_title = 'Lab 10 Point 1 - '
    savefig_path = images_folder / (
                'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_bitrate_' + \
                '_with_M_' + str(M) + '_and_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Bit Rates with M = ' + str(M) + ', N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_histogram(figure_num=fig_num, list_data=[
        connection_list_data_extractor(connection_list=results[label]['connections'], type_data='Rb') for label in
        results],
                   xlabel='Gbps', ylabel='Number of results', nbins=15, alpha=0.75,
                   label=[label.replace('_', ' ') for label in results], title=title, savefig_path=savefig_path)
    #######################################################################################################################
    ########################################            FIGURE     num CONNECTIONS       #################################################
    fig_num = fig_num + 1
    # preface_title = 'Lab 10 Point 1 - '
    savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-',
                                                                                                               '') + '_number_connections_' + \
                                    '_with_M_' + str(M) + '_and_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Number of connections with M = ' + str(M) + ', N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num, list_data=[results[label]['number_connections'] for label in results],
             x_ticks=None, xlabel='M=' + str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35),
             loc='lower center',
             label=[label.replace('_', ' ') for label in results], title=title, savefig_path=savefig_path)
    #######################################################################################################################
    ########################################            FIGURE     num BLOCKING EVENTS       #################################################
    fig_num = fig_num + 1
    # preface_title = 'Lab 10 Point 1 - '
    savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-',
                                                                                                               '') + '_number_blocking_events_' + \
                                    '_with_M_' + str(M) + '_and_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Number of blocking events with M = ' + str(M) + ', N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num, list_data=[results[label]['number_blocking_events'] for label in results],
             x_ticks=None, xlabel='M=' + str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35),
             loc='lower center',
             label=[label.replace('_', ' ') for label in results], title=title, savefig_path=savefig_path)
    #######################################################################################################################
    ########################################            FIGURE     CAPACITIES       #################################################
    fig_num = fig_num + 1
    # preface_title = 'Lab 10 Point 1 - '
    savefig_path = images_folder / (
                'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_capacities_' + \
                '_with_M_' + str(M) + '_and_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Capacities with M = ' + str(M) + ', N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num, list_data=np.array([results[label]['capacity'] for label in results]) * 1e-3,
             ylabel='[Tbps]',
             x_ticks=None, xlabel='M=' + str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35),
             loc='lower center',
             label=[label.replace('_', ' ') for label in results], title=title, savefig_path=savefig_path)
    #######################################################################################################################
    ########################################            FIGURE     BITRATE MAX       #################################################
    fig_num = fig_num + 1
    # preface_title = 'Lab 10 Point 1 - '
    savefig_path = images_folder / (
                'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_max_bitrate_' + \
                '_with_M_' + str(M) + '_and_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Maximum Bit Rates with M = ' + str(M) + ', N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num, list_data=[results[label]['bitrate_max'] for label in results], ylabel='[Gbps]',
             x_ticks=None, xlabel='M=' + str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35),
             loc='lower center',
             label=[label.replace('_', ' ') for label in results], title=title, savefig_path=savefig_path)
    #######################################################################################################################
    ########################################            FIGURE     BITRATE MIN       #################################################
    fig_num = fig_num + 1
    # preface_title = 'Lab 10 Point 1 - '
    savefig_path = images_folder / (
                'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_min_bitrate_' + \
                '_with_M_' + str(M) + '_and_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Minimum Bit Rates with M = ' + str(M) + ', N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num, list_data=[results[label]['bitrate_min'] for label in results], ylabel='[Gbps]',
             x_ticks=None, xlabel='M=' + str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35),
             loc='lower center',
             label=[label.replace('_', ' ') for label in results], title=title, savefig_path=savefig_path)
    #######################################################################################################################
    ########################################            FIGURE     SNR per link       #################################################
    fig_num = fig_num + 1
    # preface_title = 'Lab 10 Point 1 - '
    savefig_path = images_folder / (
                'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_SNR_per_link_' + \
                '_with_M_' + str(M) + '_and_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'SNRs per link (average) with M = ' + str(M) + ', N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num, list_data=[results[label]['SNR_ave_per_link'] for label in results], ylabel='[dB]',
             x_ticks=None, xlabel='M=' + str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35),
             loc='lower center',
             label=[label.replace('_', ' ') for label in results], title=title, savefig_path=savefig_path)
    #######################################################################################################################
    ########################################            FIGURE     SNR MAX       #################################################
    fig_num = fig_num + 1
    # preface_title = 'Lab 10 Point 1 - '
    savefig_path = images_folder / (
                'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_max_SNR_' + \
                '_with_M_' + str(M) + '_and_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Maximum SNRs with M = ' + str(M) + ', N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num, list_data=[results[label]['SNR_max'] for label in results], ylabel='[dB]',
             x_ticks=None, xlabel='M=' + str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35),
             loc='lower center',
             label=[label.replace('_', ' ') for label in results], title=title, savefig_path=savefig_path)
    #######################################################################################################################
    ########################################            FIGURE     SNR MIN       #################################################
    fig_num = fig_num + 1
    # preface_title = 'Lab 10 Point 1 - '
    savefig_path = images_folder / (
                'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_min_SNR_' + \
                '_with_M_' + str(M) + '_and_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Minimum SNRs with M = ' + str(M) + ', N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num, list_data=[results[label]['SNR_min'] for label in results], ylabel='[dB]',
             x_ticks=None, xlabel='M=' + str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35),
             loc='lower center',
             label=[label.replace('_', ' ') for label in results], title=title, savefig_path=savefig_path)
    #######################################################################################################################
    final_fig = fig_num + 1
    return final_fig
########################################################################################################################
def lab10_point2_graphs(results_per_M, M_list, images_folder, N_iterations, set_latency_or_snr, initial_fig , colors = None):
    #################################################### results
    ## number conncetions
    Number_connections_per_M_fixed = []
    Number_connections_per_M_flex = []
    Number_connections_per_M_shannon = []
    ## number blocking events
    Number_blocking_events_per_M_fixed = []
    Number_blocking_events_per_M_flex = []
    Number_blocking_events_per_M_shannon = []
    ## SNR ave per link
    SNRs_ave_per_link_per_M_fixed = []
    SNRs_ave_per_link_per_M_flex = []
    SNRs_ave_per_link_per_M_shannon = []
    ## SNR max
    SNRs_max_per_M_fixed = []
    SNRs_max_per_M_flex = []
    SNRs_max_per_M_shannon = []
    ## Latency average
    Latency_average_per_M_fixed = []
    Latency_average_per_M_flex = []
    Latency_average_per_M_shannon = []
    ## Latency max
    Latency_max_per_M_fixed = []
    Latency_max_per_M_flex = []
    Latency_max_per_M_shannon = []
    ## Latency min
    Latency_min_per_M_fixed = []
    Latency_min_per_M_flex = []
    Latency_min_per_M_shannon = []
    ## SNR min
    SNRs_min_per_M_fixed = []
    SNRs_min_per_M_flex = []
    SNRs_min_per_M_shannon = []
    ## Capacity
    Capacity_per_M_fixed = []
    Capacity_per_M_flex = []
    Capacity_per_M_shannon = []
    ## average_bitrate
    Average_Bit_Rate_per_M_fixed = []
    Average_Bit_Rate_per_M_flex = []
    Average_Bit_Rate_per_M_shannon = []
    ## bitrate_max
    Bit_Rate_max_per_M_fixed = []
    Bit_Rate_max_per_M_flex = []
    Bit_Rate_max_per_M_shannon = []
    ## bitrate_min
    Bit_Rate_min_per_M_fixed = []
    Bit_Rate_min_per_M_flex = []
    Bit_Rate_min_per_M_shannon = []
    ##
    for M in results_per_M:
        results_per_actual_M = results_per_M[str(M)]
        ###### Simulations
        ## Number connections
        Number_connections_per_M_fixed.append(results_per_actual_M['Fixed_Rate']['number_connections'])
        Number_connections_per_M_flex.append(results_per_actual_M['Flex_Rate']['number_connections'])
        Number_connections_per_M_shannon.append(results_per_actual_M['Shannon_Rate']['number_connections'])
        ## Number blocking events
        Number_blocking_events_per_M_fixed.append(results_per_actual_M['Fixed_Rate']['number_blocking_events'])
        Number_blocking_events_per_M_flex.append(results_per_actual_M['Flex_Rate']['number_blocking_events'])
        Number_blocking_events_per_M_shannon.append(results_per_actual_M['Shannon_Rate']['number_blocking_events'])
        ###### SNRs
        ## SNR ave per link
        SNRs_ave_per_link_per_M_fixed.append(results_per_actual_M['Fixed_Rate']['SNR_ave_per_link'])
        SNRs_ave_per_link_per_M_flex.append(results_per_actual_M['Flex_Rate']['SNR_ave_per_link'])
        SNRs_ave_per_link_per_M_shannon.append(results_per_actual_M['Shannon_Rate']['SNR_ave_per_link'])
        ## SNR max
        SNRs_max_per_M_fixed.append(results_per_actual_M['Fixed_Rate']['SNR_max'])
        SNRs_max_per_M_flex.append(results_per_actual_M['Flex_Rate']['SNR_max'])
        SNRs_max_per_M_shannon.append(results_per_actual_M['Shannon_Rate']['SNR_max'])
        ## SNR min
        SNRs_min_per_M_fixed.append(results_per_actual_M['Fixed_Rate']['SNR_min'])
        SNRs_min_per_M_flex.append(results_per_actual_M['Flex_Rate']['SNR_min'])
        SNRs_min_per_M_shannon.append(results_per_actual_M['Shannon_Rate']['SNR_min'])
        ######## Latencies
        ## Latency average
        Latency_average_per_M_fixed.append(results_per_actual_M['Fixed_Rate']['latency_average'])
        Latency_average_per_M_flex.append(results_per_actual_M['Flex_Rate']['latency_average'])
        Latency_average_per_M_shannon.append(results_per_actual_M['Shannon_Rate']['latency_average'])
        ## Latency max
        Latency_max_per_M_fixed.append(results_per_actual_M['Fixed_Rate']['latency_max'])
        Latency_max_per_M_flex.append(results_per_actual_M['Flex_Rate']['latency_max'])
        Latency_max_per_M_shannon.append(results_per_actual_M['Shannon_Rate']['latency_max'])
        ## Latency min
        Latency_min_per_M_fixed.append(results_per_actual_M['Fixed_Rate']['latency_min'])
        Latency_min_per_M_flex.append(results_per_actual_M['Flex_Rate']['latency_min'])
        Latency_min_per_M_shannon.append(results_per_actual_M['Shannon_Rate']['latency_min'])
        ###### Bit Rates
        ## Capacity
        Capacity_per_M_fixed.append(results_per_actual_M['Fixed_Rate']['capacity'])
        Capacity_per_M_flex.append(results_per_actual_M['Flex_Rate']['capacity'])
        Capacity_per_M_shannon.append(results_per_actual_M['Shannon_Rate']['capacity'])
        ## average_bitrate
        Average_Bit_Rate_per_M_fixed.append(results_per_actual_M['Fixed_Rate']['average_bitrate'])
        Average_Bit_Rate_per_M_flex.append(results_per_actual_M['Flex_Rate']['average_bitrate'])
        Average_Bit_Rate_per_M_shannon.append(results_per_actual_M['Shannon_Rate']['average_bitrate'])
        ## bitrate_max
        Bit_Rate_max_per_M_fixed.append(results_per_actual_M['Fixed_Rate']['bitrate_max'])
        Bit_Rate_max_per_M_flex.append(results_per_actual_M['Flex_Rate']['bitrate_max'])
        Bit_Rate_max_per_M_shannon.append(results_per_actual_M['Shannon_Rate']['bitrate_max'])
        ## bitrate_min
        Bit_Rate_min_per_M_fixed.append(results_per_actual_M['Fixed_Rate']['bitrate_min'])
        Bit_Rate_min_per_M_flex.append(results_per_actual_M['Flex_Rate']['bitrate_min'])
        Bit_Rate_min_per_M_shannon.append(results_per_actual_M['Shannon_Rate']['bitrate_min'])

    fig_num = initial_fig
    preface_title = 'Lab 10 Point 2 - '
    savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-',
                                                                                                               '') + '_number_connections_' + \
                                    '_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Number of connections with N = ' + str(
        N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num, list_data=[Number_connections_per_M_fixed, Number_connections_per_M_flex,
                                            Number_connections_per_M_shannon],
             x_ticks=M_list, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], alpha=0.75, xlabel='M',
             ylabel='Number of results',
             bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center', color=colors,
             title=title, savefig_path=savefig_path)
    fig_num += 1
    preface_title = 'Lab 10 Point 2 - '
    savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-',
                                                                                                               '') + '_number_blocking_events_' + \
                                    '_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Number of blocking events with N = ' + str(
        N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num, list_data=[Number_blocking_events_per_M_fixed, Number_blocking_events_per_M_flex,
                                            Number_blocking_events_per_M_shannon],
             x_ticks=M_list, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], alpha=0.75, xlabel='M',
             ylabel='Number of results',
             bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center', color=colors,
             title=title, savefig_path=savefig_path)
    fig_num += 1
    preface_title = 'Lab 10 Point 2 - '
    savefig_path = images_folder / (
                'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_SNR_per_link_' + \
                '_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'SNRs average per link with N = ' + str(
        N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num,
             list_data=[SNRs_ave_per_link_per_M_fixed, SNRs_ave_per_link_per_M_flex, SNRs_ave_per_link_per_M_shannon],
             x_ticks=M_list, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], alpha=0.75, xlabel='M', ylabel='[dB]',
             bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center', color=colors,
             title=title, savefig_path=savefig_path)
    fig_num = fig_num + 1
    preface_title = 'Lab 10 Point 2 - '
    savefig_path = images_folder / (
                'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_SNR_max_' + \
                '_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Maximum SNRs with N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num, list_data=[SNRs_max_per_M_fixed, SNRs_max_per_M_flex, SNRs_max_per_M_shannon],
             x_ticks=M_list, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], alpha=0.75, xlabel='M', ylabel='[dB]',
             bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center', color=colors,
             title=title, savefig_path=savefig_path)
    fig_num += 1
    preface_title = 'Lab 10 Point 2 - '
    savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_SNR_min_' + \
                '_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Minimum SNRs with N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num, list_data=[SNRs_min_per_M_fixed, SNRs_min_per_M_flex, SNRs_min_per_M_shannon],
             x_ticks=M_list, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], alpha=0.75, xlabel='M', ylabel='[dB]',
             bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center', color=colors,
             title=title, savefig_path=savefig_path)
    fig_num += 1
    preface_title = 'Lab 10 Point 2 - '
    savefig_path = images_folder / (
            'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_average_latency_' + \
            '_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Average latencies with N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num, list_data=[np.array(Latency_average_per_M_fixed)*1e3, np.array(Latency_average_per_M_flex)*1e3, np.array(Latency_average_per_M_shannon)*1e3],
             x_ticks=M_list, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], alpha=0.75, xlabel='M', ylabel='[ms]',
             bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center', color=colors,
             title=title, savefig_path=savefig_path)
    fig_num += 1
    preface_title = 'Lab 10 Point 2 - '
    savefig_path = images_folder / (
            'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_latency_max_' + \
            '_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Maximum latencies with N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num,
             list_data=[np.array(Latency_max_per_M_fixed) * 1e3, np.array(Latency_max_per_M_flex) * 1e3,
                        np.array(Latency_max_per_M_shannon) * 1e3],
             x_ticks=M_list, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], alpha=0.75, xlabel='M', ylabel='[ms]',
             bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center', color=colors,
             title=title, savefig_path=savefig_path)
    fig_num += 1
    preface_title = 'Lab 10 Point 2 - '
    savefig_path = images_folder / (
            'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_latency_min_' + \
            '_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Minimum latencies with N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num,
             list_data=[np.array(Latency_min_per_M_fixed) * 1e3, np.array(Latency_min_per_M_flex) * 1e3,
                        np.array(Latency_min_per_M_shannon) * 1e3],
             x_ticks=M_list, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], alpha=0.75, xlabel='M', ylabel='[ms]',
             bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center', color=colors,
             title=title, savefig_path=savefig_path)
    fig_num += 1
    preface_title = 'Lab 10 Point 2 - '
    savefig_path = images_folder / (
                'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_capacity_' + \
                '_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Capacities with N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num, list_data=[np.array(Capacity_per_M_fixed) * 1e-3, np.array(Capacity_per_M_flex) * 1e-3,
                                            np.array(Capacity_per_M_shannon) * 1e-3],
             x_ticks=M_list, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], alpha=0.75, xlabel='M', ylabel='[Tbps]',
             bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center', color=colors,
             title=title, savefig_path=savefig_path)
    fig_num += 1
    preface_title = 'Lab 10 Point 2 - '
    savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-',
                                                                                                               '') + '_average_bitrate_' + \
                                    '_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Average Bit Rates with N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num,
             list_data=[Average_Bit_Rate_per_M_fixed, Average_Bit_Rate_per_M_flex, Average_Bit_Rate_per_M_shannon],
             x_ticks=M_list, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], alpha=0.75, xlabel='M', ylabel='[Gbps]',
             bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center', color=colors,
             title=title, savefig_path=savefig_path)
    fig_num += 1
    preface_title = 'Lab 10 Point 2 - '
    savefig_path = images_folder / (
                'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_bitrate_max_' + \
                '_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Maximum Bit Rates with N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num,
             list_data=[Bit_Rate_max_per_M_fixed, Bit_Rate_max_per_M_flex, Bit_Rate_max_per_M_shannon],
             x_ticks=M_list, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], alpha=0.75, xlabel='M', ylabel='[Gbps]',
             bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center', color=colors,
             title=title, savefig_path=savefig_path)
    fig_num += 1
    preface_title = 'Lab 10 Point 2 - '
    savefig_path = images_folder / (
                'lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_bitrate_min_' + \
                '_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png')
    title = preface_title + 'Minimum Bit Rates with N = ' + str(N_iterations) + ' and find best ' + set_latency_or_snr
    plot_bar(figure_num=fig_num,
             list_data=[Bit_Rate_min_per_M_fixed, Bit_Rate_min_per_M_flex, Bit_Rate_min_per_M_shannon],
             x_ticks=M_list, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], alpha=0.75, xlabel='M', ylabel='[Gbps]',
             bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center', color=colors,
             title=title, savefig_path=savefig_path)
    fig_num += 1