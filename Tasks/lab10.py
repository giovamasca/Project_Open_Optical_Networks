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
print_and_save(text='M=' + str(M) + ':', file=file_console)

results = {'Fixed':{}, 'Flex':{}, 'Shannon':{}}
N_iterations = 100
results['Fixed'] = lab10_point1(network=network_fixed_rate, M = M, set_latency_or_snr=set_latency_or_snr, N_iterations=N_iterations, label='Fixed Rate', file_console=file_console)
results['Flex'] = lab10_point1(network=network_flex_rate, M = M, set_latency_or_snr=set_latency_or_snr, N_iterations=N_iterations, label='Flex Rate', file_console=file_console)
results['Shannon'] = lab10_point1(network=network_shannon, M = M, set_latency_or_snr=set_latency_or_snr, N_iterations=N_iterations, label='Shannon', file_console=file_console)

########################################            FIGURE    SNR         #################################################
fig_num = 1
preface_title = 'Lab 10 Point 1 - '
savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_SNR_' + \
               '_with_M_' + str(M) + '_and_find_best_' + set_latency_or_snr + '.png')
title = preface_title + 'SNRs with M = ' + str(M) + ' and find best ' + set_latency_or_snr
plot_histogram(figure_num=fig_num, list_data=[connection_list_data_extractor(connection_list=results[label]['connections'], type_data='SNR') for label in results],
               xlabel='SNR [dB]', ylabel='Number of results', nbins=15, alpha=0.75,
               label=[label for label in results], title=title, savefig_path=savefig_path)
#######################################################################################################################
########################################            FIGURE   LAT         #################################################
fig_num = 2
# preface_title = 'Lab 10 Point 1 - '
savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_latency_' + \
               '_with_M_' + str(M) + '_and_find_best_' + set_latency_or_snr + '.png')
title = preface_title + 'Latencies with M = ' + str(M) + ' and find best ' + set_latency_or_snr
plot_histogram(figure_num=fig_num, list_data=[np.array(connection_list_data_extractor(connection_list=results[label]['connections'], type_data='LAT'))*1e3 for label in results],
               xlabel='delay [ms]', ylabel='Number of results', nbins=15, alpha=0.75,
               label=[label for label in results], title=title, savefig_path=savefig_path)
#######################################################################################################################
########################################            FIGURE    BITRATES        #################################################
fig_num = 3
# preface_title = 'Lab 10 Point 1 - '
savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_bitrate_' + \
               '_with_M_' + str(M) + '_and_find_best_' + set_latency_or_snr + '.png')
title = preface_title + 'Bit Rates with M = ' + str(M) + ' and find best ' + set_latency_or_snr
plot_histogram(figure_num=fig_num, list_data=[connection_list_data_extractor(connection_list=results[label]['connections'], type_data='Rb') for label in results],
               xlabel='Gbps', ylabel='Number of results', nbins=15, alpha=0.75,
               label=[label for label in results], title=title, savefig_path=savefig_path)
#######################################################################################################################
########################################            FIGURE     num CONNECTIONS       #################################################
fig_num = 4
# preface_title = 'Lab 10 Point 1 - '
savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_number_connections_' + \
               '_with_M_' + str(M) + '_and_find_best_' + set_latency_or_snr + '.png')
title = preface_title + 'Number of connections with M = ' + str(M) + ' and find best ' + set_latency_or_snr
plot_bar(figure_num=fig_num, list_data=[results[label]['number_connections'] for label in results],
         x_ticks=None, xlabel='M='+str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center',
        label=[label for label in results], title=title, savefig_path=savefig_path)
#######################################################################################################################
########################################            FIGURE     num BLOCKING EVENTS       #################################################
fig_num = 5
# preface_title = 'Lab 10 Point 1 - '
savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_number_blocking_events_' + \
               '_with_M_' + str(M) + '_and_find_best_' + set_latency_or_snr + '.png')
title = preface_title + 'Number of blocking events with M = ' + str(M) + ' and find best ' + set_latency_or_snr
plot_bar(figure_num=fig_num, list_data=[results[label]['number_blocking_events'] for label in results],
         x_ticks=None, xlabel='M='+str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center',
        label=[label for label in results], title=title, savefig_path=savefig_path)
#######################################################################################################################
########################################            FIGURE     CAPACITIES       #################################################
fig_num = 6
# preface_title = 'Lab 10 Point 1 - '
savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_capacities_' + \
               '_with_M_' + str(M) + '_and_find_best_' + set_latency_or_snr + '.png')
title = preface_title + 'Capacities with M = ' + str(M) + ' and find best ' + set_latency_or_snr
plot_bar(figure_num=fig_num, list_data=np.array([results[label]['capacity'] for label in results])*1e-3, ylabel='[Tbps]',
         x_ticks=None, xlabel='M='+str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center',
        label=[label for label in results], title=title, savefig_path=savefig_path)
#######################################################################################################################
########################################            FIGURE     BITRATE MAX       #################################################
fig_num = 7
# preface_title = 'Lab 10 Point 1 - '
savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_max_bitrate_' + \
               '_with_M_' + str(M) + '_and_find_best_' + set_latency_or_snr + '.png')
title = preface_title + 'Maximum Bit Rates with M = ' + str(M) + ' and find best ' + set_latency_or_snr
plot_bar(figure_num=fig_num, list_data=[results[label]['bitrate_max'] for label in results], ylabel='[Gbps]',
         x_ticks=None, xlabel='M='+str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center',
        label=[label for label in results], title=title, savefig_path=savefig_path)
#######################################################################################################################
########################################            FIGURE     BITRATE MIN       #################################################
fig_num = 8
# preface_title = 'Lab 10 Point 1 - '
savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_min_bitrate_' + \
               '_with_M_' + str(M) + '_and_find_best_' + set_latency_or_snr + '.png')
title = preface_title + 'Minimum Bit Rates with M = ' + str(M) + ' and find best ' + set_latency_or_snr
plot_bar(figure_num=fig_num, list_data=[results[label]['bitrate_min'] for label in results], ylabel='[Gbps]',
         x_ticks=None, xlabel='M='+str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center',
        label=[label for label in results], title=title, savefig_path=savefig_path)
#######################################################################################################################
########################################            FIGURE     SNR per link       #################################################
fig_num = 9
# preface_title = 'Lab 10 Point 1 - '
savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_SNR_per_link_' + \
               '_with_M_' + str(M) + '_and_find_best_' + set_latency_or_snr + '.png')
title = preface_title + 'SNRs per link (average) with M = ' + str(M) + ' and find best ' + set_latency_or_snr
plot_bar(figure_num=fig_num, list_data=[results[label]['SNR_ave_per_link'] for label in results], ylabel='[dB]',
         x_ticks=None, xlabel='M='+str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center',
        label=[label for label in results], title=title, savefig_path=savefig_path)
#######################################################################################################################
########################################            FIGURE     SNR MAX       #################################################
fig_num = 10
# preface_title = 'Lab 10 Point 1 - '
savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_max_SNR_' + \
               '_with_M_' + str(M) + '_and_find_best_' + set_latency_or_snr + '.png')
title = preface_title + 'Maximum SNRs with M = ' + str(M) + ' and find best ' + set_latency_or_snr
plot_bar(figure_num=fig_num, list_data=[results[label]['SNR_max'] for label in results], ylabel='[dB]',
         x_ticks=None, xlabel='M='+str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center',
        label=[label for label in results], title=title, savefig_path=savefig_path)
#######################################################################################################################
########################################            FIGURE     SNR MAX       #################################################
fig_num = 11
# preface_title = 'Lab 10 Point 1 - '
savefig_path = images_folder / ('lab10_fig' + str(fig_num) + '_' + preface_title.replace(' ', '_').replace('-', '') + '_min_SNR_' + \
               '_with_M_' + str(M) + '_and_find_best_' + set_latency_or_snr + '.png')
title = preface_title + 'Minimum SNRs with M = ' + str(M) + ' and find best ' + set_latency_or_snr
plot_bar(figure_num=fig_num, list_data=[results[label]['SNR_min'] for label in results], ylabel='[dB]',
         x_ticks=None, xlabel='M='+str(M), alpha=0.75, bottom=0.25, bbox_to_anchor=(0.5, -0.35), loc='lower center',
        label=[label for label in results], title=title, savefig_path=savefig_path)
#######################################################################################################################
plt.show()



