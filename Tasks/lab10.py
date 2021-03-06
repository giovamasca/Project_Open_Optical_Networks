############## Monte Carlo Analysis for OON ############à

# from Project_Open_Optical_Networks.Core.parameters import *
from Project_Open_Optical_Networks.Core.utils import *
from Project_Open_Optical_Networks.Core.science_utils import *

trial_path = 'best_' + set_latency_or_snr + '_of_' + time.strftime('%Y_%m_%d__%H_%M_%S')
if not os.path.isdir( root / 'Results' / 'Lab10' / trial_path / 'Images' ):  # if Results doesn't exists, it creates it
    os.makedirs( root / 'Results' / 'Lab10' / trial_path / 'Images' )
file_console = root / 'Results' / 'Lab10' / trial_path / 'console.txt'
images_folder = root / 'Results' / 'Lab10' / trial_path / 'Images'

################             N  simulations            ####################
N_iterations = 250
###########################################################################
file_print = open(file_console, 'w')
file_print.close()
############ NETWORKs GENERATION
# these 3 networks has defined transceiver instance
network_fixed_rate = network_generation_from_file(file_nodes_full_fixed_rate)
network_flex_rate = network_generation_from_file(file_nodes_full_flex_rate)
network_shannon = network_generation_from_file(file_nodes_full_shannon)

####### Network size analysis ######
CD_distance = network_flex_rate.lines['CD'].length
DE_distance = np.sqrt(np.sum( np.power(np.array(network_flex_rate.nodes['D'].position) - np.array(network_flex_rate.nodes['E'].position), 2) ))
max_ray = max(CD_distance, DE_distance)
print_and_save('The Network has a ray almost around ' + str(np.round(max_ray*1e-3, 3)) + ' km.', file=file_console)
area_network = network_flex_rate.area_network()*1e-6 # km^2
print_and_save('The Network cover an area of ' + str(np.round(area_network,3)) + ' km square\n', file=file_console)
####################################

# LABELS
fixed = 'fixed_rate'
flex = 'flex_rate'
shannon = 'shannon'

print_and_save(text='Lab 10 Point 1 - Monte Carlo Simulation with Single Traffic Matrix', file=file_console)
########################            Fixed M - Single Traffic Matrix Scenario          ###############################
M = 20 # fixed value of M
print_and_save(text='M=' + str(M) + ':', file=file_console)

results = {}#{'Fixed_Rate':{}, 'Flex_Rate':{}, 'Shannon_Rate':{}}
results['Fixed_Rate'] = lab10_point1_results(network=network_fixed_rate, M = M, set_latency_or_snr=set_latency_or_snr, N_iterations=N_iterations, label='Fixed Rate', file_console=file_console)
results['Flex_Rate'] = lab10_point1_results(network=network_flex_rate, M = M, set_latency_or_snr=set_latency_or_snr, N_iterations=N_iterations, label='Flex Rate', file_console=file_console)
results['Shannon_Rate'] = lab10_point1_results(network=network_shannon, M = M, set_latency_or_snr=set_latency_or_snr, N_iterations=N_iterations, label='Shannon Rate', file_console=file_console)

initial_fig = lab10_point1_graphs(initial_fig=1, images_folder=images_folder, results=results, set_latency_or_snr=set_latency_or_snr, M=M, N_iterations=N_iterations)

###################################      Increment of M - Network Congestion           ############################
print_and_save(text='\n\nLab 10 Point 2 - Monte Carlo Simulation with Network Congestion', file=file_console)
results_per_M = {}

## traffic matrix saturation
is_saturated_per_M_fixed = []
is_saturated_per_M_flex = []
is_saturated_per_M_shannon = []

M_list = [15, 35, 45, 55]# range(1, 10, 2) # [15, 35, 45, 55]
for M in M_list:
    print_and_save(text='\nM=' + str(M) + ':', file=file_console)
    results_per_M[str(M)] = {}#{'Fixed_Rate':{}, 'Flex_Rate':{}, 'Shannon_Rate':{}}
    results_per_actual_M = results_per_M[str(M)] # save address

    results_per_actual_M['Fixed_Rate'] = lab10_point1_results(network=network_fixed_rate, M = M, set_latency_or_snr=set_latency_or_snr, N_iterations=N_iterations, label='Fixed Rate', file_console=file_console)
    results_per_actual_M['Flex_Rate'] = lab10_point1_results(network=network_flex_rate, M = M, set_latency_or_snr=set_latency_or_snr, N_iterations=N_iterations, label='Flex Rate', file_console=file_console)
    results_per_actual_M['Shannon_Rate'] = lab10_point1_results(network=network_shannon, M = M, set_latency_or_snr=set_latency_or_snr, N_iterations=N_iterations, label='Shannon Rate', file_console=file_console)

    is_saturated_per_M_fixed.append( 1 if network_fixed_rate.traffic_matrix_saturated() else 0 )
    is_saturated_per_M_flex.append(1 if network_fixed_rate.traffic_matrix_saturated() else 0)
    is_saturated_per_M_shannon.append(1 if network_fixed_rate.traffic_matrix_saturated() else 0)

colors = ['r', 'y', 'b']
fig = lab10_point2_graphs(initial_fig=initial_fig, results_per_M=results_per_M, M_list=M_list, images_folder=images_folder, N_iterations=N_iterations, set_latency_or_snr=set_latency_or_snr, colors=colors)

plot_bar(figure_num=fig, list_data=[[is_saturated_per_M_fixed[i] for i in range(0, len(M_list))],
                                           [is_saturated_per_M_flex[i] for i in range(0, len(M_list))],
                                           [is_saturated_per_M_shannon[i] for i in range(0, len(M_list))]],
         x_ticks = [M for M in M_list], bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center',
         edge_color='k', color=None, label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'],
         xlabel='M value for traffic matrix', ylabel='Saturated Traffic Matricies', remove_y_ticks=True,
         title=('Lab 10 Point 2 - Traffic Matrix Saturation - with N = ' + str(N_iterations) + ' and best '+ set_latency_or_snr), alpha=0.75,
         savefig_path=images_folder / (
                'lab10_fig' + str(fig) + '_Lab_10_Point_2_traffic_matrix_saturation_' + \
                '_N_' + str(N_iterations) + '_and_find_best_' + set_latency_or_snr + '.png'))

plt.show()