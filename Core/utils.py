import random
from Project_Open_Optical_Networks.Core.elements import Connection

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