import json # to read and write json files
# import numpy as np # to use array and other facilities matlab-like
import matplotlib.pyplot as plt # to plot
import pandas as pd # for dataframes

from Project_Open_Optical_Networks.Core.science_utils import * # and also import parameters as defined in science utils

class SignalInformation: # this is the class of the signal
    def __init__(self, signal_power=1.0, path=None):
        self._signal_power = float(signal_power) # Power of the transmitted signal (therefore the power along the line)
        self._noise_power = float(0) # noise floor, will be defined by length of each line and power of signal
        self._latency = float(0) # latency in seconds
        self._path = path if path else 'ABD' # if a path is defined, else there is a standard definition
    @property # property is to protect attributes
    def signal_power(self):
        return self._signal_power
    @property
    def noise_power(self):
        return self._noise_power
    @property
    def latency(self):
        return self._latency
    @property
    def path(self):
        return self._path
    @signal_power.setter # to set external vaslues to private attributes
    def signal_power(self, signal_power):
        self._signal_power=signal_power
    @noise_power.setter
    def noise_power(self, noise_power):
        self._noise_power=noise_power
    @latency.setter
    def latency(self, latency):
        self._latency=latency
    @path.setter
    def path(self, path):
        self._path=path
    def decrement_power(self, power_to_update): # reduce power along path, not yet used
        self.signal_power -= power_to_update
    def increment_noise(self, noise_to_update): # add noise along path
        self.noise_power += noise_to_update
    def increment_latency(self, latency_to_update): # each line introduce a latency
        self.latency += latency_to_update
    def path_update(self): # to update the path
        # once a node is crossed, we have to update path, moving to the next one on and removing first one
        self.path = self.path[1:]

#### This class is for Lightpath objects
class Lightpath( SignalInformation ): # inherited class from signal information
    def __init__(self, signal_power=1.0, path=None, channel=None):
        SignalInformation.__init__(self, signal_power, path) # same initial state as signal information
        # channel goes from 0 to 9, 10 slots overall
        self._channel = int(channel) if channel >= 0 and channel <= number_of_active_channels - 1 else (print('Error, channel', str(channel), 'avoided'), exit(2))
        # if number channel = 10 -> 0 <= channel <= 9
        # there is a channel attribute more than signal information class, by the way this attribute has constrains
        # if these constrains are not respected the code exits with an error state
        ####### LAB 8 attributes
        self._Rs = None # symbol rate
        self._df = None # channel spacing between two adjacent frequencies
    @property
    def channel(self): # channel is a private value, once defined could not be changed
        return self._channel
    # @channel.setter
    # def channel(self, channel):
    #     self._channel=channel
    @property
    def Rs(self):
        return self._Rs
    @Rs.setter
    def Rs(self, Rs):
        self._Rs = Rs
    @property
    def df(self):
        return self._df
    @df.setter
    def df(self, df):
        self._df = df


#### This class is for Node objects
class Node: # class for node definition
    def __init__(self, node_dictionary): # dictionary passed as input like {'label':'A', 'position':[x,y], 'connected_nodes':['B', 'D', 'C']}
        self._label = node_dictionary['label']
        self._position = tuple(node_dictionary['position'])
        self._connected_nodes = node_dictionary['connected_nodes']
        self._successive = dict() # will be useful for connect method in network and propagate method
        self._switching_matrix = node_dictionary['switching_matrix'] # this is the switching matrix that will be modified
        self._transceiver = node_dictionary['transceiver'] if 'transceiver' in node_dictionary else 'fixed_rate'
    @property
    def label(self):
        return self._label
    @property
    def position(self):
        return self._position
    @property
    def connected_nodes(self):
        return self._connected_nodes
    @property
    def successive(self):
        return self._successive
    @property
    def switching_matrix(self):
        return self._switching_matrix
    @property
    def transceiver(self):
        return self._transceiver
    @label.setter
    def label(self, label):
        self._label = label
    @position.setter
    def position(self, position):
        self._position = position
    @connected_nodes.setter
    def connected_nodes(self, connected_nodes):
        self._connected_nodes = connected_nodes
    @successive.setter
    def successive(self, successive):
        self._successive=successive
    @switching_matrix.setter
    def switching_matrix(self, switching_matrix):
        self._switching_matrix = switching_matrix
    def probe(self, signal_information): # define a propagation without modify state
        path = signal_information.path # extracts path
        if len(path)>1: # verify if there are still lines to go through path
            label = path[0:2] # the actual line is defined by the first teo nodes of path
            line = self.successive[label] # line object address is in successive with specific label (defined in connect() method)
            # we extract line of interest defined by actual line, for example we are in node A and path[:2]=AB, so line will be AB line object address
            signal_information.path_update() # remove first node on path
            signal_information = line.probe(signal_information) # to continue probe it recall same method in line class (line is AB in the example, so it will propagate on these line)
        # probe propagation will continue until the path as at least two node, if it has only one node the probe is finished and return all recursive functions
        return signal_information # return always the signal with changes

class Line: # class for line objects
    def __init__(self, label, length, gain=None, noise_figure=None, loss_coefficient=None, propagation_constant=None, gamma_NL=None):
        self._label = label # this is the line label, for example 'AB'
        self._length = length # this is the length in meter for the line
        self._successive = {} # this will be useful for network connect method and propagate/probe functions
        self._state = np.ones(number_of_active_channels, dtype='int') # state defined by 0 or 1, 1 is free state and at the beginning they are put all free
        # state is a numpy array and are defined by integer numbers
        ### WE CANNOT STATE THEM "A PRIORI"
        ### ASE parameters (standard values in parameters.py)
        self._n_amplifier = self.n_amplifier_evaluation(length)
        self._gain = gain if gain else G_gain_ct # standard 16 dB
        self._noise_figure = noise_figure if noise_figure else NF_noise_figure_ct # standard 3 dB
        ### Linear and non-linear loss parameters (standard values in parameters.py)
        self._alpha_in_dB = loss_coefficient if loss_coefficient else alpha_in_dB_ct # standard 0.2 dB/km
        self._beta_abs_for_CD = propagation_constant if propagation_constant else beta_abs_for_CD_ct # standard 2.13e-26 # 1/(m*Hz^2)
        self._gamma_non_linearity = gamma_NL if gamma_NL else gamma_non_linearity_ct # 1.27e-3 # 1/(W*m)
        self._L_effective = 1 / (2 * alpha_from_dB_to_linear_value(alpha_in_dB=self.alpha_in_dB)) # effective length
    @property
    def label(self):
        return self._label
    @property
    def length(self):
        return self._length
    @property
    def successive(self):
        return self._successive
    @property
    def state(self):
        return self._state
    @property
    def n_amplifier(self):
        return self._n_amplifier
    @property
    def gain(self):
        return self._gain
    @property
    def noise_figure(self):
        return self._noise_figure
    @property
    def alpha_in_dB(self):
        return self._alpha_in_dB
    @property
    def beta_abs_for_CD(self):
        return self._beta_abs_for_CD
    @property
    def gamma_non_linearity(self):
        return self._gamma_non_linearity
    @property
    def L_effective(self):
        return self._L_effective
    @label.setter
    def label(self, label):
        self._label=label
    @length.setter
    def length(self, length):
        self._length=length
    @successive.setter
    def successive(self, successive):
        self._successive=successive
    @state.setter
    def state(self, state):
        self._state=state
    def n_amplifier_evaluation(self, length):  # evaluate the number of amplifiers for the line
        n_amplifier = int(np.floor(length / span_length))  # span length from parameters
        return n_amplifier
    def ase_generation(self):
        ASE = self.n_amplifier * (h_Plank * frequency * Bn_noise_band * dB_to_linear_conversion_power(self.noise_figure) * (dB_to_linear_conversion_power(self.gain)-1) )
        return ASE
    def probe(self, signal_information): # this function is called by node method
        latency = latency_evaluation(self.length) # generates latency for current line
        noise_power = noise_generation(signal_power=signal_information.signal_power, length=self.length) # generates noise, requires signal power
        signal_information.increment_latency(latency) # update latency accumulated in signal
        signal_information.increment_noise(noise_power) # update noise accumulated in signal

        node = self.successive[signal_information.path[0]] # remember that before calling this class method tha path has removed first node,
        # so we are calling as successive the first node of the current path, that will be next node to analyze.

        signal_information = node.probe(signal_information) # recall the probe method in node class, but now with new path
        return signal_information
    def nli_generation(self, power_of_the_channel):
        ### maximum number of channels because it is the worst case approach
        eta_NLI = eta_NLI_evaluation(alpha_dB=self.alpha_in_dB, beta=self.beta_abs_for_CD, gamma_NL=self.gamma_non_linearity, Rs=Rs_symbol_rate, DeltaF=channel_spacing, N_channels=maximum_number_of_channels, L_eff=self.L_effective)
        NLI = np.power(power_of_the_channel, 3)*eta_NLI*self.n_amplifier*Bn_noise_band
        return NLI

class Network: # this is the most important class and define the network from the file
    def __init__(self, json_file):
        self._nodes = {} # dictionary of nodes of network
        self._lines = {} # dictionary of lines of networks
        # then there are the dataframes for latency/snr (weighted paths) and state per channel (route space)
        self._weighted_paths = pd.DataFrame(columns=['path', 'latency', 'noise', 'snr'])
        self._route_space = pd.DataFrame(columns=['path', 'availability_per_ch'])
        self._file_name = json_file # to restore switching matrix

        self.node_reading_from_file() # read file and creates nodes
        for first_node in self.nodes: # first node is node label, being equal to key
            for second_node in self.nodes[first_node].connected_nodes: # for each connected node is defined a line and successives
                line_label = first_node + second_node # line by starting node and its connected one
                # the length is evaluated thanks to numpy array:
                # first of all there is a difference between x and y (x_length = x2 - x1, y_length = y2 - y1), they may be negative
                # then is obtained the power of two of each coordinate length: x_length^2, y_length^2, no abs required thanks to it
                # then they are added, for Pythagorean theorem: sum=(x_length^2)+(y_length^2)
                # then the length is obtained with square root: sqrt( sum )
                line_length = np.sqrt( np.sum( np.power( np.array(self.nodes[first_node].position) - np.array(self.nodes[second_node].position) , 2) ) )
                self.lines[line_label] = Line(line_label, line_length) # each line is defined
        # call automatically the other methods of initialization
        self.connect()
        self.probe()
        self.route_space_update()
    @property
    def nodes(self):
        return self._nodes
    @property
    def lines(self):
        return self._lines
    @property
    def weighted_paths(self):
        return self._weighted_paths
    @property
    def route_space(self):
        return self._route_space
    @property
    def file_name(self):
        return self._file_name
    @nodes.setter
    def nodes(self, nodes):
        self._nodes=nodes
    @lines.setter
    def lines(self, lines):
        self._lines=lines
    @weighted_paths.setter
    def weighted_paths(self, weighted_paths):
        self._weigthed_paths=weighted_paths
    @route_space.setter
    def route_space(self, route_space):
        self._route_space = route_space
    def connect(self): # this method connect each line and node, creating all successive attributes
        # take each node and make successive, so for each line, cycle for as before
        for first_node in self.nodes: # for each node
            for second_node in self.nodes[first_node].connected_nodes: # for each connected node
                line_label = first_node + second_node # line definition as first node and second node
                # the successive element of current line will be the second node address, and labeled as this second node
                self.lines[line_label].successive[second_node] = self.nodes[second_node] # only one element in dict, these objects work as address
                # the successive element of current first node will be the line address defined by line
                # (by each connected node are defined all successive lines)
                # each line is labeled by line name
                self.nodes[first_node].successive[line_label] = self.lines[line_label]
                # for example lines['AB'].successive['B'] is nodes['B'] object
                # nodes['A'].successive['AB'] is lines['AB'] object
    def restore_state_lines(self):
        for line in self.lines:
            self.lines[line].state = np.ones(number_of_active_channels, dtype='int') # re-initialize state
        # self.probe() # not changed
        self.route_space_update() # required to have correct channel availabilities
    def node_reading_from_file(self): # restore only switching matrix condition
        # to restore the switching matrix let's read again the corresponding value on the file
        # and recreate the node objects
        with open(self.file_name, 'r') as json_read: # open json file to read it
            imported_data = json.load(json_read) # save imported file as dictionary

        for key in imported_data: # each key will be each node of network
            # let's define node dictionary for the nodes with node object definition
            node_dict = { 'label': key, 'position': imported_data[key]['position'], 'connected_nodes': imported_data[key]['connected_nodes'], 'switching_matrix': imported_data[key]['switching_matrix'] }
            if 'transceiver' in imported_data[key]: # verify if it is defined the transceiver instance
                node_dict['transceiver'] = imported_data[key]['transceiver']
            self.nodes[key] = Node(node_dict) # dictionary as input of node class
    def route_space_update(self): # update route space analyzing the state of each line
        # it is required a probe in any case to have weighted path, by the way it is done in initialization
        if self.weighted_paths.empty:
            self.probe()
        # define lists for dataframe
        availabilities_dataframe=[]
        titles=[]
        # take the paths from weighted paths and analyze the availability of channels
        for path in self.weighted_paths['path']: # extracts each path from weighted paths
            titles.append(path) # save each path label for route space definition
            path = path.replace('->', '') # removes arrows
            availability_per_channel = np.ones(number_of_active_channels, dtype='int') # define an initial state of all ones, that means free, on the way to refresh for condition
            # Each time will be updated this availability along path
            # if len(path)==2:
            #     availability_per_channel = self.lines[path].state
            # else:
            start=True
            previous_node = ''
            while len(path)>1: # as propagate does, let's analyze path until there is at least a line
                if start: # if it is the first node, let's define availability only by line states
                    availability_per_channel = self.lines[path[:2]].state
                    start = False
                else:
                    block = self.nodes[path[0]].switching_matrix[previous_node][path[1]]  # this is the switching matrix of the line
                    line_state = self.lines[path[:2]].state # this is the array of line state
                    # each time we go through the line we update availabilities of channels by thre elements:
                    # 1 - updated availabilities for this path
                    # 2 - block of switching matrix
                    # 3 - state of the actual line
                    availability_per_channel = availability_per_channel * block * line_state #depends on this path, previuos blocks and the switching matrix
                    # update path to go on the path and have next line
                previous_node = path[0]
                path=path[1:]
            availabilities_dataframe.append(availability_per_channel) # save the availabilities of the channels
        #produce route space dataframe
        self.route_space['path']=titles
        self.route_space['availability_per_ch']=availabilities_dataframe
        return

    def find_paths(self, first_input_node, second_input_node):
        # let's produce all nodes in between and possible lines (with also input and output nodes)
        nodes_in_between = list( self.nodes.keys() ) # takes all nodes
        nodes_in_between.remove(first_input_node) # remove from the list the first one
        nodes_in_between.remove(second_input_node) # remove from the list the second one
        possible_lines = list( self.lines.keys() ) # take all lines

        possible_paths = ['init']*(len(nodes_in_between)+1) # each time we restart the path is required to increase the index to maintain previous ones
        possible_paths[0] = first_input_node # starts with input node
        for i in range(0, len(nodes_in_between)): # maximum length of analisys
            for path in possible_paths[i]: # from the "previous" list of paths analize each portion registered (the first will be simply input node)
                for node in nodes_in_between: #take each node in between
                    if path[-1]+node in possible_lines: # verify if this node could be added at the end of path
                        if node not in path: # obvioulsy we don't want to go back with copies of same node
                            if possible_paths[i+1] == 'init':
                                possible_paths[i+1] = [path + node] #add this new path
                            else:
                                possible_paths[i+1] += [path + node] # increase list of this paths list with new node
        # better with dictionary

        # we don't have last node as output, we have to find it from all possible paths in between.
        paths=[]
        for i in range( len(nodes_in_between)+1 ): # for each slot
            for path in possible_paths[i]: # extract each paths from previous list of list
                if path[-1]+second_input_node in possible_lines: # if has a possible output as we want, save it on return variables
                    paths.append( path+second_input_node )

        return paths
    def propagate(self, signal): # if signal has channel occupacy it changes states of lines and update route space
        if self.route_space.empty:
            self.route_space_update()

        path = signal.path
        ############ FOR LABEL DF ################
        path_label=''               ##############
        for node in path:           ##############
            path_label+=node+'->'   ##############
        path_label=path_label[:-2]  ##############
        ##########################################

        index_weighted_paths = self.weighted_paths.loc[self.weighted_paths['path'] == path_label].index.item()  # extract index of dataframe
        # then put as values for propagation the ones from weighted paths
        signal.latency = self.weighted_paths.loc[index_weighted_paths, 'latency']

        if hasattr(signal, 'channel'): # if there is an attribute called channel condition
            index_route_space = self.route_space.loc[ self.route_space['path']==path_label ].index.item() # need it as a number value, not index
            states_route_space = self.route_space.loc[index_route_space, 'availability_per_ch']
            state = states_route_space[signal.channel] #extract single state value of interest for if condition

            if state==OCCUPIED: # occupied
                signal.latency = None # the path is already occupied, change signal power into None value
            elif state==FREE: # free
                start = True
                previous_node=''
                # then update line state
                while len(path)>1:
                    line_label = path[:2]
                    line = self.lines[line_label]
                    line.state[signal.channel]=OCCUPIED # occupied
                    if start:
                        start=False
                    else: # changes the switching matrix of adjacent channels
                        actual_node = path[0]
                        next_node = path[1]
                        switch_matrix_address=self.nodes[actual_node].switching_matrix[previous_node][next_node]
                        if signal.channel == 0: # if channel is the first one
                            switch_matrix_address[1] = OCCUPIED
                        elif signal.channel == number_of_active_channels - 1: # if channel is the first one
                            switch_matrix_address[number_of_active_channels - 2] = OCCUPIED
                        else: # others position
                            switch_matrix_address[signal.channel - 1] = OCCUPIED
                            switch_matrix_address[signal.channel + 1] = OCCUPIED
                    previous_node = path[0]
                    path = path[1:]
            else:
                print('Error in state definition')
                exit(4)

        ########### because we have done it at the beginning with weighted paths
        ##### THEN CONITNUE AFTER IF CONDITION OF ATTRIBUTE CHANNEL with noise power, that is required also for occupied
        signal.noise_power = self.weighted_paths.loc[index_weighted_paths, 'noise'] * signal.signal_power  # because is normalized to one we multiply it for signal power

        #################### TO HAVE NOT A DYNAMIC SWITCHING MATRIX LET'S COMMENT LINE BELOW ###########################
        # self.node_reading_from_file() # made available if not dynamic
        ################################################################################################################
        self.route_space_update() # to update new switching matricies and their route space paths

        return signal
    def draw(self): # draw the network
        # creates list for points
        x_points = []
        y_points = []
        node_text=[]
        for node in self.nodes: # extracts data node by node
            x_node = self.nodes[node].position[0] # save x position
            y_node = self.nodes[node].position[1] # save y position
            # for text
            x_points.append(x_node) # all x positions are in a list
            y_points.append(y_node) # all y positions are in a list
            node_text.append(node) # save name of the node for text
            for connected_node in self.nodes[node].connected_nodes: # let's connect each node to the adjacent
                x_connected_node = self.nodes[connected_node].position[0] # x extracts coordinations of connected nodes
                y_connected_node = self.nodes[connected_node].position[1] # y extracts coordinations of connected nodes

                plt.plot( [x_node, x_connected_node], [y_node, y_connected_node], 'y', linewidth=1.5 ) # plot the line between nodes

        plt.plot(x_points, y_points, 'ob', markersize=8) # then after is plot the marker of nodes, so they will be up the lines
        for i in range(0, len(x_points)): # repeat for each node
            plt.text( x_points[i]-25000, y_points[i]+5000, node_text[i] ) # remain the text of each node label

        #plt.show()
        # put outside
    # def node_switching_analyse(self, node_label, disp=None): # this method displays the switching matrix by nodes and their connected nodes
    #     print('For node ', node_label, ':')
    #     node = self.nodes[node_label]
    #     for input_connected_node in node.switching_matrix:
    #         for output_connected_node in node.switching_matrix[input_connected_node]:
    #             if output_connected_node != input_connected_node:
    #                 if np.sum(node.switching_matrix[input_connected_node][output_connected_node])==OCCUPIED:
    #                     if disp == None or disp == 'OFF':
    #                         print('\tConnection ', input_connected_node, '->', node_label, '->', output_connected_node, ' is OFF')
    #                 else:
    #                     if disp == None or disp == 'ON':
    #                         print('\tConnection ', input_connected_node, '->', node_label, '->', output_connected_node, ' is ON')

    def probe(self): # this method take a signal and analyze all the variables of interest without modifying state line
        nodes = self.nodes.keys() # take all node labels
        couples = []
        # produces al possible couples
        for first_node in nodes: # take first node
            for second_node in nodes: # take first node
                if first_node != second_node: # a couple is generated when the two nodes are different
                    couples.append(first_node + second_node)
        # creates lists for dataframe
        titles = []
        latencies = []
        snrs = []
        noises = []
        for couple in couples: # couple by couple is generated the dataframe
            # let's find all possible paths between two nodes of interest
            for path in self.find_paths(couple[0], couple[1]):
                # produce title for each path for dataframe
                title = ''
                for node in path:
                    title += node + '->' # this title requires an arrow to define direction
                titles.append(title[:-2]) # removes last arrow

                signal = SignalInformation(1, path) # in point 5 of lab 3 is required 1 mW of signal power, by the way is not possible to view noise power
                # let's define signal power as 1 W, the noise will be proportional to it
                # SNR doesn't change

                ### Function for propagation
                path = signal.path # extract path
                first_node = self.nodes[path[0]] # extract first node
                signal = first_node.probe(signal) # propagate it
                #############################

                # snr defined as dB, so 10 log10 of ratio between signal power and noise power
                # if condition seems useless here, because there is no channel, is just a probe
                snr = linear_to_dB_conversion_power(signal.signal_power / signal.noise_power) # if signal.latency is not None else 0

                latencies.append(signal.latency)
                snrs.append(snr)
                noises.append(signal.noise_power)

        # define dataframe of interest
        self.weighted_paths['path'] = titles
        self.weighted_paths['latency'] = latencies
        self.weighted_paths['snr'] = snrs
        self.weighted_paths['noise'] = noises

        # once done it shouldn't be repeated if the network is the same
        return
    def find_best(self, first_node, last_node):
        # ############## TEST ################à
        # self.lines['AB'].state= [OCCUPIED]*10 # verifies the case of all occupied state
        # self.lines['DB'].state= [OCCUPIED]*10
        # self.lines['FB'].state= [OCCUPIED]*10
        # ####################################
        # self.route_space_update()

        # define snr dataframe
        best_dataframe = pd.DataFrame(columns=['path', 'snr', 'number_channels_available'])

        # create lists for dataframe
        paths = []
        snrs = []
        latencies = []
        number_channels_available = []
        for path_label in self.weighted_paths['path']: # extract each path obtained in weighted paths
            if path_label[0] == first_node and path_label[-1] == last_node: # we are interested in only some specific paths with input and output defined
                path = path_label.replace('->','') # remove arrows
                free = True # needed to remember out the while cycle that there is at least a free status or not
                line = path[0:2] # extract line name

                states_path = np.ones(number_of_active_channels, dtype='int') # this array updates states along path, array of integers
                while len(line)>1: # if there is enough nodes to define a path, at least two nodes

                    # extract the states from route space for this path
                    states = self.route_space['availability_per_ch'].loc[self.route_space['path']==path_label].item() # extracted as array
                    states_path = states_path * states # save states and update them while they are going line by line
                    if np.sum(states) == OCCUPIED: # at least required one free state, so let's find where all accupancies are (all zeros, zero sum)
                        free = False # let's remember that everything is occupied
                        break # useless go along the line, let's break out the while
                    path = path[1:] # update path for while (remove first node and go on)
                    line = path[0:2] # update line
                if free==True: # only to do if at least there is one free state
                    channels = np.array([i for i in range(0, number_of_active_channels) if states_path[i] == FREE]) # extracts positions [numbers that could be from 0 to max ch]
                    if len(channels)>0: # if there are available channels it appends in list the results for dataframe
                        paths.append(path_label)
                        snrs.append(self.weighted_paths['snr'].loc[self.weighted_paths['path']==path_label].item()) # needed snr fot the corresponding path
                        latencies.append(self.weighted_paths['latency'].loc[self.weighted_paths['path'] == path_label].item())  # needed latency for the corresponding path
                        number_channels_available.append(channels)
        if len(paths)==0: # if there is not any path available, return none
            return {'snr': {'path':None, 'channels':None}, 'latency': {'path':None, 'channels':None}} # there is not any available path for all possible channels

        # dataframe filling
        best_dataframe['path'] = paths
        best_dataframe['snr'] = snrs
        best_dataframe['latency'] = latencies
        best_dataframe['number_channels_available'] = number_channels_available

        #### Find MAX SNR
        #max_snr = snr_dataframe['snr'].max() # this was the maximum value, but is useless
        idx_max = best_dataframe['snr'].idxmax() # find maximum snr index
        #### FIND MIN LATENCY
        # min_latency = latency_dataframe['latency'].min() # this was the minimum value, but is useless
        idx_min = best_dataframe['latency'].idxmin()  # find minimum latency index
        # Find path for each max and min
        path_best_snr = best_dataframe.loc[[idx_max],:]['path'].item() # extract path for this maximum SNR
        path_best_latency = best_dataframe.loc[[idx_min], :]['path'].item()  # extract path for this minimum latency

        channel_availability_snr = best_dataframe.loc[[idx_max], :]['number_channels_available'].item() # extract channels for best SNR
        channel_availability_latency = best_dataframe.loc[[idx_min], :]['number_channels_available'].item()  # extract channels for best latency

        best_snr = {'path': path_best_snr, 'channels': channel_availability_snr} # creates a dictionary that will be returned
        best_latency = {'path': path_best_latency, 'channels': channel_availability_latency}  # creates a dictionary that will be returned
        best = {'latency': best_latency, 'snr': best_snr}
        return best

    def stream(self, connection, set_latency_or_snr=None, use_state=None): # this function streams a conncetion with specific set
        use_state = use_state if use_state else False # if we want to use the state of the connections, let's put use_state=True
        #connection = Connection(input_node, output_node, signal_power)
        set_lat_snr = set_latency_or_snr if set_latency_or_snr else 'latency' # select which set we want to use, if not defined is 'latency'
        remove_connection = False # if some conditions are not satisfied let's avoid propagation
        Rb = 0 # initialized Rb

        if set_lat_snr != 'latency' and set_lat_snr!='snr': # THEY USE STATE
            print('ERROR! Set for stream function in class Network avoided.') # if the setter is wrongly defined, it gives an error and exit
            exit(3)

        best = self.find_best(connection.input, connection.output) # find the best path for connection
        best = best[set_lat_snr] # take the best as required by setter in input

        path = best['path'] # extracts path from best
        if path == None: # verify if there is at least one path, or it avoid the connection
            remove_connection = True
        else:
            path = path.replace('->','')

            #### evaluation of bit rate
            Rb = self.calculate_bit_rate(path=path, strategy=self.nodes[path[0]].transceiver) # evaluate the bit rate of path with the transceiver condition of first node
            if Rb == 0: # if bit rate null let's avoid connection
                remove_connection = True
        if remove_connection: # if at least one condition on bit rate and path is not respected, let's avoid it
            connection.latency = np.NaN # set Not-A-Number for all components, except for channel that is None
            connection.snr = np.NaN
            connection.channel = None
            connection.bit_rate = np.NaN
            return connection

        channel = None
        if use_state: # verify the use state condition
            channel = best['channels'] # extracts channels from best
            channel = channel[0] # choose the first one available
            signal = Lightpath(connection.signal_power, path, channel) # creates lightpath
        else:
            signal = SignalInformation(connection.signal_power, path) # creates signal without channel

        self.propagate(signal) # propagation of signal or lightpath

        connection.latency = signal.latency # if signal.latency is not None else np.NaN
        connection.snr = linear_to_dB_conversion_power( signal.signal_power / signal.noise_power )
        connection.channel = channel # set channel
        connection.bit_rate = Rb # set bit rate

        return connection
    def calculate_bit_rate(self, path, strategy=None):
        strategy = strategy if strategy else 'fixed_rate'

        path_label = ''
        for node in path:
            path_label += node + '->'
        path_label = path_label[:-2]

        # let's find the corresponding SNR for path label at input and obtain it as a floating number
        GSNR_dB = self.weighted_paths['snr'].loc[self.weighted_paths['path'] == path_label].item()
        GSNR_lin = dB_to_linear_conversion_power(GSNR_dB) # in linear value

        return bit_rate_evaluation(GSNR_lin, strategy)

class Connection:  # class that define a connection between two nodes
    def __init__(self, input_node, output_node, signal_power=None, channel=None, bit_rate=None):
        self._input = input_node  # starting node
        self._output = output_node  # ending node
        self._signal_power = signal_power if signal_power else float(1)  # signal power definition, if not defined set to 1 W
        self._latency = float(0)  # latency set to 0
        self._snr = float(0)  # snr set to 0
        self._channel = channel if channel else int(0) # channel of interest if defined
        self._bit_rate = bit_rate  # the bit rate of the connection
    @property
    def input(self):
        return self._input
    @property
    def output(self):
        return self._output
    @property
    def signal_power(self):
        return self._signal_power
    @signal_power.setter
    def signal_power(self, signal_power):
        self._signal_power = signal_power
    @property
    def latency(self):
        return self._latency
    @latency.setter
    def latency(self, latency):
        self._latency = latency
    @property
    def snr(self):
        return self._snr
    @property
    def channel(self):
        return self._channel
    @property
    def bit_rate(self):
        return self._bit_rate
    @snr.setter
    def snr(self, snr):
        self._snr = snr
    @channel.setter
    def channel(self, channel):
        self._channel = channel
    @bit_rate.setter
    def bit_rate(self, bit_rate):
        self._bit_rate = bit_rate