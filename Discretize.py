import pandas
import numpy as np

class Bin():
    def __init__(self, max_val, num_bins, min_val=None):
        '''
        A Bin object has fields num_bins and bin_edges
        The bin_edges are formed by evenly spacing from 
        min_val to max_val with extra bins for -inf to 
        min_val and max_val to inf. 

        If min_val is not given then min_val = -max_val
        '''

        max_val = max_val
        num_bins = num_bins

        if min_val is None:
            min_val = -max_val
        min_val = min_val

        bin_edges = np.linspace(min_val, max_val, num=num_bins-1)
        inf = float("inf")

        bin_edges = np.insert(bin_edges, 0, -inf)
        bin_edges = np.append(bin_edges, inf)

        self.num_bins = num_bins
        self.bin_edges = bin_edges

class Discretize():
    def __init__(self):
        self.bin_dict = {}

    def add_bins(self, name, bins):
        self.bin_dict[name] = bins

    def discretize_var(self, continuous, name):
        '''
        Uses pandas.cut to get discretized version of 
        continuous. 
        Looks up the bin_edges in self.bin_dict[name]
        '''

        curr_bin = pandas.cut([continuous], bins=self.bin_dict[name].bin_edges, labels=False)
        curr_bin = curr_bin[0]
        # print("val: {:.2f}, bins: {}, bin: {}".format(
            # continuous, self.bin_dict[name].bin_edges, curr_bin))
        return curr_bin

    def discretize_1d(self, L):
        '''
        L is a list of tuples (continuous, name)
        containing the continuous value and name of each
        variable

        For example L[0] = (relpos[0], 'relposx')

        This function discretizes the continuous values and
        then combines the discrete values into a single state
        using np.ravel_multi_index and the bin sizes
        '''

        # For each continuous value in L convert to a discrete
        # value and get the num_bins for that variable
        discretes = []
        sizes = []
        for continuous, name in L:
            discretes.append(self.discretize_var(continuous, name))
            sizes.append(self.bin_dict[name].num_bins)


        # Convert n-dimensional state representation to 1 dimension
        discrete_1d = np.ravel_multi_index(discretes, tuple(sizes))

        # print("nd: {}, size: {}, 1d: {}".format(
            # discretes, sizes, discrete_1d))

        return discrete_1d

    def size(self):
        s = 1
        for key, bins in self.bin_dict.items():
            s *= bins.num_bins
        return s

    def unravel(self, discrete_1d, L):
        sizes = []
        for name in L:
            sizes.append(self.bin_dict[name].num_bins)

        return np.unravel_index(discrete_1d, tuple(sizes))