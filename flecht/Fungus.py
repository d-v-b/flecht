# Classes and methods for running cellular automata models


class Fungus(object):
    """
    A fungus object for cellular automata simulations
    """
    def __init__(self, filters, num_iter=1):
        # ensure that filters is an iterable
        if not hasattr(filters, '__getitem__'):
            raise TypeError('Filters must be an iterable, got ' + str(type(filters)) + ' instead.')

        self.filters = filters
        self.num_iter = num_iter
        self._num_states = None

    @property
    def num_states(self):
        if self._num_states is None:
            self._num_states = len(self.filters)
        return self._num_states


#    def run_sim(self, seed, t):
#        sMask = seed * 0
#            for s in range(self.num_states):
#                sMask += update(curField, nbs, nStates, gos, s)
#        curField = sMask
#        vol[t] = curField

    def grow(self, seed_array):
        """
        Apply filters to seed_array, returning a volume
        """
        from numpy import zeros

        # initialize the array containing the result

        dims = list(seed_array.shape)
        dims.append(self.num_iter)
        vol = zeros(dims).astype('int')
        vol[:, :, 0] = seed_array

        for t in range(1, self.num_iter):
            for s in range(self.num_states):
                # vol[:,:,t] += update(curField, nbs, nStates, gos, s)
                vol[:, :, t] += self.__advance(s, vol[:, :, t-1])

        return vol

    def __advance(self, cur_state, seed_array):
        from numpy import in1d
        from scipy.ndimage import correlate

        # advance the cellular automaton simulation by one frame for a given state
        s = cur_state
        next_state = (s + 1) % self.num_states
        s_mask = seed_array == s
        u_mask = (seed_array == next_state).astype('int')
        kernel = self.filters[s].kernel
        rule = self.filters[s].rule

        conved = correlate(u_mask, kernel, mode='wrap') * s_mask
        ix = in1d(conved.ravel(), rule).reshape(conved.shape)
        s_mask = s_mask.astype('int') * s
        s_mask[ix] = next_state

        return s_mask

    @staticmethod
    def to_file(path):
        """
        Save simulation object to disk
        """
        pass


class Filter(object):

    def __init__(self, kernel, rule):
        self.kernel = kernel
        self.rule = rule