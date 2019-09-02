# Classes and methods for running cellular automata models
from numpy import zeros, in1d, array, where
from scipy.ndimage import correlate
from itertools import combinations
from operator import concat
from functools import reduce

class Fungus:
    """
    A fungus object for cellular automata simulations
    """
    def __init__(self, filters, trigger_state=1):
        # ensure that filters is an iterable
        if not hasattr(filters, '__getitem__'):
            raise TypeError('Filters must be indexible, got ' + str(type(filters)) + ' instead.')

        self.filters = filters
        self._num_states = None
        self._trigger_state = trigger_state

    @property
    def num_states(self):
        if self._num_states is None:
            self._num_states = len(self.filters)
        return self._num_states

    @property
    def trigger_state(self):
        return self._trigger_state

    def grow(self, seed_array, num_iter=1, stability_check=None):
        """
        Iteratively apply filters to seed_array, returning a volume
        """

        # initialize the array containing the result
        dims = list(seed_array.shape)
        dims.insert(0, num_iter)
        vol = zeros(dims, dtype='int')
        vol[0] = seed_array

        for t in range(1, num_iter):
            # embarrassingly parallelizable over states
            for s in range(self.num_states):
                vol[t] += self.advance(s, vol[t-1])

            if stability_check is not None:
                if stability_check(t, vol):
                    break

        return vol

    def advance(self, cur_state, seed_array):
        """
        advance the cellular automaton simulation by one frame for a given state
        """

        # the next state is the value we will increment to if a cell advances
        next_state = (cur_state + 1) % self.num_states

        # the trigger state is the value used to satisfy the filter conditions
        trigger_state = (cur_state + self.trigger_state) % self.num_states
        s_mask = seed_array == cur_state
        u_mask = (seed_array == trigger_state).astype('int')

        stencil = self.filters[cur_state].stencil
        rule = self.filters[cur_state].rule
        # todo: make the mode depend on the topology of the field
        conved = correlate(u_mask, stencil, mode='wrap') * s_mask
        ix = in1d(conved.ravel(), rule).reshape(conved.shape)

        s_mask = s_mask.astype('int') * cur_state
        s_mask[ix] = next_state

        return s_mask

    @staticmethod
    def to_file(path):
        """
        Save simulation object to disk
        """
        pass


class Filter(object):
    """
    A combination of a stencil and a rule which together determine conditions for a state transition
    """
    def __init__(self, stencil, rule):
        self.stencil = stencil
        self.rule = rule

    def generate_domain(self):
        """
        Return the all the configurations that will trigger a state change for this filter object
        """

        # Find all the possible ways for the rule to match the filter
        kernel_inds = where(self.stencil.ravel() > 0)[0]
        cases = list(list(combinations(kernel_inds, r)) for r in self.rule)
        domain_size = sum(map(len, cases))
        domain = zeros((domain_size, *self.stencil.shape))

        if len(cases) > 1:
            cases_flat = reduce(concat, cases)
        else:
            cases_flat = cases[0]

        for ind, case in enumerate(cases_flat):
            domain[ind].ravel()[list(case)] = 1

        return domain

