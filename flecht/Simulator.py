from numpy import zeros, in1d, array, where, zeros_like, nan, asarray
from scipy.ndimage import correlate
from itertools import combinations
from operator import concat
from functools import reduce
from dataclasses import dataclass
import xarray as xr

class Simulator:
    """
    An object that performs cellular automata simulations. This class wraps a function, `run`, that takes an initial 
    condition and generates some number of timesteps of the cellular automaton simulation based on that seed. 
    """
    def __init__(self, rules, trigger_state=1, boundary='torus'):
        self.rules = rules
        self._num_states = None
        self._trigger_state = trigger_state
        self.boundary = boundary

    @property
    def num_states(self):
        if self._num_states is None:
            self._num_states = len(self.rules)
        return self._num_states

    @property
    def trigger_state(self):
        return self._trigger_state

    @trigger_state.setter
    def trigger_state(self, trigger_state):
        self._trigger_state = trigger_state

    def run(self, seed, num_iter=1, stability_func=None):
        """
        Advance the automaton by at least one frame.
        """        
        try:
            assert num_iter > 0
        except AssertionError:
            raise ValueError(f'The number of iterations must be a positive integer, not {num_iter}')

        result = zeros(shape=(num_iter, *seed.shape), dtype=seed.dtype)
        
        for t in range(num_iter):
            
            if t == 0:
                prev = seed
            else:
                prev = result[t-1]
            result[t] = prev
            for ind, rule in enumerate(self.rules):
                seed_mask = (prev == ind).astype('uint8')
                trigger_mask = (prev == ((ind + self.trigger_state) % self.num_states)).astype('uint8')
                changed = rule.apply(seed_mask, trigger_mask)
                result[t][changed.astype('bool')] += 1 
            result[t] = result[t] % self.num_states

            if stability_func is not None:
                if stability_func(t, result):
                    break

        return result



class TotalisticRule:
    """
    A combination of a stencil, and a condition which together determine state transitions. A stencil is a
    binary numpy array where 1s denote locations that influence the state transition. A condition is a collection of integers
    that define the conditions for the cell to advance. 
    """
    def __init__(self, stencil, condition):
        self.stencil = stencil
        self.condition = condition


    def apply(self, seed_mask, trigger_mask):
        """
        Apply the rule to an input array, generating a new array.

        seed_mask: binary numpy array
            mask representing indices that may be updated by this rule

        trigger_mask: binary numpy array
            mask representing the indices influencing whether an index is updated or not

        returns a binary numpy array
        """
        result = zeros_like(seed_mask)
        # todo: make the mode depend on the topology of the field
        conved = asarray(correlate(trigger_mask, self.stencil, mode='wrap').astype(seed_mask.dtype) * seed_mask)
        ix = in1d(conved.ravel(), self.condition).reshape(conved.shape)
        result[ix] = 1
        return result


    def generate_domain(self):
        """
        Return all the configurations that will trigger a state change for this filter object
        """

        # Find all the possible ways for the condition to match the filter
        kernel_inds = where(self.stencil.ravel() > 0)[0]
        cases = list(list(combinations(kernel_inds, r)) for r in self.condition)
        domain_size = sum(map(len, cases))
        domain = zeros((domain_size, *self.stencil.shape))

        if len(cases) > 1:
            cases_flat = reduce(concat, cases)
        else:
            cases_flat = cases[0]

        for ind, case in enumerate(cases_flat):
            domain[ind].ravel()[list(case)] = 1

        return domain


class ElementaryRule:
    """
    Implements the so-called elementary automata 
    """
    def __init__(self, stencil, condition):
        self.stencil = stencil
        self.condition = condition


    def apply(self, seed_mask, trigger_mask):
        """
        Apply the rule to an input array, generating a new array.

        seed_mask: binary numpy array
            mask representing indices that may be updated by this rule

        trigger_mask: binary numpy array
            mask representing the indices influence whether an index is updated or not

        returns a binary numpy array
        """
        result = zeros_like(seed_mask)
        # todo: make the mode depend on the topology of the field
        conved = correlate(trigger_mask, self.stencil, mode='wrap').astype(seed_mask.dtype) * seed_mask
        ix = in1d(conved.ravel(), self.condition).reshape(conved.shape)
        result[ix] = 1
        return result


    def generate_domain(self):
        """
        Return all the configurations that will trigger a state change for this filter object
        """

        # Find all the possible ways for the condition to match the filter
        kernel_inds = where(self.stencil.ravel() > 0)[0]
        cases = list(list(combinations(kernel_inds, r)) for r in self.condition)
        domain_size = sum(map(len, cases))
        domain = zeros((domain_size, *self.stencil.shape))

        if len(cases) > 1:
            cases_flat = reduce(concat, cases)
        else:
            cases_flat = cases[0]

        for ind, case in enumerate(cases_flat):
            domain[ind].ravel()[list(case)] = 1

        return domain
