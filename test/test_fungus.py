
from numpy import eye, zeros, array_equal
from flecht.Fungus import Fungus, Rule


def test_growth_n2(self):

    """ Simple growth with 2 states on a 3x3 array"""
    rule = Rule(np.ones((3,3)))

    sim = Fungus(filters, num_iter=10)
    seed_dims = [9, 9]
    seed_array = zeros(seed_dims)
    seed_array[seed_dims[0] / 2, seed_dims[1] / 2] = 1
    vol = sim.grow(seed_array)

    assert array_equal(eye(vol.shape[0]).astype('int'), vol[:, :, -1])


test_growth_n2()