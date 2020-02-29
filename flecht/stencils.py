import numpy as np

def nsphere(radius, order, ndim):
    """
    Generate an n-dimensional symmetrical stencil with a given radius by returning a binary array 
    with 1s at indices with a distance less than or equal to radius. The minkowski order of the distance metric
    can be set by the `order` kwarg.
    """
    base = np.zeros((2 * radius + 1,) * ndim)
    coords = np.indices(base.shape) - radius
    distance = np.linalg.norm(coords, ord=order, axis=0)
    base[distance <= radius] = 1
    # remove the central index
    base[distance < 1] = 0
    return base

def moore(radius, ndim):
    return nsphere(radius, order=1, ndim=ndim)

def vonNeumann(radius, ndim):
    return nsphere(radius, order=np.inf, ndim=ndim)

def random(radius, ndim):
    # multiply with vonNeumann stencil to remove the midpoint
    return np.random.randint(0,2, size=(2 * radius + 1, ) * ndim) * vonNeumann(radius, ndim)
    
