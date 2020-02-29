import numpy as np
from flecht import Simulator, TotalisticRule
from flecht.stencils import vonNeumann
examples = dict()


def make_conway(mode):
    stencil = vonNeumann(1, 2)

    # Any live cell with fewer than two live neighbours dies, as if by underpopulation.
    # Any live cell with two or three live neighbours lives on to the next generation.
    # Any live cell with more than three live neighbours dies, as if by overpopulation.
    # Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

    death = Rule(stencil=stencil, condition=(1, 2, 3, 4, 7, 8))
    birth = Rule(stencil=stencil, condition=(3,))

    if mode == 'glider':
        glider_base = np.zeros((3,3), dtype='uint8')
        glider_base[2, :] = 1
        glider_base[1, -1] = 1
        glider_base[0, 1] = 1
        field = np.pad(glider_base,  2 * ((0, 10),))
    elif mode == 'random':
        field = np.random.randint(0,2, (200,200))
    sim = Fungus((birth, death))
    result = sim.grow(num_iter=50, seed=field)
    result = np.concatenate([field[np.newaxis,...], result])
    return result

if __name__ == '__main__':
    import napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        glider = make_conway('glider')
        random = make_conway('random')
        viewer.add_image(glider, name="Conway's Game of Life: glider", contrast_limits=(0,1))
        viewer.add_image(random, name="Conway's Game of Life: random", contrast_limits=(0,1))


