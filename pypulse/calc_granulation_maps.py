import matplotlib.pyplot as plt
import numpy as np

def calc_overlap_area(square1_x, square1_y, square2_x, square2_y, half_size=0.5):
    """ Calculate the overlap of two squares with same size.

        :param square1_x: X coordinate of square 1 center
        :param square1_y: Y coordinate of square 1 center
        :param square2_x: X coordinate of square 2 center
        :param square1_y: Y coordinate of square 2 center
        :param half_size: Half the size of a square border
    """
    # Define the borders of the squares
    square_1_left = square1_x - half_size
    square_1_right = square1_x + half_size
    square_1_bottom = square1_y - half_size
    square_1_top = square1_y + half_size
    square_2_left = square2_x - half_size
    square_2_right = square2_x + half_size
    square_2_bottom = square2_y - half_size
    square_2_top = square2_y + half_size

    # Calculate the width, height and overlapping area
    width_overlap = max(0, min(square_1_right, square_2_right) - max(square_1_left, square_2_left))
    height_overlap = max(0, min(square_1_top, square_2_top) - max(square_1_bottom, square_2_bottom))
    area_overlap = width_overlap * height_overlap

    return area_overlap

def calc_num_nodes(num_incoming_nodes, num_outgoing_nodes, idx_x, idx_y, shifted_xx, shifted_yy, xx, yy):
    """ Calculate the number of nodes (outgoing for the local cell, incoming for the neighbors).

        Return array num_incoming_nodes and num_outoing_nodes
    """
    for ix in range(idx_x - 1, idx_x + 2):
        for iy in range(idx_y - 1, idx_y + 2):
            if ix == idx_x and iy == idx_y:
                continue
            overlap_area = calc_overlap_area(shifted_xx[idx_x, idx_y],
                                             shifted_yy[idx_x, idx_y],
                                             xx[ix, iy],
                                             yy[ix, iy])
            if overlap_area:
                num_incoming_nodes[ix, iy] += 1
                num_outgoing_nodes[idx_x, idx_y] += 1

    return num_incoming_nodes, num_outgoing_nodes


def test_case():
    # Define some test arrays
    temp = np.zeros((7, 7))
    grad_x = np.zeros((7, 7))
    grad_y = np.zeros((7, 7))
    v_vertical = np.zeros((7, 7))

    # Throw in some test temps and gradients
    test_idx_x_list = [1, 3, 3]
    test_idx_y_list = [2, 2, 4]

    test_idx_x = 1
    test_idx_y = 2

    temp[test_idx_x, test_idx_y] = 5000
    grad_x[test_idx_x, test_idx_y] = 1
    grad_y[test_idx_x, test_idx_y] = 0.5
    v_vertical[test_idx_x, test_idx_y] = 1000

    test_idx_x = 3
    test_idx_y = 2

    temp[test_idx_x, test_idx_y] = 5000
    grad_x[test_idx_x, test_idx_y] = 1
    grad_y[test_idx_x, test_idx_y] = -0.5
    v_vertical[test_idx_x, test_idx_y] = 1000

    test_idx_x = 3
    test_idx_y = 4

    temp[test_idx_x, test_idx_y] = 5000
    grad_x[test_idx_x, test_idx_y] = -1
    grad_y[test_idx_x, test_idx_y] = 0.5
    v_vertical[test_idx_x, test_idx_y] = 1000


    # Calculate the normalization
    normalization = np.sqrt(np.square(grad_x) + np.square(grad_y))

    simulation_cell_size = temp.shape[0]
    xx, yy = np.meshgrid(range(simulation_cell_size), range(simulation_cell_size))

    grad_x_norm = grad_x / normalization
    grad_y_norm = grad_y / normalization
    shifted_xx = xx + grad_x_norm
    shifted_yy = yy + grad_y_norm

    # We will need some helper arrays
    num_incoming_nodes = np.zeros(xx.shape)
    # Sum up the incoming horizontal velocity for each cell
    incoming_v_hor = np.zeros(xx.shape)
    num_outgoing_nodes = np.zeros(xx.shape)

    # First calculate for each cell the number of all incoming nodes and all outgoing nodes
    for idx_x in range(1, simulation_cell_size-1):
        for idx_y in range(1, simulation_cell_size-1):
            num_incoming_nodes, num_outgoing_nodes = calc_num_nodes(num_incoming_nodes,
                                                                    num_outgoing_nodes,
                                                                    idx_x,
                                                                    idx_y,
                                                                    shifted_xx,
                                                                    shifted_yy,
                                                                    xx,
                                                                    yy)


    # pct_array = area_array / np.sum(area_array)
    #
    # # For visualization
    # pct_array[idx_x, idx_y] = 1


    fig, ax = plt.subplots(1, 3, figsize=(16, 9))
    img = ax[0].imshow(temp, origin="lower")
    fig.colorbar(img, label="Fake Temp", ax=ax[0])
    ax[0].set_title("Fake Temperature")

    img = ax[1].imshow(num_incoming_nodes, origin="lower")
    ax[1].set_title("Nr of incoming nodes")
    fig.colorbar(img, label="# incoming nodes", ax=ax[1])


    half_cell = 0.5

    for a in ax:
        a.scatter(shifted_xx, shifted_yy)
        a.quiver(xx, yy, grad_x_norm, grad_y_norm, scale=10)
        for test_idx_x, test_idx_y in zip(test_idx_x_list, test_idx_y_list):
            elem_xx = shifted_xx[test_idx_x, test_idx_y]
            elem_yy = shifted_yy[test_idx_x, test_idx_y]
            a.vlines(elem_xx - half_cell, elem_yy - half_cell, elem_yy + half_cell)
            a.vlines(elem_xx + half_cell, elem_yy - half_cell, elem_yy + half_cell)
            a.hlines(elem_yy - half_cell, elem_xx - half_cell, elem_xx + half_cell)
            a.hlines(elem_yy + half_cell, elem_xx - half_cell, elem_xx + half_cell)
    plt.show()

if __name__ == "__main__":
    test_case()