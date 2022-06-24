import matplotlib.pyplot as plt
import numpy as np
from dataloader import granulation_map
from physics import radiance_to_temperature

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

def calc_raw_num_nodes_and_fluxes(num_incoming_nodes, num_outgoing_nodes, idx_x, idx_y, shifted_xx, shifted_yy, xx, yy):
    """ Calculate the raw or naive number of nodes (outgoing for the local cell, incoming for the neighbors)
        and the flux array for the local cell.

        Naive in the sense that cells pointing at each other are not yet resolved.

        Return array num_incoming_nodes and num_outoing_nodes
    """
    overlap_areas = np.zeros((3, 3))
    shifted_x = shifted_xx[idx_x, idx_y]
    shifted_y = shifted_yy[idx_x, idx_y]
    local_xx = xx[idx_x - 1: idx_x + 2, idx_y - 1: idx_y + 2]
    local_yy = yy[idx_x - 1: idx_x + 2, idx_y - 1: idx_y + 2]
    for ix in range(3):
        for iy in range(3):
            if ix == 1 and iy == 1:
                continue
            overlap_area = calc_overlap_area(shifted_x,
                                             shifted_y,
                                             local_xx[ix, iy],
                                             local_yy[ix, iy])
            overlap_areas[ix, iy] = overlap_area
            if overlap_area:
                num_incoming_nodes[idx_x-1+ix, idx_y-1+iy] += 1
                num_outgoing_nodes[idx_x, idx_y] += 1

    # To avoid dividing by zero
    if np.sum(overlap_areas):
        flux_percentage = overlap_areas / np.sum(overlap_areas)
    else:
        flux_percentage = overlap_areas

    return num_incoming_nodes, num_outgoing_nodes, flux_percentage

def calc_flux_percentages(idx_x, idx_y, shifted_xx, shifted_yy, xx, yy):
    """ Calculate the number of nodes (outgoing for the local cell, incoming for the neighbors).
    """
    overlap_areas = np.zeros(xx.shape)
    for ix in range(idx_x - 1, idx_x + 2):
        for iy in range(idx_y - 1, idx_y + 2):
            if ix == idx_x and iy == idx_y:
                continue
            overlap_area = calc_overlap_area(shifted_xx[idx_x, idx_y],
                                             shifted_yy[idx_x, idx_y],
                                             xx[ix, iy],
                                             yy[ix, iy])
            overlap_areas[ix, iy] = overlap_area

    # To avoid dividing by zero
    if np.sum(overlap_areas):
        flux_percentage = overlap_areas / np.sum(overlap_areas)
    else:
        flux_percentage = overlap_areas

    return flux_percentage


def test_case():
    test = True
    # Define some test arrays
    if test:
        grad_x, grad_y, temp, test_idx_x_list, test_idx_y_list, v_vertical = create_test_data()
    else:
        granulation_radiance = granulation_map()
        temperature = radiance_to_temperature(granulation_radiance)
        temp = temperature[992 + 1292]

        temp = temp[20:80, 20:80]

        grad_x, grad_y = np.gradient(temp)
        v_vertical = -2000 + 3000 * ((temp - np.min(temp)) / (np.max(temp) - np.min(temp)))
        v_vertical -= np.mean(v_vertical)

        test_idx_x_list = []
        test_idx_y_list = []

    # Calculate the normalization
    normalization = np.sqrt(np.square(grad_x) + np.square(grad_y))
    normalization[normalization == 0] = 1
    # Normalize and add a minus since the flux should be downhill
    grad_x_norm = -grad_x / normalization
    grad_y_norm = -grad_y / normalization

    simulation_cell_size = temp.shape[0]
    xx, yy = np.meshgrid(range(simulation_cell_size), range(simulation_cell_size))

    shifted_xx = xx + grad_y_norm
    shifted_yy = yy + grad_x_norm

    # We will need some helper arrays
    num_incoming_nodes = np.zeros(xx.shape, dtype=int)
    # Sum up the incoming horizontal velocity for each cell
    incoming_v_hor = np.zeros(xx.shape, dtype=float)
    num_outgoing_nodes = np.zeros(xx.shape, dtype=int)
    finished_nodes = np.zeros(xx.shape, dtype=bool)
    troubling_cells = np.zeros(xx.shape, dtype=bool)

    # Save the flux percentages (i.e. how much flux does each neighboring cell get from each cell)
    # Save as flattened NxN 3x3 arrays, so the pixels are stacked on the first axis
    # And for each pixel there is a 3x3 matrix that calculates the fluxes
    # First the raw
    raw_flux_percentages = np.zeros((xx.shape[0], xx.shape[1], 3, 3))
    # And then the corrected one. It should be corrected for cells that point to each other
    flux_percentages = np.zeros((xx.shape[0], xx.shape[1], 3, 3))

    # First calculate the raw flux percentages
    for idx_x in range(1, simulation_cell_size-1):
        for idx_y in range(1, simulation_cell_size-1):

            (num_incoming_nodes,
             num_outgoing_nodes,
             raw_flux_percentages[idx_x, idx_y, :, :]) = calc_raw_num_nodes_and_fluxes(num_incoming_nodes,
                                                                                       num_outgoing_nodes,
                                                                                       idx_x,
                                                                                       idx_y,
                                                                                       shifted_xx,
                                                                                       shifted_yy,
                                                                                       xx,
                                                                                       yy)

    flux_percentages = raw_flux_percentages.copy()

    # Now correct the flux for cells that influence each other
    for idx_x in range(1, simulation_cell_size-1):
        for idx_y in range(1, simulation_cell_size-1):
            local_raw_flux = raw_flux_percentages[idx_x, idx_y, :, :]
            # Check all the neighboring cells
            for dix in range(-1, 2):
                for diy in range(-1, 2):
                    # If it's the same cell, skip
                    if dix == 0 and diy == 0:
                        continue
                    # If there is no flux there is no problem
                    # ix and iy are the indices in the 3x3 array of the source pixel
                    ix = dix + 1
                    iy = diy + 1
                    if not local_raw_flux[ix, iy]:
                        continue
                    target_raw_flux = raw_flux_percentages[idx_x + dix, idx_y + diy, : ,: ]
                    # Dictionary to find the source cell in the 3x3 flux array of the target cell
                    map_to_relative = {1: 1,
                                       0: 2,
                                       2: 0}
                    relative_idx_x = map_to_relative[ix]
                    relative_idx_y = map_to_relative[iy]
                    flux_to_source_cell = target_raw_flux[relative_idx_x, relative_idx_y]
                    # When there is no flux flowing back to the source cell, there is no problem
                    if not flux_to_source_cell:
                        continue

                    # In the case that there is a stronger flux flowing into the cell than it is giving back to the
                    # same cell
                    if flux_to_source_cell >= local_raw_flux[ix, iy]:
                        # Set the flux into the target cell to 0
                        local_raw_flux[ix, iy] = 0
                        # And renormalize, i.e. the flux will be distributed to the other cells
                        local_flux = local_raw_flux / np.sum(local_raw_flux)
                        flux_percentages[idx_x, idx_y, :, :] = local_flux
                        # Reduce the number of incoming nodes
                        num_incoming_nodes[idx_x + dix, idx_y + diy] -= 1
                        num_outgoing_nodes[idx_x, idx_y] -= 1
                    else:
                        target_raw_flux[map_to_relative[ix], map_to_relative[ix]] = 0
                        target_flux = target_raw_flux / np.sum(target_raw_flux)
                        flux_percentages[idx_x + dix, idx_y + diy, :, :] = target_flux
                        num_incoming_nodes[idx_x, idx_y] -= 1
                        num_outgoing_nodes[idx_x + dix, idx_y + diy] -= 1





    num_remaining_incoming_nodes = num_incoming_nodes.copy()

    # Now we want to calc the incoming flux for all cells
    # Loop over all cells starting with the ones that have ideally no incoming nodes
    done = False
    counter = 0
    while not done:
        counter += 1
        tmp_remaining_nodes = num_remaining_incoming_nodes.copy()
        tmp_remaining_nodes[finished_nodes] = 1e5
        flattened_idx = np.argsort(tmp_remaining_nodes.flatten())[0]
        # Retrieve the original 2D index
        (idx_x, idx_y) = np.unravel_index(flattened_idx, (simulation_cell_size, simulation_cell_size))

        # Don't calc on the borders
        if (not idx_x or
                idx_x == simulation_cell_size-1 or
            not idx_y or
                idx_y == simulation_cell_size-1):
            finished_nodes[idx_x, idx_y] = True
            if finished_nodes.all():
                done = True
                break
            continue
        # Get the number of remaining nodes (it should be 0)
        num_remaining = num_remaining_incoming_nodes[idx_x, idx_y]
        #assert num_remaining == 0, f"{num_remaining} incoming flux vectors remaining"
        if num_remaining != 0:
            troubling_cells[idx_x, idx_y] = True

        # Calculate the flux that is transported to the different cells
        local_flux_percentages = flux_percentages[idx_x, idx_y, :, :]

        global_flux_percentage = np.zeros(xx.shape)
        global_flux_percentage[idx_x-1: idx_x+2, idx_y-1: idx_y+2] = local_flux_percentages

        # Calculate the incoming flux for the local cell = vertical + incoming horizontal
        incoming_flux_local_cell = (v_vertical[idx_x, idx_y] +
                                    incoming_v_hor[idx_x, idx_y])

        # And give that flux to the neighboring cells
        incoming_v_hor += global_flux_percentage * incoming_flux_local_cell

        # Remove the nodes that have received flux from the num_remaining_nodes_array
        num_remaining_incoming_nodes[global_flux_percentage > 0.] -= 1

        # Now mark the current node as finished
        finished_nodes[idx_x, idx_y] = True

        print(f"{np.sum(finished_nodes)}/{np.size(finished_nodes)}")
        print(counter)

        if finished_nodes.all():
            done = True
            break

        # if counter >= 1e4:
        #     print(f"Break Loop")
        #     break


    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    img = ax[0, 0].imshow(temp, origin="lower", cmap="hot")
    fig.colorbar(img, label="Temp", ax=ax[0, 0])
    ax[0, 0].set_title("Temperature")

    # img = ax[0, 1].imshow(troubling_cells, origin="lower")
    # fig.colorbar(img, label="Temp", ax=ax[0, 1])
    # ax[0, 1].set_title("Trouble Cells")

    flux_image = np.zeros(xx.shape)
    test_idx = 3
    flux_image[test_idx_x_list[test_idx] - 1:test_idx_x_list[test_idx] + 2,
    test_idx_y_list[test_idx] - 1:test_idx_y_list[test_idx] + 2] = flux_percentages[
        test_idx_x_list[test_idx], test_idx_y_list[test_idx]]
    img = ax[0, 1].imshow(flux_image, origin="lower")
    ax[0, 1].set_title(f"Flux percentage - Cell {test_idx_x_list[test_idx], test_idx_y_list[test_idx]}")
    fig.colorbar(img, label=f"Flux percentage", ax=ax[0, 1])

    img = ax[1, 0].imshow(num_incoming_nodes, origin="lower")
    ax[1, 0].set_title("Nr of incoming nodes")
    fig.colorbar(img, label="# incoming nodes", ax=ax[1, 0])

    #flux_image = np.zeros(xx.shape)
    #test_idx = 2
    #flux_image[test_idx_x_list[test_idx]-1:test_idx_x_list[test_idx]+2,
    #test_idx_y_list[test_idx]-1:test_idx_y_list[test_idx]+2] = flux_percentages[
    #    test_idx_x_list[test_idx], test_idx_y_list[test_idx]]
    img = ax[1,1].imshow(incoming_v_hor, origin="lower")
    ax[1, 1].set_title(f"Incoming flux")
    fig.colorbar(img, label="Incoming Flux [m/s]", ax=ax[1, 1])


    half_cell = 0.5

    for a in ax.flatten():
        a.scatter(shifted_xx, shifted_yy)
        a.quiver(xx, yy, grad_y_norm, grad_x_norm, scale=10)
        # test_idx_x_list = [2, 2]
        # test_idx_y_list = [8, 8 ]
        for test_idx_x, test_idx_y in zip(test_idx_x_list, test_idx_y_list):
            elem_xx = shifted_xx[test_idx_x, test_idx_y]
            elem_yy = shifted_yy[test_idx_x, test_idx_y]
            a.vlines(elem_xx - half_cell, elem_yy - half_cell, elem_yy + half_cell)
            a.vlines(elem_xx + half_cell, elem_yy - half_cell, elem_yy + half_cell)
            a.hlines(elem_yy - half_cell, elem_xx - half_cell, elem_xx + half_cell)
            a.hlines(elem_yy + half_cell, elem_xx - half_cell, elem_xx + half_cell)
    plt.tight_layout()

    from pathlib import Path
    out_dir = Path("/home/dane/Documents/PhD/Sabine_overviews/12.07.2022")
    plt.savefig(out_dir / "corrected_flux_physical.png", dpi=300)
    plt.show()


def create_test_data():
    temp = np.zeros((7, 7))
    grad_x = np.zeros((7, 7))
    grad_y = np.zeros((7, 7))
    v_vertical = np.zeros((7, 7))

    # Throw in some test temps and gradients
    test_idx_x_list = [1, 3, 3, 3]
    test_idx_y_list = [2, 2, 4, 5]
    test_idx_x = 1
    test_idx_y = 2
    temp[test_idx_x, test_idx_y] = 5000
    grad_x[test_idx_x, test_idx_y] = 1
    grad_y[test_idx_x, test_idx_y] = 0.5
    v_vertical[test_idx_x, test_idx_y] = 1000.
    test_idx_x = 3
    test_idx_y = 2
    temp[test_idx_x, test_idx_y] = 5000
    grad_x[test_idx_x, test_idx_y] = 1
    grad_y[test_idx_x, test_idx_y] = -0.5
    v_vertical[test_idx_x, test_idx_y] = 1000.
    test_idx_x = 3
    test_idx_y = 4
    temp[test_idx_x, test_idx_y] = 5000
    grad_x[test_idx_x, test_idx_y] = 1
    grad_y[test_idx_x, test_idx_y] = -0.5
    v_vertical[test_idx_x, test_idx_y] = 1000.

    test_idx_x = 3
    test_idx_y = 5
    temp[test_idx_x, test_idx_y] = 5000
    grad_x[test_idx_x, test_idx_y] = -1
    grad_y[test_idx_x, test_idx_y] = 1
    v_vertical[test_idx_x, test_idx_y] = 1000.
    return grad_x, grad_y, temp, test_idx_x_list, test_idx_y_list, v_vertical


if __name__ == "__main__":
    test_case()