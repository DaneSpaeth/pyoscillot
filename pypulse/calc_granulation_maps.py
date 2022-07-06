import matplotlib.pyplot as plt
import numpy as np
from dataloader import granulation_map
from physics import radiance_to_temperature

def calc_overlap_area(square1_row, square1_col, square2_row, square2_col, half_size=0.5):
    """ Calculate the overlap of two squares with same size.

        :param square1_row: Row coordinate of square 1 center
        :param square1_col: Column coordinate of square 1 center
        :param square2_row: Row coordinate of square 2 center
        :param square2_col: Column coordinate of square 2 center
        :param half_size: Half the size of a square border
    """
    # Define the borders of the squares
    square_1_left = square1_row - half_size
    square_1_right = square1_row + half_size
    square_1_bottom = square1_col - half_size
    square_1_top = square1_col + half_size
    square_2_left = square2_row - half_size
    square_2_right = square2_row + half_size
    square_2_bottom = square2_col - half_size
    square_2_top = square2_col + half_size

    # Calculate the width, height and overlapping area
    width_overlap = max(0, min(square_1_right, square_2_right) - max(square_1_left, square_2_left))
    height_overlap = max(0, min(square_1_top, square_2_top) - max(square_1_bottom, square_2_bottom))
    area_overlap = width_overlap * height_overlap

    return area_overlap

def calc_raw_num_nodes_and_fluxes(num_incoming_nodes, num_outgoing_nodes, row, col, shifted_rr, shifted_cc, rr, cc):
    """ Calculate the raw or naive number of nodes (outgoing for the local cell, incoming for the neighbors)
        and the flux array for the local cell.

        Naive in the sense that cells pointing at each other are not yet resolved.

        Return array num_incoming_nodes and num_outoing_nodes
    """
    overlap_areas = np.zeros((3, 3))
    shifted_x = shifted_rr[row, col]
    shifted_y = shifted_cc[row, col]
    local_xx = rr[row - 1: row + 2, col - 1: col + 2]
    local_yy = cc[row - 1: row + 2, col - 1: col + 2]
    for local_row in range(3):
        for local_col in range(3):
            if local_row == 1 and local_col == 1:
                continue
            overlap_area = calc_overlap_area(shifted_x,
                                             shifted_y,
                                             local_xx[local_row, local_col],
                                             local_yy[local_row, local_col])
            overlap_areas[local_row, local_col] = overlap_area
            if overlap_area:
                num_incoming_nodes[row - 1 + local_row, col - 1 + local_col] += 1
                num_outgoing_nodes[row, col] += 1

    # To avoid dividing by zero
    if np.sum(overlap_areas):
        flux_percentage = overlap_areas / np.sum(overlap_areas)
    else:
        flux_percentage = overlap_areas

    return num_incoming_nodes, num_outgoing_nodes, flux_percentage

def calc_flux_percentages(row, col, shifted_rr, shifted_cc, rr, cc):
    """ Calculate the number of nodes (outgoing for the local cell, incoming for the neighbors).
    """
    overlap_areas = np.zeros(rr.shape)
    for local_row in range(row - 1, row + 2):
        for local_col in range(col - 1, col + 2):
            if local_row == row and local_col == col:
                continue
            overlap_area = calc_overlap_area(shifted_rr[row, col],
                                             shifted_cc[row, col],
                                             rr[local_row, local_col],
                                             cc[local_row, local_col])
            overlap_areas[local_row, local_col] = overlap_area

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
        grad_row, grad_col, temp, test_idx_row_list, test_idx_col_list, v_vertical = create_test_data()
    else:
        granulation_radiance = granulation_map()
        temperature = radiance_to_temperature(granulation_radiance)
        temp = temperature[992 + 1292]

        temp = temp[20:80, 20:80]

        grad_row, grad_col = np.gradient(temp)
        v_vertical = -2000 + 3000 * ((temp - np.min(temp)) / (np.max(temp) - np.min(temp)))
        v_vertical -= np.mean(v_vertical)

        test_idx_row_list = []
        test_idx_col_list = []

    # Calculate the normalization
    normalization = np.sqrt(np.square(grad_row) + np.square(grad_col))
    normalization[normalization == 0] = 1
    # Normalize and add a minus since the flux should be downhill
    grad_row_norm = -grad_row / normalization
    grad_col_norm = -grad_col / normalization

    simulation_cell_size = temp.shape[0]
    # rr and cc are short for rowrow and columncolumn
    # These are 2d arrays containing for each cell the row or column respectively
    rr, cc = np.meshgrid(range(simulation_cell_size), range(simulation_cell_size))

    shifted_rr = rr + grad_col_norm
    shifted_cc = cc + grad_row_norm

    # We will need some helper arrays
    num_incoming_nodes = np.zeros(rr.shape, dtype=int)
    # Sum up the incoming horizontal velocity for each cell
    incoming_v_hor = np.zeros(rr.shape, dtype=float)
    num_outgoing_nodes = np.zeros(rr.shape, dtype=int)
    finished_nodes = np.zeros(rr.shape, dtype=bool)
    troubling_cells = np.zeros(rr.shape, dtype=bool)

    # Save the flux percentages (i.e. how much flux does each neighboring cell get from each cell)
    # Save as flattened NxN 3x3 arrays, so the pixels are stacked on the first axis
    # And for each pixel there is a 3x3 matrix that calculates the fluxes
    # First the raw
    raw_flux_percentages = np.zeros((rr.shape[0], rr.shape[1], 3, 3))
    # And then the corrected one. It should be corrected for cells that point to each other
    flux_percentages = np.zeros((rr.shape[0], rr.shape[1], 3, 3))

    # First calculate the raw flux percentages
    for row in range(1, simulation_cell_size-1):
        for col in range(1, simulation_cell_size-1):

            (num_incoming_nodes,
             num_outgoing_nodes,
             raw_flux_percentages[row, col, :, :]) = calc_raw_num_nodes_and_fluxes(num_incoming_nodes,
                                                                                       num_outgoing_nodes,
                                                                                       row,
                                                                                       col,
                                                                                       shifted_rr,
                                                                                       shifted_cc,
                                                                                       rr,
                                                                                       cc)

    flux_percentages = raw_flux_percentages.copy()

    # Now correct the flux for cells that influence each other
    for row in range(1, simulation_cell_size-1):
        for col in range(1, simulation_cell_size-1):
            local_raw_flux = raw_flux_percentages[row, col, :, :]
            # Check all the neighboring cells
            for drow in range(-1, 2):
                for dcol in range(-1, 2):
                    # If it's the same cell, skip
                    if drow == 0 and dcol == 0:
                        continue
                    # If there is no flux there is no problem
                    # local_row and local_col are the indices in the 3x3 array of the source pixel
                    local_row = drow + 1
                    local_col = dcol + 1
                    if not local_raw_flux[local_row, local_col]:
                        continue
                    target_raw_flux = raw_flux_percentages[row + drow, col + dcol, : ,: ]
                    # Dictionary to find the source cell in the 3x3 flux array of the target cell
                    map_to_relative = {1: 1,
                                       0: 2,
                                       2: 0}
                    relative_row = map_to_relative[local_row]
                    relative_col = map_to_relative[local_col]
                    flux_to_source_cell = target_raw_flux[relative_row, relative_col]
                    # When there is no flux flowing back to the source cell, there is no problem
                    if not flux_to_source_cell:
                        continue

                    # In the case that there is a stronger flux flowing into the cell than it is giving back to the
                    # same cell
                    if flux_to_source_cell >= local_raw_flux[local_row, local_col]:
                        # Set the flux into the target cell to 0
                        local_raw_flux[local_row, local_col] = 0
                        # And renormalize, i.e. the flux will be distributed to the other cells
                        local_flux = local_raw_flux / np.sum(local_raw_flux)
                        flux_percentages[row, col, :, :] = local_flux
                        # Reduce the number of incoming nodes
                        num_incoming_nodes[row + drow, col + dcol] -= 1
                        num_outgoing_nodes[row, col] -= 1
                    else:
                        target_raw_flux[map_to_relative[local_row], map_to_relative[local_row]] = 0
                        target_flux = target_raw_flux / np.sum(target_raw_flux)
                        flux_percentages[row + drow, col + dcol, :, :] = target_flux
                        num_incoming_nodes[row, col] -= 1
                        num_outgoing_nodes[row + drow, col + dcol] -= 1





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
        (row, col) = np.unravel_index(flattened_idx, (simulation_cell_size, simulation_cell_size))

        # Don't calc on the borders
        if (not row or
                row == simulation_cell_size-1 or
            not col or
                col == simulation_cell_size-1):
            finished_nodes[row, col] = True
            if finished_nodes.all():
                done = True
                break
            continue
        # Get the number of remaining nodes (it should be 0)
        num_remaining = num_remaining_incoming_nodes[row, col]
        #assert num_remaining == 0, f"{num_remaining} incoming flux vectors remaining"
        if num_remaining != 0:
            troubling_cells[row, col] = True

        # Calculate the flux that is transported to the different cells
        local_flux_percentages = flux_percentages[row, col, :, :]

        global_flux_percentage = np.zeros(rr.shape)
        global_flux_percentage[row-1: row+2, col-1: col+2] = local_flux_percentages

        # Calculate the incoming flux for the local cell = vertical + incoming horizontal
        incoming_flux_local_cell = (v_vertical[row, col] +
                                    incoming_v_hor[row, col])

        # And give that flux to the neighboring cells
        incoming_v_hor += global_flux_percentage * incoming_flux_local_cell

        # Remove the nodes that have received flux from the num_remaining_nodes_array
        num_remaining_incoming_nodes[global_flux_percentage > 0.] -= 1

        # Now mark the current node as finished
        finished_nodes[row, col] = True

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

    flux_image = np.zeros(rr.shape)
    test_idx = 3
    flux_image[test_idx_row_list[test_idx] - 1:test_idx_row_list[test_idx] + 2,
    test_idx_col_list[test_idx] - 1:test_idx_col_list[test_idx] + 2] = flux_percentages[
        test_idx_row_list[test_idx], test_idx_col_list[test_idx]]
    img = ax[0, 1].imshow(flux_image, origin="lower")
    ax[0, 1].set_title(f"Flux percentage - Cell {test_idx_row_list[test_idx], test_idx_col_list[test_idx]}")
    fig.colorbar(img, label=f"Flux percentage", ax=ax[0, 1])

    img = ax[1, 0].imshow(num_incoming_nodes, origin="lower")
    ax[1, 0].set_title("Nr of incoming nodes")
    fig.colorbar(img, label="# incoming nodes", ax=ax[1, 0])

    #flux_image = np.zeros(rr.shape)
    #test_idx = 2
    #flux_image[test_idx_row_list[test_idx]-1:test_idx_row_list[test_idx]+2,
    #test_idx_col_list[test_idx]-1:test_idx_col_list[test_idx]+2] = flux_percentages[
    #    test_idx_row_list[test_idx], test_idx_col_list[test_idx]]
    img = ax[1,1].imshow(incoming_v_hor, origin="lower")
    ax[1, 1].set_title(f"Incoming flux")
    fig.colorbar(img, label="Incoming Flux [m/s]", ax=ax[1, 1])


    half_cell = 0.5

    for a in ax.flatten():
        a.scatter(shifted_rr, shifted_cc)
        a.quiver(rr, cc, grad_col_norm, grad_row_norm, scale=10)
        # test_idx_row_list = [2, 2]
        # test_idx_col_list = [8, 8 ]
        for test_idx_x, test_idx_y in zip(test_idx_row_list, test_idx_col_list):
            elem_xx = shifted_rr[test_idx_x, test_idx_y]
            elem_yy = shifted_cc[test_idx_x, test_idx_y]
            a.vlines(elem_xx - half_cell, elem_yy - half_cell, elem_yy + half_cell)
            a.vlines(elem_xx + half_cell, elem_yy - half_cell, elem_yy + half_cell)
            a.hlines(elem_yy - half_cell, elem_xx - half_cell, elem_xx + half_cell)
            a.hlines(elem_yy + half_cell, elem_xx - half_cell, elem_xx + half_cell)
        a.set_ylabel("Row")
        a.set_xlabel("Col")
    plt.tight_layout()

    from pathlib import Path
    # out_dir = Path("/home/dane/Documents/PhD/Sabine_overviews/12.07.2022")
    # plt.savefig(out_dir / "corrected_flux_physical.png", dpi=300)
    plt.show()


def create_test_data():
    temp = np.zeros((7, 7))
    grad_row = np.zeros((7, 7))
    grad_col = np.zeros((7, 7))
    v_vertical = np.zeros((7, 7))

    # Throw in some test temps and gradients
    test_idx_row_list = [1, 3, 3, 3]
    test_idx_col_list = [2, 2, 4, 5]
    test_idx_row = 1
    test_idx_col = 2
    temp[test_idx_row, test_idx_col] = 5000
    grad_row[test_idx_row, test_idx_col] = 1
    grad_col[test_idx_row, test_idx_col] = 0
    v_vertical[test_idx_row, test_idx_col] = 1000.
    test_idx_row = 3
    test_idx_col = 2
    temp[test_idx_row, test_idx_col] = 5000
    grad_row[test_idx_row, test_idx_col] = 1
    grad_col[test_idx_row, test_idx_col] = -0.5
    v_vertical[test_idx_row, test_idx_col] = 1000.
    test_idx_row = 3
    test_idx_col = 4
    temp[test_idx_row, test_idx_col] = 5000
    grad_row[test_idx_row, test_idx_col] = 1
    grad_col[test_idx_row, test_idx_col] = -0.5
    v_vertical[test_idx_row, test_idx_col] = 1000.

    test_idx_row = 3
    test_idx_col = 5
    temp[test_idx_row, test_idx_col] = 5000
    grad_row[test_idx_row, test_idx_col] = -1
    grad_col[test_idx_row, test_idx_col] = 1
    v_vertical[test_idx_row, test_idx_col] = 1000.
    return grad_row, grad_col, temp, test_idx_row_list, test_idx_col_list, v_vertical


if __name__ == "__main__":
    test_case()
    # img = np.array([[0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0]])
    # row = 4
    # col = 1
    # img[row, col] = 1
    # plt.imshow(img)
    # plt.show()