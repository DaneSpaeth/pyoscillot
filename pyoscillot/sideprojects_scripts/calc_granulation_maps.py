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


def calc_raw_num_nodes_and_fluxes(num_incoming_nodes, num_outgoing_nodes,
                                  row, col, shifted_row, shifted_col, rr, cc,
                                  rr3, cc3):
    """ Calculate the raw or naive number of nodes (outgoing for the local cell, incoming for the neighbors)
        and the flux array for the local cell.

        Naive in the sense that cells pointing at each other are not yet resolved.

        Return array num_incoming_nodes and num_outgoing_nodes
    """
    overlap_areas = np.zeros((3, 3))
    cell_size_row = rr.shape[0]
    cell_size_col = rr.shape[1]
    shifted_row3 = shifted_row[row, col] + cell_size_row
    shifted_col3 = shifted_col[row, col] + cell_size_col

    num_incoming_nodes3 = np.zeros(rr3.shape, dtype=int)

    # To solve the bordering cases create add simulation cells at all borders
    # Then select the local_rr and local_cc 3x3 array from that fat array
    # You have to add one shape of your simulation cells on top
    fat_row = cell_size_row + row
    fat_col = cell_size_col + col
    local_rr = rr3[fat_row - 1: fat_row + 2, fat_col - 1: fat_col + 2]
    local_cc = cc3[fat_row - 1: fat_row + 2, fat_col - 1: fat_col + 2]
    for local_row in range(3):
        for local_col in range(3):
            if local_row == 1 and local_col == 1:
                continue
            overlap_area = calc_overlap_area(shifted_row3,
                                             shifted_col3,
                                             local_rr[local_row, local_col],
                                             local_cc[local_row, local_col])
            overlap_areas[local_row, local_col] = overlap_area
            if overlap_area:
                num_incoming_nodes3[fat_row - 1 + local_row, fat_col - 1 + local_col] += 1
                num_outgoing_nodes[row, col] += 1

    # To avoid dividing by zero
    if np.sum(overlap_areas):
        flux_percentage = overlap_areas / np.sum(overlap_areas)
    else:
        flux_percentage = overlap_areas

    num_incoming_nodes += fold3x3_to_central_cell(num_incoming_nodes3)

    return num_incoming_nodes, num_outgoing_nodes, flux_percentage

def fold3x3_to_central_cell(array3):
    """ Fold a 3x3 array back to the central cell"""
    cell_size_row = int(array3.shape[0]/3)
    cell_size_col = int(array3.shape[1]/3)
    central_array = np.zeros((cell_size_row, cell_size_col), dtype=array3.dtype)
    # Now fold the 3x3 arrays back onto the central cell
    for i in range(3):
        for j in range(3):
            central_array += array3[i * cell_size_row:(i + 1) * cell_size_row,
                                    j * cell_size_col:(j + 1) * cell_size_col]
    return central_array

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

def get_row_col_conflicting_cell(source_row, source_col, flux_percentages):
    """ Return the row and col of the target cell which receives flux from the source cell and returns it
        to the source cell
    """
    source_flux_percentages = flux_percentages[source_row, source_col, :, :]
    for local_row in range(3):
        for local_col in range(3):
            if local_row == 1 and local_col == 1:
                continue
        drow = local_row - 1
        dcol = local_col - 1
        if source_flux_percentages[local_row, local_col]:
            target_row = source_row + drow
            target_col = source_col + dcol
            target_flux_percentage = flux_percentages[target_row, target_col]

            # Dictionary to find the source cell in the 3x3 flux array of the target cell
            map_to_target =   {1: 1,
                               0: 2,
                               2: 0}
            target_to_source_row = map_to_target[local_row]
            target_to_source_col = map_to_target[local_col]

            if target_flux_percentage[target_to_source_row, target_to_source_col]:
                return target_row, target_col


def calc_outgoing_flux(row, col, flux_percentages, incoming_flux):
    """ Calculate a global outgoing flux array for the local cell defined by row and col"""
    simulation_cell_size = flux_percentages.shape[0]
    # Get the local 3x3 flux percentages for the local cell
    local_flux_percentages = flux_percentages[row, col, :, :]
    # Define a fat 3*simulation_cell_size x 3*simulation_cell_size array for the flux percentages
    global_flux_percentage3 = np.zeros((int(simulation_cell_size*3), int(simulation_cell_size*3)))
    # And fill in the small 3x3 array at the right position in the center cell
    global_flux_percentage3[row + simulation_cell_size - 1: row + simulation_cell_size + 2,
                            col + simulation_cell_size - 1: col + simulation_cell_size + 2] = local_flux_percentages
    # Next fold the fat array into the original array of size simulation_cell_size x simulation_cell_size
    global_flux_percentage = fold3x3_to_central_cell(global_flux_percentage3)
    # Calculate the incoming flux for the array

    outgoing_flux = global_flux_percentage * incoming_flux

    return outgoing_flux



def test_case():
    test = 1
    # Define some test arrays
    if test == 1:
        grad_row, grad_col, temp, test_idx_row_list, test_idx_col_list, v_vertical = create_test_data()
        simulation_cell_size = temp.shape[0]
    elif test == 2:

        cc, rr = np.meshgrid(range(simulation_cell_size), range(simulation_cell_size))
        temp = 5000 - 500*cc
        v_vertical = np.ones_like(temp)*1000
        grad_row, grad_col = np.gradient(temp)

        test_idx_row_list = [3]
        test_idx_col_list = [2]
    elif test == 3:
        grad_row, grad_col, temp, test_idx_row_list, test_idx_col_list, v_vertical = create_test_data_over_border()
    else:
        granulation_radiance = granulation_map()
        temperature = radiance_to_temperature(granulation_radiance)
        temp = temperature[992 + 1292]

        # temp = temp[0:10, 0:10]
        simulation_cell_size = temp.shape[0]

        temp3 = np.vstack((np.hstack((temp, temp, temp)),
                           np.hstack((temp, temp, temp)),
                           np.hstack((temp, temp, temp))))

        # temp = temp[40:60, 40:60]

        grad3_row, grad3_col = np.gradient(temp3)
        grad_row = grad3_row[simulation_cell_size:2 * simulation_cell_size,
                   simulation_cell_size:2 * simulation_cell_size]
        grad_col = grad3_col[simulation_cell_size:2 * simulation_cell_size,
                   simulation_cell_size:2 * simulation_cell_size]
        v_vertical = -2000 + 3000 * ((temp - np.min(temp)) / (np.max(temp) - np.min(temp)))
        v_vertical -= np.mean(v_vertical)

        test_idx_row_list = []
        test_idx_col_list = []

    # The gradient e.g. grad_row is intended to be the component of the gradient across the rows
    # i.e. if you have grad_row = 1 and grad_col = 0
    # then your vector should point across the rows but not across the cols

    # Calculate the normalization
    normalization = np.sqrt(np.square(grad_row) + np.square(grad_col))
    normalization[normalization == 0] = 1
    # Normalize and add a minus since the flux should be downhill
    grad_row_norm = -grad_row / normalization
    grad_col_norm = -grad_col / normalization

    # rr and cc are short for rowrow and columncolumn
    # These are 2d arrays containing for each cell the row or column respectively
    cc, rr = np.meshgrid(range(simulation_cell_size), range(simulation_cell_size))

    # Define meshgrids with 3x3 the size but continuously in coordinates (not repetitive)
    cc3, rr3 = np.meshgrid(range(rr.shape[0] * 3), range(rr.shape[1] * 3))

    # Create cc and rr but 3x3 simulation cells added together
    # We need these later for determining the flux for different cells
    # fat_rr = np.vstack((np.hstack((rr, rr, rr)), np.hstack((rr, rr, rr)), np.hstack((rr, rr, rr))))
    # fat_cc = np.vstack((np.hstack((cc, cc, cc)), np.hstack((cc, cc, cc)), np.hstack((cc, cc, cc))))

    shifted_rr = rr + grad_row_norm
    shifted_cc = cc + grad_col_norm

    # We will need some helper arrays
    num_incoming_nodes = np.zeros(rr.shape, dtype=int)
    # num_incoming_nodes3 = np.zeros(rr3.shape, dtype=int)
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
    for row in range(0, simulation_cell_size):
        for col in range(0, simulation_cell_size):

            (num_incoming_nodes,
             num_outgoing_nodes,
             raw_flux_percentages[row, col, :, :]) = calc_raw_num_nodes_and_fluxes(num_incoming_nodes,
                                                                                   num_outgoing_nodes,
                                                                                   row,
                                                                                   col,
                                                                                   shifted_rr,
                                                                                   shifted_cc,
                                                                                   rr,
                                                                                   cc,
                                                                                   rr3,
                                                                                   cc3)

    flux_percentages = raw_flux_percentages.copy()

    # Now correct the flux for cells that influence each other
    trouble_counter = 0
    for row in range(0, simulation_cell_size):
        for col in range(0, simulation_cell_size):
            continue
            local_raw_flux = flux_percentages[row, col, :, :]
            # If the cells does not give any flux away at all continue
            if not local_raw_flux.any():
                continue
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
                    if row + drow < 0:
                        target_row = simulation_cell_size - 1
                    elif row + drow >= simulation_cell_size:
                        target_row = 0
                    else:
                        target_row = row + drow
                    if col + dcol < 0:
                        target_col = simulation_cell_size - 1
                    elif col + dcol >= simulation_cell_size:
                        target_col = 0
                    else:
                        target_col = col + dcol

                    target_raw_flux = flux_percentages[target_row, target_col, :, :]
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

                    nr_neighbors_cell = np.sum(local_raw_flux > 0)
                    nr_neighbors_target = np.sum(target_raw_flux > 0)

                    influx_remaining = None
                    if nr_neighbors_cell > 1 and nr_neighbors_target == 1:
                        influx_remaining = True
                    elif nr_neighbors_target > 1 and nr_neighbors_cell == 1:
                        influx_remaining = False
                    elif nr_neighbors_target > 1 and nr_neighbors_cell > 1:
                        # In the case that there is a stronger flux flowing into the cell than it is giving back to the
                        # same cell
                        if flux_to_source_cell >= local_raw_flux[local_row, local_col]:
                            influx_remaining = True
                        else:
                            influx_remaining = False
                    elif nr_neighbors_target == 1 and nr_neighbors_cell == 1:
                        # TODO solve that case
                        # raise NotImplementedError
                        trouble_counter += 1
                        influx_remaining = False

                    # In the case that there is a stronger flux flowing into the cell than it is giving back to the
                    # same cell
                    # TODO make sure that cells which only have one neighbor are not reduced (in this case you will get NANs)
                    if influx_remaining:
                        # Set the flux into the target cell to 0
                        local_raw_flux[local_row, local_col] = 0
                        # And renormalize, i.e. the flux will be distributed to the other cells
                        local_flux = local_raw_flux / np.sum(local_raw_flux)
                        flux_percentages[row, col, :, :] = local_flux
                        # Reduce the number of incoming nodes
                        num_incoming_nodes[target_row, target_col] -= 1
                        num_outgoing_nodes[row, col] -= 1
                    else:
                        target_raw_flux[relative_row, relative_col] = 0
                        target_flux = target_raw_flux / np.sum(target_raw_flux)
                        # TODO fix that otherwise
                        # One idea would be to slightly alter the vector before that so that the flux in this case goes
                        # to another cell (or do it here and let the flux flow to another cell that is not involved)
                        if np.isnan(target_flux).all():
                            target_flux = np.zeros(target_flux.shape)
                        flux_percentages[target_row, target_col, :, :] = target_flux
                        num_incoming_nodes[row, col] -= 1
                        num_outgoing_nodes[target_row, target_col] -= 1


    print(trouble_counter)
    # exit()


    num_remaining_incoming_nodes = num_incoming_nodes.copy()

    # Now we want to calc the incoming flux for all cells
    # Loop over all cells starting with the ones that have ideally no incoming nodes
    done = False
    counter = 0
    sequence = np.ones(temp.shape)*100
    while not done:
        tmp_remaining_nodes = num_remaining_incoming_nodes.copy()
        tmp_remaining_nodes[finished_nodes] = 1e5
        flattened_idx = np.argsort(tmp_remaining_nodes.flatten())[0]
        # Retrieve the original 2D index
        (row, col) = np.unravel_index(flattened_idx, (simulation_cell_size, simulation_cell_size))

        # Get the number of remaining nodes (it should be 0)
        num_remaining = num_remaining_incoming_nodes[row, col]
        # assert num_remaining == 0, f"{num_remaining} incoming flux vectors remaining"
        if num_remaining != 0:
            # try to detect if it is two cells influencing each other
            # TODO implement the detection
            if num_remaining == 1:
                target_row, target_col = get_row_col_conflicting_cell(row, col, flux_percentages)
                if num_remaining_incoming_nodes[target_row, target_col] != 1:
                    raise ValueError

                # calc the flux from the source to the target in a naive way
                source_incoming_flux = (v_vertical[row, col] + incoming_v_hor[row, col])
                source_outgoing_flux = calc_outgoing_flux(row, col, flux_percentages, source_incoming_flux)

                target_incoming_flux = (v_vertical[target_row, target_col] + incoming_v_hor[target_row, target_col])
                target_outgoing_flux = calc_outgoing_flux(target_row, target_col, flux_percentages, target_incoming_flux)


                # And give that flux to the neighboring cells
                loop_incoming_v_hor = source_outgoing_flux + target_outgoing_flux
                cumulative_v_hor = loop_incoming_v_hor.copy()
                sum_source_incoming_flux = 0
                sum_target_incoming_flux = 0

                ### Possibility
                # fig, ax = plt.subplots(2, 3, figsize=(16,9))
                # for a, n in zip(ax.flatten(), range(9)):
                #     # The plotting
                #     img = a.imshow(cumulative_v_hor, vmin=0, vmax=1000)
                #     a.quiver(cc, rr, grad_col_norm, -grad_row_norm, scale=10)
                #     for r in range(2,5):
                #         for c in range(4,6):
                #             a.text(c, r, round(cumulative_v_hor[r,c]))
                #     a.set_title(f"Step {n}")
                #     fig.colorbar(img, label=f"Incoming Flux [m/s]", ax=a)
                #     source_incoming_flux = loop_incoming_v_hor[row, col]
                #     target_incoming_flux = loop_incoming_v_hor[target_row, target_col]
                #
                #     new_source_incoming_flux = source_incoming_flux #- sum_source_incoming_flux
                #     new_target_incoming_flux = target_incoming_flux #- sum_target_incoming_flux
                #
                #
                #
                #     source_outgoing_flux = calc_outgoing_flux(row,
                #                                               col,
                #                                               flux_percentages,
                #                                               new_source_incoming_flux)
                #     loop_incoming_v_hor[row, col] = 0
                #     target_outgoing_flux = calc_outgoing_flux(target_row,
                #                                               target_col,
                #                                               flux_percentages,
                #                                               new_target_incoming_flux)
                #
                #     sum_source_incoming_flux += new_source_incoming_flux
                #     sum_target_incoming_flux += new_target_incoming_flux
                #     loop_incoming_v_hor[target_row, target_col] = 0
                #     loop_incoming_v_hor = (source_outgoing_flux + target_outgoing_flux)
                #     cumulative_v_hor += loop_incoming_v_hor
                #
                # fig.set_tight_layout(True)
                # fig.suptitle("Possibility 2: Incoming flux is still accounted for in troubling cells - Upper cells get more, but total flux has increased?")
                # from pathlib import Path
                # out_dir = Path("/home/dspaeth/data/simulations/tmp_plots")
                # # plt.savefig(out_dir / "POSSIBILITY_2.png", dpi=300)
                # plt.show()
                # exit()

                ### Possibility 3
                fig, ax = plt.subplots(2, 3, figsize=(16, 9))
                a = ax[0,0]

                img = a.imshow(loop_incoming_v_hor, vmin=0, vmax=1000)
                a.quiver(cc, rr, grad_col_norm, -grad_row_norm, scale=10)
                for r in range(2,5):
                    for c in range(4,6):
                        a.text(c, r, round(loop_incoming_v_hor[r,c]))
                # a.set_title(f"Step {n}")
                fig.colorbar(img, label=f"Incoming Flux [m/s]", ax=a)

                source_incoming_flux = loop_incoming_v_hor[row, col]
                target_incoming_flux = loop_incoming_v_hor[target_row, target_col]

                if source_incoming_flux > target_incoming_flux:
                    source_incoming_flux -= target_incoming_flux
                    target_incoming_flux = 0
                else:
                    target_incoming_flux -= source_incoming_flux
                    source_incoming_flux = 0
                loop_incoming_v_hor[row, col] = source_incoming_flux
                loop_incoming_v_hor[target_row, target_col] = target_incoming_flux

                a = ax[0,1]
                img = a.imshow(loop_incoming_v_hor, vmin=0, vmax=1000)
                a.quiver(cc, rr, grad_col_norm, -grad_row_norm, scale=10)
                for r in range(2, 5):
                    for c in range(4, 6):
                        a.text(c, r, round(loop_incoming_v_hor[r, c]))
                # a.set_title(f"Step {n}")
                fig.colorbar(img, label=f"Incoming Flux [m/s]", ax=a)


                fig.set_tight_layout(True)
                fig.suptitle(
                    "Possibility 3: Calc the difference!")
                from pathlib import Path
                out_dir = Path("/home/dspaeth/data/simulations/tmp_plots")
                plt.savefig(out_dir / "POSSIBILITY_3.png", dpi=300)
                plt.show()
                exit()

                # break




            # try:
            #     # Probably you have catched a loop!
            #     smallest_flux_pct = 1
            #     smallest_fl_idx = 0
            #     for fl_idx in np.argsort(tmp_remaining_nodes.flatten()):
            #         # Find the element with the smallest flux_percentage
            #         _row, _col = np.unravel_index(fl_idx, (simulation_cell_size, simulation_cell_size))
            #         if num_remaining_incoming_nodes[_row, _col] == 0:
            #             continue
            #         local_flux_percentages = flux_percentages[_row, _col, :, :]
            #         nonzero_flux_percentages = local_flux_percentages[local_flux_percentages > 0.]
            #         min_flux_pct = np.min(nonzero_flux_percentages)
            #         if min_flux_pct < smallest_flux_pct:
            #             smallest_flux_pct = min_flux_pct
            #             smallest_fl_idx = fl_idx
            #
            #     # Now that you have the element with the smallest flux percentage
            #     smallest_row, smallest_col = np.unravel_index(smallest_fl_idx, (simulation_cell_size, simulation_cell_size))
            #     local_flux_percentages = flux_percentages[smallest_row, smallest_col, :, :]
            #     local_target_row, local_target_col = np.where(local_flux_percentages == smallest_flux_pct)
            #     local_target_row = local_target_row[0]
            #     local_target_col = local_target_col[0]
            #     global_target_row = smallest_row-1+(local_target_row)
            #     global_target_col = smallest_col-1+(local_target_col)
            #     # Reduce the number of remaining incoming nodes
            #     num_remaining_incoming_nodes[global_target_row, global_target_col] -= 1
            #     num_outgoing_nodes[smallest_row, smallest_col] -= 1
            #     flux_percentages[smallest_row, smallest_col, :, :][flux_percentages[smallest_row, smallest_col, :, :] == smallest_flux_pct] = 0.
            #     flux_percentages[smallest_row, smallest_col, :, :] /= np.sum(flux_percentages[smallest_row, smallest_col, :, :])
            #     continue
            # except:
            #     break
            # print(flux_percentages[row, col, :, :])
            # break
            # troubling_cells[row, col] = True
        sequence[row, col] = counter

        # Calculate the flux that is transported to the different cells
        local_flux_percentages = flux_percentages[row, col, :, :]

        global_flux_percentage3 = np.zeros(rr3.shape)
        global_flux_percentage3[row+simulation_cell_size-1: row+simulation_cell_size+2,
                                col+simulation_cell_size-1: col+simulation_cell_size+2] = local_flux_percentages

        global_flux_percentage = fold3x3_to_central_cell(global_flux_percentage3)

        # global_flux_percentage = np.zeros(rr.shape)
        # global_flux_percentage[row-1:row+2, col-1:col+2] = local_flux_percentages


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

        if finished_nodes.all():
            done = True
            break

        # if counter >= 1e4:
        #     print(f"Break Loop")
        #     break
        counter += 1



    plot_origin = "upper"
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    img = ax[0, 0].imshow(temp, origin=plot_origin, cmap="hot")
    fig.colorbar(img, label="Temp", ax=ax[0, 0])
    ax[0, 0].set_title("Temperature")

    # img = ax[0, 1].imshow(num_remaining_incoming_nodes, origin="lower")
    # fig.colorbar(img, label="Num Remaining Nodes", ax=ax[0, 1])
    # ax[0, 1].set_title("Num Remaining Nodes")
    img = ax[0, 1].imshow(sequence, origin=plot_origin)
    fig.colorbar(img, label="Sequence", ax=ax[0, 1])
    ax[0, 1].set_title("Sequence")


    # test_idx = 0
    # flux_image[test_idx_row_list[test_idx] - 1:test_idx_row_list[test_idx] + 2,
    # test_idx_col_list[test_idx] - 1:test_idx_col_list[test_idx] + 2] = flux_percentages[
    #     test_idx_row_list[test_idx], test_idx_col_list[test_idx]]
    img = ax[1, 0].imshow(v_vertical, origin=plot_origin)
    ax[1, 0].set_title(f"Vertical Velocity, Sum={np.sum(v_vertical)}")
    fig.colorbar(img, label=f"Vertical Velocity", ax=ax[1, 0])

    # img = ax[1, 0].imshow(num_incoming_nodes, origin=plot_origin)
    # ax[1, 0].set_title("Nr of incoming nodes")
    # fig.colorbar(img, label="# incoming nodes", ax=ax[1, 0])

    # img = ax[1, 0].imshow(rr, origin=plot_origin)
    # ax[1, 0].set_title("RowRow")
    # fig.colorbar(img, label="Row", ax=ax[1, 0])

    # flux_image = np.zeros(rr.shape)
    # test_idx = 1
    # flux_image[test_idx_row_list[test_idx]-1:test_idx_row_list[test_idx]+2,
    # test_idx_col_list[test_idx]-1:test_idx_col_list[test_idx]+2] = flux_percentages[
    #    test_idx_row_list[test_idx], test_idx_col_list[test_idx]]
    img = ax[1, 1].imshow(loop_incoming_v_hor, origin=plot_origin)
    ax[1, 1].set_title(f"Incoming flux, Sum={np.sum(loop_incoming_v_hor)}")
    fig.colorbar(img, label=f"Incoming Flux [m/s]", ax=ax[1, 1])


    # img = ax[1, 1].imshow(num_remaining_incoming_nodes, origin=plot_origin)
    # ax[1, 1].set_title(f"Troubling Cells")
    # fig.colorbar(img, label="Num Remaining Incoming Nodes", ax=ax[1, 1])


    half_cell = 0.5

    for a in ax.flatten():
        # Remember: scatter, quiver, vlines and hlines think in x and y but imshow in rows and cols
        # So the col coordinate is corresponding to x, and rows to y
        if test:
            a.scatter(shifted_cc, shifted_rr)
            if plot_origin == "lower":
                a.quiver(cc, rr, grad_col_norm, grad_row_norm, scale=10)
            else:
                a.quiver(cc, rr, grad_col_norm, -grad_row_norm, scale=10)

            # test_idx_row_list = [2, 2]
            # test_idx_col_list = [8, 8 ]
            for test_idx_row, test_idx_col in zip(test_idx_row_list, test_idx_col_list):
                elem_rr = shifted_rr[test_idx_row, test_idx_col]
                elem_cc = shifted_cc[test_idx_row, test_idx_col]

                print(elem_rr, elem_cc)

                a.vlines(elem_cc - half_cell, elem_rr - half_cell, elem_rr + half_cell)
                a.vlines(elem_cc + half_cell, elem_rr - half_cell, elem_rr + half_cell)
                a.hlines(elem_rr - half_cell, elem_cc - half_cell, elem_cc + half_cell)
                a.hlines(elem_rr + half_cell, elem_cc - half_cell, elem_cc + half_cell)
        else:
            quiver_scale = 150
            if plot_origin == "lower":
                a.quiver(cc, rr, grad_col_norm, grad_row_norm, scale=quiver_scale)
            else:
                a.quiver(cc, rr, grad_col_norm, -grad_row_norm, scale=quiver_scale)
        a.set_ylabel("Row")
        a.set_xlabel("Col")
    fig.set_tight_layout(True)

    from pathlib import Path
    out_dir = Path("/home/dspaeth/data/simulations/tmp_plots")
    plt.savefig(out_dir / "SOLUTION_NO_FLUX.png", dpi=300)
    plt.show()


def create_test_data():
    temp = np.zeros((7, 7))
    grad_row = np.zeros((7, 7))
    grad_col = np.zeros((7, 7))
    v_vertical = np.zeros((7, 7))

    # Throw in some test temps and gradients
    test_idx_row_list = []
    test_idx_col_list = []
    # test_idx_row = 1
    # test_idx_col = 2
    # temp[test_idx_row, test_idx_col] = 5000
    # grad_row[test_idx_row, test_idx_col] = 1
    # grad_col[test_idx_row, test_idx_col] = 0
    # v_vertical[test_idx_row, test_idx_col] = 1000.
    # test_idx_row_list.append(test_idx_row)
    # test_idx_col_list.append(test_idx_col)

    # test_idx_row = 3
    # test_idx_col = 2
    # temp[test_idx_row, test_idx_col] = 5000
    # grad_row[test_idx_row, test_idx_col] = 1
    # grad_col[test_idx_row, test_idx_col] = -0.5
    # v_vertical[test_idx_row, test_idx_col] = 1000.
    # test_idx_row_list.append(test_idx_row)
    # test_idx_col_list.append(test_idx_col)

    test_idx_row = 3
    test_idx_col = 4
    temp[test_idx_row, test_idx_col] = 5000
    grad_row[test_idx_row, test_idx_col] = 1
    grad_col[test_idx_row, test_idx_col] = -0.5
    v_vertical[test_idx_row, test_idx_col] = 1000.
    test_idx_row_list.append(test_idx_row)
    test_idx_col_list.append(test_idx_col)

    test_idx_row = 3
    test_idx_col = 5
    temp[test_idx_row, test_idx_col] = 5000
    grad_row[test_idx_row, test_idx_col] = -1
    grad_col[test_idx_row, test_idx_col] = 1
    v_vertical[test_idx_row, test_idx_col] = 1000.
    test_idx_row_list.append(test_idx_row)
    test_idx_col_list.append(test_idx_col)

    # Throw in some test temps and gradients
    # test_idx_row = 3
    # test_idx_col = 6
    # temp[test_idx_row, test_idx_col] = 5000
    # grad_row[test_idx_row, test_idx_col] = -1
    # grad_col[test_idx_row, test_idx_col] = -1
    # v_vertical[test_idx_row, test_idx_col] = 1000.
    # test_idx_row_list.append(test_idx_row)
    # test_idx_col_list.append(test_idx_col)
    return grad_row, grad_col, temp, test_idx_row_list, test_idx_col_list, v_vertical

def create_test_data_over_border():
    temp = np.zeros((7, 7))
    grad_row = np.zeros((7, 7))
    grad_col = np.zeros((7, 7))
    v_vertical = np.zeros((7, 7))

    # Throw in some test temps and gradients
    test_idx_row_list = []
    test_idx_col_list = []
    test_idx_row = 3
    test_idx_col = 3
    temp[test_idx_row, test_idx_col] = 5000
    grad_row[test_idx_row, test_idx_col] = -1
    grad_col[test_idx_row, test_idx_col] = 1
    v_vertical[test_idx_row, test_idx_col] = 1000.

    test_idx_row_list.append(test_idx_row)
    test_idx_col_list.append(test_idx_col)

    test_idx_row = 4
    test_idx_col = 2
    temp[test_idx_row, test_idx_col] = 5000
    grad_row[test_idx_row, test_idx_col] = 1
    grad_col[test_idx_row, test_idx_col] = -0.5
    v_vertical[test_idx_row, test_idx_col] = 1000.

    test_idx_row_list.append(test_idx_row)
    test_idx_col_list.append(test_idx_col)
    return grad_row, grad_col, temp, test_idx_row_list, test_idx_col_list, v_vertical


if __name__ == "__main__":
    test_case()
    exit()
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

    simulation_cell_size = 7
    cc, rr = np.meshgrid(range(simulation_cell_size), range(simulation_cell_size))
    row = 6
    col = 6
    row_shape = rr.shape[0]
    col_shape = rr.shape[1]

    # To solve the bordering cases create add simulation cells at all borders
    # Then select the local_rr and local_cc 3x3 array from that fat array
    fat_rr = np.vstack((np.hstack((rr, rr, rr)), np.hstack((rr, rr, rr)), np.hstack((rr, rr, rr))))
    fat_cc = np.vstack((np.hstack((cc, cc, cc)), np.hstack((cc, cc, cc)), np.hstack((cc, cc, cc))))
    # You have to add one shape of your simulation cells on top
    fat_row = row_shape + row
    fat_col = col_shape + col
    local_rr = fat_rr[fat_row-1: fat_row+2, fat_col-1: fat_col+2]
    local_cc = fat_cc[fat_row-1: fat_row+2, fat_col-1: fat_col+2]

    fig, ax = plt.subplots(1, 2, figsize=(16,9))
    img = ax[0].imshow(local_rr, vmin=0, vmax=6)
    fig.colorbar(img, label="Row", ax=ax[0])

    img = ax[1].imshow(local_cc, vmin=0, vmax=6)
    fig.colorbar(img, label="Col", ax=ax[1])
    fig.set_tight_layout(True)
    fig.suptitle(f"Selected Cell {row, col}")
    for a in ax:
        a.set_ylabel("Row")
        a.set_xlabel("Column")
    plt.show()