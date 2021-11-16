import subprocess
from simulation_controller import SimulationController
from parse_ini import parse_global_ini, parse_ticket
from datasaver import DataSaver
from check_time_series import check_time_series
import socket
from theoretical_rvs import calc_theoretical_results
laptop = socket.gethostname() == "dane-ThinkPad-E460"


def main(ticket, run_laptop=False):
    """ Run a simulation specified in ticket. Run serval. Copy all files
        and plot the result.
    """
    global_dict = parse_global_ini()
    conf_dict = parse_ticket(ticket)
    name = str(conf_dict["name"])

    if not laptop:
        # Run the Simulation
        SimulationController(ticket)
    else:
        if run_laptop:
            # Run the Simulation even if on laptop
            SimulationController(ticket)
        # Run Serval
        # if conf_dict["instrument"].upper() == "ALL":

        #     subprocess.run(["bash", "run_serval.sh", str(global_dict["rvlibpath"]),
        #                     name, f"HIP{int(conf_dict['hip'])}",
        #                     "CARMENES_VIS"])

        #     subprocess.run(["bash", "run_serval.sh", str(global_dict["rvlibpath"]),
        #                     name, f"HIP{int(conf_dict['hip'])}",
        #                     "HARPS"])
        # else:
        #     subprocess.run(["bash", "run_serval.sh", str(global_dict["rvlibpath"]),
        #                     name, f"HIP{int(conf_dict['hip'])}",
        #                     conf_dict["instrument"].upper()])

        # Copy the flux and the ticket to the new folders
        saver = DataSaver(name)
        saver.copy_ticket(ticket)
        saver.copy_flux()

        calc_theoretical_results(name)
        check_time_series(name)


if __name__ == "__main__":
    # ticket = "small_amplitude.ini"
    # ticket = "ngc2423-3_ticket.ini"
    ticket = "hip73620_ticket.ini"
    # ticket = "example_ticket.ini"
    main(ticket, run_laptop=False)
