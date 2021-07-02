import subprocess
from simulation_controller import SimulationController
from parse_ini import parse_global_ini, parse_ticket
from datasaver import DataSaver
from check_time_series import check_time_series


def main(ticket):
    """ Run a simulation specified in ticket. Run serval. Copy all files
        and plot the result.
    """
    global_dict = parse_global_ini()
    conf_dict = parse_ticket(ticket)

    # Run the Simulation
    SimulationController(ticket)
    # Run Serval
    exit()
    subprocess.run(["bash", "run_serval.sh", str(global_dict["rvlibpath"]),
                    str(conf_dict["name"]), f"HIP{int(conf_dict['hip'])}"])

    # Copy the flux and the ticket to the new folders
    saver = DataSaver(str(conf_dict["name"]))
    saver.copy_ticket(ticket)
    saver.copy_flux()

    check_time_series(str(conf_dict["name"]))


if __name__ == "__main__":
    ticket = "example_ticket.ini"
    main(ticket)
