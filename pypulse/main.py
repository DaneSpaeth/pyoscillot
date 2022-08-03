import subprocess
from simulation_controller import SimulationController
from parse_ini import parse_global_ini, parse_ticket
from datasaver import DataSaver
import socket
from datetime import datetime
from theoretical_rvs import theoretical_main
from pathlib import Path
try:
    from check_time_series import check_time_series
except ModuleNotFoundError:
    pass
laptop = socket.gethostname() == "dane-ThinkPad-E460"

import sys
try:
    sys.path.append("/home/dane/Documents/PhD/pyCARM/pyCARM")
    from correct_nzps import create_correction, read_in_nzps
except:
    pass


def main(ticket, run_laptop=False):
    """ Run a simulation specified in ticket. Run serval. Copy all files
        and plot the result.
    """
    start = datetime.now()

    global_dict = parse_global_ini()
    if laptop:
        global_dict["datapath"] = "/home/dane/Documents/Phd/pypulse/mounted_data"
    conf_dict = parse_ticket(ticket)
    name = str(conf_dict["name"])

    saver = DataSaver(name)

    if not laptop:
        # Run the Simulation
        SimulationController(ticket)
        saver.copy_ticket_spectrafolder(ticket)

        # Now also calculate the theoretical results
        # theoretical_main(name)
        stop = datetime.now()
        timedelta = stop - start
        minutes = timedelta.total_seconds() / 60
        seconds = (minutes - int(minutes))*60
        minutes = int(minutes)
        seconds = round(seconds)
        print(f"Program Took {minutes}:{seconds}")

    else:
        if run_laptop:
            # Run the Simulation even if on laptop
            SimulationController(ticket)
            exit()
        # Run Serval
        try:
            star = f"HIP{int(conf_dict['hip'])}"
        except ValueError:
            star = conf_dict['hip']
        instruments = conf_dict["instrument"].upper()
        if instruments == "ALL":

            reduce_CARMENES_VIS(global_dict, name, star)

            reduce_CARMENES_NIR(global_dict, name, star)

            reduce_HARPS(global_dict, name, star)
        elif instruments == "CARMENES":
            reduce_CARMENES_VIS(global_dict, name, star)

            # reduce_CARMENES_NIR(global_dict, name, star)

        elif instruments == "CARMENES_VIS":
            reduce_CARMENES_VIS(global_dict, name, star)

        elif instruments == "CARMENES_NIR":
            reduce_CARMENES_NIR(global_dict, name, star)

        elif instruments == "HARPS":
            reduce_HARPS(global_dict, name, star)

        # Copy the flux and the ticket to the new folders
        saver = DataSaver(name)
        saver.copy_ticket_servalfolder(ticket)
        try:
            saver.copy_flux()
        except FileNotFoundError:
            print("Flux could not be copied!")
            pass

        # check_time_series(name, reduction="serval")
        # check_time_series(name, reduction="raccoon")

def reduce_CARMENES_VIS(global_dict, name, star):
    """ Convenience function to reduce CARMENES_VIS spectra"""
    subprocess.run(["bash", "run_serval.sh",
                    str(global_dict["datapath_laptop"]),
                    str(global_dict["rvlibpath"]),
                    name, star,
                    "CARMENES_VIS"])

    # subprocess.run(["bash", "run_raccoon.sh",
    #                 str(global_dict["datapath_laptop"]),
    #                 str(global_dict["rvlibpath"]),
    #                 name, star,
    #                 "CARMENES_VIS"])
    #
    # # For raccoon also create the csv file
    # # A bit ugly but take the existing nzp correction code
    # # TODO: Refactor at some point
    # outfile = global_dict["rvlibpath"] / "raccoon" / "SIMULATION" / name / "CARMENES_VIS_CCF" / "None.par.dat"
    # nzps = read_in_nzps("vis")
    # create_correction(outfile, nzps, raccoon=True)

def reduce_CARMENES_NIR(global_dict, name, star):
    """ Convenience function to reduce CARMENES_NIR spectra"""
    subprocess.run(["bash", "run_serval.sh",
                    str(global_dict["datapath_laptop"]),
                    str(global_dict["rvlibpath"]),
                    name, star,
                    "CARMENES_NIR"])
    # TODO add raccoon NIR

def reduce_HARPS(global_dict, name, star):
    """ Convenience function to reduce HARPS spectra"""
    subprocess.run(["bash", "run_serval.sh",
                    str(global_dict["datapath_laptop"]),
                    str(global_dict["rvlibpath"]),
                    name, star,
                    "HARPS"])



if __name__ == "__main__":

    root = Path().cwd() / "tickets"
    ticket = root / "EV_Lac_spot_configurations" / "EV_Lac_4SPOTS.ini"

    # for ticket in reversed(tickets):
    #     if ticket.name ==  "EV_Lac_4SPOTS.ini":
    #         continue
    main(ticket, run_laptop=False)

    exit()

