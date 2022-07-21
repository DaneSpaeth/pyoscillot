import subprocess
from simulation_controller import SimulationController
from parse_ini import parse_global_ini, parse_ticket
from datasaver import DataSaver
import socket
from datetime import datetime
from theoretical_rvs import theoretical_main
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
        if conf_dict["instrument"].upper() == "ALL":

            subprocess.run(["bash", "run_serval.sh",
                            str(global_dict["datapath_laptop"]),
                            str(global_dict["rvlibpath"]),
                            name, star,
                            "CARMENES_VIS"])

            subprocess.run(["bash", "run_serval.sh",
                            str(global_dict["datapath_laptop"]),
                            str(global_dict["rvlibpath"]),
                            name, star,
                            "HARPS"])
        else:
            subprocess.run(["bash", "run_serval.sh",
                            str(global_dict["datapath_laptop"]),
                            str(global_dict["rvlibpath"]),
                            name, star,
                            conf_dict["instrument"].upper()])

            subprocess.run(["bash", "run_raccoon.sh",
                            str(global_dict["datapath_laptop"]),
                            str(global_dict["rvlibpath"]),
                            name, star,
                            conf_dict["instrument"].upper()])

            outfile = global_dict["rvlibpath"] / "raccoon" / "SIMULATION" / name / "CARMENES_VIS_CCF" / "None.par.dat"
            nzps = read_in_nzps("vis")
            create_correction(outfile, nzps, raccoon=True)

        # Copy the flux and the ticket to the new folders
        saver = DataSaver(name)
        saver.copy_ticket_servalfolder(ticket)
        try:
            saver.copy_flux()
        except FileNotFoundError:
            print("Flux could not be copied!")
            pass

        check_time_series(name, reduction="serval")
        check_time_series(name, reduction="raccoon")




if __name__ == "__main__":

    ticket = "two_spots.ini"
    main(ticket, run_laptop=False)
    # ticket2 = "talk_ticket2.ini"
    # main(ticket2, run_laptop=False)
    # ticket3 = "talk_ticket3.ini"
    # main(ticket3, run_laptop=False)
    # ticket4 = "talk_ticket4.ini"
    # main(ticket4, run_laptop=False)
