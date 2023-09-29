import subprocess
from simulation_controller import SimulationController
from cfg import parse_global_ini, parse_ticket
from datasaver import DataSaver
import socket
from datetime import datetime
from theoretical_rvs import theoretical_main
from create_nzp_files import create_nzp_file, create_ascii_file
from pathlib import Path
try:
    from check_time_series import check_time_series
except ModuleNotFoundError:
    pass
laptop = socket.gethostname() == "dane-ThinkPad-E460"


def main(ticket, run=True, serval=True, raccoon=True, run_laptop=False):
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
        if run:
            # Run the Simulation
            SimulationController(ticket)
            saver.copy_ticket_spectrafolder(ticket)

            # Now also calculate the theoretical results
            # theoretical_main(name)

    else:
        if run_laptop:
            # Run the Simulation even if on laptop
            SimulationController(ticket)
            exit()

    # Run Reduction
    try:
        star = f"HIP{int(conf_dict['hip'])}"
    except ValueError:
        star = conf_dict['hip']
    instruments = conf_dict["instrument"].upper()
    if instruments == "ALL":
        reduce_CARMENES_VIS(global_dict, name, star, serval=serval, raccoon=raccoon)

        reduce_CARMENES_NIR(global_dict, name, star, serval=serval, raccoon=raccoon)

        reduce_HARPS(global_dict, name, star, serval=serval, raccoon=raccoon)
    elif instruments == "CARMENES":
        reduce_CARMENES_VIS(global_dict, name, star, serval=serval, raccoon=raccoon)

        reduce_CARMENES_NIR(global_dict, name, star, serval=serval, raccoon=raccoon)

    elif instruments == "CARMENES_VIS":
        reduce_CARMENES_VIS(global_dict, name, star, serval=serval, raccoon=raccoon)

    elif instruments == "CARMENES_NIR":
        reduce_CARMENES_NIR(global_dict, name, star, serval=serval, raccoon=raccoon)

    elif instruments == "HARPS":
        reduce_HARPS(global_dict, name, star, serval=serval, raccoon=raccoon)

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
    stop = datetime.now()
    timedelta = stop - start
    minutes = timedelta.total_seconds() / 60
    seconds = (minutes - int(minutes)) * 60
    minutes = int(minutes)
    seconds = round(seconds)
    print(f"Program Took {minutes}:{seconds}")

def reduce_CARMENES_VIS(global_dict, name, star, serval=True, raccoon=True):
    """ Convenience function to reduce CARMENES_VIS spectra"""
    if serval:
        subprocess.run(["bash", "run_serval_srv.sh",
                        str(global_dict["outpath"]),
                        str(global_dict["rvlibpath"]),
                        name, star,
                        "CARMENES_VIS"])
        outfile = global_dict["rvlibpath"] / "serval" / "SIMULATION" / name / "CARMENES_VIS" / f"{name}.rvc.dat"
        csv_file = create_nzp_file(outfile, raccoon=False)
        create_ascii_file(csv_file)

    if raccoon:
        subprocess.run(["bash", "run_raccoon_srv.sh",
                        str(global_dict["outpath"]),
                        str(global_dict["rvlibpath"]),
                        name, star,
                        "CARMENES_VIS"])

        outfile = global_dict["rvlibpath"] / "raccoon" / "SIMULATION" / name / "CARMENES_VIS_CCF" / "None.par.dat"
        csv_file = create_nzp_file(outfile, raccoon=True)
        create_ascii_file(csv_file)


def reduce_CARMENES_NIR(global_dict, name, star, serval=True, raccoon=True):
    """ Convenience function to reduce CARMENES_NIR spectra"""
    if serval:
        subprocess.run(["bash", "run_serval_srv.sh",
                        str(global_dict["outpath"]),
                        str(global_dict["rvlibpath"]),
                        name, star,
                        "CARMENES_NIR"])
        outfile = global_dict["rvlibpath"] / "serval" / "SIMULATION" / name / "CARMENES_NIR" / f"{name}.rvc.dat"
        csv_file = create_nzp_file(outfile, raccoon=False)
        create_ascii_file(csv_file)
    if raccoon:
        subprocess.run(["bash", "run_raccoon_srv.sh",
                        str(global_dict["outpath"]),
                        str(global_dict["rvlibpath"]),
                        name, star,
                        "CARMENES_NIR"])

        outfile = global_dict["rvlibpath"] / "raccoon" / "SIMULATION" / name / "CARMENES_NIR_CCF" / "None.par.dat"
        csv_file = create_nzp_file(outfile, raccoon=True)
        create_ascii_file(csv_file)


def reduce_HARPS(global_dict, name, star, serval=True, raccoon=True):
    """ Convenience function to reduce HARPS spectra"""
    if serval:
        subprocess.run(["bash", "run_serval_srv.sh",
                        str(global_dict["outpath"]),
                        str(global_dict["rvlibpath"]),
                        name, star,
                        "HARPS"])
    if raccoon:
        subprocess.run(["bash", "run_raccoon_srv.sh",
                        str(global_dict["outpath"]),
                        str(global_dict["rvlibpath"]),
                        name, star,
                        "HARPS"])
        outfile = global_dict["rvlibpath"] / "raccoon" / "SIMULATION" / name / "HARPS_pre2015_CCF" / "None.par.dat"
        csv_file = create_nzp_file(outfile, raccoon=True)
        create_ascii_file(csv_file)



if __name__ == "__main__":
    root = Path().cwd() / "tickets"

    # define ticket
    tickets = []
    #for i in range(25, 30):
    #     ticket = root / "NGC4349-127" / f"test{i}.ini"
    #     tickets.append(ticket)
    #         # main(ticket, run=True, serval=True, raccoon=True, run_laptop=False)
    #
    # tickets.append(root / "spots_dT_2/test5.ini")

    # rot_dir = Path(root / "NGC4349_TestRot")
    # tickets = [rot_dir / "test67.ini"]
    # for i in range(222, 223):
    #     ticket = root / "NGC4349_TestMacro" / f"test{i}.ini"
    #     tickets.append(ticket)
    
    # tickets = [root / "test_V_flux.ini"]
    
    grid_folder = root / "NGC4349_blindsearch_grid"
    tickets = sorted(list(grid_folder.glob("*_blindsearch_grid_*.ini")))

    for ticket in tickets:
        try:
            main(ticket, run=True, serval=True, raccoon=True, run_laptop=False)
        except:
            continue
        
    # i = 24
    # ticket = root / "NGC4349-127" / f"test{i}.ini"
    # main(ticket, run=True, serval=True, raccoon=True, run_laptop=False)



