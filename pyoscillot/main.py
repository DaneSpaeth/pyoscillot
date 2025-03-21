import subprocess
from simulation_controller import SimulationController
from cfg import parse_global_ini, parse_ticket
from datasaver import DataSaver
import socket
from datetime import datetime
from create_nzp_files import create_nzp_file, create_ascii_file
from pathlib import Path

laptop = socket.gethostname() == "dane-ThinkPad-E460"


def main(ticket, run=True, serval=True, raccoon=True, run_laptop=False):
    """ Run a simulation specified in ticket. Run serval. Copy all files
        and plot the result.
    """
    start = datetime.now()

    global_dict = parse_global_ini()
    if laptop:
        global_dict["datapath"] = "/home/dane/Documents/Phd/pyoscillot/mounted_data"
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
    # try:
    #     star = f"HIP{int(conf_dict['hip'])}"
    # except ValueError:
    #     star = conf_dict['hip']
    star = conf_dict.get("star", "ngc4349-127")
        
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
    import time
    
    # ticket = root / "NGC4349_fine_tuning" / "NGC4349_improved_fine_grid_163+10.ini"
    # main(ticket, run=True, serval=True, raccoon=True, run_laptop=False)
    # ticket = root / "NGC4349_fine_tuning" / "NGC4349_improved_fine_grid_163+3.ini"
    # main(ticket, run=True, serval=True, raccoon=True, run_laptop=False)
    # exit()
    root = Path("/home/dspaeth/pyoscillot/pyoscillot/tickets")
    # gridname = "NGC4349_l2_m-2_vp_grid"
    
    
        
        
    # tickets = (sorted(list(grid_folder.rglob(f"{gridname}_*.ini")), reverse=True))
    # print(len(tickets))
    # exit()
    
    # tickets = [Path("/home/dspaeth/pyoscillot/pyoscillot/tickets/NGC4349_p_modes/NGC4349_p_mode_05.ini")]
    # tickets = [Path("/home/dspaeth/pyoscillot/pyoscillot/tickets/NGC4349_very_fine_RV_grid/NGC4349_very_fine_RV_grid_022+02+10.ini")]
    
    
    # tickets = [Path("/home/dspaeth/pyoscillot/pyoscillot/tickets/NGC4349_p_modes/NGC4349_p_mode_04.ini")]
    # second_grid_folder = root / "NGC4349_l1m0_broad_grid"
    # second_tickets = sorted(list(second_grid_folder.glob("NGC4349_*.ini")))
    # tickets = tickets + second_tickets
    # gridname = "NGC4249_l1_dT0"
    # tickets = sorted(list((root / gridname).glob("*.ini")))
    
    # tickets = [grid_folder / "NGC4349_l2_m-2_vp_grid_04+01.ini"]
    
    # tickets = [Path("/home/dspaeth/pyoscillot/pyoscillot/tickets/PhD_parameter_grids/base_ngc4349_127.ini")]
    
    # tickets = [root / "PAPER_NGC4349-127_K-1_phase_full_dT2p5_vp03.ini",
    #            root / "PAPER_NGC4349-127_K-1_phase_full_dT0_vp03.ini"]
    
    # lm_tickets = sorted(list((root / "UPDATED_PAPER_lm_grid").glob("*.ini")))
    
    # tickets = tickets + lm_tickets
    
    # print(tickets)
    # exit()
    
    # tickets = [Path(                                                                                                                                                                                                                                                "/home/dspaeth/pyoscillot/pyoscillot/tickets/UPDATED_PAPER_lm_grid/updated_paper_lm_grid_l1m0_dT0.ini")]
    grid_folder = root / "CARM_param_grids"
    
    # tickets = [root / "CHECK_PHOTON_FLUX_NGC4349-127_K-1_phase_full_dT2p5_vp03.ini"]
    
    tickets = []
    # tickets += list(sorted(list(grid_folder.rglob("CARM_param_grid_vp_05.ini"))))
    # tickets += list(sorted(list(grid_folder.rglob("CARM_param_grid_dT_*.ini"))))
    # tickets += list(sorted(grid_folder.rglob("CARM_param_grid_vrot_*.ini")))
    tickets += list(sorted(list(grid_folder.rglob("CARM_random_grid_*.ini"))))
    # print(len(tickets))
    # exit()
    
    # tickets += list(sorted(grid_folder.rglob("base_ngc4349_127_CARMVIS.ini")))
    # tickets = list(sorted((root / "TICKETS_USED_FOR_PAPER_PHOTON_FLUX").glob("*.ini")))
    
    # tickets = 
    
    # tickets = (grid_folder / "base_ngc4349_127_BOTHCARM.ini",)
    # ticket = grid_folder / "CARM_param_grid_vp" / "CARM_param_grid_vp_04.ini" 
    # main(ticket, run=False, serval=True, raccoon=False)
    # exit()
    
    
    for idx, ticket in enumerate(tickets):
        simname = ticket.name.replace(".ini","")
        if Path(f"/data/dspaeth/pyoscillot_reduced/serval/SIMULATION/{simname}").is_dir():
            print(f"SKIP {simname}")
            continue
        
        # print(f"RUN {simname}")
        # continue
        # if idx < 14:
        #     continue
        print("=======================================")
        print(f"RUN TICKET {ticket.stem}")
        print("=======================================")
        time.sleep(2)
        # if idx <= 13:
        #     print(f"SKIP {ticket}")
        #     continue
        
        # continue
        try:
            main(ticket, run=True, serval=True, raccoon=True, run_laptop=False)
        except Exception as e:
            # raise e
            print(f"TICKET {ticket.stem} failed!")
            continue
        
          
        
    # i = 24
    # ticket = root / "NGC4349-127" / f"test{i}.ini"
    # main(ticket, run=True, serval=True, raccoon=True, run_laptop=False)



