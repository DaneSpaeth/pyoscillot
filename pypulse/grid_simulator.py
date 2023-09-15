from cfg import parse_global_ini, parse_ticket
from pathlib import Path
import numpy as np
from datetime import timedelta, datetime
import configparser

global_dict = parse_global_ini()
TICKETROOT = global_dict["ticketpath"]

#### INSERT YOUR NAMES ETC HERE ####
gridname = "NGC4349_broadeninggrid"
baseticket = TICKETROOT / gridname / "base.ini"
overviewfile = TICKETROOT / gridname / "gridoverview.csv"

#### DEFINE THE RANGES THAT YOU WANT TO SIMULATE ####
# dt = np.arange(21, 24, 1, dtype=int)
v_rot = np.arange(1000, 6500, 500, dtype=int)
v_macro = np.arange(1000, 6500, 500, dtype=int)
v_rots, v_macros = np.meshgrid(v_rot, v_macro)

for idx, (v_rot, v_macro) in enumerate(zip(v_rots.flatten(), v_macros.flatten())):
    idx_plus = idx + 1
    
    config = configparser.ConfigParser()
    config.read(baseticket)
    
    simname = f"{gridname}_{idx_plus}"
    config["GLOBAL"]["name"] = simname
    config["GLOBAL"]["date"] = datetime.today().strftime("%d.%m.%Y")
    config["GLOBAL"]["v_rot"] = str(v_rot)
    config["GLOBAL"]["v_macro"] = str(v_macro)
    
    
    outfile = TICKETROOT / gridname / f"{simname}.ini"
    with open(outfile, "w") as f:
        config.write(f)

est_time = v_rots.size * timedelta(minutes=35)
print(est_time)

