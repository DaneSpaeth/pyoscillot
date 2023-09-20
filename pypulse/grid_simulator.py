from cfg import parse_global_ini, parse_ticket
from pathlib import Path
import numpy as np
from datetime import timedelta, datetime
import configparser

global_dict = parse_global_ini()
TICKETROOT = global_dict["ticketpath"]

#### INSERT YOUR NAMES ETC HERE ####
gridname = "NGC4349_inclinationgrid"
baseticket = TICKETROOT / gridname / "base.ini"
overviewfile = TICKETROOT / gridname / "gridoverview.txt"

#### DEFINE THE RANGES THAT YOU WANT TO SIMULATE ####
# dt = np.arange(21, 24, 1, dtype=int)
v_rot = np.arange(4000, 6000, 500, dtype=int)
v_macro = np.arange(4000, 6000, 500, dtype=int)
inclination = np.arange(15, 90, 15)
v_rots, v_macros, inclinations = np.meshgrid(v_rot, v_macro, inclination)

# Empty the file from last time
with open(overviewfile, "w") as f:
    pass

for idx, (v_rot, v_macro, incl) in enumerate(zip(v_rots.flatten(), v_macros.flatten(), inclinations.flatten())):
    idx_plus = idx + 1
    
    config = configparser.ConfigParser()
    config.read(baseticket)
    
    simname = f"{gridname}_{idx_plus}"
    config["GLOBAL"]["name"] = simname
    config["GLOBAL"]["date"] = datetime.today().strftime("%d.%m.%Y")
    config["GLOBAL"]["v_rot"] = str(v_rot)
    config["GLOBAL"]["v_macro"] = str(v_macro)
    config["GLOBAL"]["inclination"] = str(incl)
    
    
    outfile = TICKETROOT / gridname / f"{simname}.ini"
    with open(outfile, "w") as f:
        config.write(f)
    with open(overviewfile, "a") as f:
        f.write(f"{idx_plus}    {v_rot}    {v_macro}\n")

# Save the settings
est_time = v_rots.size * timedelta(minutes=30)
print(est_time)

